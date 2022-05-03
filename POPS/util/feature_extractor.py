# -*- coding: UTF-8 -*-

# from preprocessing.invdisttree import Invdisttree
# from util.raster_util import assign_pts_to_cells, convert_index_single_to_multi
from util.util import last_integer, next_integer, percentile
from util.cuda_util import *
from util.var_util import INVALID
import os
import pycuda.driver as cuda
import pycuda.autoinit
import subprocess
from copy import deepcopy
# from util.test_util import *
from feature_extraction.extract_pca_feat import *
import scipy.stats as stats
import cupy
from util.test_util import *
from time import perf_counter


# 所有的trim_brink一律为假！
# 对于divide_z的情况，忽略为空的格子，对不等长的features做aggregation（用单个维度以及所有维度的统计量）
class PtCloudFeatExtractor(object):

    core_met_info_map = {
        'PCA_ISPRS12': {'core_id': 0, 'feat_ndim': 2, 'post_processing': False},
        'STAT': {'core_id': 1, 'feat_ndim': 8, 'post_processing': True},
    }

    # 待检查
    # example的scale允许x, y不一致，但是feat不允许; example不可以有重叠部分，但是feat可以
    def __init__(self, ex_scale_x, ex_scale_y, feat_scales, feat_xpts_per_step_by_scale=None,
                 feat_ypts_per_step_by_scale=None, feat_xstep_to_scale_ratio=0.5, feat_ystep_to_scale_ratio=0.5,
                 threads_per_block=None):
        if threads_per_block is not None and threads_per_block > 0 and threads_per_block % 16 != 0:
            print("Error! threads_per_block is not divisible by half warp size!")
            exit(1)

        self.threads_per_block = threads_per_block
        self.ex_scale_x = ex_scale_x
        self.ex_scale_y = ex_scale_y
        self.feat_xpts_per_step_by_scale = feat_xpts_per_step_by_scale
        self.feat_ypts_per_step_by_scale = feat_ypts_per_step_by_scale
        self.feat_xstep_to_scale_ratio = feat_xstep_to_scale_ratio
        self.feat_ystep_to_scale_ratio = feat_ystep_to_scale_ratio
        try:
            self.sorted_feat_scales = np.sort(feat_scales)  # 注意一定是sort过的
        except:  # 标量
            self.sorted_feat_scales = np.array([feat_scales])

        os.chdir('/users/pss/Liid/Pycharm_projects/PARKER/c_and_cuda_code')
        subprocess.run(
            ['/usr/local/cuda-11.2/bin/nvcc', '--cubin', '-arch=sm_75', '--disable-warnings', 'extractFeats.cu'])

    def _core_wrapper(self, core, args):
        func = eval(f'self._{core}')
        if core == 'PCA_ISPRS12':
            return func(args['sub_x'], args['sub_y'], args['sub_z'])
        elif core == 'STAT':
            return func(args['sub_z'])

    def _post_core_wrapper(self, core, args):
        raw_feats = args['raw_feats']
        if core == 'PCA_ISPRS12':
            return raw_feats, PtCloudFeatExtractor.core_met_info_map[core]['feat_ndim']

        func = eval(f'self._post_{core}')
        if core == 'STAT':
            return func(raw_feats, args['std_norm_z'])

    @staticmethod
    def _PCA_ISPRS12(sub_x, sub_y, sub_z):
        sub_xyz = np.c_[sub_x, sub_y, sub_z]
        comp, _ = np.linalg.eig(np.matmul(sub_xyz.T, sub_xyz) / (len(sub_x) - 1))
        comp = -np.sort(-comp / np.sum(comp))[np.newaxis, :]
        return estimate_ternary_coord2d(comp)

    @staticmethod
    def _STAT(sub_z):
        return np.array([
            np.std(sub_z), stats.skew(sub_z), stats.kurtosis(sub_z), np.min(sub_z),
            percentile(sub_z, 0.25), percentile(sub_z, 0.5),
            percentile(sub_z, 0.75), np.max(sub_z)
        ])

    @staticmethod
    def _post_STAT(raw_feats, std_norm_z):
        feat_ndim = PtCloudFeatExtractor.core_met_info_map['STAT']['feat_ndim']
        if std_norm_z:
            valid_inds = np.array([i for i in range(raw_feats.shape[1]) if i % feat_ndim != 0])
            raw_feats = raw_feats[:, valid_inds]
            feat_ndim -= 1

        return raw_feats, feat_ndim

    # 待检查：只针对train和test之一来做（roi已经确定了）
    # 不divide_z！也不在这里aggregate
    # 这个n_ex_per_batch的选项只是方便测试的，不过也没多大影响，留着吧
    def safe_batch_extract_feats_gridded(self, ori_z, resol_x, resol_y, is_volumetric,
                                         std_norm_xy=True, std_norm_z=True, ny=None, num_valid_th=10,
                                         n_ex_per_batch=None, core='PCA_ISPRS12', use_incre_shmem=True,
                                         use_multi_scale=True, rand_state=1, sh_nx=2e9, sh_ny=2e9,
                                         timing_only=False, time_out=float('inf')):  # 注意这里num_valid_th默认是2e9而不是float('inf')，是因为gpu里没法放这么大的值

        tic = perf_counter()

        feat_ndim, core_id = self.core_met_info_map[core]['feat_ndim'], self.core_met_info_map[core]['core_id']
        use_gpu = self.threads_per_block is not None

        z = deepcopy(ori_z) if ori_z.ndim == 2 else deepcopy(ori_z.reshape((-1, ny)))
        n_feat_scales, nx, ny = len(self.sorted_feat_scales), z.shape[0], z.shape[1]
        ex_nx, ex_ny = map(int, (self.ex_scale_x // resol_x, self.ex_scale_y // resol_y))
        ex_nv = ex_nx * ex_ny
        n_ex_x, n_ex_y = map(int, (nx // ex_nx, ny // ex_ny))
        if n_ex_x * n_ex_y == 0:
            print('Error. ROI size too small. Cannot extract examples!')
            exit(1)
        nx, ny = n_ex_x * ex_nx, n_ex_y * ex_ny
        z = np.float32(z[:nx, :ny])  # 裁掉边角，并转为np.float32（为了让gpu和cpu结果一致）

        feat_xpts_per_step_by_scale = np.maximum(1, [np.round(f_scale * self.feat_xstep_to_scale_ratio / resol_x)
                                                     for f_scale in self.sorted_feat_scales]) \
            if self.feat_xpts_per_step_by_scale is None else self.feat_xpts_per_step_by_scale
        feat_ypts_per_step_by_scale = np.maximum(1, [np.round(f_scale * self.feat_ystep_to_scale_ratio / resol_y)
                                                     for f_scale in self.sorted_feat_scales]) \
            if self.feat_ypts_per_step_by_scale is None else self.feat_ypts_per_step_by_scale
        feat_xpts_per_step_by_scale, feat_ypts_per_step_by_scale = \
            map(np.int32, (feat_xpts_per_step_by_scale, feat_ypts_per_step_by_scale))

        # ex_scale_x, ex_scale_y = ex_nx * resol_x, ex_ny * resol_y
        n_sub_x_by_scale, n_sub_y_by_scale = np.empty(n_feat_scales, dtype=int), np.empty(n_feat_scales, dtype=int)

        # feat_xpts_per_step_by_scale, feat_ypts_per_step_by_scale = \
        #     np.empty(n_feat_scales, dtype=int), np.empty(n_feat_scales, dtype=int)
        feat_nx_by_scale, feat_ny_by_scale = \
            np.empty(n_feat_scales, dtype=int), np.empty(n_feat_scales, dtype=int)
        cum_num_sub_by_scale = np.zeros(n_feat_scales + 1, dtype=int)
        # n_sub_x_by_scale, n_sub_y_by_scale = np.zeros(n_feat_scales, dtype=int), np.zeros(n_feat_scales, dtype=int)

        if last_integer(self.sorted_feat_scales[0] / max(resol_x, resol_y), 0) == 0:
            print('Error! Feature scale too small!')
            exit(1)
        if last_integer(self.sorted_feat_scales[-1] / resol_x, 0) > ex_nx \
                or last_integer(self.sorted_feat_scales[-1] / resol_y, 0) > ex_ny:
            print('Error! Feature scale too large!')
            exit(1)

        # to do: get rid of the for loop. This can be done more pythonically!
        for i, f_scale in enumerate(self.sorted_feat_scales):

            feat_nx_by_scale[i], feat_ny_by_scale[i] = \
                last_integer(f_scale / resol_x, 1), last_integer(f_scale / resol_y, 1)

            n_sub_x_by_scale[i] = 1 + last_integer((ex_nx - feat_nx_by_scale[i]) / feat_xpts_per_step_by_scale[i], 1)
            n_sub_y_by_scale[i] = 1 + last_integer((ex_ny - feat_ny_by_scale[i]) / feat_ypts_per_step_by_scale[i], 1)
            cum_num_sub_by_scale[i + 1] = cum_num_sub_by_scale[i] + n_sub_x_by_scale[i] * n_sub_y_by_scale[i]

        n_ex, n_feats_per_ex = n_ex_x * n_ex_y, cum_num_sub_by_scale[-1] * feat_ndim
        if not timing_only:
            raw_feats = 2e9 * np.ones(n_ex * n_feats_per_ex, dtype=float)  # 存放全部raw feats

        # raw feature extraction (and maybe aggregation)
        if use_gpu:

            d_sorted_feat_scales = safe_cuda_memcpy_htod(np.float32(self.sorted_feat_scales))
            d_feat_nx_by_scale = safe_cuda_memcpy_htod(np.int32(feat_nx_by_scale))
            d_feat_ny_by_scale = safe_cuda_memcpy_htod(np.int32(feat_ny_by_scale))
            d_feat_xpts_per_step_by_scale = safe_cuda_memcpy_htod(np.int32(feat_xpts_per_step_by_scale))
            d_feat_ypts_per_step_by_scale = safe_cuda_memcpy_htod(np.int32(feat_ypts_per_step_by_scale))
            max_float_sh = get_max_shmem_bytes_per_block() // 4
            shared = max_float_sh * 4

            max_n_ex_per_batch = int(np.min([MAX_BYTES // (4 * n_feats_per_ex), MAX_BYTES // (4 * ex_nv), n_ex]))
            n_ex_per_batch = max_n_ex_per_batch if n_ex_per_batch is None else min(n_ex_per_batch, max_n_ex_per_batch)
            d_z = safe_cuda_malloc(4 * n_ex_per_batch * ex_nv)
            h_z = np.empty(n_ex_per_batch * ex_nv, dtype=np.float32)
            if not timing_only:
                h_feats = 2e9 * np.ones(n_ex_per_batch * n_feats_per_ex, dtype=np.float32)  # 预置为INF
            # d_feats = safe_cuda_memcpy_htod(h_feats)
            d_feats = safe_cuda_malloc(n_ex_per_batch * n_feats_per_ex * 4)

            d_n_feat_scales, d_ny, d_ex_nx, d_ex_ny, d_is_volumetric, d_std_norm_xy, d_std_norm_z, \
                d_max_float_sh, d_num_valid_th, d_core_id, d_feat_ndim, d_use_incre_shmem, d_use_multi_scale = \
                map(np.int32, (n_feat_scales, ny, ex_nx, ex_ny, is_volumetric,
                               std_norm_xy, std_norm_z, max_float_sh, num_valid_th, core_id, feat_ndim,
                               use_incre_shmem, use_multi_scale))
            d_resol_x, d_resol_y = map(np.float32, (resol_x, resol_y))
            mod = pycuda.driver.module_from_file \
                ("/users/pss/Liid/Pycharm_projects/PARKER/c_and_cuda_code/extractFeats.cubin")  # 直接粘下来的
            func = mod.get_function("extractFeats_gridded")

            cnt, start_ex = 0, 0
            while cnt < n_ex:
                # 以下完全按照interpolation那里的逻辑照着改的
                print(f'Extracting raw features for examples on GPU. #remaining = {n_ex - cnt}...')
                finish_ex = min(start_ex + n_ex_per_batch, n_ex)
                start_ex_x, finish_ex_x, = start_ex // n_ex_y, (finish_ex - 1) // n_ex_y + 1
                if start_ex_x == finish_ex_x - 1:  # 当前涉及的所有样本在一条直线上
                    start_ex_y, finish_ex_y = start_ex % n_ex_y, (finish_ex - 1) % n_ex_y + 1
                    offset_ex_y = 0
                else:  # 当前涉及的所有样本进入了下一列
                    start_ex_y, finish_ex_y = 0, n_ex_y
                    offset_ex_y = start_ex % n_ex_y

                start_x, finish_x, start_y, finish_y = \
                    start_ex_x * ex_nx, min(finish_ex_x * ex_nx, nx), start_ex_y * ex_ny, min(finish_ex_y * ex_ny, ny)
                z_, nx_ = z[start_x: finish_x, start_y: finish_y], finish_x - start_x

                # 处理第一列
                finish_x, start_y = min(ex_nx, nx_), offset_ex_y * ex_ny
                cur_z = z_[:finish_x, start_y:].flatten()

                n_ex_col = finish_ex_x - start_ex_x
                n_ex_last_col = (finish_ex - start_ex) - (n_ex_y * (n_ex_col - 1) - offset_ex_y)
                if n_ex_col >= 3:  # 至少有三列样本
                    start_x, finish_x = ex_nx, (n_ex_col - 1) * ex_nx
                    cur_z = np.concatenate((cur_z, z_[start_x: finish_x].flatten()))
                    start_x, finish_y = finish_x, min(ny, n_ex_last_col * ex_ny)
                    cur_z = np.concatenate((cur_z, z_[start_x:, :finish_y].flatten()))
                elif n_ex_col == 2:  # 有两列样本
                    start_x, finish_y = ex_nx, min(ny, n_ex_last_col * ex_ny)
                    cur_z = np.concatenate((cur_z, z_[start_x:, :finish_y].flatten()))
                h_z[: len(cur_z)] = deepcopy(cur_z.astype(np.float32))
                cuda.memcpy_htod(d_z, h_z)

                grid_size = finish_ex - start_ex
                np.random.seed(rand_state)
                rand_props = np.random.rand(grid_size * self.threads_per_block).astype(np.float32)
                d_rand_props = safe_cuda_memcpy_htod(rand_props)

                # call feature extraction kernel
                func(d_feats, d_z, d_sorted_feat_scales, d_feat_nx_by_scale, d_feat_ny_by_scale,
                     d_feat_xpts_per_step_by_scale, d_feat_ypts_per_step_by_scale, d_rand_props, d_n_feat_scales, d_ny,
                     d_resol_x, d_resol_y, d_ex_nx, d_ex_ny, d_max_float_sh, d_is_volumetric, d_std_norm_xy,
                     d_std_norm_z, np.int32(cnt), d_core_id, d_feat_ndim, d_num_valid_th, np.int32(sh_nx),
                     np.int32(sh_ny), d_use_incre_shmem, d_use_multi_scale,
                     shared=shared, grid=(finish_ex - start_ex, 1, 1), block=(self.threads_per_block, 1, 1))
                if not timing_only:
                    cuda.memcpy_dtoh(h_feats, d_feats)
                    start, finish = cnt * n_feats_per_ex, min(cnt + n_ex_per_batch, n_ex) * n_feats_per_ex
                    raw_feats[start: finish] = h_feats[: finish - start].astype(float)

                cnt, start_ex = cnt + n_ex_per_batch, finish_ex  # 最后一轮的时候，cnt可以超过n_ex，没关系的

                if perf_counter() - tic >= time_out:
                    return None
        else:
            # raw_feats是二维的，每行一个样本，每行按（scale -> x_coordinates -> y_coordinates） -> dim的顺序排
            cnt_feat = 0
            for i_ex_x in range(n_ex_x):  # x轴上每一个样本
                ix = ex_nx * i_ex_x
                for i_ex_y in range(n_ex_y):  # y轴上每一个样本
                    # print(f'Extracting raw features of example {i_ex_x * n_ex_y + i_ex_y + 1}/{n_ex} on CPU...')
                    iy = ex_ny * i_ex_y
                    cur_z = z[ix: ix + ex_nx, iy: iy + ex_ny]  # 当前样本
                    for i, f_scale in enumerate(self.sorted_feat_scales):  # 每一个scale

                        feat_nx, feat_ny = feat_nx_by_scale[i], feat_ny_by_scale[i]
                        n_sub_x, n_sub_y = n_sub_x_by_scale[i], n_sub_y_by_scale[i]

                        if not is_volumetric:  # ball query
                            dist_th = f_scale / 2  # f_scale是直径
                            ic_x, ic_y = last_integer(feat_nx / 2, 0), last_integer(feat_ny / 2, 0)

                        for i_sub_x in range(n_sub_x):  # 每一个x_coordinate
                            cur_ix = int(feat_xpts_per_step_by_scale[i] * i_sub_x)
                            sub_x_1d = np.array(range(cur_ix, cur_ix + feat_nx)) * resol_x  # 相对于样本(而非sub-cube)左下角的偏移量，和GPU代码一致，方便检查
                            if not is_volumetric:
                                xc = sub_x_1d[ic_x]  # 中心点的x坐标

                            for i_sub_y in range(n_sub_y):
                                cur_iy = int(feat_ypts_per_step_by_scale[i] * i_sub_y)
                                sub_y_1d = np.array(range(cur_iy, cur_iy + feat_ny)) * resol_y
                                sub_x, sub_y = np.meshgrid(sub_x_1d, sub_y_1d,
                                                           indexing='ij')  # 当前sub_cube的x和y值。由于后面还要做noramlization，没必要考虑偏移量
                                sub_z = cur_z[cur_ix: cur_ix + feat_nx, cur_iy: cur_iy + feat_ny]  # 当前sub_cube的z值
                                # print(sub_z.shape, "xxx", cur_iy, feat_ny, ex_ny)
                                if not is_volumetric:
                                    yc, zc = sub_y_1d[ic_y], sub_z[ic_x, ic_y]  # 中心点的y和z坐标
                                sub_x, sub_y, sub_z = sub_x.flatten(), sub_y.flatten(), sub_z.flatten()

                                i_valid = np.where(~np.isnan(sub_z))
                                if len(i_valid) != 0:
                                    sub_x, sub_y, sub_z = sub_x[i_valid], sub_y[i_valid], sub_z[i_valid]
                                    # print('!!! Not all valid !!!')
                                else:
                                    sub_x, sub_y, sub_z = [], [], []
                                # print(sub_x)

                                # 这一步一定在normalization前面！
                                if not is_volumetric:
                                    if np.isnan(zc):
                                        sub_x, sub_y, sub_z = [], [], []
                                    else:
                                        dists = np.sqrt((sub_x - xc) ** 2 +
                                                        (sub_y - yc) ** 2 +
                                                        (sub_z - zc) ** 2)
                                        i_valid = np.where(dists < dist_th)[0]
                                        # if cnt_feat == 0:
                                        #     exec(gen_cmd_print_variables('xc, yc, zc, sub_x, sub_y, sub_z'))
                                        #     exec(gen_cmd_print_variables('dists, len(dists), i_valid'))
                                        sub_x, sub_y, sub_z = sub_x[i_valid], sub_y[i_valid], sub_z[i_valid]

                                # print(len(sub_x), num_valid_th, "******")
                                if len(sub_x) >= num_valid_th:
                                    sub_x, sub_y, sub_z = map(np.float64, (sub_x, sub_y, sub_z))

                                    # # 测试代码，记得删掉
                                    # mat = np.c_[sub_x, sub_y, sub_z]

                                    sub_x -= np.mean(sub_x)  # 一律中心化
                                    sub_y -= np.mean(sub_y)
                                    sub_z -= np.mean(sub_z)
                                    if std_norm_xy:
                                        sub_x /= (np.std(sub_x) if np.std(sub_x) > 0 else 1)
                                        sub_y /= (np.std(sub_y) if np.std(sub_y) > 0 else 1)
                                    if std_norm_z:
                                        sub_z /= (np.std(sub_z) if np.std(sub_z) > 0 else 1)

                                    # core method
                                    if not timing_only:
                                        raw_feats[cnt_feat: cnt_feat + feat_ndim] = self._core_wrapper(core, {
                                            'sub_x': sub_x, 'sub_y': sub_y, 'sub_z': sub_z
                                        })
                                    else:
                                        self._core_wrapper(core, {
                                            'sub_x': sub_x, 'sub_y': sub_y, 'sub_z': sub_z
                                        })


                                    # tgtInd = 0
                                    # if cnt_feat <= tgtInd < cnt_feat + feat_ndim:
                                    #     exec(gen_cmd_print_variables(
                                    #         'mat'
                                    #     ))
                                    #     # exec(gen_cmd_print_variables(
                                    #     #     'i_ex_x, i_ex_y, f_scale, i_sub_x, i_sub_y, i_ex_x * n_ex_y + i_ex_y, '
                                    #     #     'i_sub_x * n_sub_y + i_sub_y'))
                                    #     # exec(gen_cmd_print_variables('np.sum(mat[:,2]), np.sum(mat[:,2]**2), np.sum(mat[:,2]**3), np.sum(mat[:,2]**4)'))
                                    #     # exec(gen_cmd_print_variables(
                                    #     #     'np.mean(mat[:,2]), np.std(mat[:,2]), np.sum(sub_z), np.sum(sub_z**2), np.sum(sub_z**3), np.sum(sub_z**4)'))
                                    #
                                    #     q1_nn, median_nn, q3_nn = percentile(mat[:,2], 0.25), \
                                    #                      percentile(mat[:,2], 0.5), \
                                    #                      percentile(mat[:,2], 0.75)
                                    #     # exec(gen_cmd_print_variables(
                                    #     #     'np.min(mat[:,2]), q1_nn, median_nn, q3_nn, np.max(mat[:,2])'))
                                    #
                                    #
                                    #     # q1, median, q3 = percentile(sub_z, 0.25), percentile(sub_z, 0.5), percentile(sub_z, 0.75)
                                    #     # exec(gen_cmd_print_variables(
                                    #     #     'stats.skew(sub_z), stats.kurtosis(sub_z), '
                                    #     #     'np.min(sub_z), q1, median, q3, np.max(sub_z)'))
                                    #     # sum4 = np.sum(sub_z**4)
                                    #     mat2 = np.float64(mat[:, 2])
                                    #     # sum4_ = np.sum(mat2**4)
                                    #     # sum4_ += -4 * np.sum(mat2**3) * np.mean(mat2) + \
                                    #     #         6 * np.sum(mat2**2) * (np.mean(mat2) ** 2) - 4 * np.sum(mat2) * (np.mean(mat2) ** 3) + \
                                    #     #         len(mat2) * (np.mean(mat2) ** 4)
                                    #     # print(f'sum4_ = {np.sum(mat2**4)} -{4 * np.sum(mat2**3) * np.mean(mat2)} + '
                                    #     #       f'{6 * np.sum(mat2**2) * (np.mean(mat2) ** 2)} - {4 * np.sum(mat2) * (np.mean(mat2) ** 3)} + '
                                    #     #       f'{len(mat2) * (np.mean(mat2) ** 4)} = {sum4_}')
                                    #     Sum = np.sum(sub_z)
                                    #     sum2 = np.sum(sub_z**2)
                                    #     sum3 = np.sum(sub_z**3)
                                    #     sum4 = np.sum(sub_z**4)
                                    #
                                    #     n = len(mat2)
                                    #     avg_ = np.mean(mat2)
                                    #     stdev_ = np.std(mat2)
                                    #     sum_ = np.sum(mat2)
                                    #     sum2_ = np.sum(mat2**2)
                                    #     sum3_ = np.sum(mat2 ** 3)
                                    #     sum4_ = np.sum(mat2 ** 4)
                                    #     avg = 0
                                    #     stdev = np.std(sub_z)
                                    #     exec(gen_cmd_print_variables('avg, stdev, avg_, stdev_'))
                                    #     exec(gen_cmd_print_variables('sum_, sum2_, sum3_, sum4_'))
                                    #     # sum_m = 0
                                    #     # sum2_m = np.sum(mat2**2) - 2 * avg_ * np.sum(mat2) + n * (avg_ ** 2)
                                    #     # sum3_m = np.sum(mat2**3) - 3 * np.sum(mat2**2) * avg_ + 3 * np.sum(mat2) * (avg_**2) - n * (avg_ ** 3)
                                    #     # sum4_m = np.sum(mat2**4) - 4 * np.sum(mat2**3) * avg_ + 6 * np.sum(mat2**2) * (avg_**2) - 4 * np.sum(mat2) * (avg_**3) + n * (avg_ ** 4)
                                    #     # exec(gen_cmd_print_variables('sum_m, sum2_m, sum3_m, sum4_m'))
                                    #     #
                                    #     # sum_ = 0
                                    #     # sum2_ = sum2_m / (stdev_ ** 2)
                                    #     # sum3_ = sum3_m / (stdev_ ** 3)
                                    #     # sum4_ = sum4_m / (stdev_ ** 4)
                                    #     # exec(gen_cmd_print_variables('sum_, sum2_, sum3_, sum4_'))
                                    #     #
                                    #
                                    #     avg = 0
                                    #     stdev = np.std(sub_z)
                                    #     skew = (sum3 - 3 * avg * sum2 + 3 * (avg**2) * Sum - n * (avg**3)) / (n * (stdev**3))
                                    #     kurt = (sum4 - 4 * avg * sum3 + 6 * (avg**2) * sum2 - 4 * (avg**3) * Sum + n * (avg**4)) / (n * (stdev ** 4)) - 3
                                    #     skew_ = (sum3_ - 3 * avg_ * sum2_ + 3 * (avg_**2) * sum_ - n * (avg_**3)) / (n * (stdev_**3))
                                    #     kurt_ = (sum4_ - 4 * avg_ * sum3_ + 6 * (avg_ ** 2) * sum2_ - 4 * (
                                    #             avg_ ** 3) * sum_ + n * (avg_ ** 4)) / (n * (stdev_ ** 4)) - 3
                                    #     exec(gen_cmd_print_variables(
                                    #         'Sum, avg_, stdev_, sum_, skew, kurt, skew_, kurt_'
                                    #     ))
                                    #     print(f'skew = ({sum3} - {3 * avg * sum2} + {3 * (avg**2) * Sum} - {n * (avg**3)} / {(n * stdev**3)} = {skew}')
                                    #     print(f'skew_ = ({sum3_} - {3 * avg_ * sum2_} + {3 * (avg_ ** 2) * sum_} - {n * (avg_ ** 3)}) / {(n * (stdev_ ** 3))} = {skew_}')
                                    #     print(f'kurt = ({sum4} - {4 * avg * sum3} + {6 * (avg**2) * sum2} - {4 * (avg**3) * Sum} + {n * (avg**4)}) / {(n * (stdev ** 4))} - 3 = {kurt}')
                                    #     print(f'kurt_ = ({sum4_} - {4 * avg_ * sum3_} + {6 * (avg_ ** 2) * sum2_} - {4 * (avg_ ** 3) * sum_} + {n * (avg_ ** 4)}) / {(n * (stdev_ ** 4))} - 3 = {kurt_}')
                                else:
                                    # tgtInd = 1330
                                    # mat = np.c_[sub_x, sub_y, sub_z]
                                    # if cnt_feat <= tgtInd < cnt_feat + feat_ndim:
                                    #     exec(gen_cmd_print_variables(
                                    #         'mat'
                                    #     ))
                                    #     exec(gen_cmd_print_variables(
                                    #         'i_ex_x, i_ex_y, f_scale, i_sub_x, i_sub_y, i_ex_x * n_ex_y + i_ex_y, '
                                    #         'i_sub_x * n_sub_y + i_sub_y'))
                                    if not timing_only:
                                        raw_feats[cnt_feat: cnt_feat + feat_ndim] = np.array([INVALID] * feat_ndim)
                                    # print('!!! Totally invalid !!!')
                                cnt_feat += feat_ndim
                                if perf_counter() - tic >= time_out:
                                    return None
            if not timing_only:
                raw_feats = raw_feats.reshape((n_ex, -1))

        if use_gpu and not timing_only:
            raw_feats = raw_feats.reshape((n_ex, feat_ndim, -1))  # 样本 -> dim -> (scale -> x_coordinates -> y_coordinates)
            raw_feats = np.swapaxes(raw_feats, 1, 2).reshape((n_ex, -1))  # 换dim和(scale -> x_coordinates -> y_coordinates)轴，然后reshape即可

        # for i_ex in range(len(raw_feats)):
        #
        #     # for cnt_feat in range(0, raw_feats.shape[1], 2):
        #     #     print(raw_feats[i_ex][int(cnt_feat / 2)], raw_feats[i_ex][int(cnt_feat / 2 + 1)])
        #     # print()
        #
        #     cnt_feat = 0
        #     while cnt_feat != len(raw_feats[i_ex]):
        #
        #         if raw_feats[i_ex][cnt_feat] + raw_feats[i_ex][cnt_feat + 1] > 1:
        #             exec(gen_cmd_print_variables('i_ex, cnt_feat, '
        #                                          'raw_feats[i_ex][cnt_feat], raw_feats[i_ex][cnt_feat + 1]'))
        #             input("Paused")
        #
        #         cnt_feat += 2

        if not timing_only:
            # 记得一定要加回来！
            raw_feats, feat_ndim = self._post_core_wrapper(core, {'raw_feats': raw_feats, 'std_norm_z': std_norm_z})

            return raw_feats, cum_num_sub_by_scale, n_feat_scales, n_ex_x, n_ex_y, n_ex, feat_ndim
        else:
            return True

    # 注意raw_feats对于用gpu和不用gpu的情形是不一样的
    # raw_feats是二维的，每行一个样本，每行按（scale -> x_coordinates -> y_coordinates） -> dim的顺序排
    def safe_aggregate_features(self, raw_feats, cum_num_sub_by_scale, n_feat_scales, n_ex, feat_ndim,
                                n_ex_per_batch=None, n_valid_th=3, timing_only=False, time_out=None):

        # exec(gen_cmd_print_variables('raw_feats', pfx_msg='))) '))
        if time_out is not None:
            tic = perf_counter()

        s_agg_feats = ['mean', 'std', 'skewness', 'kurtosis', 'min', 'q1', 'median', 'q3', 'max']
        n_agg_stats, n_agg_per_ex = len(s_agg_feats), feat_ndim * n_feat_scales
        use_gpu = self.threads_per_block is not None

        agg_feats = np.empty((n_ex, n_agg_per_ex * n_agg_stats), dtype=float) # agg_feats是二维的，每行一个样本，每行按照scale -> dim -> stat的顺序排
        if use_gpu:
            # exec(gen_cmd_print_variables('feat_ndim, s_agg_feats, n_agg_stats, n_agg_per_ex'))

            sorted_raw_feats = deepcopy(raw_feats.reshape((n_ex, -1, feat_ndim)))  # reshape返回的是视图！
            # exec(gen_cmd_print_variables('sorted_raw_feats[0,:10]'))
            sorted_raw_feats = np.swapaxes(sorted_raw_feats, 1, 2)  # 三维，外层是样本，中层是dim，内层是（scale -> x -> y)
            # exec(gen_cmd_print_variables('sorted_raw_feats[0,:10]'))

            # exec(gen_cmd_print_variables('raw_feats', pfx_msg='@@@ '))

            # pre sort
            for i in range(n_feat_scales):
                cur = cupy.array(sorted_raw_feats[:, :, cum_num_sub_by_scale[i]: cum_num_sub_by_scale[i + 1]])
                # exec(gen_cmd_print_variables('i, cur, cupy.sort(cur,0)'))
                sorted_raw_feats[:, :, cum_num_sub_by_scale[i]: cum_num_sub_by_scale[i + 1]] = \
                    cupy.asnumpy(cupy.sort(cur))
            sorted_raw_feats = sorted_raw_feats.flatten()  # 一维，按样本 -> dim ->（scale -> x -> y)排

            # exec(gen_cmd_print_variables('raw_feats', pfx_msg='喵喵喵！ '))

            n_raw_per_ex = raw_feats.shape[1]
            max_n_ex_per_batch = int(np.min([MAX_BYTES // (4 * n_raw_per_ex)]))
            n_ex_per_batch = max_n_ex_per_batch if n_ex_per_batch is None else min(n_ex_per_batch, max_n_ex_per_batch)
            # exec(gen_cmd_print_variables('n_ex_per_batch'))
            mod = pycuda.driver.module_from_file(
                "/users/pss/Liid/Pycharm_projects/PARKER/c_and_cuda_code/extractFeats.cubin")  # 直接粘下来的
            func = mod.get_function("aggregate_feats")
            d_cum_num_sub_by_scale = safe_cuda_memcpy_htod(cum_num_sub_by_scale.astype(np.int32))

            # exec(gen_cmd_print_variables('raw_feats', pfx_msg='!!! '))

            for i in range(next_integer(n_ex / n_ex_per_batch, True)):

                start_ex, finish_ex = i * n_ex_per_batch, min((i + 1) * n_ex_per_batch, n_ex)
                print(f'Aggregating features for examples {start_ex} - {finish_ex} /{n_ex} on GPU...')
                cur_n_ex = finish_ex - start_ex
                start, finish = start_ex * n_raw_per_ex, finish_ex * n_raw_per_ex

                h_sorted_raw_feats = sorted_raw_feats[start: finish].astype(np.float32)
                d_sorted_raw_feats = safe_cuda_memcpy_htod(h_sorted_raw_feats)

                d_agg_feats = safe_cuda_malloc(4 * cur_n_ex * n_agg_per_ex * n_agg_stats)
                func(d_agg_feats, d_sorted_raw_feats, d_cum_num_sub_by_scale, np.int32(n_feat_scales),
                     np.int32(feat_ndim), np.int32(n_valid_th), shared=6 * self.threads_per_block * 4,
                     grid=(cur_n_ex * n_agg_per_ex, 1, 1), block=(self.threads_per_block, 1, 1))

                h_agg_feats = np.empty(cur_n_ex * n_agg_per_ex * n_agg_stats, dtype=np.float32)
                cuda.memcpy_dtoh(h_agg_feats, d_agg_feats)
                agg_feats[start_ex: finish_ex] = h_agg_feats.astype(float).reshape((cur_n_ex, -1))

                if time_out is not None and perf_counter() - tic >= time_out:
                    return None

        else:  # 输入是二维的，每行一个样本，每行按scale -> x_coordinates -> y_coordinates -> dim的顺序排
            # agg_feats是二维的，每行一个样本，每行按照scale -> dim -> stat的顺序排
            for i_ex, cur_raw_feats_ex in enumerate(raw_feats):
                # print(f'Aggregating features for example {i_ex + 1}/{n_ex} on CPU...')
                for i_scale in range(n_feat_scales):
                    start, finish = cum_num_sub_by_scale[i_scale], cum_num_sub_by_scale[i_scale + 1]
                    n_sub = finish - start
                    cur_raw_feats_scale = cur_raw_feats_ex[start * feat_ndim: finish * feat_ndim].reshape(
                        (n_sub, -1)).T  # 当前scale的raw_feats，外dim，内(x_coor, ycoor)，
                    for i_dim, cur_raw_feats in enumerate(cur_raw_feats_scale):

                        cur_raw_feats = cur_raw_feats[np.where(cur_raw_feats != INVALID)[0]]
                        if len(cur_raw_feats) >= n_valid_th:
                            cur_mean = np.mean(cur_raw_feats)
                            cur_std = np.std(cur_raw_feats)
                            cur_skewness = stats.skew(cur_raw_feats)
                            cur_kurtosis = stats.kurtosis(cur_raw_feats)
                            cur_min = np.min(cur_raw_feats)
                            cur_max = np.max(cur_raw_feats)
                            cur_median = percentile(cur_raw_feats, 0.5)
                            cur_q1 = percentile(cur_raw_feats, 0.25)
                            cur_q3 = percentile(cur_raw_feats, 0.75)

                            cur_agg_feats = np.array([cur_mean, cur_std, cur_skewness, cur_kurtosis,
                                                      cur_min, cur_q1, cur_median, cur_q3, cur_max])
                        else:
                            cur_agg_feats = INVALID * np.ones(n_agg_stats, dtype=float)  # invalid by default

                        start = (i_scale * feat_ndim + i_dim) * n_agg_stats
                        agg_feats[i_ex, start: start + n_agg_stats] = deepcopy(cur_agg_feats)

                        if time_out is not None and perf_counter() - tic >= time_out:
                            return None

                        # if i_ex == 18 and i_scale == 2 and i_dim == 1:
                        #     cur_std = np.std(cur_raw_feats)
                        #     exec(gen_cmd_print_variables(
                        #         'np.sum(cur_raw_feats), np.sum(cur_raw_feats**2), np.sum(cur_raw_feats**3), np.sum(cur_raw_feats**4), '
                        #         'cur_mean, cur_var, cur_std, cur_skewness, cur_kurtosis'))
                        #     exec(gen_cmd_print_variables('cur_min, cur_q1, cur_median, cur_q3, cur_max'))


        # # INVALID -> NaN
        # for i_ex, cur_agg_feats_ex in enumerate(agg_feats):
        #     cur_agg_feats_ex[np.where(cur_agg_feats_ex == INVALID)[0]] = float('nan')
        #     agg_feats[i_ex] = cur_agg_feats_ex
        if not timing_only:
            return agg_feats, n_agg_stats, n_agg_per_ex
        else:
            return True

    @staticmethod
    def filter_examples(agg_feats, agg_min_valid_ratio=1):
        n, valid_inds = agg_feats.shape[1], []
        for i_ex, cur_agg_feats_ex in enumerate(agg_feats):
            n_valid = len(np.intersect1d(np.where(cur_agg_feats_ex != INVALID)[0],
                                         np.where(~np.isnan(cur_agg_feats_ex))[0]))

            if n_valid / n >= agg_min_valid_ratio:
                valid_inds.append(i_ex)
        return np.array(valid_inds)
