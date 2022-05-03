# -*- coding: UTF-8 -*-

from util.util import last_integer, next_integer, percentile
from util.cuda_util import *
import os
import numpy as np
import subprocess
from copy import deepcopy
import scipy.stats as stats
import cupy
from time import perf_counter

INVALID = INF = 2e9


class PtCloudFeatExtractor(object):

    core_met_info_map = {
        'PCA_ISPRS12': {'core_id': 0, 'feat_ndim': 2, 'post_processing': False},
        'STAT': {'core_id': 1, 'feat_ndim': 8, 'post_processing': True},
    }

    def __init__(self, ex_scale, feat_scales, feat_steps,
                 raw_feat_extr_threads_per_block=None, feat_agg_threads_per_block=None):  # use threads_per_block=None for CPU
        if raw_feat_extr_threads_per_block is not None and raw_feat_extr_threads_per_block > 0 \
                and raw_feat_extr_threads_per_block % 16 != 0:
            print("Error! raw_feat_extr_threads_per_block is not divisible by half warp size!")
            exit(1)
        if feat_agg_threads_per_block is not None and feat_agg_threads_per_block > 0 \
                and feat_agg_threads_per_block % 16 != 0:
            print("Error! feat_agg_threads_per_block is not divisible by half warp size!")
            exit(1)

        self.raw_feat_extr_threads_per_block = raw_feat_extr_threads_per_block
        self.feat_agg_threads_per_block = feat_agg_threads_per_block
        self.ex_scale = ex_scale

        order = np.argsort(feat_scales)
        self.sorted_feat_scales = np.array(feat_scales)[order]
        self.sorted_feat_steps = np.array(feat_steps)[order]

        # os.chdir('/users/pss/Liid/Pycharm_projects/PARKER/c_and_cuda_code')
        # subprocess.run(
        #     ['/usr/local/cuda-11.2/bin/nvcc', '--cubin', '-arch=sm_75', '--disable-warnings', 'extractFeats.cu'])

    @staticmethod
    def _core_wrapper(self, core, args):
        func = eval(f'self._{core}')
        if core == 'PCA_ISPRS12':
            return func(args['sub_x'], args['sub_y'], args['sub_z'])
        elif core == 'STAT':
            return func(args['sub_z'])

    @staticmethod
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
        return PtCloudFeatExtractor._estimate_ternary_coord2d(comp)

    @staticmethod
    def _estimate_ternary_coord2d(comp):

        # Conversion towards a,b,c space
        ct = 3 * comp[:, 2]

        xp1p2 = (1 - (comp[:, 1] - comp[:, 0])) / 2
        yp1p2 = 1 - xp1p2

        dS2 = np.sqrt((.5 - xp1p2) ** 2 + (.5 - yp1p2) ** 2)

        at = (1 - ct) * dS2 / (np.sqrt((0.5 - 1) ** 2 + (0.5 - 0) ** 2))
        bt = 1 - (ct + at)
        v = (at + bt + ct)

        # Conversion towards ternary graph
        x_ter = np.maximum(.5 * (2. * bt + ct) / v, 0)
        y_ter = np.maximum(0.5 * np.sqrt(3) * ct / v, 0)
        if len(x_ter) == 1:
            x_ter = x_ter[0]
        if len(y_ter) == 1:
            y_ter = y_ter[0]

        return x_ter, y_ter

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

    def safe_batch_extract_feats_gridded(self, ori_z, resol_x, resol_y, is_volumetric,
                                         std_norm_xy=True, std_norm_z=True, ny=None, num_valid_th=10,
                                         n_ex_per_batch=None, core='PCA_ISPRS12', use_incre_shmem=True,
                                         use_multi_scale=True, rand_state=1, sh_nx=INF, sh_ny=INF,
                                         time_out=float('inf')):

        tic = perf_counter()

        feat_ndim, core_id = self.core_met_info_map[core]['feat_ndim'], self.core_met_info_map[core]['core_id']
        use_gpu = self.raw_feat_extr_threads_per_block is not None

        z = deepcopy(ori_z) if ori_z.ndim == 2 else deepcopy(ori_z.reshape((-1, ny)))
        n_feat_scales, nx, ny = len(self.sorted_feat_scales), z.shape[0], z.shape[1]
        ex_nx, ex_ny = map(int, (self.ex_scale // resol_x, self.ex_scale // resol_y))
        ex_nv = ex_nx * ex_ny
        n_ex_x, n_ex_y = map(int, (nx // ex_nx, ny // ex_ny))
        if n_ex_x * n_ex_y == 0:
            print('Error. ROI size too small. Cannot extract examples!')
            exit(1)
        nx, ny = n_ex_x * ex_nx, n_ex_y * ex_ny
        z = np.float32(z[:nx, :ny])

        feat_xpts_per_step_by_scale = np.maximum(
            1, [np.round(f_step / resol_x) for f_step in self.sorted_feat_steps]
        )
        feat_ypts_per_step_by_scale = np.maximum(
            1, [np.round(f_step / resol_y) for f_step in self.sorted_feat_steps]
        )
        feat_xpts_per_step_by_scale, feat_ypts_per_step_by_scale = \
            map(np.int32, (feat_xpts_per_step_by_scale, feat_ypts_per_step_by_scale))

        n_sub_x_by_scale, n_sub_y_by_scale = np.empty(n_feat_scales, dtype=int), np.empty(n_feat_scales, dtype=int)
        feat_nx_by_scale, feat_ny_by_scale = \
            np.empty(n_feat_scales, dtype=int), np.empty(n_feat_scales, dtype=int)
        cum_num_sub_by_scale = np.zeros(n_feat_scales + 1, dtype=int)

        # to do: get rid of the for loop. This can be done more pythonically!
        for i, f_scale in enumerate(self.sorted_feat_scales):

            feat_nx_by_scale[i], feat_ny_by_scale[i] = \
                last_integer(f_scale / resol_x, 1), last_integer(f_scale / resol_y, 1)

            n_sub_x_by_scale[i] = 1 + last_integer((ex_nx - feat_nx_by_scale[i]) / feat_xpts_per_step_by_scale[i], 1)
            n_sub_y_by_scale[i] = 1 + last_integer((ex_ny - feat_ny_by_scale[i]) / feat_ypts_per_step_by_scale[i], 1)
            cum_num_sub_by_scale[i + 1] = cum_num_sub_by_scale[i] + n_sub_x_by_scale[i] * n_sub_y_by_scale[i]

        n_ex, n_feats_per_ex = n_ex_x * n_ex_y, cum_num_sub_by_scale[-1] * feat_ndim
        raw_feats = INVALID * np.ones(n_ex * n_feats_per_ex, dtype=float)

        # raw feature extraction
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
            h_feats = INVALID * np.ones(n_ex_per_batch * n_feats_per_ex, dtype=np.float32)
            d_feats = safe_cuda_malloc(n_ex_per_batch * n_feats_per_ex * 4)

            d_n_feat_scales, d_ny, d_ex_nx, d_ex_ny, d_is_volumetric, d_std_norm_xy, d_std_norm_z, \
                d_max_float_sh, d_num_valid_th, d_core_id, d_feat_ndim, d_use_incre_shmem, d_use_multi_scale = \
                map(np.int32, (n_feat_scales, ny, ex_nx, ex_ny, is_volumetric,
                               std_norm_xy, std_norm_z, max_float_sh, num_valid_th, core_id, feat_ndim,
                               use_incre_shmem, use_multi_scale))
            d_resol_x, d_resol_y = map(np.float32, (resol_x, resol_y))
            mod = cuda.module_from_file("./extractFeats.cubin")
            func = mod.get_function("extractFeats_gridded")

            cnt, start_ex = 0, 0
            while cnt < n_ex:
                print(f'Extracting raw features for examples on GPU. #remaining = {n_ex - cnt}...')
                finish_ex = min(start_ex + n_ex_per_batch, n_ex)
                start_ex_x, finish_ex_x, = start_ex // n_ex_y, (finish_ex - 1) // n_ex_y + 1
                if start_ex_x == finish_ex_x - 1:
                    start_ex_y, finish_ex_y = start_ex % n_ex_y, (finish_ex - 1) % n_ex_y + 1
                    offset_ex_y = 0
                else:
                    start_ex_y, finish_ex_y = 0, n_ex_y
                    offset_ex_y = start_ex % n_ex_y

                start_x, finish_x, start_y, finish_y = \
                    start_ex_x * ex_nx, min(finish_ex_x * ex_nx, nx), start_ex_y * ex_ny, min(finish_ex_y * ex_ny, ny)
                z_, nx_ = z[start_x: finish_x, start_y: finish_y], finish_x - start_x

                finish_x, start_y = min(ex_nx, nx_), offset_ex_y * ex_ny
                cur_z = z_[:finish_x, start_y:].flatten()

                n_ex_col = finish_ex_x - start_ex_x
                n_ex_last_col = (finish_ex - start_ex) - (n_ex_y * (n_ex_col - 1) - offset_ex_y)
                if n_ex_col >= 3:
                    start_x, finish_x = ex_nx, (n_ex_col - 1) * ex_nx
                    cur_z = np.concatenate((cur_z, z_[start_x: finish_x].flatten()))
                    start_x, finish_y = finish_x, min(ny, n_ex_last_col * ex_ny)
                    cur_z = np.concatenate((cur_z, z_[start_x:, :finish_y].flatten()))
                elif n_ex_col == 2:
                    start_x, finish_y = ex_nx, min(ny, n_ex_last_col * ex_ny)
                    cur_z = np.concatenate((cur_z, z_[start_x:, :finish_y].flatten()))
                h_z[: len(cur_z)] = deepcopy(cur_z.astype(np.float32))
                cuda.memcpy_htod(d_z, h_z)

                grid_size = finish_ex - start_ex
                np.random.seed(rand_state)
                rand_props = np.random.rand(grid_size * self.raw_feat_extr_threads_per_block).astype(np.float32)
                d_rand_props = safe_cuda_memcpy_htod(rand_props)

                # call feature extraction kernel
                func(d_feats, d_z, d_sorted_feat_scales, d_feat_nx_by_scale, d_feat_ny_by_scale,
                     d_feat_xpts_per_step_by_scale, d_feat_ypts_per_step_by_scale, d_rand_props, d_n_feat_scales, d_ny,
                     d_resol_x, d_resol_y, d_ex_nx, d_ex_ny, d_max_float_sh, d_is_volumetric, d_std_norm_xy,
                     d_std_norm_z, np.int32(cnt), d_core_id, d_feat_ndim, d_num_valid_th, np.int32(sh_nx),
                     np.int32(sh_ny), d_use_incre_shmem, d_use_multi_scale,
                     shared=shared, grid=(finish_ex - start_ex, 1, 1),
                     block=(self.raw_feat_extr_threads_per_block, 1, 1))
                cuda.memcpy_dtoh(h_feats, d_feats)
                start, finish = cnt * n_feats_per_ex, min(cnt + n_ex_per_batch, n_ex) * n_feats_per_ex
                raw_feats[start: finish] = h_feats[: finish - start].astype(float)

                cnt, start_ex = cnt + n_ex_per_batch, finish_ex  # 最后一轮的时候，cnt可以超过n_ex，没关系的

                if perf_counter() - tic >= time_out:
                    return None
        else:
            cnt_feat = 0
            for i_ex_x in range(n_ex_x):
                ix = ex_nx * i_ex_x
                for i_ex_y in range(n_ex_y):
                    iy = ex_ny * i_ex_y
                    cur_z = z[ix: ix + ex_nx, iy: iy + ex_ny]
                    for i, f_scale in enumerate(self.sorted_feat_scales):

                        feat_nx, feat_ny = feat_nx_by_scale[i], feat_ny_by_scale[i]
                        n_sub_x, n_sub_y = n_sub_x_by_scale[i], n_sub_y_by_scale[i]

                        if not is_volumetric:  # ball query
                            dist_th = f_scale / 2  # f_scale is the diameter
                            ic_x, ic_y = last_integer(feat_nx / 2, 0), last_integer(feat_ny / 2, 0)

                        for i_sub_x in range(n_sub_x):
                            cur_ix = int(feat_xpts_per_step_by_scale[i] * i_sub_x)
                            sub_x_1d = np.array(range(cur_ix, cur_ix + feat_nx)) * resol_x
                            if not is_volumetric:
                                xc = sub_x_1d[ic_x]  # coordinate of center point of the patch

                            for i_sub_y in range(n_sub_y):
                                cur_iy = int(feat_ypts_per_step_by_scale[i] * i_sub_y)
                                sub_y_1d = np.array(range(cur_iy, cur_iy + feat_ny)) * resol_y
                                sub_x, sub_y = np.meshgrid(sub_x_1d, sub_y_1d,
                                                           indexing='ij')
                                sub_z = cur_z[cur_ix: cur_ix + feat_nx, cur_iy: cur_iy + feat_ny]
                                if not is_volumetric:
                                    yc, zc = sub_y_1d[ic_y], sub_z[ic_x, ic_y]
                                sub_x, sub_y, sub_z = sub_x.flatten(), sub_y.flatten(), sub_z.flatten()

                                i_valid = np.where(~np.isnan(sub_z))
                                if len(i_valid) != 0:
                                    sub_x, sub_y, sub_z = sub_x[i_valid], sub_y[i_valid], sub_z[i_valid]
                                else:
                                    sub_x, sub_y, sub_z = [], [], []

                                if not is_volumetric:
                                    if np.isnan(zc):
                                        sub_x, sub_y, sub_z = [], [], []
                                    else:
                                        dists = np.sqrt((sub_x - xc) ** 2 +
                                                        (sub_y - yc) ** 2 +
                                                        (sub_z - zc) ** 2)
                                        i_valid = np.where(dists < dist_th)[0]
                                        sub_x, sub_y, sub_z = sub_x[i_valid], sub_y[i_valid], sub_z[i_valid]

                                if len(sub_x) >= num_valid_th:
                                    sub_x, sub_y, sub_z = map(np.float64, (sub_x, sub_y, sub_z))

                                    sub_x -= np.mean(sub_x)
                                    sub_y -= np.mean(sub_y)
                                    sub_z -= np.mean(sub_z)
                                    if std_norm_xy:
                                        sub_x /= (np.std(sub_x) if np.std(sub_x) > 0 else 1)
                                        sub_y /= (np.std(sub_y) if np.std(sub_y) > 0 else 1)
                                    if std_norm_z:
                                        sub_z /= (np.std(sub_z) if np.std(sub_z) > 0 else 1)

                                    # core method
                                    raw_feats[cnt_feat: cnt_feat + feat_ndim] = self._core_wrapper(core, {
                                        'sub_x': sub_x, 'sub_y': sub_y, 'sub_z': sub_z
                                    })
                                else:
                                    raw_feats[cnt_feat: cnt_feat + feat_ndim] = np.array([INVALID] * feat_ndim)
                                cnt_feat += feat_ndim
                                if perf_counter() - tic >= time_out:
                                    return None
            raw_feats = raw_feats.reshape((n_ex, -1))

        raw_feats = raw_feats.reshape((n_ex, feat_ndim, -1))
        raw_feats = np.swapaxes(raw_feats, 1, 2).reshape((n_ex, -1)) # example -> [(scale -> x_coordinates -> y_coordinates) -> dim]

        raw_feats, feat_ndim = self._post_core_wrapper(core, {'raw_feats': raw_feats, 'std_norm_z': std_norm_z})
        return raw_feats, cum_num_sub_by_scale, n_feat_scales, n_ex_x, n_ex_y, n_ex, feat_ndim

    def safe_aggregate_features(self, raw_feats, cum_num_sub_by_scale, n_feat_scales, n_ex, feat_ndim,
                                n_ex_per_batch=None, n_valid_th=3, time_out=None):

        if time_out is not None:
            tic = perf_counter()

        s_agg_feats = ['mean', 'std', 'skewness', 'kurtosis', 'min', 'q1', 'median', 'q3', 'max']
        n_agg_stats, n_agg_per_ex = len(s_agg_feats), feat_ndim * n_feat_scales
        use_gpu = self.feat_agg_threads_per_block is not None

        agg_feats = np.empty((n_ex, n_agg_per_ex * n_agg_stats), dtype=float) # ex -> [scale -> dim -> stat]
        if use_gpu:

            sorted_raw_feats = deepcopy(raw_feats.reshape((n_ex, -1, feat_ndim)))
            sorted_raw_feats = np.swapaxes(sorted_raw_feats, 1, 2)

            # pre sort
            for i in range(n_feat_scales):
                cur = cupy.array(sorted_raw_feats[:, :, cum_num_sub_by_scale[i]: cum_num_sub_by_scale[i + 1]])
                sorted_raw_feats[:, :, cum_num_sub_by_scale[i]: cum_num_sub_by_scale[i + 1]] = \
                    cupy.asnumpy(cupy.sort(cur))
            sorted_raw_feats = sorted_raw_feats.flatten()

            n_raw_per_ex = raw_feats.shape[1]
            max_n_ex_per_batch = int(np.min([MAX_BYTES // (4 * n_raw_per_ex)]))
            n_ex_per_batch = max_n_ex_per_batch if n_ex_per_batch is None else min(n_ex_per_batch, max_n_ex_per_batch)
            mod = cuda.module_from_file("./extractFeats.cubin")
            func = mod.get_function("aggregate_feats")
            d_cum_num_sub_by_scale = safe_cuda_memcpy_htod(cum_num_sub_by_scale.astype(np.int32))

            for i in range(next_integer(n_ex / n_ex_per_batch, True)):

                start_ex, finish_ex = i * n_ex_per_batch, min((i + 1) * n_ex_per_batch, n_ex)
                print(f'Aggregating features for examples {start_ex} - {finish_ex} /{n_ex} on GPU...')
                cur_n_ex = finish_ex - start_ex
                start, finish = start_ex * n_raw_per_ex, finish_ex * n_raw_per_ex

                h_sorted_raw_feats = sorted_raw_feats[start: finish].astype(np.float32)
                d_sorted_raw_feats = safe_cuda_memcpy_htod(h_sorted_raw_feats)

                d_agg_feats = safe_cuda_malloc(4 * cur_n_ex * n_agg_per_ex * n_agg_stats)
                func(d_agg_feats, d_sorted_raw_feats, d_cum_num_sub_by_scale, np.int32(n_feat_scales),
                     np.int32(feat_ndim), np.int32(n_valid_th), shared=6 * self.feat_agg_threads_per_block * 4,
                     grid=(cur_n_ex * n_agg_per_ex, 1, 1), block=(self.feat_agg_threads_per_block, 1, 1))

                h_agg_feats = np.empty(cur_n_ex * n_agg_per_ex * n_agg_stats, dtype=np.float32)
                cuda.memcpy_dtoh(h_agg_feats, d_agg_feats)
                agg_feats[start_ex: finish_ex] = h_agg_feats.astype(float).reshape((cur_n_ex, -1))

                if time_out is not None and perf_counter() - tic >= time_out:
                    return None

        else:
            for i_ex, cur_raw_feats_ex in enumerate(raw_feats):
                print(f'Aggregating features for example {i_ex + 1}/{n_ex} on CPU...')
                for i_scale in range(n_feat_scales):
                    start, finish = cum_num_sub_by_scale[i_scale], cum_num_sub_by_scale[i_scale + 1]
                    n_sub = finish - start
                    cur_raw_feats_scale = cur_raw_feats_ex[start * feat_ndim: finish * feat_ndim].reshape(
                        (n_sub, -1)).T
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
                            cur_agg_feats = INVALID * np.ones(n_agg_stats, dtype=float)

                        start = (i_scale * feat_ndim + i_dim) * n_agg_stats
                        agg_feats[i_ex, start: start + n_agg_stats] = deepcopy(cur_agg_feats)

                        if time_out is not None and perf_counter() - tic >= time_out:
                            return None

        return agg_feats, n_agg_stats, n_agg_per_ex
