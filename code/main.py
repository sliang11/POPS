import sys

sys.path.append('/mypath/POPS/')  # change "mypath" to your own

from feature_extractor import PtCloudFeatExtractor
from util.file_util import *
from time import perf_counter
import inspect

class FeatExtrRunner(object):

    def __init__(
            self, data_path, result_path, data_id, idw_alpha, ex_scale, feat_scales, feat_steps,
            core_met, raw_feat_extr_framework, feat_agg_met,
            raw_feat_extr_threads_per_block=256, feat_agg_threads_per_block=256
    ):

        """
        :param data_path: path to the data grids. All grids have a resolution of 60m on both the x- and y-axes
        :param result_path: path to the raw feature extraction and aggregation results
        :param data_id: the ID of the dataset to run on,
            such as 'GEDI_Ngazima_Drainage', 'MOLA_Olympus_Volcano'
        :param idw_alpha: a parameter for the interpolation method we used to obtain the data grids.
            Changing this will cause running on different grids for the same data_id.
            You can set it to one of 1, 2, 3, 4, 5.
        :param ex_scale: the example scale in meters. Recommended value is 2000.
            Example sampling step is hard-coded to be equal to ex_scale at the moment.
        :param feat_scales: the feature scales in meters.
        :param feat_steps: the feature sampling steps in meters.
        :param core_met: core method used for raw feature extraction,
            currently supports 'PCA_ISPRS12' (which is the same as 'PCA' mentioned in our paper), 'STAT'.
        :param raw_feat_extr_framework: the raw feature extraction framework to use,
            currently supports 'CPU', 'POPS-NMS', 'POPS'
        :param feat_agg_met: the feature aggregation method to use, currently supports 'CPU', 'GPU'.
        :param raw_feat_extr_threads_per_block: Number of GPU threads per block for raw feature extraction,
            only effective when raw_feat_extr_framework is not 'CPU'. Make sure this is an integer multiple of 16!
        :param feat_agg_threads_per_block: Number of GPU threads per block for feature aggregation,
            only effective when feat_agg_met is not 'CPU'. Make sure this is an integer multiple of 16!
        """

        for para in inspect.signature(self.__init__).parameters.keys():
            exec(f'self.{para} = {para}')

        if raw_feat_extr_framework == 'CPU':
            raw_feat_extr_threads_per_block = None
        elif raw_feat_extr_framework == 'POPS-NMS':
            self.use_multi_scale = False
        else:
            self.use_multi_scale = True
        if feat_agg_met == 'CPU':
            feat_agg_threads_per_block = None
        self.feat_extractor = PtCloudFeatExtractor(
            ex_scale=ex_scale, feat_scales=feat_scales, feat_steps=feat_steps,
            raw_feat_extr_threads_per_block=raw_feat_extr_threads_per_block,
            feat_agg_threads_per_block=feat_agg_threads_per_block
        )

        data_fname = \
            f'raw_results_with_ROIs_interpol_raster_60_60_{data_id}_' \
            f'fast_idw_gpu_raster_10000_{idw_alpha}_0.pkl'
        self.data_fname = os.path.join(data_path, data_fname)
        result_fname = FileNameProcessor.create_fname(
            [
                'feat_extr_results_60_60', data_id, 'fast_idw_gpu_raster_10000', idw_alpha, ex_scale,
                self.feat_extractor.sorted_feat_scales, self.feat_extractor.sorted_feat_steps, core_met,
                raw_feat_extr_framework, raw_feat_extr_threads_per_block, feat_agg_met, feat_agg_threads_per_block
            ]
        )
        self.result_fname = os.path.join(result_fname, result_path)

    def raw_feature_extraction(self):

        _, _, all_z2s, _ = FileReader.load_pickle(self.data_fname)

        tic = perf_counter()
        all_raw_ret = []
        for z2 in all_z2s:  # There can be multiple ROIs (i.e. multiple data grids) in one data file.

            tic = perf_counter()
            results = self.feat_extractor.safe_batch_extract_feats_gridded(
                z2, 60, 60, True, std_norm_xy=False, std_norm_z=False,
                core=self.core_met, use_multi_scale=self.use_multi_scale
            )
            all_raw_ret.append(results)
        raw_time = perf_counter() - tic
        return all_raw_ret, raw_time

    def feature_aggregation(self, all_raw_ret):  # takes the return of raw_feature_extraction
        tic = perf_counter()
        all_agg_feats = []
        for raw_ret in all_raw_ret:
            raw_feats, cum_num_sub_by_scale, n_feat_scales, n_ex_x, n_ex_y, n_ex, feat_ndim = raw_ret
            agg_feats, _, _ = self.feat_extractor.safe_aggregate_features(
                raw_feats, cum_num_sub_by_scale, n_feat_scales, n_ex, feat_ndim)
            all_agg_feats.append(agg_feats)
        agg_time = perf_counter() - tic
        return all_agg_feats, agg_time

    def run(self):
        print("*********************************************\n")
        print(
            f'Running feature extraction for {self.data_id} with idw_alpha = {self.idw_alpha}, with: \n'
            f'ex_scale={self.ex_scale}, feat_scales={self.feat_scales}, feat_steps={self.feat_steps}, '
            f'core_met={self.core_met}, raw_feat_extr_framework={self.raw_feat_extr_framework}, '
            f'feat_agg_met={self.feat_agg_met}, '
            f'raw_feat_extr_threads_per_block={self.raw_feat_extr_threads_per_block}, '
            f'feat_agg_threads_per_block={self.feat_agg_threads_per_block}'
        )
        all_raw_ret, raw_time = self.raw_feature_extraction()
        all_raw_feats = [raw_ret[0] for raw_ret in all_raw_ret]
        print(f'Raw feature extraction time: {raw_time}s')
        all_agg_feats, agg_time = self.feature_aggregation(all_raw_ret)
        print(f'Feature aggregation time: {raw_time}s\n')
        FileWriter.dump_pickle((all_raw_feats, all_agg_feats, raw_time, agg_time), self.result_fname)


if __name__ == '__main__':

    data_path = sys.argv[1]
    result_path = sys.argv[2]
    data_id = sys.argv[3]
    idw_alpha = int(sys.argv[4])
    core_met = sys.argv[5]
    raw_feat_extr_framework = sys.argv[6]
    feat_agg_met = sys.argv[7]
    ex_scale = float(sys.argv[8]) if len(sys.argv) > 8 else 2000
    feat_scales = [float(scale) for scale in sys.argv[9].split('_')] if len(sys.argv) > 9 else \
        np.array((300, 400, 500, 700, 800, 900, 1100, 1200, 1300), dtype=float)
    feat_steps = [float(step) for step in sys.argv[10].split('_')] if len(sys.argv) > 10 else \
        feat_scales * .5
    raw_feat_extr_threads_per_block = sys.argv[11] if len(sys.argv) > 11 else 256
    feat_agg_threads_per_block = sys.argv[12] if len(sys.argv) > 12 else 256

    runner = FeatExtrRunner(
        data_path, result_path, data_id, idw_alpha, ex_scale, feat_scales, feat_steps,
        core_met, raw_feat_extr_framework, feat_agg_met,
        raw_feat_extr_threads_per_block=raw_feat_extr_threads_per_block,
        feat_agg_threads_per_block=feat_agg_threads_per_block)
    runner.run()
