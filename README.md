# POPS
This repository holds the source code and data for our paper: *POPS: An Efficient GPU-based Framework for Multi-scale Feature
Extraction from Massive Planetary LiDAR Data*, submitted to **VLDB 2024 (Scalable Data Science Track)**. 

This repository is intended as supporting materials for the reviewing process of our paper only. Please do not use it for any other purposes.

## Prerequisites
To use the code, you need:
- NVIDIA-GPU with computate capability >= 7.5
- Python 3
- CUDA Toolkit >= 11.0
- Eigen library

## How to use

Please follow the steps below to use our code.
1. Download the sample datasets in the data folder to your local directory. If you wish to use all the data, go to https://drive.google.com/file/d/1t0v7RZLFka-qfPEO5nz55k6gYDQT79kd/view?usp=sharing. Note however that the entire dataset is as large as 4.84GB even when compressed!
2. Download the code folder to your local directory, and change the directory in the third line of main.py to where you have downloaded the code to.
3. Use the following command to run the code:
```
python main.py $data_path $result_path $data_id $idw_alpha $core_met $raw_feat_extr_framework $feat_agg_met $ex_scale $feat_scales $feat_steps  $raw_feat_extr_threads_per_block $feat_agg_threads_per_block
```
The meanings of the paramters are as follows:

- data_path: path to the data grids. All grids have a resolution of 60m on both the x- and y-axes
- result_path: path to the raw feature extraction and aggregation results
- data_id: ID of the dataset to conduct feature extraction on, such as "GEDI\_Ngazima_Drainage", "MOLA\_Olympus\_Volcano".
- core_met: core method used for raw feature extraction, currently supports "PCA_ISPRS12" (which is the same as "PCA" mentioned in our paper), "STAT".
- raw_feat_extr_framework: the raw feature extraction framework to use, currently supports "CPU", "POPS-NMS", "POPS"
- feat_agg_met: the feature aggregation method to use, currently supports "CPU", "GPU".
- idw_alpha: a parameter for the interpolation method we used to obtain the data grids. Changing this will cause running on different grids for the same data_id. You can set it to one of 1, 2, 3, 4, 5.
- ex_scale (optional): the example scale in meters. Default is 2000. Example sampling step is hard-coded to be equal to ex_scale at the moment.
- feat_scales (optional): the feature scales in meters. If there are more than one feature scales you wish to set, use "\_" to join them. Default is 300\_400\_500\_700\_800\_900\_1100\_1200\_1300.
- feat_steps (optional): the feature sampling steps in meters, each correspond to one of feat_scales. Use "\_" to join multiple feature steps. Default is half of all the default feature scales.
- raw_feat_extr_threads_per_block (optional): Number of GPU threads per block for raw feature extraction, only effective when raw_feat_extr_framework is not 'CPU'. Make sure this is an integer multiple of 16. Default is 256.
- feat_agg_threads_per_block (optional): Number of GPU threads per block for feature aggregation, only effective when feat_agg_met is not 'CPU'. Make sure this is an integer multiple of 16. Default is 256.

## About the input and output files
The data files are named as *raw_results_with_ROIs_interpol_raster_60_60_$data_id_fast_idw_gpu_raster_10000_$idw_alpha_0.pkl*. Each data file contains one or more data grids corresponding to $data_id.

The output files are named as *feat_extr_results_60_60_$data_id_fast_idw_gpu_raster_10000_$idw_alpha_$ex_scale_$feat_scales_$feat_steps_$core_met_$raw_feat_extr_framework_$raw_feat_extr_threads_per_block_$feat_agg_met_$feat_agg_threads_per_block.pkl'*. This file can be loaded using the following Python script:
```
import pickle
with open(output_file_name, 'rb') as f:
  all_raw_feats, all_agg_feats, raw_time, agg_time = pickle.load(f)
```
There are four outputs:
- all_raw_feats: the raw features extracted, which is a list of numpy ndarrays. Each array is raw features of one data grid that corresponds to $data_id.
- all_agg_feats: the aggregated features, which is a list of numpy ndarrays. Each array is the aggregated features of one data grid that corresponds to $data_id.
- raw_time: the raw feature extraction time in seconds.
- agg_time: the feature aggregation time in seconds.
