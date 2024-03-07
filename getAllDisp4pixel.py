#!/usr/bin/env python

import os
import glob
from tqdm.auto import tqdm 


def getAllDisp4pixel(dataset = None):
    assert dataset is not None, "Please provide dataset name"
    inf_dir = glob.glob(f'/data2/aryan/{dataset}/*')
    # left = glob.glob('/data2/raghav/datasets/Pixel4_3DP/rectified/B/Video_data/*')
    # right = glob.glob('/data2/raghav/datasets/Pixel4_3DP/rectified/A/Video_data/*')
    
    pbar_inf = tqdm(inf_dir)
    for dir_ in pbar_inf:
        # dirR = dirL.replace('B', 'A')
        videoName = dir_.split('/')[-1]
        pbar_inf.set_description(f"Processing {videoName}")
        # print(videoName)

        # Call main_stereo.py with params from gmstereo_demo.sh
        os.system(f"CUDA_VISIBLE_DEVICES=2 python main_stereo.py \
                    --inference_dir {dir_} \
                    --inference_size 512 768 \
                    --output_path dp_otherDS/{dataset}/{videoName} \
                    --resume pretrained/mixdata.pth \
                    --padding_factor 32 \
                    --upsample_factor 4 \
                    --num_scales 2 \
                    --attn_type self_swin2d_cross_swin1d \
                    --attn_splits_list 2 8 \
                    --corr_radius_list -1 4 \
                    --prop_radius_list -1 1 \
                    --reg_refine \
                    --num_reg_refine 3 \
                    --save_pfm_disp")

if __name__ == '__main__':
    datasets = ["TAMULF", "Stanford", "Kalantari", "Hybrid"]
    for dataset in datasets:
        getAllDisp4pixel(dataset)