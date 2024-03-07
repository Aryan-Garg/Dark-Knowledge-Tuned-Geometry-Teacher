#!/usr/bin/env bash


# gmstereo-scale2-regrefine3 model
CUDA_VISIBLE_DEVICES=1 python main_stereo.py \
--inference_dir /data2/aryan/TAMULF/train_dp \
--inference_size 512 768 \
--output_path ./dp_otherDS/TAMULF/train_dp \
--resume pretrained/mixdata.pth \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_type self_swin2d_cross_swin1d \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--reg_refine \
--num_reg_refine 3
# --save_pfm_disp
# optionally predict both left and right disparities
#--pred_bidir_disp




