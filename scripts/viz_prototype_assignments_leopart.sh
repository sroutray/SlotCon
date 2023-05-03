#!/bin/bash
export PATH=/h/sroutray/.conda/envs/dfl-3/bin:$PATH

set -e
set -x

data_dir="/scratch/gobi3/chaenayo/coco/images"
output_dir="./output/interimage-ex1/"
model_path="/h/chaenayo/leopart/output/interimage-ex1/latest.ckpt"

CUDA_VISIBLE_DEVICES=0 python viz_prototype_assignments_leopart.py \
    --dataset COCOval \
    --data_dir ${data_dir} \
    \
    --arch vit-small \
    --patch_size 16 \
    --num_prototypes 300 \
    --projection_hidden_dim 2048 \
    --model_path "${model_path}" \
    \
    --batch_size 64 \
    \
    --mode knn \
    --topk 14 \
    --alpha 0.6 \
    --save_path ${output_dir} \
    --temp 0.07
    # --proj_bn \
