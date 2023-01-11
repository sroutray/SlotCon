#!/bin/bash
export PATH=/h/sroutray/.conda/envs/dfl-3/bin:$PATH

set -e
set -x

data_dir="/scratch/gobi3/chaenayo/coco/images"
output_dir="./output/slotcon_proto_assingments_viz_orig/"
model_path="/h/sroutray/SlotCon/output/slotcon_coco_baseline_800ep/slotcon_coco_r50_800ep_full_checkpoint.pth"

CUDA_VISIBLE_DEVICES=0 python viz_prototype_assignments.py \
    --dataset COCOval \
    --data_dir ${data_dir} \
    \
    --arch resnet50 \
    --dim_hidden 4096 \
    --dim_out 256 \
    --num_prototypes 256 \
    --model_path ${model_path} \
    \
    --batch_size 64 \
    \
    --mode knn \
    --topk 14 \
    --alpha 0.6 \
    --save_path ${output_dir} \
    --temp 0.07

