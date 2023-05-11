#!/bin/bash
export PATH=/h/sroutray/.conda/envs/dfl-3/bin:$PATH

set -e
set -x

exp_name="interimage-pos-mining3-softmax-r3-pw0.1"
data_dir="/scratch/hdd001/home/sroutray/dataset/coco/images"
output_dir="./output/$exp_name-debug/"
model_path="/scratch/hdd001/home/sroutray/leopart/output/$exp_name/latest.ckpt"
# model_path="/scratch/hdd001/home/sroutray/leopart/output/$exp_name/ckp-epoch=49.ckpt"

CUDA_VISIBLE_DEVICES=0 python viz_backbone_features_pca.py \
    --dataset_name coco-thing \
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
    --topk 4 \
    --alpha 0.6 \
    --save_path ${output_dir} \
    --temp 0.07
    # --proj_bn \
