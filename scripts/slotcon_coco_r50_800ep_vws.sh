#!/bin/bash
export PATH=/h/sroutray/.conda/envs/dfl-3/bin:$PATH

set -e
set -x

data_dir="/scratch/gobi3/chaenayo/coco/images"
output_dir="./output/slotcon_coco_baseline_1gpu_800ep"

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 12348 --nproc_per_node=1 \
    main_pretrain.py \
    --dataset COCO \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch resnet50 \
    --dim-hidden 4096 \
    --dim-out 256 \
    --num-prototypes 256 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 0.5 \
    \
    --batch-size 80 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 800 \
    --fp16 \
    \
    --print-freq 50 \
    --save-freq 10 \
    --auto-resume \
    --num-workers 8 \
    --wandb-logging
