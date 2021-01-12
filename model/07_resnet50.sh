#!/bin/bash

python3 main.py \
    --image_dim 224 \
    --texture \
    --name 07_resnet50 \
    --load_checkpoint ckpt/07_resnet50/e020.pth \
    --model resnet50 \
    --total_epoch 40 \
    --checkpoint_epoch 5
