#!/bin/bash

python3 main.py \
    --image_dim 224 \
    --texture \
    --name 06_resnet18 \
    --load_checkpoint ckpt/06_resnet18/e020.pth \
    --model resnet18 \
    --total_epoch 40 \
    --checkpoint_epoch 5
