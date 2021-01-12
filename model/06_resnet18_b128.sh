#!/bin/bash

python3 main.py \
    --image_dim 224 \
    --batch_size 128 \
    --texture \
    --name 06_resnet18_b128 \
    --model resnet18 \
    --total_epoch 40 \
    --checkpoint_epoch 5
