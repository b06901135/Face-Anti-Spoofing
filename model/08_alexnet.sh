#!/bin/bash

python3 main.py \
    --image_dim 224 \
    --texture \
    --name 08_alexnet \
    --model alexnet \
    --total_epoch 20 \
    --checkpoint_epoch 5
