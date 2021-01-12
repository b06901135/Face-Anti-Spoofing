#!/bin/bash

python3 main.py \
    --name 05_dense \
    --limit_num 10 \
    --image_dim 256 \
    --batch_size 16 \
    --total_epoch 20 \
    --checkpoint_epoch 5
