#!/bin/bash

NAME="01_resnet18"
MODEL="resnet18"

python3 predict.py \
    --image_dim 224 \
    --texture \
    --category \
    --load_checkpoint ckpt_final/${NAME}.pth \
    --model ${MODEL} \
    --test_dir $1 \
    --output_csv $2
