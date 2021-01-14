#!/bin/bash

# Texture model
NAME=01_resnet18
MODEL=resnet18

python3 predict.py \
    --image_dim 224 \
    --texture \
    --category \
    --load_checkpoint ckpt/$NAME/e50.pth \
    --model $MODEL \
    --output_csv output/"test_cat.csv" \
