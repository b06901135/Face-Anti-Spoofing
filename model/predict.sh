#!/bin/bash

python3 predict.py \
    --image_dim 224 \
    --texture \
    --load_checkpoint ckpt/06_resnet18_b128/e040.pth \
    --model resnet18 \
    --output_csv output/06_resnet18_b128_e40_fivecrop.csv

python3 predict.py \
    --image_dim 224 \
    --texture \
    --load_checkpoint ckpt/06_resnet18_b128/e040.pth \
    --model resnet18 \
    --output_csv output/06_resnet18_b128_e40_fivecrop_siw.csv \
    --test_dir siw_test
