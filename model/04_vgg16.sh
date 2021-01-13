#!/bin/bash

NAME=04_vgg16
MODEL=vgg16

python3 main.py \
    --image_dim 224 \
    --batch_size 96 \
    --texture \
    --name $NAME \
    --model $MODEL \
    --total_epoch 50 \
    --checkpoint_epoch 5

python3 predict.py \
    --image_dim 224 \
    --texture \
    --load_checkpoint ckpt/$NAME/e50.pth \
    --model $MODEL \
    --output_csv output/$NAME.csv

python3 predict.py \
    --image_dim 224 \
    --texture \
    --load_checkpoint ckpt/$NAME/e50.pth \
    --model $MODEL \
    --output_csv output/$NAME.csv \
    --test_dir siw_test
