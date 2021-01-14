#!/bin/bash

# Video model
NAME=08_resnet_r21d
MODEL=resnet_r21d

python3 main.py \
    --image_dim 112 \
    --batch_size 24 \
    --name $NAME \
    --model $MODEL \
    --total_epoch 50 \
    --checkpoint_epoch 10

python3 predict.py \
    --image_dim 112 \
    --load_checkpoint ckpt/$NAME/e50.pth \
    --model $MODEL \
    --output_csv output/"${NAME}.csv"

python3 predict.py \
    --image_dim 112 \
    --load_checkpoint ckpt/$NAME/e50.pth \
    --model $MODEL \
    --output_csv output/"${NAME}_siw.csv" \
    --test_dir siw_test

python3 predict.py \
    --image_dim 112 \
    --category \
    --load_checkpoint ckpt/$NAME/e50.pth \
    --model $MODEL \
    --output_csv output/"${NAME}_cat.csv" \
    --test_dir siw_test
