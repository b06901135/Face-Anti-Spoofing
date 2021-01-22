#!/bin/bash

TAG="_class"
declare -a NAME=("01_resnet18" "02_resnet50" "03_vgg11" "04_vgg16" "05_vgg19")
declare -a MODEL=("resnet18" "resnet50" "vgg11" "vgg16" "vgg19")

for i in $(seq 0 4);
do
    python3 predict.py \
        --image_dim 224 \
        --texture \
        --category \
        --load_checkpoint ckpt_final/${NAME[i]}_e50.pth \
        --model ${MODEL[i]} \
        --test_dir $1 \
        --output_csv output/${NAME[i]}${TAG}.csv
done

python3 blend.py --category --tag ${TAG} --output $2
