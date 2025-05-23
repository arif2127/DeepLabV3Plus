#!/bin/bash

# Run the Python script with parameters
python prediction.py \
    --image_path test_output/74_image.png \
    --save_path test_output \
    --ckpt checkpoints/best_deeplabv3Plus_modified.pth \
    --model deeplabv3Plus_modified \
    --upsample bilinear  \
    --gpu_id 0