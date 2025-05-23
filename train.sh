#!/bin/bash

# Run the Python script with parameters
# --model options are deeplabv3Plus_modified , deeplabv3Plus, deeplabv3Plus_mobilenet, deeplabv3Plus_mobilenet_modified

python main.py \
    --data_root /path to image \
    --model deeplabv3Plus \
    --upsample bilinear  \
    --epoch 200 \
    --batch_size 16 \
    --gpu_id 0 \
    --object_wise_cutmix \
#    --ckpt None \
#    --continue_training \
#    --save_val_results