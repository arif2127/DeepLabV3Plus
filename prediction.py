import os
import argparse
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn

import torchvision.transforms.functional as F
import transform  as tr
from model import DeepLabV3Plus, DeepLabV3Plus_Modified, DeepLabV3PlusMobilenet, DeepLabV3PlusMobilenet_Modified
from utils import decode_test_prediction

from PIL import Image




def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--image_path", type=str, default='/home/opin/Desktop/deltaX/cityscapes', help="path to image")
    parser.add_argument("--save_path", type=str, default='outputs', help="path to save")
    parser.add_argument("--num_classes", type=int, default=19, help="num classes (default: None)")

    # # Model Options
    parser.add_argument("--model", type=str, default='deeplabv3Plus_mobilenet', choices='deeplabv3Plus_modified , deeplabv3, deeplabv3Plus_mobilenet, deeplabv3Plus_mobilenet_modified', help='model name')
    parser.add_argument("--upsample", type=str, default='bilinear', choices=['bilinear', 'pixel_shuffle', 'transposed'])

    parser.add_argument("--ckpt", default="checkpoints/best_deeplabv3Plus_mobilenet.pth", type=str, help="restore from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")


    return parser

if __name__ == '__main__':



    opts = get_argparser().parse_args()

    # Setup Device Cuda or CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Set up model
    if opts.model == 'deeplabv3Plus':
        model = DeepLabV3Plus(num_classes=opts.num_classes)
    if opts.model == 'deeplabv3Plus_modified':
        model = DeepLabV3Plus_Modified(num_classes=opts.num_classes, upsample=opts.upsample)
    if opts.model == 'deeplabv3Plus_mobilenet':
        model = DeepLabV3PlusMobilenet(num_classes=opts.num_classes)
    if opts.model == 'deeplabv3Plus_mobilenet_modified':
        model = DeepLabV3PlusMobilenet_Modified(num_classes=opts.num_classes, upsample=opts.upsample)

    # Load the trained model
    if os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        del checkpoint  # free memory
    else:
        print("Please provide correct model path")
        exit()

    model.eval()

    # Process the input image
    test_transform = tr.ExtCompose([
        tr.ExtToTensor(),
        tr.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(opts.image_path).convert('RGB') # Load the images
    input_tensor, mask = test_transform(image, image) # Transform the image
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device, dtype=torch.float32)


    outputs = model(input_tensor) # Get Model output

    pred = outputs.detach().max(dim=1)[1].cpu().numpy()
    pred = pred.squeeze(0) # Remove the batch dimention
    pred = decode_test_prediction(pred).astype(np.uint8) # Decode the class id to color map

    if not os.path.exists(opts.save_path):
        os.mkdir(opts.save_path)

    Image.fromarray(pred).save(opts.save_path + '/result.png') #save the ouput color map
    print("The output is saved in folder " , opts.save_path)









