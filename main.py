import os
import argparse
import numpy as np
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils import data

from dataset import Cityscapes
import transform  as tr
from model import DeepLabV3Plus, DeepLabV3Plus_Modified, DeepLabV3PlusMobilenet, DeepLabV3PlusMobilenet_Modified
from utils import save_ckpt, mkdir, Denormalize, object_wise_cutmix
from strean_matrix import StreamSegMetrics


from PIL import Image
import matplotlib
import matplotlib.pyplot as plt




def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='/home/opin/Desktop/deltaX/cityscapes', help="path to Dataset")
    parser.add_argument("--num_classes", type=int, default=19, help="num classes (default: None)")

    # # Model Options
    parser.add_argument("--model", type=str, default='deeplabv3Plus_mobilenet', choices='deeplabv3Plus_modified , deeplabv3Plus, deeplabv3Plus_mobilenet, deeplabv3Plus_mobilenet_modified', help='model name')
    parser.add_argument("--upsample", type=str, default='bilinear', choices=['bilinear', 'pixel_shuffle', 'transposed'])

    # # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=True, help="save segmentation results to \"./results\"")
    parser.add_argument("--epoch", type=int, default=300, help="epoch number (default: 1000)")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--step_size", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16, help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=4, help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--object_wise_cutmix", action='store_true', default=False, help="object wise cutmix for data augmentation")

    parser.add_argument("--ckpt", default="checkpoints/best_deeplabv3Plus_mobilenet.pth", type=str, help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--print_interval", type=int, default=10, help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=10, help="epoch interval for eval (default: 10)")

    # # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False, help="use visdom for visualization")

    return parser


def get_dataset(opts):

    # Training Data Augmentation
    train_transform = tr.ExtCompose([
        # tr.ExtResize((1024, 512)),
        tr.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
        tr.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        tr.ExtRandomHorizontalFlip(),
        tr.ExtToTensor(),
        tr.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = tr.ExtCompose([
        # et.ExtResize( 512 ),
        # tr.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
        tr.ExtToTensor(),
        tr.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    train_dst = Cityscapes(root=opts.data_root, split='train', transform=train_transform)
    val_dst = Cityscapes(root=opts.data_root, split='val', transform=val_transform)

    return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):

    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs= model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    # Save the image , pred , and target with respective class color
                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    # Save the image and pred overlaped
                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score




def main():

    opts = get_argparser().parse_args()

    # Setup Device Cuda or CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup dataloader
    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %("cityscapse fine only", len(train_dst), len(val_dst)))
    total_iteration = int( len(train_dst) / opts.batch_size)


    # Set up model
    if opts.model == 'deeplabv3Plus':
        model = DeepLabV3Plus(num_classes=opts.num_classes)
    if opts.model == 'deeplabv3Plus_modified':
        model = DeepLabV3Plus_Modified(num_classes=opts.num_classes, upsample= opts.upsample)
    if opts.model == 'deeplabv3Plus_mobilenet':
        model = DeepLabV3PlusMobilenet(num_classes=opts.num_classes)
    if opts.model == 'deeplabv3Plus_mobilenet_modified':
        model = DeepLabV3PlusMobilenet_Modified(num_classes=opts.num_classes, upsample= opts.upsample)



    # Set up metrics for evaluation
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD( model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    #Set up criterion (loss)
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    #checkpoints setup and restore for pretraining
    mkdir('checkpoints')

    best_score = 0.0
    epoch_curent = 0

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            epoch_curent = checkpoint["epoch"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)

        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)




    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples, np.int32) if opts.enable_vis else None  # sample idxs for visualization


    # only for validation with pretrained model
    if opts.test_only:
        if opts.ckpt is not None and os.path.isfile(opts.ckpt):
            print("validation start")
            model.eval()
            val_score = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                ret_samples_ids=vis_sample_id)
            print(metrics.to_str(val_score))
            exit(0)
        else:
            print("Please provide the model file for only testing")
            exit(0)

    ############################################################################
    # train:
    ############################################################################

    for epoch in range(epoch_curent+1, opts.epoch):
        model.train()
        cur_itrs = 0
        interval_loss = 0
        for (images, labels) in train_loader:

            cur_itrs += 1
            if opts.object_wise_cutmix:
                images, labels = object_wise_cutmix(images, labels, class_ids=None)
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            loss= criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            # Print loss in every opts.print_interval
            if (cur_itrs) % opts.print_interval == 0:
                interval_loss = interval_loss / opts.print_interval
                print("Epoch %d/%d, Itrs %d/%d, Loss=%f" %
                      (epoch, opts.epoch, cur_itrs, total_iteration, interval_loss))
                interval_loss = 0.0

            scheduler.step()

        # run validation in every opts.val_interval
        if (epoch+1) % opts.val_interval == 0:
            # save the latest model
            save_ckpt('checkpoints/latest_%s.pth' %opts.model, epoch, model, optimizer, best_score)
            print("validation...")
            model.eval()
            val_score  = validate(
                opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                ret_samples_ids=vis_sample_id)
            print(metrics.to_str(val_score))
            if val_score['Mean IoU'] > best_score:  # save best model
                best_score = val_score['Mean IoU']
                save_ckpt('checkpoints/best_%s.pth' %opts.model, epoch, model, optimizer, best_score)


            model.train()


if __name__ == '__main__':
    main()