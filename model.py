import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3, DeepLabV3_ResNet50_Weights, deeplabv3_mobilenet_v3_large
# import os
# import functools


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio = 16):
        super(CBAM, self).__init__()

        self.CAM = ChannelAttention(in_planes, ratio = ratio)
        self.SAM = SpatialAttention()

    def forward(self, x):

        x = self.CAM(x) * x
        return self.SAM(x) * x


class Decoder(nn.Module):
    def __init__(self, in_channels=256+256, out_channels= 256):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(),
                                     nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU()
                                     )

    def forward(self, x):

        return self.decoder(x)



class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone = 'resnet50', num_classes = 20):
        super(DeepLabV3Plus, self).__init__()

        model = models.segmentation.deeplabv3_resnet50(pretrained=True)

        # Access the backbone (ResNet) and split it into its parts
        backbone = model.backbone

        self.head = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.ASPP = model.classifier[0]

        # process low-features
        self.low_feature_process = nn.Conv2d(256, 256, 1)


        self.decoder = nn.Sequential(nn.Conv2d(256+256, 256, kernel_size=3, stride=1, padding=1, bias=False),
               nn.BatchNorm2d(256),
               nn.ReLU(),
               nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
               nn.BatchNorm2d(256),
               nn.ReLU()
        )

        self.classifier  = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)



    def forward(self, x):

        input_size = x.size()[2:]

        # Backbone resnet50 pretrained
        x = self.head(x)
        low_features = self.layer1(x)
        low_features_size = low_features.size()[2:]
        x = self.layer2(low_features)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP pretrained
        x = self.ASPP(x)

        # Upsample the ASPP features
        x = F.upsample(x, size=low_features_size, mode="bilinear")

        #process the low_features for concate
        low_features = self.low_feature_process(low_features)

        # Concate low_feature and upsampled aspp features
        x = torch.cat((x, low_features), dim=1)
        x = self.decoder(x)
        return F.upsample(self.classifier(x), size=input_size, mode="bilinear")




class DeepLabV3Plus_Modified(nn.Module):
    def __init__(self, num_classes = 19, upsample = 'bilinear'):
        super(DeepLabV3Plus_Modified, self).__init__()
        self.upsample = upsample

        # Load deeplabv3_resnet50 and the pretrained model was trained with COCO dataset
        model = models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)


        # Access the backbone (ResNet) and split it into its parts
        backbone = model.backbone

        # backbone Head
        self.head = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        # backbone intermidiate features
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Load ASPP from deeplabv3_resnet50
        self.ASPP = model.classifier[0]


        # process low-features

        self.cbam = CBAM(256)


        if self.upsample == "pixel_shuffle" :
            self.decoder = Decoder(64+64+256, 256)
        else:
            self.decoder = Decoder(256+256, 256)

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)



    def forward(self, x):

        input_size = x.size()[2:]

        # Backbone resnet50 pretrained
        x = self.head(x)
        x = self.layer1(x)
        low_features_size = x.size()[2:]
        # process the low_features for concate
        low_features = x + self.cbam(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ASPP(x)

        # Upsample the ASPP features
        if self.upsample == 'bilinear':
            x = F.upsample(x, size=low_features_size, mode="bilinear")
        elif self.upsample == 'pixel_shuffle':
            x = F.pixel_shuffle(x, upscale_factor=2)
            x = F.upsample(x, size=low_features_size, mode="bilinear")


        # Concate low_feature and upsampled aspp features
        x = torch.cat((x, low_features), dim=1)

        x = self.decoder(x)
        x = self.classifier(x)

        return F.upsample(x, size=input_size, mode="bilinear")


class DeepLabV3PlusMobilenet(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3PlusMobilenet, self).__init__()

        mobilenet = models.mobilenet_v3_large(pretrained=True)
        mobilenet.features[13].block[1][0] = nn.Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), dilation=(2, 2), groups=672, bias=False)
        deeplabv3 = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)


        # Extract layers from the model
        self.low_level = mobilenet.features[:4]  # First 2 layers (Conv + Bottleneck 1)
        # self.mid_level = mobilenet.features[4:7]  # Intermediate layers (Bottlenecks 2-6)
        self.high_level = mobilenet.features[4:]  # Final layers (Bottlenecks 7-end)


        # Extract ASPP from the deeplabv3
        self.ASPP = deeplabv3.classifier[0]

        # process low-features
        self.low_feature_process = nn.Conv2d(24, 256, 1)

        self.decoder = nn.Sequential(nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU()
                                     )

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)




    def forward(self, x):
        input_size = x.size()[2:]

        low_features = self.low_level(x)
        low_features_size = low_features.size()[2:]
        x = self.high_level(low_features)

        x = self.ASPP(x)

        # Upsample the ASPP features
        x = F.upsample(x, size=low_features_size, mode="bilinear")

        # process the low_features for concate
        low_features = self.low_feature_process(low_features)

        # Concate low_feature and upsampled aspp features
        x = torch.cat((x, low_features), dim=1)
        x = self.decoder(x)

        return F.upsample(self.classifier(x), size=input_size, mode="bilinear")


class DeepLabV3PlusMobilenet_Modified(nn.Module):
    def __init__(self, num_classes, upsample = 'bilinear'):
        super(DeepLabV3PlusMobilenet_Modified, self).__init__()

        self.upsample =upsample

        mobilenet = models.mobilenet_v3_large(pretrained=True)
        mobilenet.features[13].block[1][0] = nn.Conv2d(672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(4, 4), dilation=(2, 2), groups=672, bias=False)
        deeplabv3 = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)


        # Extract layers from the model
        self.level1 = mobilenet.features[:2]  # First 2 layers (Conv + Bottleneck 1)
        self.level2 = mobilenet.features[2:4]  # Intermediate layers (Bottlenecks 2-6)
        self.level3 = mobilenet.features[4:]  # Final layers (Bottlenecks 7-end)

        self.cbam1 = CBAM(16, ratio = 2)
        self.cbam2 = CBAM(24, ratio = 4)
        self.cbam3 = CBAM(960, ratio = 16)

        # Extract ASPP from the deeplabv3
        self.ASPP = deeplabv3.classifier[0]

        # process low-features
        self.low_feature_process = nn.Conv2d(24, 256, 1)


        # pixel_shuffle reduce the number of channel (2^scale_factor) times
        if self.upsample == "pixel_shuffle" :
            self.decoder = nn.Sequential(nn.Conv2d((256/16) + 256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(),
                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU()
                                         )
        else:
            self.decoder = nn.Sequential(nn.Conv2d(256 + 256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU(),
                                         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                         nn.BatchNorm2d(256),
                                         nn.ReLU()
                                         )



        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)




    def forward(self, x):
        input_size = x.size()[2:]

        x =self.cbam1(self.level1(x))
        low_features =self.cbam2(self.level2(x))
        low_features_size = low_features.size()[2:]
        x = self.cbam3(self.level3(low_features))

        x = self.ASPP(x)

        # Upsample the ASPP features
        if self.upsample == "pixel_shuffle":
            x = F.upsample(x, size=low_features_size, mode="bilinear")

        if self.upsample == 'bilinear':
            x = F.upsample(x, size=low_features_size, mode="bilinear")
        elif self.upsample == 'pixel_shuffle':
            x = F.pixel_shuffle(x, upscale_factor=4)
            x = F.upsample(x, size=low_features_size, mode="bilinear")

        # process the low_features for concate
        low_features = self.low_feature_process(low_features)

        # Concate low_feature and upsampled aspp features
        x = torch.cat((x, low_features), dim=1)
        x = self.decoder(x)

        return F.upsample(self.classifier(x), size=input_size, mode="bilinear")



import transform  as tr
from dataset import Cityscapes
import torch.utils.data as data


if __name__ == '__main__':

    # model = CustomDeepLabV3()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = tr.ExtCompose([
        tr.ExtResize( (1024,512) ),
        tr.ExtRandomCrop(size=(512, 512)),
        tr.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        tr.ExtRandomHorizontalFlip(),
        tr.ExtToTensor(),
        tr.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = tr.ExtCompose([
        # et.ExtResize( 512 ),
        tr.ExtToTensor(),
        tr.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    train_dst = Cityscapes(root='/home/opin/Desktop/deltaX/cityscapes', split='train', transform=train_transform)
    val_dst = Cityscapes(root='/home/opin/Desktop/deltaX/cityscapes', split='val', transform=val_transform)

    train_loader = data.DataLoader(train_dst, batch_size=16, shuffle=True, num_workers=2,drop_last=True)

    model = DeepLabV3PlusMobilenet(num_classes = 19)
    model.to(device)

    for (images, labels) in train_loader:


        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)


        outputs = model(images)

        print(outputs.shape)