import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
from models.layers.convolution import ConvBnAct

from models.initialize import weight_initialize


class VGG16(nn.Module):
    def __init__(self, in_channel, classes, init_weight=True):
        '''
        block_list [in_channels, out_channels, kernel_size, stride, block_itr, is_max]
        init_weight: weight initialize
        '''
        super(VGG16, self).__init__()
        block_1 = [in_channel, 64, (3, 3), True] 
        block_2 = [64, 128, (3, 3), True]
        block_3 = [128, 256, (3, 3, 3), True]
        block_4 = [256, 512, (3, 3, 3), True]
        block_5 = [512, 512, (3, 3, 3), False]

        self.vgg_block_1= self.VGGBlock(block_1)
        self.vgg_block_2= self.VGGBlock(block_2)
        self.vgg_block_3= self.VGGBlock(block_3)
        self.vgg_block_4= self.VGGBlock(block_4)
        self.vgg_block_5= self.VGGBlock(block_5)
        self.extra_input = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        if init_weight:
            weight_initialize(self)
    
    def forward(self, x):
        s1 = self.vgg_block_1(x)
        s2 = self.vgg_block_2(s1)
        s3 = self.vgg_block_3(s2)
        s4 = self.vgg_block_4(s3)
        s5 = self.vgg_block_5(s4)
        x = self.extra_input(s5)
        return x

    def VGGBlock(self, vgg_info):
        '''vgg_info = [in_ch, out_ch, kernels, is_maxpool]
        '''
        layers= list()
        in_ch, out_ch, kernels, is_maxpool = vgg_info
        for i in range(len(kernels)):
            if i == 0:
                layers.append(ConvBnAct(in_channel = in_ch, out_channel = out_ch, kernel_size = kernels[i], stride = 1, act=None))
            else:
                layers.append(ConvBnAct(in_channel = out_ch, out_channel = out_ch, kernel_size = kernels[i], stride = 1, act=None))
        if is_maxpool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)


class Extra_layer(nn.Module):
    def __init__(self, layer_info) -> None:
        super(Extra_layer, self).__init__()
        '''layer info : 
        [256, False]
        []
        '''
        self.layer_info = layer_info

    
    def forward(self, x):
        return x

class VGG(nn.Module):
    def __init__(self) -> None:
        super(VGG, self).__init__()


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


if __name__ == "__main__":
    # import torchvision
    # from torchvision import models, transforms

    # use_pretrained = True
    # pretrained_net = models.vgg16_bn(pretrained=use_pretrained)
    # pretrained_net.eval()
    # pretrained_net.classifier = Identity()
    # print(pretrained_net)

    vgg = VGG(in_channel=3, classes=100)
    x = vgg(torch.rand(1, 3, 300, 300))
    # print(vgg)

