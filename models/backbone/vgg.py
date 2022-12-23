import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
from models.layers.convolution import ConvBnAct

from models.initialize import weight_initialize

class block(nn.Module):
    def __init__(self, info) -> None:
        super(block, self).__init__()
        '''vgg_info = [in_ch, out_ch, kernels, is_maxpool]
        '''
        self.in_ch, self.out_ch, self.kernels, self.maxpool = info
        self.block = self.create_block()

    def forward(self, x):
        return self.block(x)

    def create_block(self):
        layers= list()
        in_ch = self.in_ch
        for i in range(len(self.kernels)):
            layers.append(ConvBnAct(in_channel = in_ch, out_channel = self.out_ch, kernel_size = self.kernels[i], stride = 1))
            in_ch = self.out_ch
        if self.maxpool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)

class VGG16(nn.Module):
    def __init__(self, in_channel, init_weight=True):
        '''
        block_list [in_channels, out_channels, kernel_sizes, maxpool]
        init_weight: weight initialize
        '''
        super(VGG16, self).__init__()
        block_1 = [in_channel, 64, (3, 3), True] 
        block_2 = [64, 128, (3, 3), True]
        block_3 = [128, 256, (3, 3, 3), False]
        block_4 = [256, 512, (3, 3, 3), False]
        block_5 = [512, 512, (3, 3, 3), False]
        block_6 = [512, 1024]
        block_7 = [1024, 1024, (1,), False]

        self.conv1= block(block_1)
        self.conv2= block(block_2)
        self.conv3= block(block_3)
        self.conv3_mp = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv4= block(block_4)
        self.conv4_mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5= block(block_5)
        self.conv5_mp = nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv6= ConvBnAct(in_channel=block_6[0], out_channel=block_6[1], kernel_size=3, stride=1, padding=6, dilation=6)
        self.conv7= block(block_7)

        if init_weight:
            weight_initialize(self)
    
    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c3_mp = self.conv3_mp(c3)
        c4 = self.conv4(c3_mp) # feature map 1
        print(c4.shape)
        c4_mp = self.conv4_mp(c4)
        c5 = self.conv5(c4_mp)
        c5_mp = self.conv5_mp(c5)
        c6 = self.conv6(c5_mp)
        c7 = self.conv7(c6)
        print(c7.shape)
        return c4, c7

class Extra_layer(nn.Module):
    def __init__(self, layer_info) -> None:
        super(Extra_layer, self).__init__()
        '''layer info : 
                [
                    [in_channel, out_channel, is_stride],
                    ...
                ]
        '''
        self.extra_conv1 = self.block(layer_info[0])
        self.extra_conv2 = self.block(layer_info[1])
        self.extra_conv3 = self.block(layer_info[2])
        self.extra_conv4 = self.block(layer_info[3])
    
    def forward(self, x):
        c1 = self.extra_conv1(x)
        print(c1.shape)
        c2 = self.extra_conv2(c1)
        print(c2.shape)
        c3 = self.extra_conv3(c2)
        print(c3.shape)
        c4 = self.extra_conv4(c3)
        print(c4.shape)
        return c1, c2, c3, c4

    def block(self, layer_info):
        in_channel, out_channel, is_stride = layer_info
        layer = list()
        layer += [
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel//2, kernel_size=1, stride=1),
            nn.Conv2d(in_channels=out_channel//2, out_channels=out_channel, kernel_size=3, \
                      stride=2 if is_stride else 1, padding=1 if is_stride else 0),
        ]
        return nn.Sequential(*layer)

class VGG_SSD(nn.Module):
    def __init__(self, in_channel, init_weight = True) -> None:
        super(VGG_SSD, self).__init__()
        self.vgg16 = VGG16(in_channel=in_channel, init_weight=init_weight)
        """
            about vgg16-ssd, extract total 4 conv blocks at extra layer
        """
        self.extra = Extra_layer(layer_info=[
            [1024, 512, True],
            [512, 256, True], 
            [256, 256, False],
            [256, 256, False],
        ])
    def forward(self, x):
        c4, c7 = self.vgg16(x)
        c8, c9, c10, c11 = self.extra(c7)
        return c4, c7, c8, c9, c10, c11


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def forward(self, x):
        return x


if __name__ == "__main__":
    vgg_ssd = VGG_SSD(in_channel=3, init_weight=False)
    x = vgg_ssd(torch.rand(1, 3, 300, 300))

