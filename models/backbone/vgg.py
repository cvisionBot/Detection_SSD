import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
from models.layers.convolution import ConvBnAct


class VGG(nn.Module):
    def __init__(self, in_channel, classes, init_weight=True):
        '''
        block_list [in_channels, out_channels, kernel_size, stride, block_itr, is_max]
        init_weight: weight initialize
        '''
        super(VGG, self).__init__()
        block_1 = [in_channel, 64, (3, 3), True] 
        block_2 = [64, 128, (3, 3), True]
        block_3 = [128, 256, (3, 3, 3), True]
        block_4 = [256, 512, (3, 3, 3), True]
        block_5 = [512, 512, (3, 3, 3), True]
        self.vgg_block_1= self.VGGBlock(block_1)
        self.vgg_block_2= self.VGGBlock(block_2)
        self.vgg_block_3= self.VGGBlock(block_3)
        self.vgg_block_4= self.VGGBlock(block_4)
        self.vgg_block_5= self.VGGBlock(block_5)
        self.classification = nn.Sequential(
            ConvBnAct(in_c=512, out_c=1280, k=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, classes, 1)
        )
       
    def forward(self, x):
        s1 = self.vgg_block_1(x)
        s2 = self.vgg_block_2(s1)
        s3 = self.vgg_block_3(s2)
        s4 = self.vgg_block_4(s3)
        s5 = self.vgg_block_5(s4)
        x = self.classification(s5)
        b, c, _, _ = x.size()
        x = x.view(b, c)
        return x

    def VGGBlock(self, vgg_info):
        '''vgg_info = [in_ch, out_ch, kernels, is_maxpool]
        '''
        layers= list()
        in_c, out_c, kernels, is_maxpool = vgg_info
        for i in range(len(kernels)):
            if i == 0:
                layers.append(ConvBnAct(in_c = in_c, out_c = out_c, k = kernels[i], s = 1, act=None))
            else:
                layers.append(ConvBnAct(in_c = out_c, out_c = out_c, k = kernels[i], s = 1, act=None))
        if is_maxpool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)

if __name__ == "__main__":
    vgg = VGG(in_channel=3, classes=100)
    x = vgg(torch.rand(1, 3, 224, 224))

