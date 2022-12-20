import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn

from models.backbone.vgg import VGG_SSD


def ssd_head(vgg_base, num_classes, boxes_cfg):
    ''' 
        vgg_base    : conv5, conv7, conv8, conv9, conv10, conv11  
            conv5, 7         -> backbone stages  
            conv8, 9, 10, 11 -> extra layer stages  
        num_classes : number of classes  
        boxes_cfg   :   [4, 6, 6, 6, 4, 4]
    '''
    loc_layers = []
    conf_layers = []
    for i, c in enumerate(vgg_base):
        loc_layers += [
            nn.Conv2d(in_channels= c.shape[1], out_channels= boxes_cfg[i] * 4, kernel_size=3, padding=1)
        ]
        conf_layers += [
            nn.Conv2d(in_channels= c.shape[1], out_channels= boxes_cfg[i] * num_classes, kernel_size=3, padding=1)
        ]
    return loc_layers, conf_layers

'''input params : vgg_base, extra_layers, (loc_layer, conf_layer)
'''
class SSD(nn.Module):
    def __init__(self, vgg_ssd_info):
        super(SSD, self).__init__()
        self.img_size = vgg_ssd_info['img_size']
        self.num_classes = vgg_ssd_info['num_classes']
        self.boxes_cfg = vgg_ssd_info['boxes_cfg']
        self.vgg_ssd = VGG_SSD(in_channel=3, init_weight=False)
        self.loc_layers, self.conf_layers = ssd_head(self.vgg_ssd, self.num_classes, self.boxes_cfg)
        
    


if __name__ == "__main__":
    vgg_ssd_info = {
        "img_size":300,
        "num_classes":21,
        "boxes_cfg":[4, 6, 6, 6, 4, 4]
    }
    vgg_ssd = VGG_SSD(in_channel=3, init_weight=False)
    x = vgg_ssd(torch.rand(1, 3, 300, 300)) # 6 stages
    ssd = SSD(vgg_base=x, info_dict=vgg_ssd_info)
    loc_layer, conf_layer = ssd(x)
    print(len(loc_layer))
