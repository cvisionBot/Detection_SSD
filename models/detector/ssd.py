import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn

from models.backbone.vgg import VGG_SSD


def ssd_head(vgg_base, num_classes, boxes_cfg) -> None:
    ''' 
        vgg_base    : conv5, conv7, conv8, conv9, conv10, conv11  
            conv5, 7         -> backbone stages  
            conv8, 9, 10, 11 -> extra layer stages  
        num_classes : number of classes  
        boxes_cfg   :   
    '''
    return 

class SSD(nn.Module):
    def __init__(self, vgg_base, info_dict) -> None:
        super(SSD, self).__init__()
        ''' ssd base : vgg_ssd
            info dict : 'num_classes',
                        'num_boxes'
        '''
        self.num_classes = info_dict['num_classes']
        self.boxes_cfg = info_dict['num_boxes']
        self.loc_layer = self.localization_model(vgg_base, self.boxes_cfg)
        self.conf_layer = self.classification_model(vgg_base, self.boxes_cfg, self.num_classes)

    def forward(self, x):
        loc_layers = list()
        conf_layers = list()
        for i, c in enumerate(x):
            loc_layers += [self.loc_layer[i](c)]
            conf_layers += [self.conf_layer[i](c)]
        return loc_layers, conf_layers
    
    def localization_model(self, ssd_base, boxes_cfg):
        loc_layers = list()
        for i, c in enumerate(ssd_base):
            loc_layers += [
                nn.Conv2d(in_channels= c.shape[1], out_channels= boxes_cfg[i] * 4, kernel_size=3, padding=1)
            ]
        
        return loc_layers
        
    def classification_model(self, ssd_base, boxes_cfg, num_classes):
        conf_layers = list()
        for i, c in enumerate(ssd_base):
            conf_layers += [
                nn.Conv2d(in_channels= c.shape[1], out_channels= boxes_cfg[i] * num_classes, kernel_size=3, padding=1)
            ]
        
        return conf_layers


if __name__ == "__main__":
    vgg_ssd_info = {
        "num_classes":21,
        "num_boxes":[4, 6, 6, 6, 4, 4]
    }
    vgg_ssd = VGG_SSD(in_channel=3, init_weight=False)
    x = vgg_ssd(torch.rand(1, 3, 300, 300)) # 6 stages
    ssd = SSD(vgg_base=x, info_dict=vgg_ssd_info)
    loc_layer, conf_layer = ssd(x)
    print(len(loc_layer))
