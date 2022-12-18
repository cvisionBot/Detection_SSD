import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn

from models.backbone.vgg import VGG_SSD

class SSD(nn.Module):
    def __init__(self, info_dict) -> None:
        super(SSD, self).__init__()
        ''' ssd base : vgg_ssd
            info dict : 'num_classes',
                        'num_boxes'
        '''
        self.num_classes = info_dict['num_classes']
        self.boxes_cfg = info_dict['num_boxes']
        

    def forward(self, x):
        loc_layers = list()
        conf_layers = list()
        for i, c in enumerate(x):
            loc_layers += [
                nn.Conv2d(in_channels= c.shape[1], out_channels= self.boxes_cfg[i] * 4, kernel_size=3, padding=1)
            ]
            conf_layers += [
                nn.Conv2d(in_channels= c.shape[1], out_channels= self.boxes_cfg[i] * self.num_classes, kernel_size=3, padding=1)
            ]
        return loc_layers, conf_layers


if __name__ == "__main__":
    vgg_ssd_info = {
        "num_classes":21,
        "num_boxes":[4, 6, 6, 6, 4, 4]
    }
    vgg_ssd = VGG_SSD(in_channel=3, init_weight=False)
    x = vgg_ssd(torch.rand(1, 3, 300, 300))
    ssd = SSD(info_dict=vgg_ssd_info)
    x = ssd(x)
    print(x)
