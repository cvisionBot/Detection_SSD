import sys
sys.path.append('')
import torch
import torch.nn as nn
from backbone.layer.common import Conv, MP

class VGG(nn.Module):
    '''

    '''
    def __init__(self, in_c, nc, info, init_weight=True):
        super(VGG, self).__init__()
        self.in_c= in_c
        self.vgg_layer= self.create_conv_layer(info)
        self.add_layer= nn.Sequential(
            nn.Conv2d(in_channels= 512, out_channels= 1024, kernel_size=3, padding=6, dilation= 6),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels= 1024, kernel_size= 1),
            nn.ReLU()
        )

        
        if init_weight:
            self._initialize_weights()
    
    def forward(self,x):
        x= self.vgg_layer(x)
        x= self.add_layer(x)
        return x
    
    def create_conv_layer(self, info):
        layers= list()
        in_c= self.in_c
        
        for x in info:
            if type(x) == int:
                out_c= x
                layers+= [Conv(in_c= in_c, out_c= out_c, k= 3, s= 1, p= None, g= 1, act= True)]
                in_c= x   
            elif x == 'M':
                layers+= [MP(k=2)]
        
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
    
    net= VGG(3, 10, VGG16, True)
    x= torch.rand((1, 3, 300, 300))
    output= net(x)
    print(output.size())