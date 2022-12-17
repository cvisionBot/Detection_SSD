import sys
sys.path.append('')
import torch
import torch.nn as nn
from backbone.layer.common import Conv, MP

class VGG(nn.Module):
    '''

    '''
    def __init__(self, in_c, info, extra_info, init_weight=True):
        super(VGG, self).__init__()
        self.in_c= in_c
        self.vgg_layer= self.create_conv_layer(info)
        self.add_layer= nn.Sequential(
            nn.Conv2d(in_channels= 512, out_channels= 1024, kernel_size=3, padding=6, dilation= 6),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels= 1024, kernel_size= 1),
            nn.ReLU()
        )

        self.extra_layer= self.create_extra_layers(extra_info)
        
        if init_weight:
            self._initialize_weights()
    
    def forward(self,x):
        x= self.vgg_layer(x)
        x= self.add_layer(x)
        x= self.extra_layer(x)
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

    def create_extra_layers(self, extra_info):
        in_c= 1024
        layers= list()

        layers += [nn.Conv2d(in_c, extra_info[0], kernel_size=1)]
        layers += [nn.Conv2d(extra_info[0], extra_info[1], kernel_size=3, stride=2, padding=1)]
        layers += [nn.Conv2d(extra_info[1], extra_info[2], kernel_size=1)]
        layers += [nn.Conv2d(extra_info[2], extra_info[3], kernel_size=3, stride=2, padding=1)]
        layers += [nn.Conv2d(extra_info[3], extra_info[4], kernel_size=1)]
        layers += [nn.Conv2d(extra_info[4], extra_info[5], kernel_size=3)]
        layers += [nn.Conv2d(extra_info[5], extra_info[6], kernel_size=1)]
        layers += [nn.Conv2d(extra_info[6], extra_info[7], kernel_size=3)]        

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

def backbone():
    vgg16_info = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M']
    extra_info= [256, 512, 128, 256, 128, 256, 128, 256]
    net= VGG(3, vgg16_info, extra_info, True)
    return net

if __name__ == '__main__':
    net= backbone()
    x= torch.rand((1, 3, 300, 300))
    print(net)
    output= net(x)
    print(output.size())