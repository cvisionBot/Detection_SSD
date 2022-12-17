import torch
import torch.nn as nn

from model.backbone.common import MP, Conv

class Cretae_Layer(nn.Module):
    def __init__(self, cfg, in_c):
        super(Cretae_Layer, self).__init__()
        self.layer= self._create_layer(cfg, in_c)

    def forward(self, x):
        x= self.layer(x)
        return x

    def _create_layer(self, cfg, in_c):
        layers= list()
        in_c= in_c
        for x in cfg:
            if type(x) == int:
                out_c= x
                layers+= [Conv(in_c, out_c, k= 3, s= 1, p= None)]
                in_c= x
            elif x == 'M':
                layers+= [MP(2)]
        return nn.Sequential(*layers)


class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase, self).__init__()
        cfg_1= [64, 64, 'M']
        cfg_2= [128, 128, 'M']
        cfg_3= [256, 256, 256, 'M']
        cfg_4= [512, 512, 512, 'M']
        cfg_5= [512, 512, 512]
        cfg_6= [512, 1024]
        cfg_7= [1024, 1024]

        self.conv1= Cretae_Layer(cfg_1, 3)
        self.conv2= Cretae_Layer(cfg_2, cfg_1[0])
        self.conv3= Cretae_Layer(cfg_3, cfg_2[0])
        self.conv4= Cretae_Layer(cfg_4, cfg_3[0])
        self.conv5= Cretae_Layer(cfg_5, cfg_4[0])
        self.pool5= nn.MaxPool2d(3, 1, 1)
        self.conv6= Cretae_Layer(cfg_6, cfg_5[0])
        self.conv7= Cretae_Layer(cfg_7, cfg_7[0])


    def forward(self, x):
        conv1= self.conv1(x)
        conv2= self.conv2(conv1)
        conv3= self.conv3(conv2)
        conv4= self.conv4(conv3)
        conv5= self.conv5(conv4)
        conv5= self.pool5(conv5)
        conv6= self.conv6(conv5)
        conv7= self.conv7(conv6)

        return conv4, conv7

class AuxiliaryConv(nn.Module):
    def __init__(self):
        super(AuxiliaryConv, self).__init__()
        cfg_8= [1024, 256, 512]
        cfg_9= [512, 128, 256]
        cfg_10= [256, 128, 256]
        cfg_11= [256, 128, 256]

        self.conv8= nn.Sequential(
            nn.Conv2d(cfg_8[0], cfg_8[1], 1, 1, 0),
            nn.Conv2d(cfg_8[1], cfg_8[2], 3, 2, 1),
            nn.ReLU()
        )

        self.conv9= nn.Sequential(
            nn.Conv2d(cfg_9[0], cfg_9[1], 1, 1, 0),
            nn.Conv2d(cfg_9[1], cfg_9[2], 3, 2, 1),
            nn.ReLU()
        )

        self.conv10= nn.Sequential(
            nn.Conv2d(cfg_10[0], cfg_10[1], 1, 1, 0),
            nn.Conv2d(cfg_10[1], cfg_10[2], 3, 1, 0),
            nn.ReLU()
        )

        self.conv11= nn.Sequential(
            nn.Conv2d(cfg_11[0], cfg_11[1], 1, 1, 0),
            nn.Conv2d(cfg_11[1], cfg_11[2], 3, 1, 0),
            nn.ReLU()
        )

    def forward(self, conv7):
        conv8= self.conv8(conv7)
        conv9= self.conv9(conv8)
        conv10= self.conv10(conv9)
        conv11= self.conv11(conv10)

        return conv8, conv9, conv10, conv11

class SSD_BackBone(nn.Module):
    def __init__(self):
        super(SSD_BackBone, self).__init__()
        self.vgg= VGGBase()
        self.extra= AuxiliaryConv()

    def forward(self, x):
        conv4, conv7= self.vgg(x)
        conv8, conv9, conv10, conv11= self.extra(conv7)

        return conv4, conv7, conv8, conv8, conv9, conv10, conv11

if __name__ == '__main__':
    model = SSD_BackBone()
    output= torch.randn((1, 3, 300, 300))
    print(model)
    conv4, conv7, conv8, conv8, conv9, conv10, conv11= model(output)





