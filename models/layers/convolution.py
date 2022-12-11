import torch
from torch import nn

def autopad(k, p=None):
    if p is None:
        p=k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ConvBnAct(nn.Module):
    def __init__(self, in_c, out_c, k= 1, s= 1, p= None, g= 1, act = None):
        super(ConvBnAct, self).__init__()
        self.act = nn.ReLU() if act == None else act
        self.conv= nn.Conv2d(in_c, out_c, k, s, autopad(k, p), groups= g, bias= False)
        self.bn= nn.BatchNorm2d(out_c)

    def forward(self, x):
        x= self.conv(x)
        x= self.bn(x)
        x= self.act(x)
        return x

