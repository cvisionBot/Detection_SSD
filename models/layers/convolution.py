import torch
from torch import nn

def autopad(k, p=None):
    if p is None:
        p=k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class ConvBnAct(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 1, stride = 1, padding = None, dilation = 1, groups = 1, act = None):
        super(ConvBnAct, self).__init__()
        self.act = nn.ReLU(inplace=True) if act == None else act
        self.conv= nn.Conv2d(in_channel, out_channel, kernel_size, stride, autopad(kernel_size, padding), dilation=dilation, groups= groups, bias= False)
        self.bn= nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x= self.conv(x)
        x= self.bn(x)
        x= self.act(x)
        return x

