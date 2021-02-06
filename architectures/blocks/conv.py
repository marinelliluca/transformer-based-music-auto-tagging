import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from functools import reduce
from operator import __add__

def _calc_same_pad(i, k, s, d):
    return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    Code adopted from: https://github.com/human-analysis/MUXConv/blob/master/conv2d_helpers.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        pad_h = _calc_same_pad(ih, kh, self.stride[0], self.dilation[0])
        pad_w = _calc_same_pad(iw, kw, self.stride[1], self.dilation[1])
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class BlockChoi(nn.Module):
    """     
    See Choi et al. 2016 https://arxiv.org/abs/1609.04243
    
    Differences from paper:
     - ELU
     - Pool before normalize
     - flag for max/avg --> False/True 
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 pool_size,
                 avgpool_flag):
        super(BlockChoi, self).__init__()
        
        self.conv = Conv2dSame(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size)
        
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.activation = nn.ELU() 
        
        if avgpool_flag:
            self.pool = nn.AvgPool2d(pool_size)
        else:
            self.pool = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout(0.1) #see paper
    
    def forward(self, inputs):
        x = self.conv(inputs)
        # pool before normalize https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/
        x = self.pool(x) 
        x = self.bn(x)
        x = self.activation(x)       
        x = self.dropout(x)
        return x