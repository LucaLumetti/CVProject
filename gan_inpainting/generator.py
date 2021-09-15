import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import GatedConv, GatedDeConv, SelfAttention

def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

# TODO: maybe this get_pad function can be removed and implemented inside the
# gated conv layer, this will also remove the dependency from the img size of
# 256x256
# The img size dependency can be removed easy by setting a variable and *2 or /2
# each time we down/upsample
class Generator(nn.Module):
    def __init__(self, input_channels=5, cnum=32):
        super(Generator, self).__init__()
        self.cnum = cnum
        self.coarse_net = nn.Sequential(
                GatedConv(input_channels, self.cnum, 5, 1, padding=get_pad(256, 5, 1)),
                #downsampling
                GatedConv(self.cnum, 2*self.cnum, 4, 2, padding=get_pad(256, 4, 2)),
                GatedConv(2*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                #downsampling
                GatedConv(2*self.cnum, 4*self.cnum, 4, 2, padding=get_pad(128, 4, 2)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1)),
                #atrous
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
                #conv
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1, 2)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1, 2)),
                #upsample
                GatedDeConv(2, 4*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                GatedConv(2*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                #upsample
                GatedDeConv(2, 2*self.cnum, self.cnum, 3, 1, padding=get_pad(256, 3, 1)),
                GatedConv(self.cnum, self.cnum//2, 3, 1, padding=get_pad(128, 3, 1)),
                GatedConv(self.cnum//2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None),
                )
        self.refine_net = nn.Sequential(
                GatedConv(input_channels, self.cnum, 5, 1, padding=get_pad(256, 5, 1)),
                #downsampling
                GatedConv(self.cnum, self.cnum, 4, 2, padding=get_pad(256, 4, 2)),
                GatedConv(self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                #downsampling
                GatedConv(2*self.cnum, 2*self.cnum, 4, 2, padding=get_pad(128, 4, 2)),
                GatedConv(2*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1)),
                #atrous
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
                #self attention
                SelfAttention(4*self.cnum, 'relu', with_attn=False),
                #conv
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1, 2)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1, 2)),
                #upsample
                GatedDeConv(2, 4*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                GatedConv(2*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                #upsample
                GatedDeConv(2, 2*self.cnum, self.cnum, 3, 1, padding=get_pad(256, 3, 1)),
                GatedConv(self.cnum, self.cnum//2, 3, 1, padding=get_pad(128, 3, 1)),
                GatedConv(self.cnum//2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None),
                )

    def forward(self, input):
        ... # TODO: implement forward
