import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import GatedConv, GatedDeConv, SelfAttention

def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

class Discriminator(nn.Module):
    def __init__(self, input_channels=5, cnum=32):
        super(Discriminator, self).__init__()
        self.cnum = cnum
        self.discriminator_net = nn.Sequential(
                GatedConv(input_channels, 2*self.cnum, 4, 2, padding=get_pad(256, 5, 2)),
                GatedConv(2*self.cnum, 4*self.cnum, 4, 2, padding=get_pad(128, 5, 2)),
                GatedConv(4*self.cnum, 8*self.cnum, 4, 2, padding=get_pad(64, 5, 2)),
                GatedConv(8*self.cnum, 8*self.cnum, 4, 2, padding=get_pad(32, 5, 2)),
                GatedConv(8*self.cnum, 8*self.cnum, 4, 2, padding=get_pad(16, 5, 2)),
                GatedConv(8*self.cnum, 8*self.cnum, 4, 2, padding=get_pad(8, 5, 2)),
                SelfAttention(8*self.cnum, 'relu'),
                GatedConv(8*self.cnum, 8*self.cnum, 4, 2, padding=get_pad(4, 5, 2)),
                nn.Linear(8*self.cnum*2*2, 1)
                )

    def forward(self, input):
        x = self.discriminator_net(input)
        return x

