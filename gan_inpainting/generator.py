import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import GatedConv, GatedDeConv, SelfAttention
from init_weights import init_weights

def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

# TODO: maybe this get_pad function can be removed and implemented inside the
# gated conv layer, this will also remove the dependency from the img size of
# 256x256
# The img size dependency can be removed easy by setting a variable and *2 or /2
# each time we down/upsample
class Generator(nn.Module):
    def __init__(self, input_channels=4, input_size=1024, cnum=16):
        super(Generator, self).__init__()
        if input_size%4 != 0:
            raise 'input_size%4 != 0'

        self.cnum = cnum
        self.size = input_size # check it to be a power of 2
        self.coarse_net = nn.Sequential(
                GatedConv(input_channels, self.cnum, 5, 1, padding=get_pad(self.size, 5, 1)),
                # downsampling
                GatedConv(self.cnum, 2*self.cnum, 4, 2, padding=get_pad(self.size, 4, 2)),
                GatedConv(2*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(self.size//2, 3, 1)),
                # downsampling
                GatedConv(2*self.cnum, 4*self.cnum, 4, 2, padding=get_pad(self.size//2, 4, 2)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1)),
                # atrous
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=2, padding=get_pad(self.size//4, 3, 1, 2)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=4, padding=get_pad(self.size//4, 3, 1, 4)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=8, padding=get_pad(self.size//4, 3, 1, 8)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=16, padding=get_pad(self.size//4, 3, 1, 16)),
                # conv
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1, 1)),
                # upsample
                GatedDeConv(2, 4*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(self.size//2, 3, 1)),
                GatedConv(2*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(self.size//2, 3, 1)),
                # upsample
                GatedDeConv(2, 2*self.cnum, self.cnum, 3, 1, padding=get_pad(self.size, 3, 1)),
                GatedConv(self.cnum, self.cnum//2, 3, 1, padding=get_pad(self.size//2, 3, 1)),
                GatedConv(self.cnum//2, 3, 3, 1, padding=get_pad(self.size//2, 3, 1), activation=None),
                )
        self.refine_conv_net = nn.Sequential(
                GatedConv(input_channels, self.cnum, 5, 1, padding=get_pad(self.size, 5, 1)),
                # downsampling
                GatedConv(self.cnum, self.cnum, 4, 2, padding=get_pad(self.size, 4, 2)),
                GatedConv(self.cnum, 2*self.cnum, 3, 1, padding=get_pad(self.size//2, 3, 1)),
                # downsampling
                GatedConv(2*self.cnum, 2*self.cnum, 4, 2, padding=get_pad(self.size//2, 4, 2)),
                GatedConv(2*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1)),
                # atrous
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=2, padding=get_pad(self.size//4, 3, 1, 2)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=4, padding=get_pad(self.size//4, 3, 1, 4)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=8, padding=get_pad(self.size//4, 3, 1, 8)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=16, padding=get_pad(self.size//4, 3, 1, 16)),
                )
        self.refine_att_net = nn.Sequential(
                GatedConv(input_channels, self.cnum, 5, 1, padding=get_pad(self.size, 5, 1)),
                # downsampling
                GatedConv(self.cnum, self.cnum, 4, 2, padding=get_pad(self.size, 4, 2)),
                GatedConv(self.cnum, 2*self.cnum, 3, 1, padding=get_pad(self.size//2, 3, 1)),
                # downsampling
                GatedConv(2*self.cnum, 2*self.cnum, 4, 2, padding=get_pad(self.size//2, 4, 2)),
                GatedConv(2*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1), activation=nn.ReLU()),
                # self attention
                # SelfAttention(4*self.cnum, 'relu', with_attn=False),
                # conv
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1, 1)),
                )
        self.refine_all_net = nn.Sequential(
                GatedConv(2*4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1, 1)),
                # upsample
                GatedDeConv(2, 4*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(self.size//2, 3, 1)),
                GatedConv(2*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(self.size//2, 3, 1)),
                # upsample
                GatedDeConv(2, 2*self.cnum, self.cnum, 3, 1, padding=get_pad(self.size, 3, 1)),
                GatedConv(self.cnum, self.cnum//2, 3, 1, padding=get_pad(self.size//2, 3, 1)),
                GatedConv(self.cnum//2, 3, 3, 1, padding=get_pad(self.size//2, 3, 1), activation=None),
                )
        self.coarse_net.apply(init_weights)
        self.refine_conv_net.apply(init_weights)
        self.refine_att_net.apply(init_weights)
        self.refine_all_net.apply(init_weights)

    def forward(self, input_images, input_masks):
        # coarse
        masked_images = input_images*(1-input_masks)
        x = torch.cat([masked_images, input_masks], dim=1)
        x = self.coarse_net(x)
        x = torch.tanh(x)
        coarse_result = x

        # refine
        masked_images = input_images*(1-input_masks) + coarse_result*input_masks
        x = torch.cat([masked_images, input_masks], dim=1)
        xnow = x

        conv_out = self.refine_conv_net(x)
        att_out = self.refine_att_net(x)

        x = torch.cat([conv_out, att_out], dim=1)

        x = self.refine_all_net(x)
        x = torch.tanh(x)

        refine_result = x

        return coarse_result, refine_result

if __name__ == '__main__':
    # remember samples x channels x height x width !
    # test if the generator can accept a Nx3x256x256 tensor + Nx1x256x256 tensor
    # and output a Nx3xSIZExSIZE tensor without any error
    N = 2 # number of images/mask to feed in the net
    SIZE = 64
    input_images = torch.rand((N, 3, SIZE, SIZE))
    input_masks = torch.randint(0, 2, (N, 1, SIZE, SIZE))

    net = Generator(input_size=SIZE)
    coarse_out, out = net(input_images, input_masks)
    if out.shape == input_images.shape and coarse_out.shape == out.shape:
        print(f'Shapes after forward are ok!')
    else:
        print(f'Something went wrong...')
        print(f'input_images.shape: {input_images.shape}')
    print(f'out.shape: {out.shape}')
    print(f'coarse_out.shape: {input_images.shape}')
