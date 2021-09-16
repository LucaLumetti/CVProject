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
                # downsampling
                GatedConv(self.cnum, 2*self.cnum, 4, 2, padding=get_pad(256, 4, 2)),
                GatedConv(2*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                # downsampling
                GatedConv(2*self.cnum, 4*self.cnum, 4, 2, padding=get_pad(128, 4, 2)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1)),
                # atrous
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
                # conv
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1, 1)),
                # upsample
                GatedDeConv(2, 4*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                GatedConv(2*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                # upsample
                GatedDeConv(2, 2*self.cnum, self.cnum, 3, 1, padding=get_pad(256, 3, 1)),
                GatedConv(self.cnum, self.cnum//2, 3, 1, padding=get_pad(128, 3, 1)),
                GatedConv(self.cnum//2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None),
                )
        self.refine_net = nn.Sequential(
                GatedConv(input_channels, self.cnum, 5, 1, padding=get_pad(256, 5, 1)),
                # downsampling
                GatedConv(self.cnum, self.cnum, 4, 2, padding=get_pad(256, 4, 2)),
                GatedConv(self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                # downsampling
                GatedConv(2*self.cnum, 2*self.cnum, 4, 2, padding=get_pad(128, 4, 2)),
                GatedConv(2*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1)),
                # atrous
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),
                # self attention
                SelfAttention(4*self.cnum, 'relu', with_attn=False),
                # conv
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1, 1)),
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(64, 3, 1, 1)),
                # upsample
                GatedDeConv(2, 4*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                GatedConv(2*self.cnum, 2*self.cnum, 3, 1, padding=get_pad(128, 3, 1)),
                # upsample
                GatedDeConv(2, 2*self.cnum, self.cnum, 3, 1, padding=get_pad(256, 3, 1)),
                GatedConv(self.cnum, self.cnum//2, 3, 1, padding=get_pad(128, 3, 1)),
                GatedConv(self.cnum//2, 3, 3, 1, padding=get_pad(128, 3, 1), activation=None),
                )

    def forward(self, input_images, input_masks):
        # coarse
        masked_images = input_images*(1-input_masks)
        x = torch.cat([masked_images, input_masks, torch.full_like(input_masks, 1.)], dim=1)
        x = self.coarse_net(x)
        x = torch.tanh(x)
        coarse_result = x

        # refine
        masked_images = input_images*(1-input_masks) + coarse_result*input_masks
        x = torch.cat([masked_images, input_masks, torch.full_like(input_masks, 1.)], dim=1)
        x = self.refine_net(x)
        x = torch.tanh(x)

        return coarse_result, x

if __name__ == '__main__':
    # remember samples x channels x height x width !
    # test if the generator can accept a Nx3x256x256 tensor + Nx1x256x256 tensor
    # and output a Nx3x256x256 tensor without any error
    N = 4 # number of images/mask to feed in the net
    input_images = torch.rand((N, 3, 256, 256))
    input_masks = torch.randint(0, 2, (N, 1, 256, 256))

    net = Generator()
    coarse_out, out = net(input_images, input_masks)
    if out.shape == input_images.shape and coarse_out.shape == out.shape:
        print(f'Shapes after forward are ok!')
    else:
        print(f'Something went wrong...')
        print(f'input_images.shape: {input_images.shape}')
        print(f'out.shape: {out.shape}')
        print(f'coarse_out.shape: {input_images.shape}')
