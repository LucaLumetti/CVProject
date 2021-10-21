import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import *
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
    def __init__(self, input_channels=4, input_size=1024, cnum=32):
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
                # MultiDilationResnetBlock8(4*self.cnum, 4*self.cnum),
                # MultiDilationResnetBlock8(4*self.cnum, 4*self.cnum),
                # MultiDilationResnetBlock8(4*self.cnum, 4*self.cnum),
                # MultiDilationResnetBlock8(4*self.cnum, 4*self.cnum),
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
                # MultiDilationResnetBlock4(4*self.cnum, 4*self.cnum),
                # self attention
                SelfAttention(4*self.cnum),
                # conv
                GatedConv(4*self.cnum, 4*self.cnum, 3, 1, padding=get_pad(self.size//4, 3, 1, 1)),
                # MultiDilationResnetBlock4(4*self.cnum, 4*self.cnum),
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

class MSSAGenerator(nn.Module):
    def __init__(self, input_channels=4, input_size=1024, cnum=32):
        super(MSSAGenerator, self).__init__()
        if input_size%4 != 0:
            raise 'input_size%4 != 0'

        self.cnum = cnum
        self.size = input_size
        self.pad = nn.ReplicationPad2d
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
                # MultiDilationResnetBlock8(4*self.cnum, 4*self.cnum),
                # MultiDilationResnetBlock8(4*self.cnum, 4*self.cnum),
                # MultiDilationResnetBlock8(4*self.cnum, 4*self.cnum),
                # MultiDilationResnetBlock8(4*self.cnum, 4*self.cnum),
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
        self.c1 = nn.Sequential(
                self.pad(3),
                nn.Conv2d(input_channels, self.cnum, 7, 1, padding=0),
                nn.LeakyReLU(0.2, True)
                )
        self.c2 = nn.Sequential(
                self.pad(1),
                nn.Conv2d(self.cnum, self.cnum*2, 4, 2, padding=0),
                nn.LeakyReLU(0.2, True)
                )
        self.c3 = nn.Sequential(
                self.pad(1),
                nn.Conv2d(self.cnum*2, self.cnum*4, 4, 2, padding=0),
                nn.LeakyReLU(0.2, True)
                )
        self.skip_c3 = nn.Sequential(
                self.pad(1),
                nn.Conv2d(self.cnum*4, self.cnum*2, 3, 1, padding=0),
                nn.LeakyReLU(0.2, True),
                SelfAttention(self.cnum*2)
                )
        self.c4 = nn.Sequential(
                self.pad(1),
                nn.Conv2d(self.cnum*4, self.cnum*8, 4, 2, padding=0),
                nn.LeakyReLU(0.2, True)
                )
        self.skip_c4 = nn.Sequential(
                self.pad(1),
                nn.Conv2d(self.cnum*8, self.cnum*4, 3, 1, padding=0),
                nn.LeakyReLU(0.2, True),
                SelfAttention(self.cnum*4)
                )
        self.middle1 = nn.Sequential(
                self.pad(1),
                nn.Conv2d(self.cnum*8, self.cnum*16, 4, 2, padding=0),
                nn.LeakyReLU(0.2, True),
                MultiDilationResnetBlock4(self.cnum*16, self.cnum*16, 3, 1, 1),
                MultiDilationResnetBlock4(self.cnum*16, self.cnum*16, 3, 1, 1),
                MultiDilationResnetBlock4(self.cnum*16, self.cnum*16, 3, 1, 1),
                )
        self.middle2 = nn.Sequential(
                SelfAttention(self.cnum*16),
                MultiDilationResnetBlock4(self.cnum*16, self.cnum*16, 3, 1, 1),
                MultiDilationResnetBlock4(self.cnum*16, self.cnum*16, 3, 1, 1),
                MultiDilationResnetBlock4(self.cnum*16, self.cnum*16, 3, 1, 1),
                )
        self.dc4 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                self.pad(1),
                nn.Conv2d(self.cnum*16, self.cnum*8, 3, 1, padding=0),
                nn.LeakyReLU(0.2, True)
                )
        self.dc3 = nn.Sequential(
                self.pad(1),
                nn.Conv2d(self.cnum*8 + self.cnum*4, self.cnum*8, 3, 1, padding=0),
                nn.LeakyReLU(0.2, True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                self.pad(1),
                nn.Conv2d(self.cnum*8, self.cnum*4, 3, 1, padding=0),
                nn.LeakyReLU(0.2, True)
                )
        self.dc2 = nn.Sequential(
                self.pad(1),
                nn.Conv2d(self.cnum*4 + self.cnum*2, self.cnum*4, 3, 1, padding=0),
                nn.LeakyReLU(0.2, True),
                nn.UpsamplingBilinear2d(scale_factor=2),
                self.pad(1),
                nn.Conv2d(self.cnum*4, self.cnum*2, 3, 1, padding=0),
                nn.LeakyReLU(0.2, True),
                )
        self.dc1 = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                self.pad(1),
                nn.Conv2d(self.cnum*2, self.cnum, 3, 1, padding=0),
                nn.LeakyReLU(0.2, True),
                )
        self.end = nn.Sequential(
                self.pad(3),
                nn.Conv2d(self.cnum, 3, 7, 1, padding=0),
                )
        self.avgpool = nn.AvgPool2d(16)

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
        x = self.c1(x)  # 256x256x32
        x = self.c2(x)  # 128x128x64
        x3 = self.c3(x) # 64x64x128
        x4 = self.c4(x3) # 32x32x256

        emb_repr = self.middle1(x4)   # 16x16x512
        x = self.middle2(emb_repr)   # 16x16x512

        x3 = self.skip_c3(x3) # 64x64x64
        x4 = self.skip_c4(x4) # 32x32x32

        x = self.dc4(x) # 32x32x256
        x = torch.cat([x, x4], dim=1) # 32x32x(256+128)
        x = self.dc3(x) # 64x64x128
        x = torch.cat([x, x3], dim=1) # 64x64x(128+64)
        x = self.dc2(x) # 128x128x64
        x = self.dc1(x) # 256x256x32

        x = self.end(x) # 256x256x3
        # x = torch.tanh(x)

        emb_repr = self.avgpool(emb_repr).squeeze()
        refine_result = x

        return emb_repr, coarse_result, refine_result

if __name__ == '__main__':
    # remember samples x channels x height x width !
    # test if the generator can accept a Nx3x256x256 tensor + Nx1x256x256 tensor
    # and output a Nx3xSIZExSIZE tensor without any error
    N = 2 # number of images/mask to feed in the net
    SIZE = 64
    input_images = torch.rand((N, 3, SIZE, SIZE))
    input_masks = torch.randint(0, 2, (N, 1, SIZE, SIZE))

    net = MSSAGenerator(input_size=SIZE)
    coarse_out, out = net(input_images, input_masks)
    if out.shape == input_images.shape and coarse_out.shape == out.shape:
        print(f'Shapes after forward are ok!')
    else:
        print(f'Something went wrong...')
        print(f'input_images.shape: {input_images.shape}')
    print(f'out.shape: {out.shape}')
    print(f'coarse_out.shape: {input_images.shape}')
