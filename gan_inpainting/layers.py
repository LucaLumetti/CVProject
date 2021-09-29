import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

class GatedConv(nn.Module):
    def __init__(self,
            input_channels,
            output_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            batch_norm=True,
            activation=nn.LeakyReLU(0.2, inplace=False)):
        super(GatedConv, self).__init__()

        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.gate = nn.Sequential(
                self.conv2d,
                nn.Sigmoid()
                )
        self.activation = activation if activation is not None else lambda x: x
        self.batch_norm = nn.BatchNorm2d(output_channels) if batch_norm else lambda x: x

    def forward(self, input):
        # the same conv layer is applied to x and mask, in the reference code x
        # and mask are joined togheter then split to apply sigmoid to mask only
        # peraphs this latter approach is better
        x = self.conv2d(input)
        mask = self.gate(input)
        x = self.activation(x) * mask
        x = self.batch_norm(x)
        return x

class GatedDeConv(nn.Module):
    def __init__(self,
            scale_factor,
            input_channels,
            output_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            batch_norm=True,
            activation=nn.LeakyReLU(0.2, inplace=False)):
        super(GatedDeConv, self).__init__()
        self.gatedconv = GatedConv(input_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)
        self.scale_factor = scale_factor

    def forward(self, input):
        x = F.interpolate(input, scale_factor=self.scale_factor)
        x = self.gatedconv(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, input_channels, activation, with_attn=False):
        super(SelfAttention, self).__init__()
        self.input_channels = input_channels
        self.activation = activation
        self.with_attn = with_attn
        self.query_conv = nn.Conv2d(input_channels, input_channels//8, 1)
        self.key_conv = nn.Conv2d(input_channels, input_channels//8, 1)
        self.value_conv = nn.Conv2d(input_channels, input_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        m_batchsize, C, width, height = input.size()
        proj_query  = self.query_conv(input).view(m_batchsize, -1, width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(input).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(input).view(m_batchsize, -1, width*height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma*out + input
        return out

class SpectralNormConv(nn.Module):
    def __init__(self,
            input_channels,
            output_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            activation=nn.LeakyReLU(0.2, inplace=False)):
        super(SpectralNormConv, self).__init__()

        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = nn.utils.spectral_norm(self.conv2d)
        self.activation = activation if activation is not None else lambda x: x

    def forward(self, input):
        x = self.conv2d(input)
        x = self.activation(x)
        return x

class MultiDilationResnetBlock8(nn.Module):
    def __init__(self,
            input_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True):
        super(MultiDilationResnetBlock8, self).__init__()

        self.branch1 = GatedConv(input_channels, output_channels//8, 3, 1, 2, 2, activation=nn.ReLU())
        self.branch2 = GatedConv(input_channels, output_channels//8, 3, 1, 3, 3, activation=nn.ReLU())
        self.branch3 = GatedConv(input_channels, output_channels//8, 3, 1, 4, 4, activation=nn.ReLU())
        self.branch4 = GatedConv(input_channels, output_channels//8, 3, 1, 5, 5, activation=nn.ReLU())
        self.branch5 = GatedConv(input_channels, output_channels//8, 3, 1, 6, 6, activation=nn.ReLU())
        self.branch6 = GatedConv(input_channels, output_channels//8, 3, 1, 8, 8, activation=nn.ReLU())
        self.branch7 = GatedConv(input_channels, output_channels//8, 3, 1, 10, 10, activation=nn.ReLU())
        self.branch8 = GatedConv(input_channels, output_channels//8, 3, 1, 1, 1, activation=nn.ReLU())

        self.concatenation = GatedConv(input_channels, output_channels, 3, 1, 1, 1, activation=None)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self.branch5(x)
        b6 = self.branch6(x)
        b7 = self.branch7(x)
        b8 = self.branch8(x)
        b9 = torch.cat((b1, b2, b3, b4, b5, b6, b7, b8), dim=1)
        out = x + self.concatenation(b9)
        return out

class MultiDilationResnetBlock4(nn.Module):
    def __init__(self,
            input_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True):
        super(MultiDilationResnetBlock4, self).__init__()

        self.branch1 = GatedConv(input_channels, output_channels//4, 3, 1, 1, 1, activation=nn.ReLU())
        self.branch2 = GatedConv(input_channels, output_channels//4, 3, 1, 2, 2, activation=nn.ReLU())
        self.branch3 = GatedConv(input_channels, output_channels//4, 3, 1, 4, 4, activation=nn.ReLU())
        self.branch4 = GatedConv(input_channels, output_channels//4, 3, 1, 8, 8, activation=nn.ReLU())

        self.concatenation = GatedConv(input_channels, output_channels, 3, 1, 1, 1, activation=None)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = torch.cat((b1, b2, b3, b4), dim=1)
        out = x + self.concatenation(b5)
        return out
