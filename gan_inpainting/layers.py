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
            activation=nn.LeakyReLU(0.2, inplace=True)):
        super(GatedConv, self).__init__()

        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.gate = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias),
                nn.Sigmoid()
                )
        self.activation = activation if activation is not None else lambda x: x
        self.batch_norm = nn.BatchNorm2d(output_channels) if batch_norm else lambda x: x

    def forward(self, input):
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
            activation=nn.LeakyReLU(0.2, inplace=True)):
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
        if self.with_attn:
            return out, attention
        else:
            return out

