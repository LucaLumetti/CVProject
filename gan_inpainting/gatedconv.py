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
            dilation=0,
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
        self.batch_norm = nn.BatchNorm2d(out_channels) if batch_norm else lambda x: x

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.gate(input)
        x = self.activation(x) * mask
        x = self.batch_norm(x)
        return x

class GatedDeConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation):
        super(GatedDeConv, self).__init__()
        self.gatedconv = GatedConv(input_channels, output_channels, kernel_size, stride, padding, dilation, groups, bias, batch_norm, activation)

    def forward(self, input):
        x = F.interpolate(input, scale_factor=2)
        x = self.gatedconv(x)
