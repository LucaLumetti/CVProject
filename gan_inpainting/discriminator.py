import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from layers import GatedConv, GatedDeConv, SelfAttention, SpectralNormConv
from init_weights import init_weights

def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

class Discriminator(nn.Module):
    def __init__(self, input_channels=4, input_size=1024, cnum=8):
        super(Discriminator, self).__init__()
        self.cnum = cnum
        self.size = input_size
        self.discriminator_net = nn.Sequential(
                SpectralNormConv(input_channels, 2*self.cnum, 4, 2, padding=get_pad(self.size, 5, 2)),
                SpectralNormConv(2*self.cnum, 4*self.cnum, 4, 2, padding=get_pad(self.size//2, 4, 2)),
                SpectralNormConv(4*self.cnum, 8*self.cnum, 4, 2, padding=get_pad(self.size//4, 4, 2)),
                SpectralNormConv(8*self.cnum, 8*self.cnum, 4, 2, padding=get_pad(self.size//8, 4, 2)),
                SpectralNormConv(8*self.cnum, 8*self.cnum, 4, 2, padding=get_pad(self.size//16, 4, 2)),
                SpectralNormConv(8*self.cnum, 8*self.cnum, 4, 2, padding=get_pad(self.size//32, 4, 2)),
                # not clear if usefull
                # SelfAttention(8*self.cnum, 'relu'),
                # GatedConv(8*self.cnum, 8*self.cnum, 4, 2, padding=get_pad(4, 5, 2)),
                )
        # self.linear = nn.Linear(8*self.cnum*2*2, 1)
        # self.sigmoid = nn.Sigmoid()
        self.discriminator_net.apply(init_weights)

    def forward(self, input_images, input_masks):
        # masked_images = input_images*(1-input_masks) # can be this usefull?
        x = torch.cat([input_images, input_masks], dim=1)
        x = self.discriminator_net(x)
        x = x.view((x.size(0),-1))
        # x = torch.mean(x, dim=1) # maybe this is wrong bc already done in the loss
        # x = self.linear(x)
        # x = self.sigmoid(x)
        return x

if __name__ == '__main__':
    # remember samples x channels x height x width !
    # test if the discriminator can accept a Nx3x256x256 tensor
    # and output a 1 or 0
    N = 4 # number of images/mask to feed in the net
    SIZE = 64
    input_images = torch.rand((N, 3, SIZE, SIZE))
    input_masks = torch.randint(0, 2, (N, 1, SIZE, SIZE))

    net = Discriminator(input_size=SIZE)
    out = net(input_images, input_masks)
    if out.shape == torch.Size([N]):
        print(f'Shapes after forward are ok!')
    else:
        print(f'Something went wrong...')
    print(f'input_images.shape: {input_images.shape}')
    print(f'out.shape: {out.shape}')

