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
                # nn.Linear(8*self.cnum*2*2, 1)
                )
        self.linear = nn.Linear(8*self.cnum*2*2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_images, input_masks):
        # masked_images = input_images*(1-input_masks) # can be this usefull?
        x = torch.cat([input_images, input_masks, torch.full_like(input_masks, 1.)], dim=1)
        x = self.discriminator_net(x)
        x = x.view((x.size(0),-1))
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

if __name__ == '__main__':
    # remember samples x channels x height x width !
    # test if the discriminator can accept a Nx3x256x256 tensor
    # and output a 1 or 0
    N = 4 # number of images/mask to feed in the net
    input_images = torch.rand((N, 3, 256, 256))
    input_masks = torch.randint(0, 2, (N, 1, 256, 256))

    net = Discriminator()
    out = net(input_images, input_masks)
    if out.shape == (N,1):
        print(f'Shapes after forward are ok!')
    else:
        print(f'Something went wrong...')
        print(f'input_images.shape: {input_images.shape}')
        print(f'out.shape: {out.shape}')

