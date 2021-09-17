import numpy as np

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

from generator import Generator
from discriminator import Discriminator
from loss import GeneratorLoss, DiscriminatorLoss
from dataset import FakeDataset

def test(netG, netD, lossG, lossD, dataloader):
    netG.eval()
    netD.eval()

    return

if __name__ == '__main__':
    ...
