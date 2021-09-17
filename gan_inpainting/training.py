import numpy as np

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

from generator import Generator
from discriminator import Discriminator
from loss import GeneratorLoss, DiscriminatorLoss, L1ReconLoss
from dataset import FakeDataset

# torch.autograd.set_detect_anomaly(True)
# a loss history should be held to keep tracking if the network is learning
# something or is doing completely random shit
# also a logger would be nice
def train(netG, netD, optimG, optimD, lossG, lossD, lossRecon, dataloader):
    netG.train()
    netD.train()

    for imgs, masks in dataloader:
        netG.zero_grad()
        netD.zero_grad()
        optimG.zero_grad()
        optimD.zero_grad()
        lossG.zero_grad()
        lossD.zero_grad()

        # change img range from [0,255] to [-1,+1]
        imgs = imgs / 127.5 - 1

        # forward G
        coarse_out, refined_out = netG(imgs, masks)
        reconstructed_imgs = refined_out*masks + imgs*(1-masks)

        pos_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
        neg_imgs = torch.cat([reconstructed_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

        # forward D
        pos_neg_imgs, masks, _ = torch.split(pos_neg_imgs, (3,1,1), dim=1)
        pred_pos_neg_imgs = netD(pos_neg_imgs, masks)
        pred_pos_imgs, pred_neg_imgs = torch.chunk(pred_pos_neg_imgs, 2, dim=0)

        # loss + backward D
        loss_discriminator = lossD(pred_pos_imgs, pred_neg_imgs)
        print(f'd_loss: {loss_discriminator}')

        # loss + backward G
        loss_generator = lossG(pred_neg_imgs)
        loss_generator += lossRecon(imgs, coarse_out, refined_out, masks)
        print(f'g_loss: {loss_generator}')

        # these operations has been rearranged to avoid a autograd problem
        # not 100% sure on why retain_graph=True
        loss_discriminator.backward(retain_graph=True)
        # maybe i got why retain graph is needed, the backward on the discr also
        # touch gradint that the generator has to use, then the first backward
        # needs to retain graph, the second does not, i will leave this comment
        # here for the future
        loss_generator.backward()

        optimD.step()
        optimG.step()
    return

if __name__ == '__main__':
    # using a fake dataset just to test the net until our dataset is not ready
    dataset = FakeDataset(None)
    dataloader = dataset.loader()

    netG = Generator()
    netD = Discriminator()

    optimG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))

    lossG = GeneratorLoss()
    lossRecon = L1ReconLoss() # in the original paper, all alphas == 1
    lossD = DiscriminatorLoss()

    train(netG, netD, optimG, optimD, lossG, lossD, lossRecon, dataloader)

    torch.save(netG.state_dict(), 'models/generator.pt')
    torch.save(netD.state_dict(), 'models/discriminator.pt')
