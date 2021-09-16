import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from generator import Generator
from discriminator import Discriminator

# a loss history should be held to keep tracking if the network is learning
# something or is doing completely random shit
# also a logger would be nice
def train(netG, netD, optimG, optimD, lossG, lossD, dataloader):
    netG.train()
    netD.train()

    for imgs, masks in dataloader:
        netG.zero_grad()
        netD.zero_grad()
        optimG.zero_grad()
        optimD.zero_grad()

        # change img range from [0,255] to [-1,+1]
        imgs = imgs / 127.5 - 1

        # forward G
        coarse_out, refined_out = netG(imgs, masks)
        reconstructed_imgs = refined_out*masks + imgs*(1-masks)

        true_imgs = torch.cat([imgs, masks, torch.full_like(masks, 1.)], dim=1)
        gen_imgs = torch.cat([reconstructed_imgs, masks, torch.full_like(masks, 1.)], dim=1)
        pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

        # forward D
        pred_pos_neg_imgs = netD(pos_neg_imgs)
        pred_pos_imgs, pred_neg_imgs = torch.chunk(pred_pos_neg_imgs, 2, dim=0)

        # loss + backward D
        loss_discriminator = lossD(pred_pos_imgs, pred_neg_imgs)
        loss_discriminator.backward()
        optimD.step()

        # loss + backward G
        loss_generator = lossG(pred_neg_imgs)
        loss_generator.backward()
        optimG.step()
    return

def main():
    ...

if __name__ == '__main__':
    main()
