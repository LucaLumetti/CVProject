import numpy as np

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt

from generator import Generator
from discriminator import Discriminator
from loss import GeneratorLoss, DiscriminatorLoss, L1ReconLoss
from dataset import FaceMaskDataset
from config import Config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.autograd.set_detect_anomaly(True)
# a loss history should be held to keep tracking if the network is learning
# something or is doing completely random shit
# also a logger would be nice
def train(netG, netD, optimG, optimD, lossG, lossD, lossRecon, dataloader):
    netG.train()
    netD.train()

    losses = {
            'g': [],
            'd': [],
            'r': [],
            }

    for i, (imgs, masks) in enumerate(dataloader):
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
        losses['d'].append(loss_discriminator)

        # loss + backward G
        loss_generator = lossG(pred_neg_imgs)
        loss_recon = lossRecon(imgs, coarse_out, refined_out, masks)
        loss_gen_recon = loss_generator + loss_recon
        losses['g'].append(loss_generator)
        losses['r'].append(loss_recon)

        # these operations has been rearranged to avoid a autograd problem
        # not 100% sure on why retain_graph=True
        loss_discriminator.backward(retain_graph=True)
        # maybe i got why retain graph is needed, the backward on the discr also
        # touch gradint that the generator has to use, then the first backward
        # needs to retain graph, the second does not, i will leave this comment
        # here for the future
        loss_gen_recon.backward()

        optimD.step()
        optimG.step()
        # every 100 img, print losses, update the graph, output an image as
        # example
        print(f"[{i}]\tloss_g: {losses['g'][-1]}, loss_d: {losses['d'][-1]}, loss_r: {losses['r'][-1]}")
        if i%5 == 0:
            fig, axs = plt.subplots(2, 1)
            x_axis = range(len(losses['g']))
            axs[0].plot(x_axis, losses['g'], x_axis, losses['r'])
            axs[0].set_xlabel('it')
            axs[0].set_ylabel('loss')
            axs[1].plot(x_axis, losses['d'])
            axs[1].set_xlabel('it')
            axs[1].set_ylabel('loss')
            axs[1].set_ylim(0,1)
            fig.tight_layout()
            fig.savefig('plots/loss.png', dpi=fig.dpi)

    return

if __name__ == '__main__':
    config = Config('config.json')
    print(config)
    # using a fake dataset just to test the net until our dataset is not ready
    dataset = FakeDataset(None)
    dataloader = dataset.loader(batch_size=config.batch_size)

    netG = Generator().to(device)
    netD = Discriminator().to(device)

    optimG = torch.optim.Adam(
            netG.parameters(),
            lr=config.learning_rate,
            betas=(0.5, 0.999)
            )
    optimD = torch.optim.Adam(
            netD.parameters(),
            lr=config.learning_rate,
            betas=(0.5, 0.999)
            )

    lossG = GeneratorLoss()
    lossRecon = L1ReconLoss() # in the original paper, all alphas == 1
    lossD = DiscriminatorLoss()

    train(netG, netD, optimG, optimD, lossG, lossD, lossRecon, dataloader)

    torch.save(netG.state_dict(), 'models/generator.pt')
    torch.save(netD.state_dict(), 'models/discriminator.pt')
