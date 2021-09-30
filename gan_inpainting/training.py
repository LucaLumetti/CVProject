import numpy as np
import cv2

import torch
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt

from generator import Generator
from discriminator import Discriminator
from loss import GeneratorLoss, DiscriminatorLoss, L1ReconLoss
from dataset import FakeDataset, FaceMaskDataset
from config import Config

from metrics import TrainingMetrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# torch.autograd.set_detect_anomaly(True)
# a loss history should be held to keep tracking if the network is learning
# something or is doing completely random shit
# also a logger would be nice
def train(netG, netD, optimG, optimD, lossG, lossD, lossRecon, dataloader):
    netG.train()
    netD.train()

    metrics = TrainingMetrics(dataloader)

    for i, (imgs, masks) in enumerate(dataloader):
        netG.zero_grad()
        netD.zero_grad()
        optimG.zero_grad()
        optimD.zero_grad()
        lossG.zero_grad()
        lossD.zero_grad()

        imgs = imgs.to(device)
        masks = masks.to(device)

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

        # loss + backward G
        loss_generator = lossG(pred_neg_imgs)
        loss_recon = lossRecon(imgs, coarse_out, refined_out, masks)
        loss_gen_recon = loss_generator + loss_recon

        loss_discriminator.backward(retain_graph=True)
        loss_gen_recon.backward()

        optimD.step()
        optimG.step()

        metrics.update({"loss_G":loss_generator,"loss_D": loss_discriminator,"loss_R": loss_recon}, pred_pos_neg_imgs, netG, netD)
    return

if __name__ == '__main__':
    config = Config('config.json')
    print(config)
    # using a fake dataset just to test the net until our dataset is not ready
    dataset = FaceMaskDataset(config.dataset_dir, 'maskffhq.csv')
    # dataset = FakeDataset()
    dataloader = dataset.loader(batch_size=config.batch_size)

    netG = Generator(input_size=config.input_size).to(device)
    netD = Discriminator(input_size=config.input_size).to(device)

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
