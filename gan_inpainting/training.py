import argparse
import logging
import os

import numpy as np
import cv2

import torch
import torchvision
from torchvision import transforms as T
from torchvision.utils import save_image
import torch.nn.functional as F
import torch.nn as nn

import matplotlib.pyplot as plt

from generator import *
from discriminator import Discriminator
from loss import *
from dataset import FakeDataset, FaceMaskDataset
from config import Config

from metrics import TrainingMetrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.autograd.set_detect_anomaly(True)
# a loss history should be held to keep tracking if the network is learning
# something or is doing completely random shit
# also a logger would be nice
def train(netG, netD, optimG, optimD, lossG, lossD, lossRecon, lossTV, lossVGG, dataloader, metrics):
    netG.train()
    netD.train()

    losses = {
            'g': [],
            'd': [],
            'r': [],
            'tv': [],
            'perc': [],
            'style': [],
            }

    accuracies = {
            'd': []
            }

    for ep in range(config.epoch):
        for i, (imgs, masks) in enumerate(dataloader):
            netG.zero_grad()
            netD.zero_grad()
            optimG.zero_grad()
            optimD.zero_grad()
            lossG.zero_grad()
            lossD.zero_grad()
            lossTV.zero_grad()
            lossVGG.zero_grad()

            imgs = imgs.to(device)
            masks = masks.to(device)

            # change img range from [0,255] to [-1,+1]
            imgs = imgs / 127.5 - 1
            masks = masks / 1.

            # forward G
            coarse_out, refined_out = netG(imgs, masks)
            reconstructed_coarses = coarse_out*masks + imgs*(1-masks)
            reconstructed_imgs = refined_out*masks + imgs*(1-masks)

            # pos_imgs = torch.cat([imgs, masks], dim=1)
            # neg_imgs = torch.cat([reconstructed_imgs, masks], dim=1)
            # pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)
            pos_neg_imgs = torch.cat([imgs, reconstructed_imgs], dim=0)
            dmasks = torch.cat([masks, masks], dim=0)

            # forward D
            # pos_neg_imgs, dmasks = torch.split(pos_neg_imgs, (3,1), dim=1)
            pred_pos_neg_imgs = netD(pos_neg_imgs, dmasks)
            pred_pos_imgs, pred_neg_imgs = torch.chunk(pred_pos_neg_imgs, 2, dim=0)

            # loss + backward D
            loss_discriminator = lossD(pred_pos_imgs, pred_neg_imgs)
            losses['d'] = loss_discriminator.item()
            loss_discriminator.backward(retain_graph=True)
            optimD.step()

            netG.zero_grad()
            netD.zero_grad()
            optimG.zero_grad()
            optimD.zero_grad()
            lossG.zero_grad()
            lossD.zero_grad()

            # loss + backward G
            pred_neg_imgs = netD(reconstructed_imgs, masks)
            loss_generator = lossG(pred_neg_imgs)
            loss_recon = lossRecon(imgs, coarse_out, refined_out, dmasks)
            loss_tv = lossTV(refined_out)
            loss_perc, loss_style = lossVGG(imgs, refined_out)
            loss_perc *= 0.05
            loss_style *= 40
            loss_gen_recon = loss_generator + loss_recon + \
                    loss_tv + loss_perc + loss_style

            losses['g'] = loss_generator.item()
            losses['r'] = loss_recon.item()
            losses['tv'] = loss_tv.item()
            losses['perc'] = loss_perc.item()
            losses['style'] = loss_style.item()

            loss_gen_recon.backward()

            optimG.step()
            # every 100 img, print losses, update the graph, output an image as
            # example
            if i % metrics.screenshot_step == 0:
                checkpoint_coarse = ((reconstructed_coarses[0] + 1) * 127.5)
                checkpoint_recon = ((reconstructed_imgs[0] + 1) * 127.5)

                save_image(checkpoint_coarse / 255, f'plots/coarse_{i}.png')
                save_image(checkpoint_recon / 255, f'plots/recon_{i}.png')

                # maybe save them in metrics.update()
                torch.save(netG.state_dict(), f'{config.checkpoint_dir}/generator.pt')
                torch.save(netD.state_dict(), f'{config.checkpoint_dir}/discriminator.pt')
                torch.save(optimG.state_dict(), f'{config.checkpoint_dir}/opt_generator.pt')
                torch.save(optimD.state_dict(), f'{config.checkpoint_dir}/opt_discriminator.pt')
            metrics.update(losses, pred_pos_neg_imgs, netG, netD)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", type=str, help="config file", required=True)
    parser.add_argument("--debug", help="debug logging level")
    parser.add_argument("--checkpoint", help="resume training")
    parser.add_argument("--screenshot_step", type=int, help="how often output metrics and imgs", default=100)
    args = parser.parse_args()

    logging_level = logging.INFO
    if args.debug:
        logging_level = logging.DEBUG

    logging.basicConfig(filename='output.log', level=logging_level)

    config = Config(args.config)
    logging.debug(config)

    dataset = FaceMaskDataset(
            config.dataset_dir,
            'maskffhq.csv',
            T.Resize(config.input_size)
        )

    dataloader = dataset.loader(batch_size=config.batch_size, shuffle=True)

    netG = MSSAGenerator(input_size=config.input_size).to(device)
    netD = Discriminator(input_size=config.input_size).to(device)

    optimG = torch.optim.Adam(
                netG.parameters(),
                lr=config.learning_rate_g,
                betas=(0.5, 0.999)
            )
    optimD = torch.optim.Adam(
                netD.parameters(),
                lr=config.learning_rate_d,
                betas=(0.5, 0.999)
            )

    if args.checkpoint is not None:
        generator_dir = config.checkpoint_dir + '/generator.pt'
        discriminator_dir = config.checkpoint_dir + '/discriminator.pt'
        opt_generator_dir = config.checkpoint_dir + '/opt_generator.pt'
        opt_discriminator_dir = config.checkpoint_dir + '/opt_discriminator.pt'

        if os.path.isfile(generator_dir):
            logging.info('resuming training of generator')
            checkpointG = torch.load(generator_dir)
            netG.load_state_dict(checkpointG)

        if os.path.isfile(discriminator_dir):
            logging.info('resuming training of discriminator')
            checkpointD = torch.load(discriminator_dir)
            netD.load_state_dict(checkpointD)

        if os.path.isfile(opt_generator_dir):
            logging.info('resuming training of opt_generator')
            checkpointOG = torch.load(opt_generator_dir)
            optimG.load_state_dict(checkpointOG)

        if os.path.isfile(opt_discriminator_dir):
            logging.info('resuming training of opt_discriminator')
            checkpointOD = torch.load(opt_discriminator_dir)
            optimD.load_state_dict(checkpointOD)

    lossG = GeneratorLoss()
    lossRecon = L1ReconLoss()
    lossTV = TVLoss()
    lossD = DiscriminatorHingeLoss()
    lossVGG = VGGLoss()

    metrics = TrainingMetrics(
            args.screenshot_step,
            config.video_dir,
            dataset
        )

    params_g = sum([ p.numel() for p in netG.parameters() ])
    params_d = sum([ p.numel() for p in netD.parameters() ])

    logging.info(f'Generator: {params_g} params')
    logging.info(f'Discriminator: {params_d} params')
    train(netG, netD, optimG, optimD, lossG, lossD, lossRecon,
            lossTV, lossVGG, dataloader, metrics)

    torch.save(netG.state_dict(), 'models/generator.pt')
    torch.save(netD.state_dict(), 'models/discriminator.pt')
