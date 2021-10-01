import argparse
import logging

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# torch.autograd.set_detect_anomaly(True)
# a loss history should be held to keep tracking if the network is learning
# something or is doing completely random shit
# also a logger would be nice
def train(netG, netD, optimG, optimD, lossG, lossD, lossRecon, lossTV, dataloader):
    netG.train()
    netD.train()

    losses = {
            'g': [],
            'd': [],
            'r': [],
            'tv': []
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

            # canculate accuracy of D
            with torch.inference_mode():
                mean_pos_pred = pred_pos_imgs.clone().detach().mean(dim=1)
                mean_neg_pred = pred_neg_imgs.clone().detach().mean(dim=1)
                mean_pos_pred = torch.where(mean_pos_pred > 0.5, 1, 0).type(torch.FloatTensor)
                mean_neg_pred = torch.where(mean_neg_pred > 0.5, 0, 1).type(torch.FloatTensor)
                accuracyD = torch.sum(mean_pos_pred) + torch.sum(mean_neg_pred)
                tot_elem = mean_pos_pred.shape[0] + mean_neg_pred.shape[0]
                accuracyD /= tot_elem
                accuracies['d'].append(accuracyD.item())

            # loss + backward D
            loss_discriminator = lossD(pred_pos_imgs, pred_neg_imgs)
            losses['d'].append(loss_discriminator.item())
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
            loss_gen_recon = loss_generator + loss_recon + loss_tv

            losses['g'].append(loss_generator.item())
            losses['r'].append(loss_recon.item())
            losses['tv'].append(loss_tv.item())

            loss_gen_recon.backward()

            optimG.step()
            # every 500 img, print losses, update the graph, output an image as
            # example
            if i % 100 == 0:
                logging.info(f"[{i}]\t" + \
                        f"loss_g: {losses['g'][-1]}, " + \
                        f"loss_d: {losses['d'][-1]}, " + \
                        f"loss_r: {losses['r'][-1]}, " + \
                        f"loss_tv: {losses['tv'][-1]}, " + \
                        f"accuracy_d: {accuracies['d'][-1]}")
                checkpoint_coarse = ((reconstructed_coarses[0]+1)*127.5)
                checkpoint_recon = ((reconstructed_imgs[0]+1)*127.5)
                checkpoint_img = ((imgs[0]+1)*127.5)

                fig, axs = plt.subplots(3, 1)
                x_axis = range(len(losses['g']))
                # loss g
                axs[0].plot(x_axis, losses['g'], x_axis, losses['r'], x_axis, losses['tv'])
                axs[0].set_xlabel('iterations')
                axs[0].set_ylabel('loss')
                # loss d
                axs[1].plot(x_axis, losses['d'])
                axs[1].set_xlabel('iterations')
                axs[1].set_ylabel('loss')
                # acc d
                axs[2].plot(x_axis, accuracies['d'])
                axs[2].set_xlabel('iterations')
                axs[2].set_ylabel('accuracy')
                axs[2].set_ylim(0,1)
                fig.tight_layout()
                fig.savefig('plots/loss.png', dpi=fig.dpi)
                plt.close(fig)

                save_image(checkpoint_coarse/255, f'plots/coarse_{i}_{ep}.png')
                save_image(checkpoint_recon/255, f'plots/recon_{i}_{ep}.png')
                save_image(checkpoint_img/255, f'plots/orig_{i}.png')
                torch.save(netG.state_dict(), 'models/generator.pt')
                torch.save(netD.state_dict(), 'models/discriminator.pt')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--config", type=str, help="config file", required=True)
    parser.add_argument("--checkpoint-gen", type=str, help="path to generator.pt")
    parser.add_argument("--checkpoint-dis", type=str, help="path to discriminator.pt")
    args = parser.parse_args()

    logging.basicConfig(filename='output.log', encoding='utf-8', level=logging.DEBUG)

    config = Config(args.config)
    logging.debug(config)

    dataset = FaceMaskDataset(config.dataset_dir, 'maskffhq.csv', T.Resize(config.input_size))
    dataloader = dataset.loader(batch_size=config.batch_size)

    netG = Generator(input_size=config.input_size).to(device)
    netD = Discriminator(input_size=config.input_size).to(device)

    if args.checkpoint_gen is not None:
        loggin.info('resuming training of generator')
        checkpointG = torch.load(args.checkpoint_gen)
        netG.load_state_dict(checkpointG)

    if args.checkpoint_dis is not None:
        loggin.info('resuming training of discriminator')
        checkpointD = torch.load(args.checkpoint_dis)
        netD.load_state_dict(checkpointD)

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

    lossG = GeneratorLoss()
    lossRecon = L1ReconLoss()
    lossTV = TVLoss()
    lossD = DiscriminatorHingeLoss()

    params_g = sum([ p.numel() for p in netG.parameters() ])
    params_d = sum([ p.numel() for p in netD.parameters() ])

    print(f'Generator: {params_g} params\n' + \
          f'Discriminator: {params_d} params')
    train(netG, netD, optimG, optimD, lossG, lossD, lossRecon, lossTV, dataloader)

    torch.save(netG.state_dict(), 'models/generator.pt')
    torch.save(netD.state_dict(), 'models/discriminator.pt')
