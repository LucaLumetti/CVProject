import sys
import torch
import os
import cv2
import logging
from config import Config
from dataset import FaceMaskDataset
from torchvision.utils import save_image
from torchvision import transforms as T
from generator import *
from discriminator import Discriminator
from loss import *
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(netG, netD, dataloader):
    netG.eval()
    netD.eval()

    metrics = {
            'l1': [],
            }

    accuracies = {
            'd': []
            }

    with torch.no_grad():
        for i, (imgs,masks) in enumerate(dataloader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            imgs = imgs / 127.5 - 1
            masks = masks / 1

            # forward G
            coarse_out, refined_out = netG(imgs,masks)
            reconstructed_imgs = refined_out*masks + imgs*(1-masks)

            pos_neg_imgs = torch.cat([imgs,reconstructed_imgs],dim=0)
            dmasks = torch.cat([masks,masks],dim=0)

            # forward D
            pred_pos_neg_imgs = netD(pos_neg_imgs, dmasks)
            pred_pos_imgs, pred_neg_imgs = torch.chunk(pred_pos_neg_imgs, 2, dim=0)

            # Calculate accuracy over the batch
            mean_pos_pred = pred_pos_imgs.clone().detach().mean(dim=1)
            mean_neg_pred = pred_neg_imgs.clone().detach().mean(dim=1)
            mean_pos_pred = torch.where(mean_pos_pred > 0.5, 1, 0).type(torch.FloatTensor)
            mean_pos_neg = torch.where(mean_pos_pred < 0.5, 0, 1).type(torch.FloatTensor)
            accuracyD = torch.sum(mean_pos_pred) + torch.sum(mean_neg_pred)
            accuracyD /= mean_pos_pred.shape[0] + mean_neg_pred.shape[0]
            accuracies['d'].append(accuracyD.item())

            # Calculate L1 loss over the batch
            l1_mean = torch.mean(torch.abs(imgs - reconstructed_imgs))
            metrics['l1'].append(l1_mean)

            output = (reconstructed_imgs[0] + 1) * 127.5
            save_image(output/255, f'{config.output_dir}/{i}.png')
    return metrics

if __name__ == '__main__':
    config = Config('config.json')
    logging.basicConfig(filename='test_output.log', encoding='utf-8', level=logging.INFO)
    logging.debug(config)

    dataset = FaceMaskDataset(config.dataset_dir, 'maskffhq.csv', T.Resize(config.input_size))
    dataloader = dataset.loader(batch_size=config.batch_size)

    netG = MSSAGenerator(input_size=config.input_size).to(device)
    netD = Discriminator(input_size=config.input_size).to(device)

    netG.load_state_dict(torch.load('models/generator.pt'))
    netD.load_state_dict(torch.load('models/discriminator.pt'))

    test(netG, netD, dataloader)
