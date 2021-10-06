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
from metrics import TestMetrics

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(netG, netD, dataloader):
    netG.eval()
    netD.eval()

    metrics_tester = TestMetrics()

    metrics = {
            'l1': [],
            'accuracy': []
            }

    with torch.no_grad():
        for i, (imgs,masks) in enumerate(dataloader):
            print(f'i:{i}')
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
            mean_neg_pred = torch.where(mean_neg_pred > 0.5, 0, 1).type(torch.FloatTensor)
            accuracyD = torch.sum(mean_pos_pred) + torch.sum(mean_neg_pred)
            tot_elem = mean_pos_pred.shape[0] + mean_neg_pred.shape[0]
            accuracyD /= tot_elem
            metrics['accuracy'].append(accuracyD.item())

            # Calculate L1 loss over the batch
            l1_mean = torch.mean(torch.abs(imgs - reconstructed_imgs))
            metrics['l1'].append(l1_mean)

            output = (reconstructed_imgs + 1) * 127.5

            print(output.shape)
            metrics_tester.update(imgs, reconstructed_imgs)
            for d in range(output.size(0)):
                save_image(output[d]/255, f'{config.output_dir}/{i*config.batch_size+d}.png')

        fid_score = metrics_tester.FID(config.test_dir,f'{config.output_dir}',config.batch_size,device)
        metrics_dict = metrics_tester.get_metrics()
        metrics_dict['FID'] = fid_score
        for key in metrics:
            metrics_dict[key] = torch.mean(torch.tensor(metrics[key])).item()

    return metrics_dict

if __name__ == '__main__':
    config = Config('config.json')
    logging.basicConfig(filename='test_output.log', level=logging.INFO)
    logging.debug(config)

    dataset = FaceMaskDataset(config.test_dir, 'maskceleba_test.csv', T.Resize(config.input_size))
    dataloader = dataset.loader(batch_size=config.batch_size)

    netG = MSSAGenerator(input_size=config.input_size).to(device)
    netD = Discriminator(input_size=config.input_size).to(device)

    netG.load_state_dict(torch.load('models/generator.pt', map_location=device))
    netD.load_state_dict(torch.load('models/discriminator.pt', map_location=device))

    metrics = test(netG, netD, dataloader)
    print(metrics)
