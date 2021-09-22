import sys

import numpy as np

import torch
import cv2
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from config import Config
from dataset import FaceMaskDataset
from torchvision.utils import save_image

from generator import Generator
from discriminator import Discriminator
from loss import GeneratorLoss, DiscriminatorLoss, L1ReconLoss
from pathlib import Path
from script_dataset import create_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(netG, netD, lossG, lossD, img, mask):
    netG.eval()
    netD.eval()

    losses = {
        'g': [],
        'd': [],
        'r': [],
    }

    accuracies = {
        'd': []
    }

    img = img.to(device)
    mask = mask.to(device)

    img = img / 127.5 - 1

    coarse_out, refined_out = netG(img,mask)
    reconstructed_imgs = refined_out*mask + img*(1-mask)

    # forward G
    pos_imgs = torch.cat([img, mask], dim=1)
    neg_imgs = torch.cat([reconstructed_imgs, mask], dim=1)
    pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)

    # forward D
    pos_neg_imgs, dmasks = torch.split(pos_neg_imgs, (3, 1), dim=1)
    pred_pos_neg_imgs = netD(pos_neg_imgs, dmasks)
    pred_pos_imgs, pred_neg_imgs = torch.chunk(pred_pos_neg_imgs, 2, dim=0)

    # inference mode will be faster but can't make inplace op inside it
    # Calculate accuracy and loss, dk if useful bc we're creating a separated file for that
    with torch.no_grad():
        mean_pos_pred = pred_pos_imgs.clone().detach().mean(dim=1)
        mean_neg_pred = pred_neg_imgs.clone().detach().mean(dim=1)
        mean_pos_pred[mean_pos_pred > 0.5] = 1
        mean_pos_pred[mean_pos_pred <= 0.5] = 0
        mean_neg_pred[mean_neg_pred > 0.5] = 0
        mean_neg_pred[mean_neg_pred <= 0.5] = 1
        accuracyD = torch.sum(mean_pos_pred) + torch.sum(mean_neg_pred)
        accuracyD /= mean_pos_pred.shape[0] + mean_neg_pred.shape[0]
        accuracies['d'].append(accuracyD.item())

        # loss + backward D
        loss_discriminator = lossD(pred_pos_imgs, pred_neg_imgs)
        losses['d'].append(loss_discriminator.item())

        # loss + backward G
        pred_neg_imgs = netD(reconstructed_imgs, mask)
        loss_generator = lossG(pred_neg_imgs)
        loss_recon = lossRecon(img, coarse_out, refined_out, dmasks)
        losses['g'].append(loss_generator.item())
        losses['r'].append(loss_recon.item())

        output = (reconstructed_imgs + 1) * 127.5
        output = torch.flip(output,[-1])

    # TODO: need to save image somewhere, this is a temporary location
    cv2.imwrite(config.test_dir+'/output/'+filename,output)
    return

if __name__ == '__main__':
    config = Config('config.json')
    # Not sure if we really need dataloader in this case
    #dataset = FaceMaskDataset(config.test_dir, 'maskffhq.csv')
    #dataloader = dataset.loader(batch_size=config.batch_size)

    netG = Generator(input_size=config.input_size).to(device)
    netD = Discriminator(input_size=config.input_size).to(device)
    netG.load_state_dict(torch.load(config.gen_path))
    netD.load_state_dict(torch.load(config.disc_path))

    lossG = GeneratorLoss()
    lossRecon = L1ReconLoss()
    lossD = DiscriminatorLoss()

    # Calculate masks for all photos in test folder
    pathname = Path(sys.argv[1])
    split_folder = sys.argv[1].split('/')
    subfolder = split_folder[-2]
    filename = split_folder[-1]
    csvf = open(f'{pathname.parent.parent.parent}/maskffhq.csv', 'a')
    path_to_mask = Path(f'{pathname.parent.parent.parent}' + \
                        f'/masked_images/{subfolder}/{filename}')
    path_to_mask.parent.mkdir(parents=True, exist_ok=True)

    if path_to_mask.exists():
        exit(0)

    img = cv2.imread(str(pathname))

    mask = create_mask(img, False)

    if mask is not None:
        #cv2.imwrite(str(path_to_mask), mask)
        #csvf.write(f'{filename.split(".")[0]},{pathname.absolute()},{path_to_mask.absolute()}\n')
        #csvf.close()
        test(netG, netD, lossG, lossD, img, mask)

    else:
        print(f"Face not found for {pathname}")
        exit(-1)


