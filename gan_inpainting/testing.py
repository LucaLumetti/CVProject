import sys
import torch
import os
import cv2
import logging
from config import Config
from dataset import FaceMaskDataset
from torchvision.utils import save_image
from generator import *
from discriminator import Discriminator
from loss import *
from pathlib import Path
from script_dataset import create_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(netG, netD, lossG, lossD, dataloader):
    netG.eval()
    netD.eval()

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

    for i,(imgs,masks) in enumerate(dataloader):

        imgs = imgs.to(device)
        masks = masks.to(device)

        imgs = imgs / 127.5 - 1
        masks = masks / 1

        # forward G
        coarse_out, refined_out = netG(imgs,masks)
        reconstructed_imgs = refined_out*masks + imgs*(1-masks)

        #pos_imgs = torch.cat([imgs, masks], dim=1)
        #neg_imgs = torch.cat([reconstructed_imgs, masks], dim=1)
        #pos_neg_imgs = torch.cat([pos_imgs, neg_imgs], dim=0)
        pos_neg_imgs = torch.cat([imgs,reconstructed_imgs],dim=0)
        dmasks = torch.cat([masks,masks],dim=0)

        # forward D
        #pos_neg_imgs, dmasks = torch.split(pos_neg_imgs, (3, 1), dim=1)
        pred_pos_neg_imgs = netD(pos_neg_imgs, dmasks)
        pred_pos_imgs, pred_neg_imgs = torch.chunk(pred_pos_neg_imgs, 2, dim=0)


        # Calculate accuracy and loss, dk if useful bc we're creating a separated file for that
        with torch.inference_mode():
            mean_pos_pred = pred_pos_imgs.clone().detach().mean(dim=1)
            mean_neg_pred = pred_neg_imgs.clone().detach().mean(dim=1)
            mean_pos_pred= torch.where(mean_pos_pred > 0.5,1,0).type(torch.FloatTensor)
            mean_pos_neg= torch.where(mean_pos_pred < 0.5,0,1).type(torch.FloatTensor)
            accuracyD = torch.sum(mean_pos_pred) + torch.sum(mean_neg_pred)
            accuracyD /= mean_pos_pred.shape[0] + mean_neg_pred.shape[0]
            accuracies['d'].append(accuracyD.item())

            # loss D
            loss_discriminator = lossD(pred_pos_imgs, pred_neg_imgs)
            losses['d'].append(loss_discriminator.item())

            # loss G
            pred_neg_imgs = netD(reconstructed_imgs, masks)
            loss_generator = lossG(pred_neg_imgs)
            loss_recon = lossRecon(imgs, coarse_out, refined_out, dmasks)
            loss_tv = lossTV(refined_out)
            loss_perc, loss_style = lossVGG(imgs,refined_out)
            losses['g'].append(loss_generator.item())
            losses['r'].append(loss_recon.item())
            losses['tv'].apppend(loss_tv.item())
            losses['perc'].append(loss_perc.item())
            losses['style'].append(loss_style.item())

        output = (reconstructed_imgs[0] + 1) * 127.5
        save_image(output/255,f'{config.dataset_dir}/output/{filename}')
    return

if __name__ == '__main__':
    config = Config('config.json')
    logging.basicConfig(filename='test_output.log',encoding='utf-8',level=logging.DEBUG)
    logging.debug(config)

    sys.path.append(config.script_dataset_dir)

    pathname = Path(sys.argv[1])
    split_folder = sys.argv[1].split('/')
    subfolder = split_folder[-2]
    filename = split_folder[-1]
    csvf = open(f'{pathname.parent.parent.parent}/maskffhq.csv', 'a')
    path_to_mask = Path(f'{pathname.parent.parent.parent}' + \
                        f'/masked_images/{subfolder}/{filename}')
    path_to_mask.parent.mkdir(parents=True, exist_ok=True)

    if path_to_mask.exists():
        logging.info('Path to mask already exists !')
        exit(0)

    img = cv2.imread(str(pathname))
    mask = create_mask(img, False)

    if mask is not None:
        cv2.imwrite(str(path_to_mask), mask)
        csvf.write(f'{filename.split(".")[0]},{pathname.absolute()},{path_to_mask.absolute()}\n')
        csvf.close()

        dataset = FaceMaskDataset(config.dataset_dir, 'maskffhq.csv')
        dataloader = dataset.loader(batch_size=config.batch_size)

        netG = MSSAGenerator(input_size=config.input_size).to(device)
        netD = Discriminator(input_size=config.input_size).to(device)

        netG.load_state_dict(torch.load(config.gen_path))
        netD.load_state_dict(torch.load(config.disc_path))

        lossG = GeneratorLoss()
        lossRecon = L1ReconLoss()
        lossTV = TVLoss()
        lossD = DiscriminatorLoss()
        lossVGG = VGGLoss()
        test(netG, netD, lossG, lossD, lossRecon, lossTV, lossVGG, dataloader)

        # deleting csv file to avoid repeating test on same images
        os.remove(config.csvf_path)

    else:
        logging.info(f"Face not found for {pathname}")
        exit(-1)


