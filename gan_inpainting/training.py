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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingMetrics:

    def __init__(self, dataloader):
        self.lossG = []
        self.lossD = []
        self.lossR = []
        self.accuracy = []
        self.dataloader = dataloader

    '''
        Parameters:
            lossG: loss for Generator
            lossD: loss for Discriminator
            lossR: loss for Refined
            D_result: result of Discriminator for accuracy
            netG: Generator Net for testing
            netD: Discriminator Net for testing
    '''
    def update(self, lossG, lossD, lossR, D_result, netG, netD):
        self.lossG.append(lossG)
        self.lossD.append(lossD)
        self.lossR.append(lossR)

        pred_pos_imgs, pred_neg_imgs = torch.chunk(D_result, 2, dim=0)

        # with torch.inference_mode():
        mean_pos_pred = pred_pos_imgs.mean(dim=1)
        mean_neg_pred = pred_neg_imgs.mean(dim=1)
        mean_pos_pred[mean_pos_pred > 0.5] = 1
        mean_pos_pred[mean_pos_pred <= 0.5] = 0

        mean_neg_pred[mean_neg_pred > 0.5] = 1
        mean_neg_pred[mean_neg_pred <= 0.5] = 0
        mean_neg_pred = ~mean_neg_pred

        accuracyD = torch.sum(mean_pos_pred) + torch.sum(mean_neg_pred)
        accuracyD /= mean_pos_pred.shape[0] + mean_neg_pred.shape[0]
        self.accuracy.append(accuracyD)

        # every 100 img, print losses, update the graph, output an image as example
        if lossG.shape[0] % 100 == 0:
            print(f"[{i}]\t" + \
                  f"loss_g: {self.lossG[-1]}, " + \
                  f"loss_d: {self.lossD[-1]}, " + \
                  f"loss_r: {self.lossR[-1]}, " + \
                  f"accuracy_d: {accuracy[-1]}")

            fig, axs = plt.subplots(3, 1)
            x_axis = range(len(self.lossG))
            # loss g
            axs[0].plot(x_axis, self.lossG, x_axis, self.lossR)
            axs[0].set_xlabel('iterations')
            axs[0].set_ylabel('loss')
            # loss d
            axs[1].plot(x_axis, self.lossD)
            axs[1].set_xlabel('iterations')
            axs[1].set_ylabel('loss')
            # acc d
            axs[2].plot(x_axis, accuracies['d'])
            axs[2].set_xlabel('iterations')
            axs[2].set_ylabel('accuracy')
            axs[2].set_ylim(0, 1)
            fig.tight_layout()
            fig.savefig('plots/loss.png', dpi=fig.dpi)
            plt.close(fig)

            img, mask = self.dataloader[np.random.random_integers(0,lossG.shape[0])]

            img = img.to(device)
            mask = mask.to(device)

            # change img range from [0,255] to [-1,+1]
            img = img / 127.5 - 1

            coarse_out, refined_out = netG(imgs, masks)
            reconstructed_imgs = refined_out * masks + imgs * (1 - masks)
            checkpoint_recon = ((reconstructed_imgs[0] + 1) * 127.5)
            checkpoint_img = ((img[0] + 1) * 127.5)
            save_image(checkpoint_recon / 255, 'plots/recon.png')
            save_image(checkpoint_img / 255, 'plots/orig.png')




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

        metrics.update(loss_generator, loss_discriminator, loss_recon, pred_pos_neg_imgs, netG, netD)
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
