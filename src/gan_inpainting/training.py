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
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import matplotlib.pyplot as plt

from generator import *
from discriminator import Discriminator
from loss import *
from dataset import FakeDataset, FaceMaskDataset

from metrics import TrainingMetrics

# torch.autograd.set_detect_anomaly(True)
# a loss history should be held to keep tracking if the network is learning
# something or is doing completely random shit
# also a logger would be nice
def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    print(f'[p{rank}] joined the training')
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    dataset = FaceMaskDataset(
            args.dataset_dir,
            'maskffhq.csv',
            T.Resize(args.input_size),
            T.Compose([
                T.ToPILImage(),
                T.RandomHorizontalFlip(p=1.0),
                T.ToTensor(),
            ])
        )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=rank
    )

    dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            sampler=sampler)

    netG = MSSAGenerator(input_size=args.input_size)
    netD = Discriminator(input_size=args.input_size)

    netG.cuda(gpu)
    netD.cuda(gpu)

    netG = DDP(netG, device_ids=[gpu])
    netD = DDP(netD, device_ids=[gpu])

    optimG = torch.optim.Adam(
                netG.parameters(),
                lr=args.learning_rate_g,
                betas=(0.5, 0.999)
            )
    optimD = torch.optim.Adam(
                netD.parameters(),
                lr=args.learning_rate_d,
                betas=(0.5, 0.999)
            )
    # Resume checkpoint if necessary
    if args.checkpoint is True:
        generator_dir = args.checkpoint_dir + '/generator.pt'
        discriminator_dir = args.checkpoint_dir + '/discriminator.pt'
        opt_generator_dir = args.checkpoint_dir + '/opt_generator.pt'
        opt_discriminator_dir = args.checkpoint_dir + '/opt_discriminator.pt'

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
    lossContra = InfoNCE()

    metrics = TrainingMetrics(
            args.screenstep,
            args.video_dir,
            dataset
        )

    netG.train()
    netD.train()

    losses = {
            'g': [],
            'd': [],
            'r': [],
            'tv': [],
            'perc': [],
            'style': [],
            'contra': []
            }

    accuracies = {
            'd': []
            }

    for ep in range(args.epochs):
        for i, (imgs, masks, aug_imgs, aug_masks) in enumerate(dataloader):
            netG.zero_grad()
            netD.zero_grad()
            optimG.zero_grad()
            optimD.zero_grad()
            lossG.zero_grad()
            lossD.zero_grad()
            lossTV.zero_grad()
            lossVGG.zero_grad()

            imgs = torch.cat([imgs, aug_imgs], dim=0)
            masks = torch.cat([masks, aug_masks], dim=0)
            imgs = imgs.cuda(gpu)
            masks = masks.cuda(gpu)

            # change img range from [0,255] to [-1,+1]
            imgs = imgs / 127.5 - 1
            masks = masks / 1.

            # forward G
            emb_repr, coarse_out, refined_out = netG(imgs, masks)
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
            loss_contra = lossContra(*emb_repr.chunk(2))
            loss_gen_recon = loss_generator + loss_recon + \
                    loss_tv + loss_perc + loss_style + loss_contra

            losses['g'] = loss_generator.item()
            losses['r'] = loss_recon.item()
            losses['tv'] = loss_tv.item()
            losses['perc'] = loss_perc.item()
            losses['style'] = loss_style.item()
            losses['contra'] = loss_contra.item()

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
                torch.save(netG.state_dict(), f'{args.checkpoint_dir}/generator.pt')
                torch.save(netD.state_dict(), f'{args.checkpoint_dir}/discriminator.pt')
                torch.save(optimG.state_dict(), f'{args.checkpoint_dir}/opt_generator.pt')
                torch.save(optimD.state_dict(), f'{args.checkpoint_dir}/opt_discriminator.pt')
            metrics.update(losses, pred_pos_neg_imgs, netG, netD)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--checkpoint", default=False, help="resume training")
    parser.add_argument("--screenstep", default=500, type=int, help="how often output metrics and imgs")
    parser.add_argument("--nodes", default=1, type=int, help="number of nodes")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus per node")
    parser.add_argument("--nr", default=0, type=int, help="ranking within the nodes")
    parser.add_argument("--epochs", default=1, type=int, help="number of total epochs to run")
    parser.add_argument("--batch_size", default=2, type=int, help="batch size")
    parser.add_argument("--input_size", default=256, type=int, help="size of the imgs")
    parser.add_argument("--learning_rate_g", default=0.0001, type=float, help="learning rate of the generator")
    parser.add_argument("--learning_rate_d", default=0.0004, type=float, help="learning rate of the discriminator")
    parser.add_argument("--dataset_dir", type=str, help="dataset location", required=True)
    parser.add_argument("--checkpoint_dir", type=str, help="where to load/save checkpoints", required=True)
    parser.add_argument("--video_dir", type=str, help="where to save the video")
    args = parser.parse_args()

    args.world_size = args.gpus*args.nodes

    logging.basicConfig(filename='output.log', level=logging.INFO)

    print(f'spawning {args.world_size} processes')
    mp.spawn(train, nprocs=args.gpus, args=(args,))
    print('end training')
