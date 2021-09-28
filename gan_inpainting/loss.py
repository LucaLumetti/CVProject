import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

# Implementation of the GanHingeLos, divided in 2 different classes
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()

    def forward(self, neg):
        loss_g = -torch.mean(neg)
        return loss_g

# code taken from https://github.com/avalonstrel/GatedConvolution_pytorch/blob/0a49013a70e77cc484ab45a5da535c2ac003b252/models/loss.py#L155
# the code as been rearranged for better readability bc the original really suck
class L1ReconLoss(nn.Module):
    def __init__(self, chole_alpha=1, cunhole_alpha=1, rhole_alpha=1, runhole_alpha=1):
        super(L1ReconLoss, self).__init__()
        self.chole_alpha = chole_alpha
        self.cunhole_alpha = cunhole_alpha
        self.rhole_alpha = rhole_alpha
        self.runhole_alpha = runhole_alpha

    def forward(self, imgs, coarse_imgs, recon_imgs, masks):
        masks, _ = torch.chunk(masks, 2, dim=0)
        masks_viewed = masks.view(masks.size(0), -1).mean(1).view(-1,1,1,1)
        norm_mask = masks / masks_viewed
        neg_norm_mask = (1 - masks) / (1. - masks_viewed)

        imgs_recon_l1 = torch.abs(imgs - recon_imgs)
        imgs_coarse_l1 = torch.abs(imgs - coarse_imgs)

        return self.rhole_alpha*torch.mean(imgs_recon_l1*norm_mask) + \
               self.runhole_alpha*torch.mean(imgs_recon_l1*neg_norm_mask) + \
               self.chole_alpha*torch.mean(imgs_coarse_l1*norm_mask) + \
               self.cunhole_alpha*torch.mean(imgs_coarse_l1*neg_norm_mask)

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, pos, neg):
        pos = - torch.mean(pos)
        neg = torch.mean(neg)
        loss_d = pos + neg
        return loss_d

class DiscriminatorHingeLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, pos, neg):
        hinge_pos = torch.mean(self.relu(1 - pos))
        hinge_neg = torch.mean(self.relu(1 + neg))
        loss_d = hinge_pos + hinge_neg
        return loss_d
