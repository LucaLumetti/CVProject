import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        super(DiscriminatorHingeLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, pos, neg):
        hinge_pos = torch.mean(self.relu(1 - pos))
        hinge_neg = torch.mean(self.relu(1 + neg))
        loss_d = hinge_pos + hinge_neg
        return loss_d

class TVLoss(nn.Module):
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        for x in range(1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 6):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 11):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11, 20):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 29):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # fixed pretrained vgg19 model for feature extraction
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    # vgg19 perceptual loss
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def gram_matrix(self, x):
        (b, ch, h, w) = x.size()
        features = x.view(b, ch, w*h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0
        style_loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
            gm_x = self.gram_matrix(x_vgg[i])
            gm_y = self.gram_matrix(y_vgg[i])
            style_loss += self.weights[i] * self.mse_loss(gm_x, gm_y.detach())
        return loss, style_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.similarity = nn.CosineSimilarity(dim=-1, eps=1e-8)
        self.temperature = temperature

    # [a1, b1, c1, d1, a2, b2, c2, d2]
    def forward(self, x):
        x = torch.squeeze(x)
        x1, x2 = x.split(2)
        # this can be improved, some cos_sim are repeated between num and den
        num_sims = self.similarity(x1, x2) \
                    .div(self.temperature) \
                    .exp()
        den_sims = self.similarity(x1.unsqueeze(1), x2.unsqueeze(2)) \
                    .div(self.temperature) \
                    .exp() \
                    .sum(dim=-1)
        print(f'num: {num_sims.shape}')
        print(f'den: {den_sims.shape}')
        return torch.sum(-torch.log(torch.div(num_sims, den_sims)))

