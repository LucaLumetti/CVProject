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

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, pos, neg):
        hinge_pos = torch.mean(self.relu(1 - pos))
        hinge_neg = torch.mean(self.relu(1 + neg))
        loss_d = 0.5*(hinge_pos + hinge_neg)
        return loss_d
