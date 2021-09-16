import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

class FakeDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def __len__(self):
        return 100

    def __getitem__(self, index):
        return torch.rand((3, 256, 256)), torch.randint(0, 2, (1, 256, 256))

    def loader(self, **args):
        return DataLoader(self, **args)
