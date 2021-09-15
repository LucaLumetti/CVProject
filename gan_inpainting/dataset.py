import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

class FFHQDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...

    def loader(self, **args):
        return DataLoader(self, **args)
