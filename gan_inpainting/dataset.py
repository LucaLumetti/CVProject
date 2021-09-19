import numpy as np

import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import io

from torch.utils.data import Dataset, DataLoader

class FakeDataset(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return 10

    def __getitem__(self, index):
        return torch.rand((3, 256, 256)), torch.randint(0, 2, (1, 256, 256))

    def loader(self, **args):
        return DataLoader(self, **args)

class FaceMaskDataset(Dataset):
    def __init__(self, csv_file, dataset_dir):
        self.dataset_dir = dataset_dir
        self.images = pd.read_csv(csv_file, dtype='str')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        # we expect the csv file format to be:
        # index,path
        # 0000,/a/0000.jpg
        # 0001,/a/0001.jpg
        # 0002,/b/0002.jpg
        img_name = os.path.join(self.dataset_dir, self.images.iloc[index, 1])
        mask_name = img_name.split(".", 2)
        mask_name = mask_name[1] + "_mask." + mask_name[2]
        return (io.read_image(img_name), io.read_image(mask_name))

    def loader(self, **args):
        return DataLoader(self, **args)
