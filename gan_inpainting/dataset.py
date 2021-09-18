import numpy as np

import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import io

from torch.utils.data import Dataset, DataLoader

class FaceMaskDataset(Dataset):
    def __init__(self, csv_file, dataset_dir):
        self.dataset_dir = dataset_dir
        self.images = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.dataset_dir,
                                self.images.iloc[index, 0])
        mask_name = img_name.split(".", 1)
        mask_name = mask_name[0] + "_mask" + mask_name[1]
        return (io.read_image(img_name), io.read_image(mask_name))

    def loader(self, **args):
        return DataLoader(self, **args)
