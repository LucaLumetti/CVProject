import numpy as np
import pandas as pd
import os
import cv2

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import io

from torch.utils.data import Dataset, DataLoader

class FakeDataset(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return 50

    def __getitem__(self, index):
        return torch.rand((3, 256, 256)), torch.randint(0, 2, (1, 256, 256))

    def loader(self, **args):
        return DataLoader(self, **args)

class FaceMaskDataset(Dataset):
    def __init__(self, dataset_dir, csv_file):
        self.dataset_dir = dataset_dir
        self.images = pd.read_csv(f'{dataset_dir}/{csv_file}', dtype='str')
        self.dataset_len = len(self.images)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.dataset_dir, self.images.iloc[index, 1])
        mask_name = os.path.join(self.dataset_dir, self.images.iloc[index, 2])

        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_name)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask // 255

        img = torch.tensor(img)
        mask = torch.tensor(mask)
        img = torch.swapaxes(img, 0, 2)
        mask = mask[None,:,:]

        return img, mask

    def loader(self, **args):
        return DataLoader(self, **args)

if __name__ == "__main__":
    dataset = FaceMaskDataset('../dataset/', 'maskffhq.csv')
    dloader = dataset.loader()
    for imgs, masks in dloader:
        cv2.imshow('img', imgs[0].numpy())
        cv2.waitKey(0)
        cv2.imshow('mask', masks[0].numpy())
        cv2.waitKey(0)
        break
