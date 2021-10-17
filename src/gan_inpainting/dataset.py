import numpy as np
import pandas as pd
import os
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms as T
import torch.nn as nn
from torchvision.io import read_image
from torchvision.utils import save_image

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
    def __init__(self, dataset_dir, csv_file, transf, aug_t):
        self.dataset_dir = dataset_dir
        self.images = pd.read_csv(f'{dataset_dir}/{csv_file}', dtype='str')
        self.dataset_len = len(self.images)
        self.transf = transf if transf is not None else lambda x: x
        self.aug_t = aug_t

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(self.dataset_dir, self.images.iloc[index, 1])
        mask_name = os.path.join(self.dataset_dir, self.images.iloc[index, 2])

        img = read_image(img_name)
        mask = read_image(mask_name)
        mask = mask // 255

        img = self.transf(img)
        mask = self.transf(mask)
        aug_img = self.aug_t(img)
        aug_mask = self.aug_t(mask)

        return img, mask, aug_img, aug_mask

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
