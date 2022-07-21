import errno
import urllib.request

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
import skimage.transform
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np

import time
import math

from utils import *


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


class ImageFile(Dataset):
    def __init__(self, img, grayscale=False, resolution=None, crop_square=False):

        super().__init__()

        # self.img = Image.open(filename)
        self.img = img
        if grayscale:
            self.img = self.img.convert('L')
        else:
            self.img = self.img.convert('RGB')

        self.img_channels = len(self.img.mode)  # 3
        self.resolution = self.img.size

        if crop_square:  # preserve aspect ratio
            self.img = crop_max_square(self.img)
            # print('crop_square:' + str(self.img.size))

        if resolution is not None:
            self.resolution = resolution
            self.img = self.img.resize(resolution, Image.ANTIALIAS)

        self.img = np.array(self.img)
        self.img = self.img.astype(np.float32) / 255.
        # print('img_numpy: ' + str(self.img.shape))

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # print('image_file: ' + str(self.img.shape))
        return self.img


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, dataset, centered=True, include_end=False):
        self.centered = centered
        self.include_end = include_end
        self.transform = Compose([
            ToTensor(),
        ])

        self.dataset = dataset
        # print('res: ' + str(self.dataset.resolution))
        self.mgrid = create_grid(self.dataset.resolution)  # -1(512 * 512) * 2

        # sample pixel centers
        self.mgrid = self.mgrid + 1 / (2 * self.dataset.resolution[0])

        img = self.transform(self.dataset[0])
        _, self.rows, self.cols = img.shape

        self.img_chw = img
        self.img = img.permute(1, 2, 0)
        # self.img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)  # [h, w, c] ==> [h*w, 3]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        coords = self.mgrid
        img = self.img

        in_dict = {'coords': coords}
        gt_dict = {'img': img}

        return in_dict, gt_dict


if __name__ == '__main__':
    img_path = 'data/lighthouse.png'
    img = Image.open(img_path)
    print(img.size)
    print(img.mode)
    print(len(img.mode))
    img = crop_max_square(img)  # 512 * 512
    print(img.size)  # Image
    print(img.mode)
    img = np.array(img)  # 512 * 512 * 3
    print(img.shape)
    transform = Compose([
        ToTensor(),
    ])
    img = transform(img)
    print(img.shape)  # 3 * 512 * 512
