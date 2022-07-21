import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np


def get_L1_loss(pred, target):
    loss = nn.L1Loss()

    return loss(pred, target)
