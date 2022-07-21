import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np

import time
import math

from utils import *
from SIREN import *
from modules import *


class NIVR(nn.Module):
    def __init__(self, res, side_len, time_len, mp_in, mp_out, mf_in, mf_out):
        super().__init__()
        self.mp = mp(in_features=mp_in, out_features=mp_out)
        self.mf = mf(in_features=mf_in, out_features=mf_out)
        self.resolution = res
        self.side_len = side_len
        self.time_len = time_len

    def forward(self, x, idx):
        frame_idx = idx
        # print(x)
        coords = x['coords']
        coords = torch.squeeze(coords)
        # print('coords.shape: ' + str(coords.shape))

        t = get_t(self.time_len, frame_idx)
        r_tensor = position_encoder_t(t, self.time_len)
        phi_t = self.mp(r_tensor.float())

        c_tensor = position_encoder_c(coords, phi_t, side_len=self.side_len).permute(1, 2, 0, 3).contiguous().view(self.resolution[1], self.resolution[0], -1)
        # print('c_tensor.shape: ' + str(c_tensor.shape))

        output = self.mf(c_tensor)

        return output
