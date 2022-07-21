import torch
import torchvision
from torchvision import transforms
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
import utils

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def position_encoder_t(t, lt):
    N_t = int(math.log(lt, 2) + 1)
    r_list = []
    for i in range(N_t):
        r_i = (math.sin(2 ** (i - 1) * math.pi * t), math.cos(2 ** (i - 1) * math.pi * t))
        r_list.append(r_i)

    # print(r_list)
    # print(len(r_list))

    r_numpy = np.array(r_list)
    r_tensor = torch.from_numpy(r_numpy).to(device)

    # print(r_tensor)
    # print(r_tensor.shape)

    return r_tensor


class mp(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = Siren(in_features=in_features, hidden_features=[256, 256, 256, 256, 256], hidden_layers=4,
                         out_features=out_features, outermost_linear=True)

    def forward(self, x):
        x = x.view(1, -1)
        output = self.mlp(x).view(-1, 1)
        # output = torch.cat((output, output), 1)

        return output


def position_encoder_c(coords, phi_t, side_len):
    N_c = int(math.log(side_len, 2) + 1)
    # print(N_c)

    sin_list = []
    cos_list = []

    for i in range(N_c):
        sin_list.append(torch.sin(2 ** (i - 1) * math.pi * coords + phi_t[2 * i]))
        cos_list.append(torch.cos(2 ** (i - 1) * math.pi * coords + phi_t[2 * i + 1]))
    sin_tensor = torch.stack(sin_list, 0)
    cos_tensor = torch.stack(cos_list, 0)

    # print(sin_list)
    # print(cos_list)

    # print('sin_tensor.shape: ' + str(sin_tensor.shape))  # Nc * side_len * side_len * 2
    # print('cos_tensor.shape: ' + str(cos_tensor.shape))

    # print('sin_tensor: ')
    # print(sin_tensor)  # Nc * 2

    # print('cos_tensor: ')
    # print(cos_tensor)

    c_tensor = torch.cat((sin_tensor, cos_tensor), dim=-1).to(device)
    # print('c_tensor: ')
    # print(c_tensor)

    return c_tensor


class mf(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.mlp = Siren(in_features=in_features, hidden_features=[256, 256, 256, 256, 256], hidden_layers=4,
                         out_features=out_features, outermost_linear=True)

    def forward(self, x):
        output = self.mlp(x)
        output = torch.tanh(output)

        return output


if __name__ == '__main__':
    time_len = 120
    side_len = 32
    frame_idx = 1
    t = get_t(time_len, frame_idx)
    print('t: ' + str(t))
    r_tensor = position_encoder_t(t, time_len)
    print('r_tensor: ' + str(r_tensor))
    print('r_tensor.shape: ' + str(r_tensor.shape))
    mp_in = int(math.log(time_len, 2) + 1) * 2
    mp_out = 2 * int(math.log(side_len, 2) + 1)
    print('mp_in: ' + str(mp_in))
    print('mp_out: ' + str(mp_out))
    m_p = mp(in_features=mp_in, out_features=mp_out)
    phi_t = m_p(r_tensor.float())
    print('phi_t: ' + str(phi_t))
    print('Nt: ' + str(int(mp_out / 2)))
    print('phi_t.shape: ' + str(phi_t.shape))  # 2N ([sin, cos] * N)
    coords = create_grid((32, 32))
    print('coords.shape：' + str(coords.shape))  # side_len * side_len * 2
    # coords = coords_list[1]
    # print(coords.shape)
    c_tensor = position_encoder_c(coords, phi_t, side_len)
    # print('c_tensor: ' + str(c_tensor))
    # print('c_tensor.shape: ' + str(c_tensor.shape))

    # gamma_c = c_tensor[:, 0, 0, :]
    # gamma_c = gamma_c.reshape(1, -1)
    # mf_in = gamma_c.shape[1]
    # print(mf_in)
    mf_in = c_tensor.shape[0] * c_tensor.shape[-1]
    mf_out = 3
    m_f = mf(in_features=mf_in, out_features=mf_out)
    output = []
    for x in range(side_len):
        for y in range(side_len):
            idx = side_len * x + y
            gamma_c = c_tensor[:, x, y, :]
            gamma_c = gamma_c.reshape(1, -1)
            rgb = m_f(gamma_c.float())
            # print(rgb.shape)
            # print(rgb)
            output.append(rgb)

    output = torch.stack(output, 0).view(side_len, side_len, -1)
    print(output)
    print(output.shape)

    output_dir = './output/pred/'
    output_path = os.path.join(output_dir, 'step_%04d/' % 1)
    utils.mkdir(output_path)
    print(output_path)
    output_path = os.path.join(output_path, 'frame_%03d.jpg' % 2)

    input_tensor = output.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).numpy()
    # print(input_tensor.shape)
    im = Image.fromarray(input_tensor)
    im.save(output_path)
