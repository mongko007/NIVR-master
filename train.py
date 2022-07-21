import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch
from functools import partial
import math
import numpy as np
import utils

from dataloader import *
from loss_functions import *
from NIVR import *

torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)

args = argparse.ArgumentParser()

# config file, output directories
args.add_argument('--logging_root', type=str, default='./logs/', help='root for logging')
args.add_argument('--model_dir', type=str, default='./model/ckpt/', help='directory for saving ckpt')
args.add_argument('--loss_dir', type=str, default='./output/loss/', help='directory for saving loss.txt')
args.add_argument('--output_dir', type=str, default='./output/pred/', help='directory for saving pred.jpg')

# general training options
args.add_argument('--batch_size', type=int, default=1)
args.add_argument('--res', type=int, default=(432, 192),
                  help='resolution of image to fit, also used to set the network-equivalent sample rate'
                       + ' i.e., the maximum network bandwidth in cycles per unit interval is half this value')
args.add_argument('--lr', type=float, default=1e-4, help='learning rate')
args.add_argument('--num_steps', type=int, default=6000, help='number of training steps')
args.add_argument('--step', type=int, default=300, help='number of training steps')
args.add_argument('--gpu', type=int, default=0, help='gpu id to use for training')
args.add_argument('--side_len', type=int, default=192, help='the smaller side of the frame for the spatial dimension')
args.add_argument('--time_len', type=int, default=120, help='temporal dimension')

# data processing and i/o
args.add_argument('--centered', action='store_true', default=False, help='centere input coordinates as -1 to 1')
args.add_argument('--img_path', type=str, default='./data/guitar/', help='path to specific png filename')
args.add_argument('--grayscale', action='store_true', default=False, help='if grayscale image')

# summary, logging options
args.add_argument('--steps_til_ckpt', type=int, default=100, help='Time interval in seconds until checkpoint is saved.')
args.add_argument('--steps_til_summary', type=int, default=100, help='Time interval in seconds until tensorboard '
                                                                     'summary is saved.')

opt = args.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

Nt = int(math.log(opt.time_len, 2) + 1)
Nc = int(math.log(opt.side_len, 2) + 1)
mp_in = 2 * Nt
mp_out = 2 * 2 * Nc


def init_model(opt):
    # mp_in = int(math.log(opt.time_len, 2) + 1) * 2
    # mp_out = 4 * int(math.log(opt.side_len, 2) + 1)

    model = NIVR(res=opt.res, side_len=opt.side_len, time_len=opt.time_len, mp_in=mp_in, mp_out=mp_out, mf_in=mp_out,
                 mf_out=3)
    model = model.to(device)

    # model.cuda()

    return model


def init_dataloader(opt, frame_idx):
    img_path = opt.img_path + str(frame_idx + 1) + '.png'

    img = Image.open(img_path)
    # img = crop_max_square(img)

    # init datasets
    trn_dataset = ImageFile(img, grayscale=opt.grayscale, resolution=None, crop_square=False)

    val_dataset = ImageFile(img, grayscale=opt.grayscale, resolution=None, crop_square=False)
    # print('trn_dataset: ' + str(trn_dataset.resolution))

    trn_dataset = ImageLoader(trn_dataset, centered=opt.centered, include_end=False)

    val_dataset = ImageLoader(val_dataset, centered=opt.centered, include_end=False)

    dataloader = DataLoader(trn_dataset, shuffle=True, batch_size=opt.batch_size, pin_memory=True, num_workers=0)

    return trn_dataset, val_dataset, dataloader


def train():
    print('Creating Output Path  ······')
    output_dir = opt.output_dir
    utils.mkdir(output_dir)
    model_dir = opt.model_dir
    utils.mkdir(model_dir)
    loss_dir = opt.loss_dir
    utils.mkdir(loss_dir)

    print('Initializing Model  ······')
    model = init_model(opt)
    model.cuda()
    model.train()

    print('Setting Optimizer  ······')
    optim = torch.optim.Adam(lr=opt.lr, params=model.parameters(), amsgrad=True)

    losses = []
    losses_frame = []

    for i in range(opt.num_steps):

        for idx in range(opt.time_len):
            trn_dataset, val_dataset, dataloader = init_dataloader(opt, idx)

            train_generator = iter(dataloader)
            model_input, gt = next(train_generator)

            model_input = dict2cuda(model_input)
            # print('model_input.shape: ' + str(model_input['coords'].shape))
            gt = dict2cuda(gt)
            # print('gt[img].shape: ' + str(gt['img'].shape))

            # print('gt.shape: ' + str(gt['img'].shape))

            # gt = gt['img'].view(opt.side_len, opt.side_len, -1)
            gt = gt['img']
            # print('gt.shape: ' + str(gt.shape))

            pred = model(model_input, idx).to(device)
            # print('pred.shape: ' + str(pred.shape))

            loss_frame = get_L1_loss(pred, gt).to(device)
            print('Step ' + str(i + 1) + ' , Frame ' + str(idx + 1) + ': ' + str(loss_frame))

            optim.zero_grad()
            loss_frame.backward()
            optim.step()

            losses_frame.append(loss_frame.item())

            if i % opt.step == 0:
                output_path = os.path.join(opt.output_dir, 'step_%04d/' % (i + 1))
                utils.mkdir(output_path)
                output_path = os.path.join(output_path, 'frame_%03d.jpg' % (idx + 1))
                pred = pred.mul_(255).add_(0.5).clamp_(0, 255).type(torch.uint8).cpu().numpy()
                img = Image.fromarray(pred)
                img.save(output_path)

        loss_step = np.array(losses_frame).mean()

        losses.append(loss_step)

        if i % opt.step == 0:
            print('Iter: ' + str(i) + ' ' + 'Loss: ' + str(loss_step))
            torch.save(model.state_dict(), os.path.join(opt.model_dir, 'model_step_%04d.pth' % (i + 1)))
            np.savetxt(os.path.join(opt.loss_dir, 'train_losses_step_%04d.txt' % (i + 1)), np.array(losses_frame))

        losses_frame.clear()

    np.savetxt(os.path.join(opt.loss_dir, 'train_losses_all_steps.txt'), np.array(losses))


if __name__ == '__main__':
    train()
