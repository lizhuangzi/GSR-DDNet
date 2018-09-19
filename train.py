import argparse
import os
from math import log10
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
import skimage.color
import skimage.measure
from Model import SRDDNet
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=120, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
parser.add_argument('--dataset', default='/home/lizhuangzi/Desktop/imageNetData2/ILSVRC2013_DET_test', type=str)
parser.add_argument('--valset', default='./Set5', type=str)

opt = parser.parse_args()

# Setting dataset
train_set = TrainDatasetFromFolder(opt.dataset, crop_size=opt.crop_size, upscale_factor=opt.upscale_factor)

val_set = ValDatasetFromFolder(opt.valset, upscale_factor=opt.upscale_factor)

train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=32, shuffle=True)

val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

# Define Network
netG = SRDDNet()
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

# Define Loss
generator_criterion = torch.nn.MSELoss()
# Cuda
if torch.cuda.is_available():
    netG.cuda()
    generator_criterion.cuda()

# Define optimizer
optimizerG = optim.Adam(netG.parameters(),lr=1e-5)
optimizerG = torch.nn.DataParallel(optimizerG,device_ids=[0,1]).module

scheduler = lr_scheduler.StepLR(optimizerG,step_size=10,gamma=0.8)

for epoch in range(1, opt.num_epochs + 1):

    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'g_loss': 0}
    scheduler.step()

    # Training begin
    netG.train()
    for data, target in train_bar:
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()

        z = Variable(data)
        netG.zero_grad()

        if torch.cuda.is_available():
            z = z.cuda()
        generted_img = netG(z)

        g_loss = generator_criterion(generted_img, real_img)

        g_loss.backward()
        optimizerG.step()

        train_bar.set_description(desc='%f' % (g_loss.data[0]))
        running_results['g_loss']+=g_loss.data[0]