import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def upsampling_block(in_channels, out_channels):
    block = nn.Sequential(
            nn.Upsample(scale_factor=3, mode='bilinear'),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(inplace = True, negative_slope=0.2))
    return block

class NNDC(nn.Module):
    def __init__(self):
         super(NNDC, self).__init__()
         self.b1 = nn.Sequential(nn.Linear(8,27),
                   nn.LeakyReLU(inplace = True, negative_slope=0.2),
                   nn.Linear(27, 876),
                   nn.LeakyReLU(inplace = True, negative_slope=0.2)
                   )
         self.b2 = upsampling_block(in_channels = 1, out_channels = 8)
         self.b5 = nn.Sequential(nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=0),
                   nn.Sigmoid(),
                   )
         #self.droupout = nn.Dropout(0.1)

    def forward(self, x):
        x1 = self.b1(x)
        x1 = x1.view((1,1,12,73))
        x2 = self.b2(x1)
        y = self.b5(x2)
        return torch.mul(y[:,:,:,:-1],-8)

