import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import sys
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=128, bias=True)
        self.fc2 = nn.Linear(in_features=128, out_features=64, bias=True)
        self.fc3 = nn.Linear(in_features=64, out_features=32, bias=True)
        self.fc4 = nn.Linear(in_features=32, out_features=1, bias=True)

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leakyrelu(x)

        x = self.fc2(x)
        x = self.leakyrelu(x)

        x = self.fc3(x)
        x = self.leakyrelu(x)

        x = self.fc4(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)

        return x

class Generator(nn.Module):
    def __init__(self, in_features, out_features):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=32, bias=True)
        self.fc2 = nn.Linear(in_features=32, out_features=64, bias=True)
        self.fc3 = nn.Linear(in_features=64, out_features=128, bias=True)
        self.fc4 = nn.Linear(in_features=128, out_features=out_features, bias=True)

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.leakyrelu(x)

        x = self.fc2(x)
        x = self.leakyrelu(x)

        x = self.fc3(x)
        x = self.leakyrelu(x)

        x = self.fc4(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.tanh(x)

        return x

