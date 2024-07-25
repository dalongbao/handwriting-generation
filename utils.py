import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import sys
import pickle
import time
import string

import numpy as np
import matplotlib.pyplot as plt

def get_alphas(batch_size, alpha_set):
    alpha_indices = torch.rand([batch_size, 1], dtype=torch.int32)
    lower_alphas = torch.gather(alpha_set, alpha_indices)
    upper_alphas = torch.gather(alpha_set, alpha_indices + 1)
    alphas = torch.rand(lower_alphas.shape) * (upper_alphas - lower_alphas)
    alphas += lower_alphas
    alphas = torch.reshape(alphas, (batch_size, 1, 1))

    return alphas
