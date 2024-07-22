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

from data import get_dataset 

device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

train_loader = get_dataset()

"""Hyperparameters"""
