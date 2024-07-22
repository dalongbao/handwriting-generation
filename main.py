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

dataset = get_dataset()

def show_image(images, label):
    image = images[0]  # Shape: (1, 180, 13)
    
    # Remove the channel dimension
    image = image.squeeze(0)  # Shape: (180, 13)
    
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='binary')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()



for img, label in dataset:
    print(img.shape)
    show_image(img, label)


"""Hyperparameters"""
