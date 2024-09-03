import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import os
import sys
import pickle
import time

import numpy as np
from PIL import Image

class CombinedDataset(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transform
        self.data = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    filename, label = parts
                    self.data.append((filename, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, label = self.data[idx]
        img_path = os.path.join(self.img_path, filename)
        image = Image.open(img_path).convert('1')

        if self.transform:
            image = self.transform(image)

        image = image.view(-1) 
        image = 1 - image
        return image, label

def get_dataset():
    IMG_PATH = os.path.join(os.getcwd(), 'data/data_labelled/IAM/image')
    LABEL_PATH = os.path.join(os.getcwd(), 'data/data_labelled/IAM/gt_test.txt')

    transform = transforms.Compose([
        transforms.Resize((30, 400)),  # Resize images to a fixed size
        transforms.ToTensor(),
    ])

    dataset = CombinedDataset(
        img_path=IMG_PATH,
        label_path=LABEL_PATH,
        transform=transform
    )

    return DataLoader(
            dataset=dataset,
            batch_size=32,
            num_workers=0,
            shuffle=True
    )
