import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

import os
import sys
import pickle
import time
import string

import numpy as np
import matplotlib.pyplot as plt

def get_alphas(batch_size, alpha_set):
    alpha_indices = torch.randint(0, len(alpha_set) - 1, (batch_size, 1))
    lower_alphas = alpha_set[alpha_indices]
    upper_alphas = alpha_set[alpha_indices + 1]
    alphas = torch.rand(lower_alphas.shape) * (upper_alphas - lower_alphas)
    alphas += lower_alphas
    alphas = alphas.reshape(batch_size, 1)
    alphas = alphas.repeat(1, 32)

    return alphas

def explin(min_val, max_val, L):
    return torch.exp(torch.linspace(torch.log(torch.tensor(min_val)), 
                                    torch.log(torch.tensor(max_val)), 
                                    L))

def get_beta_set():
    beta_set = 0.02 + explin(1e-5, 0.4, 60)
    return beta_set

def preprocess_data(path, max_text_len, max_seq_len, img_width, img_height):
    with open(path, 'rb') as f:
        ds = pickle.load(f)
        
    strokes, texts, samples = [], [], []
    for x, text, sample in ds:
        if len(text) < max_text_len:
            x = pad_stroke_seq(x, maxlength=max_seq_len)
            zeros_text = np.zeros((max_text_len-len(text), ))
            text = np.concatenate((text, zeros_text))
            h, w, _ = sample.shape

            if x is not None and sample.shape[1] < img_width: 
                sample = pad_img(sample, img_width, img_height)
                strokes.append(x)
                texts.append(text)
                samples.append(sample)
    texts = np.array(texts).astype('int32')
    samples = np.array(samples)
    return strokes, texts, samples

class HandwritingDataset(Dataset):
    def __init__(self, strokes, texts, samples, style_extractor):
        self.strokes = [torch.tensor(stroke, dtype=torch.float32) for stroke in strokes]
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.samples = torch.tensor(samples, dtype=torch.float32).permute(0, 3, 1, 2)  # Change to NCHW format
        
        # Extract style vectors
        self.style_vectors = []
        with torch.no_grad():
            for sample in self.samples.split(32):  # Process in batches of 32
                style_vec = style_extractor(sample)
                self.style_vectors.append(style_vec)
        self.style_vectors = torch.cat(self.style_vectors, dim=0)

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        return self.strokes[idx], self.texts[idx], self.style_vectors[idx]

def create_dataset(strokes, texts, samples, style_extractor, batch_size, buffer_size):
    dataset = HandwritingDataset(strokes, texts, samples, style_extractor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return dataloader
