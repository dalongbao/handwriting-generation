import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms

import os
import sys
import pickle
import time
import string
import random

import numpy as np
import matplotlib.pyplot as plt

from preprocessing import read_img, parse_stroke_xml

def get_alphas(batch_size, alpha_set):
    alpha_indices = torch.randint(0, len(alpha_set) - 1, (batch_size, 1))
    lower_alphas = alpha_set[alpha_indices]
    upper_alphas = alpha_set[alpha_indices + 1]
    alphas = torch.rand(lower_alphas.shape) * (upper_alphas - lower_alphas)
    alphas += lower_alphas
    alphas = alphas.reshape(batch_size, 1)
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

def preprocess_data(path, max_text_len, max_seq_len, img_width, img_height):
    with open(path, 'rb') as f:
        ds = pickle.load(f)

    strokes, text, samples = [], [], []
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

def create_dataset(train_dataset, style_extractor, batch_size, device):
    # Create a DataLoader for the train_dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    style_vectors = []
    
    # Use torch.no_grad() for inference
    with torch.no_grad():
        for batch in train_loader:
            strokes, texts, samples = batch
            samples = samples.to(device)
            style_vec = style_extractor(samples)
            style_vectors.append(style_vec.cpu().numpy())
    
    # Concatenate all style vectors
    style_vectors = np.concatenate(style_vectors, axis=0)
    style_vectors = style_vectors.astype('float32')
    
    # Convert style vectors to PyTorch tensor
    style_vectors_tensor = torch.tensor(style_vectors, device=device)
    
    # Create a new TensorDataset including the style vectors
    new_dataset = TensorDataset(
        train_dataset.tensors[0],  # strokes
        train_dataset.tensors[1],  # texts
        train_dataset.tensors[2],  # samples
        style_vectors_tensor
    )
    
    # Create a DataLoader with shuffling
    dataloader = DataLoader(
        new_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffling is enabled
        num_workers=0,  # No parallel workers
        pin_memory=True,  # Helps speed up data transfer to GPU
        drop_last=True
    )
    
    return dataloader

