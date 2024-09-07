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
    return torch.exp(torch.linspace(torch.log(torch.tensor(min_val)), torch.log(torch.tensor(max_val)), L))

def get_beta_set():
    beta_set = 0.02 + explin(1e-5, 0.4, 60)
    return beta_set

def show(strokes, name='', show_output=True, scale=1):
    positions = torch.cumsum(strokes, axis=0).T[:2]
    prev_ind = 0
    W, H = torch.max(positions, axis=-1)[0] - torch.min(positions, axis=-1)[0]
    W = W.detach().numpy()
    H = H.detach().numpy()
    plt.figure(figsize=(scale * W/H, scale))

    for ind, value in enumerate(strokes[:, 2]):
        if value > 0.5: 
            show_positions= positions.detach().numpy()
            plt.plot(show_positions[0][prev_ind:ind], show_positions[1][prev_ind:ind], color='black')
            prev_ind = ind
        
    plt.axis('off')
    if name: plt.savefig('./' + name + '.png', bbox_inches='tight')
    if show_output:  plt.show()
    else: plt.close()

def pad_stroke_seq(x, maxlength):
    if (x.shape[0] > maxlength) or (torch.max(torch.abs(x)) > 15):
        return None
   
    # Convert to tensor if it's not already
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    
    # Remove single dimensions
    x = torch.squeeze(x)
    
    # Ensure x is 2D
    if x.dim() == 1:
        x = x.unsqueeze(1)
    
    num_features = x.shape[1]
    pad_length = maxlength - x.shape[0]
    
    padding = torch.zeros((pad_length, num_features), dtype=torch.float32)
    
    # If x has 3 columns, set the third column of padding to 1
    if num_features == 3:
        padding[:, 2] = 1
    
    return torch.cat((x, padding), dim=0)

def pad_img(img, width, height):
    if not isinstance (img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.uint8)  # Ensure img is a torch tensor
    
    if len(img.shape) == 2:
        img = img.unsqueeze(-1)
    
    if img.shape[-1] == 1:
        img = img.expand(-1, -1, 3)
    
    pad_len = width - img.shape[1]
    padding = torch.full((height, pad_len, 3), 255, dtype=torch.uint8)
    padded_img = torch.cat((img, padding), dim=1)
    
    return padded_img[:, :, :3]


def standard_diffusion_step(xt, eps, beta, alpha, add_sigma=True):
    x_t_minus1 = (1 / torch.sqrt(1 - beta)) * (xt - (beta * eps / torch.sqrt(1 - alpha)))
    if add_sigma:
        x_t_minus1 += torch.sqrt(beta) * torch.randn_like(xt)
    return x_t_minus1

def new_diffusion_step(xt, eps, beta, alpha, alpha_next):
    x_t_minus1 = (xt - torch.sqrt(1 - alpha) * eps) / torch.sqrt(1 - beta)
    x_t_minus1 += torch.randn_like(xt) * torch.sqrt(torch.tensor(1 - alpha_next, device=xt.device))
    return x_t_minus1

def run_batch_inference(model, beta_set, text, style, tokenizer=None, time_steps=480, diffusion_mode='new', show_every=None, show_samples=True, path=None, device='mps'):
    if isinstance(text, str):
        text = torch.tensor([tokenizer.encode(text)+[1]]).to(device)
    elif isinstance(text, list) and isinstance(text[0], str):
        tmp = []
        for i in text:
            tmp.append(tokenizer.encode(i)+[1])
        text = torch.tensor(tmp).to(device)

    bs = text.shape[0]
    L = 1#len(beta_set)
    alpha_set = torch.cumprod(1 - beta_set, dim=0).to(device)
    x = torch.randn([bs, time_steps, 2]).to(device)
    
    model = model.to(device)
    style = style.to(device)
    
    for i in range(L-1, -1, -1):
        alpha = alpha_set[i].view(1, 1, 1).expand(bs, 1, 1)
        beta = beta_set[i].view(1, 1, 1).expand(bs, 1, 1)
        a_next = alpha_set[i-1] if i > 1 else 1.
        # model outputs are nan, propagates
        model_out, pen_lifts, att = model(x, text, torch.sqrt(alpha), style)
        if diffusion_mode == 'standard':
            x = standard_diffusion_step(x, model_out, beta, alpha, add_sigma=bool(i)) 
        else: 
            x = new_diffusion_step(x, model_out, beta, alpha, a_next)
        
        if show_every is not None:
            if i in show_every:
                plt.imshow(att[0][0].detach().cpu().numpy())
                plt.show()

    x = torch.cat([x, pen_lifts], dim=-1)
    for i in range(bs):
        show(x[i].cpu(), scale=1, show_output=True, name=path)

    return x.detach().cpu().numpy()

def preprocess_data(path, max_text_len, max_seq_len, img_width, img_height):
    with open(path, 'rb') as f:
        ds = pickle.load(f)

    strokes, texts, samples = [], [], []
    for x, text, sample in ds:
        if len(text) < max_text_len:
            padded_x = pad_stroke_seq(x, maxlength=max_seq_len)
            if padded_x is not None:
                zeros_text = np.zeros((max_text_len-len(text), ))
                text = np.concatenate((text, zeros_text))
                h, w  = sample.shape

                if sample.shape[1] < img_width: 
                    sample = pad_img(sample, img_width, img_height)
                    strokes.append(padded_x)
                    texts.append(text)
                    samples.append(sample)
    
    # Convert lists of arrays to single numpy arrays
    strokes = np.array(strokes, dtype=np.float32)
    texts = np.array(texts, dtype=np.int64)
    samples = np.array(samples, dtype=np.float32)
    
    # Convert numpy arrays to PyTorch tensors
    strokes_tensor = torch.from_numpy(strokes)
    texts_tensor = torch.from_numpy(texts)
    samples_tensor = torch.from_numpy(samples)
    
    return strokes_tensor, texts_tensor, samples_tensor

def build_dataset(strokes, texts, samples, style_extractor, batch_size, device):
    # Convert samples to PyTorch tensor and move to the specified device
    samples_tensor = samples.clone().detach()
    # samples_tensor = torch.tensor(samples, dtype=torch.float32, device=device)
    
    # Create a TensorDataset for samples
    samples_dataset = TensorDataset(samples_tensor)
    samples_loader = DataLoader(samples_dataset, batch_size=batch_size, shuffle=False)
    
    style_vectors = []
    
    # Use torch.no_grad() for inference
    with torch.no_grad():
        for s in samples_loader:
            s = s[0]  # Unpack the batch
            style_vec = style_extractor(s)
            style_vectors.append(style_vec.cpu().numpy())
    
    # Concatenate all style vectors
    style_vectors = np.concatenate(style_vectors, axis=0)
    style_vectors = style_vectors.astype('float32')
    
    # Convert all data to PyTorch tensors
    strokes_tensor = strokes.to(dtype=torch.float32, device=device) # torch.tensor(strokes, dtype=torch.float32, device=device)
    texts_tensor = texts.to(dtype=torch.long, device=device)  # torch.tensor(texts, dtype=torch.long, device=device)
    style_vectors_tensor = torch.tensor(style_vectors, device=device)
    
    # Create a TensorDataset
    dataset = TensorDataset(strokes_tensor, texts_tensor, style_vectors_tensor)
    
    # Create a DataLoader with shuffling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False, # set to False on Colab because data wasn't made to Dense Tensor and new dataset didn't work 
        drop_last=True
    )
    
    return dataloader
    
