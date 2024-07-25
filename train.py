import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import sys
import pickle
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import model as miku # the naming scheme clashes with the torch naming scheme
import utils

def train_step(x, pen_lifts, text, style_vectors, glob_args):
    model, alpha_set, bce, train_loss, optimizer = glob_args
    device = next(model.parameters()).device
    x = x.to(device)
    pen_lifts = pen_lifts.to(device)
    text.to_device()
    style_vectors = style_vectors.to(device)

    alphas = utils.get_alphas(len(x), alpha_set)  
    eps = torch.randn_like(x)
    x_perturbed = torch.sqrt(alphas) * x + torch.sqrt(1 - alphas) * eps

    model.train() # set model to training mode
    optimizer.zero_grad() # zero parameter grads

    score, pl_pred, att = model(x_perturbed, text, torch.sqrt(alphas), style_vectors) # forward

    loss = miku.loss_fn(eps, score, pen_lifts, pl_pred, alphas, bce)

    loss.backward()
    optimizer.step()

    train_loss(loss.item())
    return score, att

def train(dataset, iterations, optimizer, alpha_set, print_every=1000, save_every=10000):
    s = time.time() # maybe use perf counter?
    bce = nn.BCELoss(reduction='none')
    train_loss = miku.AverageMeter()
    """

    how is the dataset loaded? what was the original dataset? help

    """
    dataloader = Dataloader(dataset, batch_size=32, shuffle=True)
        
    for count, (strokes, text, style_vectors) in enumerate(dataset.repeat(5000)):
        strokes, pen_lifts = strokes[:, :, :2], strokes[:, :, 2:]
        glob_args = model, alpha_set, bce, train_loss, optimzer
        model_out, att = train_step(strokes, pen_lifts, text, style_vectors, glob_args)

