import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.optim.lr_scheduler import _LRScheduler
import os
import sys
import pickle
import time
import math

import numpy as np
import matplotlib.pyplot as plt

def create_padding_mask(seq, repeats=1):
    mask = (seq != 0).float().unsqueeze(1).unsqueeze(2)
    # # Create mask where 0 elements are 1 and others are 0
    # seq = (seq == 0).float()
    # 
    # # Repeat the mask 'repeats' times along the last dimension
    # seq = seq.repeat_interleave(repeats, dim=-1)
    # 
    # # Add two new dimensions at positions 1 and 2
    # mask = seq.unsqueeze(1).unsqueeze(2)
    
    return mask

def get_same_padding(kernel_size):
    if isinstance(kernel_size, int):
        return (kernel_size - 1) // 2
    else:
        return [(k - 1) // 2 for k in kernel_size]

def reshape_up(x, factor=2):
    return x.reshape(x.shape[0], x.shape[1]*factor, x.shape[2]//factor)

def loss_fn(eps, score_pred, pl, pl_pred, abar, bce):
    """
    eps: epsilon
    score_pred: score prediction
    pl: pen lift (list of coordinates)
    pl_pred: pen lift prediction
    abar: weighting factor for pen lift loss
    bce: boolean to decide to use binary cross-entropy
    """
    score_loss = torch.mean(torch.sum(torch.square(eps - score_pred), dim=-1)) # ()
    # abar is currently (32, 1, 1)
    bce_res = bce(pl_pred, pl) # (32, 1000, 1)
    abs = abar.squeeze(-1) # (32, 1)
    bce_res = bce_res.squeeze(-1) # (32, 1000)
    abs = abs.view(-1, 1) # (32, 1)
    pl_loss = torch.mean(bce_res * abs)
    return score_loss + pl_loss

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class InvSqrtScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self.last_epoch)
        arg1 = torch.tensor(step ** -0.5, dtype=torch.float32)  # Convert arg1 to a tensor
        arg2 = step * (self.warmup_steps ** -1.5)
        arg2 = torch.tensor(arg2, dtype=torch.float32)  # Convert arg2 to a tensor
        lr_factor = 1 / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        return [base_lr * lr_factor * torch.min(arg1, arg2) for base_lr in self.base_lrs]

def get_angles(pos, i, d_model, pos_factor=1):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates / pos_factor

def positional_encoding(max_position, d_model, pos_factor=1):
    angle_rads = get_angles(np.arange(max_position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model,
                            pos_factor=pos_factor)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.FloatTensor(pos_encoding)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, pos_factor=1):
        super().__init__()
        self.pos_encoding = positional_encoding(max_len, d_model, pos_factor)
    
    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1), :].to(x.device)

class MLP(nn.Module):
    def __init__(self, input_dims: int, hidden_dims:int = 768, output_dims: int = None, act_before: bool =True):
        super(MLP, self).__init__()
        self.layers = []
        if act_before:
            self.layers.append(nn.SiLU())

        self.output_dims = output_dims if output_dims else input_dims

        self.layers.extend([
            nn.Linear(input_dims, hidden_dims), 
            nn.SiLU(), 
            nn.Linear(hidden_dims, self.output_dims)
            ])

        self.ff = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)

class AffineTransformLayer(nn.Module):
    """
    Used for conditional normalization
    """
    def __init__(self, num_features):
        super().__init__()
        self.gamma_emb = nn.Linear(1, num_features)
        self.beta_emb = nn.Linear(1, num_features)

        nn.init.ones_(self.gamma_emb.weight)
        nn.init.zeros_(self.gamma_emb.bias)

    def forward(self, x, sigma):
        sigma = sigma.view(sigma.size(0), -1)

        gammas = self.gamma_emb(sigma).view(x.size(0), 1, -1)
        betas = self.beta_emb(sigma).view(x.size(0), 1, -1)

        return x * gammas + betas 

class ConvSubLayer(nn.Module):
    def __init__(self, filters, in_channels=None, dils=[1,1], drop_rate=0.0): # activation SiLU
        super().__init__()

        self.in_channels = in_channels if in_channels else filters

        self.silu = nn.SiLU()
        self.affine1 = AffineTransformLayer(filters // 2)
        self.affine2 = AffineTransformLayer(filters)
        self.affine3 = AffineTransformLayer(filters)
        
        self.conv_skip = nn.Conv1d(self.in_channels, filters, 3, padding=1)
        self.conv1 = nn.Conv1d(self.in_channels, filters // 2, 3, dilation=dils[0], padding='same')
        self.conv2 = nn.Conv1d(filters // 2, filters, 3, dilation=dils[0], padding='same')

        self.fc = nn.Linear(filters, filters)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, alpha):
        x = x.transpose(1, 2)
        x_skip = self.conv_skip(x)

        x = self.conv1(self.silu(x))
        x = self.dropout(self.affine1(x.transpose(1, 2), alpha)).transpose(1, 2)

        x = self.conv2(self.silu(x))
        x = self.dropout(self.affine2(x.transpose(1, 2), alpha)).transpose(1, 2)

        x = self.fc(x.transpose(1, 2))
        x = self.dropout(self.affine3(x, alpha)).transpose(1, 2)
        x += x_skip

        return x

class StyleExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).to(self.device)
        self.features = nn.Sequential(*list(self.mobilenet.features)).to(self.device)

        self.local_pool = nn.AvgPool2d((3, 3))
        self.freeze_all_layers()

    def freeze_all_layers(self):
        for param in self.mobilenet.parameters():
            param.requires_grad = False

    def forward(self, im, im2=None, get_similarity=False, training=False):
        x = im.float() / 127.5 - 1
        x = x.to(self.device) # (batch, heights, width)
        x = x.repeat(1, 3, 1, 1) # shape is (batch, channels, height, width)
        x = x.permute(0, 3, 1, 2) # shape is (batch, channels, height, width) (necessary for pytorch mobilenet dims btw)
        x = self.features(x) # mobilenet features output is (batch_size, 1280, height/32, width/32) where 1280 is output channels
        x = self.local_pool(x)

        # reshaping to fit the reshaping dims - now it's (batch_size, flattened sequence, channels)
        batch, channels, h, w = x.shape
        x = x.reshape(batch, channels, h*w).transpose(1, 2)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, text_channels: int = 384, drop_rate: float = 0.1, pos_factor: float = 1):
        super().__init__()
        self.text_pe = PositionalEncoding(d_model=d_model, max_len=2000)
        self.stroke_pe = PositionalEncoding(d_model=d_model, max_len=2000)
        self.dropout = nn.Dropout(drop_rate)
        self.layernorm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6, elementwise_affine=False)
        self.text_fc = nn.Linear(text_channels, d_model)

        self.mha1 = nn.MultiheadAttention(d_model, num_heads)
        self.mha2 = nn.MultiheadAttention(d_model, num_heads)
        self.ff = MLP(d_model, d_model*2)
        self.affine0 = AffineTransformLayer(d_model)
        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)

        self.silu = nn.SiLU()

    def forward(self, x, text, sigma, text_mask):
        text = self.text_fc(self.silu(text))
        text = self.affine0(self.layernorm(text), sigma)
        text_pe = text + self.text_pe(text) # self.text_pe[:, :text.shape[1]]  # Use square brackets instead of parentheses

        x = x.transpose(1, 2)
        x_pe = x + self.stroke_pe(x)

        text_mask = ~text_mask.squeeze(1).squeeze(1).bool() # shape (32, 50)
        x_pe = x_pe.transpose(0, 1)  # Shape: [500, 32, 192]
        text_pe = text_pe.transpose(0, 1)  # Shape: [50, 32, 192]
        text = text.transpose(0, 1)  # Shape: [50, 32, 192]

        x2, att = self.mha1(x_pe, text_pe, text, text_mask) # the issue is here? does later MHAs have the same issue?
        x2 = x2.transpose(0, 1)
        x2 = self.layernorm(self.dropout(x2))
        x2 = self.affine1(x2, sigma) + x

        x2_pe = x2 + self.stroke_pe(x2) # update this here too
        x3, _ = self.mha2(x2_pe, x2_pe, x2)
        x3 = self.layernorm(x2 + self.dropout(x3))
        x3 = self.affine2(x3, sigma)

        x4 = self.ff(x3)
        x4 = self.dropout(x4) + x3
        out = self.affine3(self.layernorm(x4), sigma)
        return out, att

class Text_Style_Encoder(nn.Module):
    def __init__(self, d_model: int, input_dims: int = 512):
        super().__init__()
        self.emb = nn.Embedding(100277, d_model)
        self.text_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=get_same_padding(3))
        self.style_mlp = MLP(256, input_dims, d_model) 
        self.mha = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.layernorm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6, elementwise_affine=False)
        self.dropout = nn.Dropout(p=0.3)

        self.affine1 = AffineTransformLayer(d_model)
        self.affine2 = AffineTransformLayer(d_model)
        self.affine3 = AffineTransformLayer(d_model)
        self.affine4 = AffineTransformLayer(d_model)
        self.text_mlp = MLP(d_model, d_model*2)

    def forward(self, text, style, sigma):
        style = self.dropout(style)
        style = reshape_up(style, 5) # style shape is now (batch_size, flattened seq of h x w, 1280)
        style = self.affine1(self.layernorm(self.style_mlp(style)), sigma)
        text = self.emb(text)
        text = self.affine2(self.layernorm(text), sigma)
        
        mha_out, _ = self.mha(text, style, style)
        text = self.affine3(self.layernorm(text + mha_out), sigma)
        text_out = self.affine4(self.layernorm(self.text_mlp(text)), sigma)
        return text_out

class DiffusionWriter(nn.Module):
    def __init__(self, num_layers: int = 4, c1: int = 128, c2: int = 192, c3: int = 256, drop_rate: float = 0.1, num_heads:int = 8):
        super(DiffusionWriter, self).__init__()
        self.input_fc = nn.Linear(2, c1)
        self.sigma_mlp = MLP(1, 2048) # MLP(c1 // 4, 2048)
        self.enc1 = ConvSubLayer(c1, c1, [1, 2])
        self.enc2 = ConvSubLayer(c2, c1, [1, 2])
        self.enc3 = DecoderLayer(c2, 3, 384, drop_rate, pos_factor=4) # 384 for text_channels
        self.enc4 = ConvSubLayer(c3, c2, [1, 2]) # 500 channel size is slapped on to make it work i'm bastardizing the model
        self.enc5 = DecoderLayer(c3, 4, 384, drop_rate, pos_factor=2)

        self.pool = nn.AvgPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)

        self.skip_conv1 = nn.Conv1d(128, c2, kernel_size=3, padding=get_same_padding(3))
        self.skip_conv2 = nn.Conv1d(192, c3, kernel_size=3, padding=get_same_padding(3))
        self.skip_conv3 = nn.Conv1d(256, c2*2, kernel_size=3, padding=get_same_padding(3))

        self.text_style_encoder = Text_Style_Encoder(c2*2, c2*4) 
        self.att_fc = nn.Linear(c3, c2*2) # stupid tensorflow taking stupid variable input dims 
        self.att_layers = [DecoderLayer(c2*2, 6, c2*2, drop_rate) for _ in range(num_layers)]

        self.dec3 = ConvSubLayer(c3, c2*2, [1,2])
        self.dec2 = ConvSubLayer(c2, c3, [1,1])
        self.dec1 = ConvSubLayer(c1, c2, [1,1])

        self.output_fc = nn.Linear(128, 2)
        self.pen_lifts_fc = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, strokes, text, sigma, style_vector):
        # sigma is 32, 1, 1
        sigma = self.sigma_mlp(sigma)
        text_mask = create_padding_mask(text)
        text = self.text_style_encoder(text, style_vector, sigma)

        x = self.input_fc(strokes)
        h1 = self.enc1(x, sigma) # (32, 128, 1000)
        h2 = self.pool(h1) # (32, 128, 500)

        h2 = self.enc2(h2.transpose(1, 2), sigma) # (32, 192, 500)
        h2, _ = self.enc3(h2, text, sigma, text_mask) # (32, 500, 192)
        h3 = self.pool(h2.transpose(1, 2)) # (32, 192, 250)

        h3 = self.enc4(h3.transpose(1, 2), sigma) # (32, 256, 250)
        h3, _ = self.enc5(h3, text, sigma, text_mask) # (32, 250, 256)
        x = self.pool(h3.transpose(1, 2)) # (32, 128, 250)

        x = self.att_fc(x.transpose(1, 2)).transpose(1, 2) # (32, 384, 125)

        for att_layer in self.att_layers:
            x, att = att_layer(x, text, sigma, text_mask)
            x = x.transpose(1, 2)
        # (32, 384, 125)
        
        # remember torch conv is BCL (batch, channels, length) and tf is (batch, length, channels)

        x = self.upsample(x) + self.skip_conv3(h3.transpose(1, 2)) # (32, 384, 250) + (32, 384, 250) -> (32, 384, 250)
        x = self.dec3(x.transpose(1, 2), sigma) # (32, 256, 500)

        x = self.upsample(x) + self.skip_conv2(h2.transpose(1, 2)) # (32, 256, 500) + (32, 256, 500) -> (32, 256, 500)
        x = self.dec2(x.transpose(1, 2), sigma) # (32, 192, 500)

        x = self.upsample(x) + self.skip_conv1(h1) # (32, 192, 1000) + (32, 192, 1000) -> (32, 192, 1000)
        x = self.dec1(x.transpose(1, 2), sigma) # (32, 128, 1000)

        output = self.output_fc(x.transpose(1, 2)) # (32, 1000, 2)
        pl = self.pen_lifts_fc(x.transpose(1, 2)) # (32, 1000, 1)
        return output, pl, att
