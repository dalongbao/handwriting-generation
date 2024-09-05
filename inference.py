import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torch.optim.lr_scheduler import _LRScheduler

import os
import tiktoken
import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils
import model as miku
import preprocessing

# slightly different from the training styleextractor because of how the input images are formatted
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
        # x = x.permute(0, 3, 1, 2) # shape is (batch, channels, height, width) (necessary for pytorch mobilenet dims btw)
        x = self.features(x) # mobilenet features output is (batch_size, 1280, height/32, width/32) where 1280 is output channels x = self.local_pool(x)

        # reshaping to fit the reshaping dims - now it's (batch_size, flattened sequence, channels)
        batch, channels, h, w = x.shape
        x = x.reshape(batch, channels, h*w).transpose(1, 2)
        return x

def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--textstring', help='the text you want to generate', default='Generating text', type=str)  
    parser.add_argument('--writersource', help="path of the image of the desired writer, (e.g. './assets/image.png'   \
                                                will use random from ./assets if unspecified", default=None)
    parser.add_argument('--name', help="path for generated image (e.g. './assets/sample.png'), \
                                             will not be saved if unspecified", default=None)
    parser.add_argument('--diffmode', help="what kind of y_t-1 prediction to use, use 'standard' for  \
                                            Eq 9 in paper, will default to prediction in Eq 12", default='new', type=str)
    parser.add_argument('--show', help="whether to show the sample (popup from matplotlib)", default=False, type=bool)
    parser.add_argument('--weights', help='the path of the loaded weights', default='./weights/model_weights.h5', type=str)
    parser.add_argument('--seqlen', help='number of timesteps in generated sequence, default 16 * length of text', default=None, type=int)
    parser.add_argument('--num_attlayers', help='number of attentional layers at lowest resolution, \
                                                 only change this if loaded model was trained with that hyperparameter', default=2, type=int)
    parser.add_argument('--channels', help='number of channels at lowest resolution, only change \
                                                 this if loaded model was trained with that hyperparameter', default=128, type=int)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    timesteps = len(args.textstring) * 16 if args.seqlen is None else args.seqlen
    timesteps = timesteps - (timesteps%8) + 8 
    #must be divisible by 8 due to downsampling layers

    if args.writersource is None:
        assetdir = os.listdir('./assets')
        sourcename = './assets/' + assetdir[np.random.randint(0, len(assetdir))]
    else: 
        sourcename = args.writersource

    L = 60
    tokenizer = tiktoken.get_encoding('cl100k_base') # using tiktoken instead of their default tokenizer
    beta_set = utils.get_beta_set().to(device)
    alpha_set = torch.cumprod(1-beta_set, dim=0).to(device)

    C1 = args.channels
    C2 = C1 * 3//2
    C3 = C1 * 2
    style_extractor = StyleExtractor()
    model = miku.DiffusionWriter(num_layers=args.num_attlayers, c1=C1, c2=C2, c3=C3).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    writer_img = preprocessing.read_img(sourcename, 96).unsqueeze(0).to(device) # AFAIK read_img is right, but is the unsqueeze supposed to replciate the batch dimension?
    print(writer_img.shape)
    style_vector = style_extractor(writer_img)
    utils.run_batch_inference(model, beta_set, args.textstring, style_vector, 
                                tokenizer=tokenizer, time_steps=timesteps, diffusion_mode=args.diffmode, 
                                show_samples=args.show, path=args.name, device=device)

if __name__ == '__main__':
    main()


