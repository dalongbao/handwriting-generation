import torch
import os
import tiktoken
import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils
import model as miku
import preprocessing

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
    beta_set = utils.get_beta_set()
    alpha_set = torch.cumprod(1-beta_set, dim=0)

    C1 = args.channels
    C2 = C1 * 3//2
    C3 = C1 * 2
    style_extractor = miku.StyleExtractor()
    model = miku.DiffusionWriter(num_layers=args.num_attlayers, c1=C1, c2=C2, c3=C3)
    
    # Load the trained model weights
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    writer_img = preprocessing.read_img(sourcename, 96).unsqueeze(0)
    print(writer_img.shape)
    style_vector = style_extractor(writer_img)
    utils.run_batch_inference(model, beta_set, args.textstring, style_vector, 
                                tokenizer=tokenizer, time_steps=timesteps, diffusion_mode=args.diffmode, 
                                show_samples=args.show, path=args.name)

if __name__ == '__main__':
    main()


