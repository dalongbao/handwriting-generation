import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import random
import os
import pickle
import argparse
import string
import tiktoken
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# max xml length = 994

class IAMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        strokes, text, image = self.data[idx]
        
        strokes = torch.Tensor(strokes).to(dtype=torch.float32)
        text = torch.Tensor(text).to(dtype=torch.int64)
        image = torch.Tensor(image).permute(2, 0, 1).to(dtype=torch.uint8)  # Change from (H, W, C) to (C, H, W)
        
        return strokes, text, image

def remove_whitespace(img, thresh, remove_middle=False):
    row_mins = np.amin(img, axis=1)
    col_mins = np.amin(img, axis=0)
    
    rows = np.where(row_mins < thresh)
    cols = np.where(col_mins < thresh)

    if remove_middle:
        return img[rows[0]][:, cols[0]]
    else:
        rows, cols = rows[0], cols[0]
        return img[rows[0]:rows[-1], cols[0]:cols[-1]]

def norms(x):
    return np.linalg.norm(x, axis=-1)

def combine_strokes(x, n):
    s, s_neighbors = x[::2, :2], x[1::2, :2]
    if len(x) % 2 != 0:
        s = s[:-1]
    values = norms(s) + norms(s_neighbors) - norms(s + s_neighbors)
    ind = np.argsort(values)[:n]
    x[ind*2] += x[ind*2+1]
    x[ind*2, 2] = np.greater(x[ind*2, 2], 0)
    x = np.delete(x, ind*2+1, axis=0)
    x[:, :2] /= np.std(x[:, :2])
    return x

def parse_page_text(dir_path, id):
    dict = {}
    with open(os.path.join(dir_path, id)) as f:
        has_started = False
        line_num = -1
        for l in f:
            if 'CSR' in l:
                has_started = True
            if has_started:
                if line_num > 0:
                    dict[f"{id[:-4]}-{line_num:02d}"] = l.strip()
                line_num += 1
    logger.info(f"Parsed page text for {id}, found {len(dict)} lines")
    return dict

def create_dict(path):
    dict = {}
    for dir in os.listdir(path):
        if dir != '.DS_Store':
            dirpath = os.path.join(path, dir)
        for subdir in os.listdir(dirpath):
            if subdir != '.DS_Store':
                subdirpath = os.path.join(dirpath, subdir)
            forms = os.listdir(subdirpath)
            for f in forms:
                dict.update(parse_page_text(subdirpath, f))
    logger.info(f"Created dictionary with {len(dict)} entries")
    return dict

def parse_stroke_xml(path): # pad the xmls to 1000 strokes so everything is consistent
    with open(path) as xml:
        xml = xml.readlines()
    strokes = []
    previous = None
    for i, l in enumerate(xml):
        if 'Point' in l:
            x_ind, y_ind, y_end = l.index('x='), l.index('y='), l.index('time=')
            x = int(l[x_ind+3:y_ind-2])
            y = int(l[y_ind+3:y_end-2])
            is_end = 1.0 if '/Stroke' in xml[i+1] else 0.0
            if previous is None:
                previous = [x, -y]
            else:
                strokes.append([x - previous[0], -y - previous[1], is_end])
                previous = [x, -y]
    
    strokes = np.array(strokes)
    strokes[:, 2] = np.roll(strokes[:, 2], 1)
    strokes[:, :2] /= np.std(strokes[:, :2])
    for i in range(3):
        strokes = combine_strokes(strokes, int(len(strokes)*0.2))
    
    logger.info(f"Parsed stroke XML for {path}, found {len(strokes)} strokes")

    return strokes

def read_img(path, height):
    img = Image.open(path)
    img_arr = np.array(img)
    img_arr = remove_whitespace(img_arr, thresh=127)
    h, w = img_arr.shape
    new_w = height * w // h  # Use integer division
    img_resized = Image.fromarray(img_arr).resize((new_w, height), Image.BILINEAR)
    return torch.Tensor(np.array(img_resized).astype('uint8'))

def create_dataset(formlist, strokes_path, images_path, tokenizer, text_dict, height): # max sentence length is 24
    dataset = []
    with open(formlist) as f:
        forms = f.readlines()

    for f in forms:
        path = os.path.join(strokes_path, f[1:4], f[1:8])
        offline_path = os.path.join(images_path, f[1:4], f[1:8])

        samples = [s for s in os.listdir(path) if f[1:-1] in s]
        offline_samples = [s for s in os.listdir(offline_path) if f[1:-1] in s]
        shuffled_offline_samples = offline_samples.copy()
        random.shuffle(shuffled_offline_samples)
        
        for i in range(len(samples)):
            sample_id = samples[i][:-4]
            if sample_id not in text_dict:
                logger.warning(f"Sample {sample_id} not found in text dictionary")
                continue
            
            """data to be added (this is here for debugging)"""
            stroke_vec = torch.Tensor(parse_stroke_xml(os.path.join(path, samples[i])))

            tokenized_string = torch.Tensor(tokenizer.encode(text_dict[sample_id]))

            img_vec = read_img(os.path.join(offline_path, shuffled_offline_samples[i]), height)
            
            dataset.append((
                stroke_vec,
                tokenized_string,
                img_vec
            ))

    logger.info(f"Created dataset with {len(dataset)} samples")
    return dataset

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('-t', '--text_path', default='./data/ascii', help='path to text labels, default ./data/ascii')
    parser.add_argument('-s', '--strokes_path', default='./data/lineStrokes',
                        help='path to stroke xml, default ./data/lineStrokes')
    parser.add_argument('-i', '--images_path', default='./data/lineImages',
                        help='path to line images, default ./data/lineImages')
    parser.add_argument('-H', '--height', type=int, default=96,
                        help='the height of offline images, default 96')

    args = parser.parse_args()
    t_path = args.text_path
    s_path = args.strokes_path
    i_path = args.images_path
    H = args.height

    train_info = './data/trainset.txt'
    val1_info = './data/testset_f.txt'
    val2_info = './data/testset_t.txt'
    test_info = './data/testset_v.txt'

    # Initialize tiktoken tokenizer
    tokenizer = tiktoken.get_encoding("p50k_base")

    labels = create_dict(t_path)
    train_strokes = create_dataset(train_info, s_path, i_path, tokenizer, labels, H)
    val1_strokes = create_dataset(val1_info, s_path, i_path, tokenizer, labels, H)
    val2_strokes = create_dataset(val2_info, s_path, i_path, tokenizer, labels, H)
    test_strokes = create_dataset(test_info, s_path, i_path, tokenizer, labels, H)
    
    train_strokes += val1_strokes
    train_strokes += val2_strokes
    random.shuffle(train_strokes)
    random.shuffle(test_strokes)

    # Create datasets
    train_dataset = IAMDataset(train_strokes)
    test_dataset = IAMDataset(test_strokes)

    # Save the datasets if needed
    with open('./data/train_strokes.p', 'wb') as f:
        pickle.dump(train_strokes, f)
    with open('./data/test_strokes.p', 'wb') as f:
        pickle.dump(test_strokes, f)
    
    logger.info("Datasets created and saved successfully")

if __name__ == '__main__':
    main()
