import mlx
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim

import os
import sys
import pickle
import time

import numpy as np

class TwoLayerLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_dims):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.lstm1 = nn.LSTM(embedding_dims, hidden_dims)
        self.lstm2 = nn.LSTM(hidden_dims, hidden_dims)
        self.fc = nn.Linear(hidden_dims, output_dims)

    def __call__(self, x, initial_state=None):
        embedded = self.embedding(x)
        x, (h_n1, c_n1) = self.lstm1(embedded, initial_state)
        x, (h_n2, c_n2) = self.lstm2(x, (h_n1, c_n1))
        x = self.fc(x)
        return x, (h_n2, c_n2)


