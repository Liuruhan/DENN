import argparse
import numpy as np
from random import shuffle

import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset



lstm = nn.LSTM(
                input_size=3,
                hidden_size=64,
                num_layers=1,
                batch_first=True
            )
linear = nn.Sequential(
            nn.Linear(64, 3)
        )
inputs = torch.randn(10, 5, 3)
print('inputs:', inputs.size())
r_out, (h_n, h_c) = lstm(inputs)
print('r_out:',r_out.size())
h = linear(r_out)
print('h', h.size())
