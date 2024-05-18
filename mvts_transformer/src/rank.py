import os
import sys
import time
import pickle
import json

# 3rd party packages
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import numpy as np

# Project modules
from .options import Options
from .utils import utils
from .models.ts_transformer import model_factory
from .models.loss import get_loss_module
from .models.ts_transformer import TSTransformerEncoderDistregressor

class Engine():
    def __init__(self, model_path):
        '''main class for QF functions'''
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        print(self.model)

    def load_model(self, model_path):
        '''ONLY loads model of type TSTransformerEncoderDistregressor, and ONLY for inference.'''
        model = TSTransformerEncoderDistregressor(feat_dim=6, max_len=500, d_model=128,
                                                    n_heads=8,
                                                    num_layers=3, dim_feedforward=256,
                                                    num_classes=1,
                                                    dropout=0.1, pos_encoding='learnable',
                                                    activation='gelu',
                                                    norm='BatchNorm', freeze=True)
        for name, param in model.named_parameters():
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = deepcopy(checkpoint['state_dict'])
        model.load_state_dict(state_dict, strict=False)
        model.mode = 'inference'
        return model.to(self.device)

    def forward(self, input, padding_mask):
        assert torch.is_tensor(input) and torch.is_tensor(padding_mask)
        assert input.dim() == 3
        assert input.shape[0] == padding_mask.shape[0] and input.shape[1] == padding_mask.shape[1]
        return self.model(input.to(self.device), padding_mask.to(self.device))

    def rank(self, input: dict, steps: int, **kwargs):
        '''input is a dict of stock data per each ticker to rank {TICKER: {feature1: y1, feature2: y2, ... etc}}
            **kwargs: must include 'steps = int' number of steps, can define other ranking functions via lambdas
            KEY WORDS:
                -int DELTA: specify number of days back to compare to, i.e. DELTA=5 gives (predicted price - price 5 days ago)
                -int MEAN: specify number of days back to get mean of, i.e. MEAN=5 gives mean(last 5 days)
                -int STD: specify number of days back to get standard deviation of, i.e. STD=5 gives std(last 5 days)
                -int AUTOREG: specify number of days *ahead* to predict change in price, i.e. AUTOREG=5 gives (predicted price 5 days from now - todays price)
                -str LABEL: the data label to perform operations on. By default is 'Close'
        By default, returns ranking of predicted mean price for the next day in the form {TICKER: float RANK}, RANK is between 0 and 1.
        '''
        ranked_out = {}
        assert steps > 0 #'''steps' should be defined for ranking beyond 1-step predicted price'''

        means = {}
        for ticker in input.keys():
            data = input[ticker].tail(self.model.max_len)
            obs = torch.Tensor(np.array(data)).view(1,-1,len(input[ticker].columns))
            padding_mask = torch.ones((obs.shape[0], obs.shape[1])).bool()
            mean, std = self.forward(obs, padding_mask)
            means[ticker] = mean
        max_val = max(means.values())
        min_val = min(means.values())
        for ticker in means.keys():
            means[ticker] = (means[ticker] - min_val)/(max_val - min_val)
            means[ticker] = means[ticker].item()
        return means #means ranked between 0-1
