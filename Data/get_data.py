# @Time   : 2021/8/25
import os
import logging
from logging import getLogger
import time
from datetime import datetime 
import argparse
import numpy as np
import pandas as pd
import pickle as pkl

import torch
from torch.utils.data import DataLoader, Dataset

def load_pkl_obj(filename):
    print('loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    with open(filename, "rb") as f: 
        obj = pkl.load(f) 
        # p = pkl.Unpickler(f)
        # obj = p.load()
    print('finish loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    return obj

class MyDataset(Dataset):
    def __init__(self, config, path, dataset_type="train"):
        self.config = config
        self.dataset_type = dataset_type

        self.path = path + self.dataset_type + '.pkl'
        self.dataset = self.load_data()

    def __getitem__(self, index):
        user_id = self.dataset.loc[index, 'user_id']
        item_id = self.dataset.loc[index, 'item_id']
        item_seq = self.dataset.loc[index, 'item_seq']
        neg_items = self.dataset.loc[index, 'neg_items']
        item_seq_len = self.dataset.loc[index, 'item_seq_len']

        if self.config['model'] == 'SRGNN':
            uni_item_seq = self.dataset.loc[index, 'uni_item_seq']
            alias_item_seq = self.dataset.loc[index, 'alias_item_seq']
            A = self._get_slice(item_seq, uni_item_seq, alias_item_seq)
            return user_id, item_id, item_seq, neg_items, item_seq_len, uni_item_seq, alias_item_seq, A

        return user_id, item_id, item_seq, neg_items, item_seq_len

    def __len__(self):
        return len(self.dataset)

    def __padding__(self, x):
        len_seq = len(x)
        if len_seq <= self.config['MAX_SEQ_LEN']:
            x = [0]*(self.config['MAX_SEQ_LEN']-len_seq) + x
        else:
            x = x[len_seq-self.config['MAX_SEQ_LEN']:]
        return x
    
    def load_data(self):
        data = load_pkl_obj(self.path)
        data = data.reset_index(drop=True)
        
        data['item_seq'] = data['item_seq'].apply(lambda x: torch.tensor(self.__padding__(x)))
        data['neg_items'] = data['neg_items'].apply(lambda x: torch.tensor(x))

        if self.config['model'] == 'SRGNN':
            data['uni_item_seq'] = data['item_seq'].apply(lambda x: torch.tensor(np.unique(x).tolist()+(self.config['MAX_SEQ_LEN']-len(np.unique(x)))*[0]))
            data['alias_item_seq'] = data['item_seq'].apply(lambda x: torch.tensor(self._alias_inputs(x)))

        return data

    def _alias_inputs(self, x):
        x = x.numpy()
        node = np.unique(x)
        tmp_alias = [np.where(node == i)[0][0] for i in x]
        return tmp_alias

    def _get_slice(self, item_seq, uni_item_seq, alias_item_seq):
        alias_item_seq = alias_item_seq.numpy()
        u_A = np.zeros((self.config['MAX_SEQ_LEN'], self.config['MAX_SEQ_LEN']))

        for _, (u, v) in enumerate(zip(alias_item_seq[:-1], alias_item_seq[1:])):
            if u == 0 or v == 0:
                continue
            u_A[u][v] = 1

        u_sum_in = np.sum(u_A, 0)
        u_sum_in[np.where(u_sum_in == 0)] = 1
        u_A_in = np.divide(u_A, u_sum_in)
        u_sum_out = np.sum(u_A, 1)
        u_sum_out[np.where(u_sum_out == 0)] = 1
        u_A_out = np.divide(u_A.transpose(), u_sum_out)
        u_A = np.concatenate([u_A_in, u_A_out]).transpose()

        A = torch.FloatTensor(u_A)
        
        return A