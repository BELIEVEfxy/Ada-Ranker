# -*- coding: utf-8 -*-
# @Time   : 2022/05/20
# @Author : Xinyan Fan

import os
import sys
import pandas as pd
import numpy as np
import copy
from datetime import datetime
from tqdm import *
import pickle as pkl
from data_process.helper import utils, datasaver
from helper.CONST_VALUE import *

def load_pkl_obj(filename):
    print('loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    with open(filename, "rb") as f: 
        obj = pkl.load(f) 
        # p = pkl.Unpickler(f)
        # obj = p.load()
    print('finish loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    return obj

def load_dict(infile, sep = '\t', is_reverse=False):
    print('loading {0}...'.format(infile))
    d = {}
    with open(infile, 'r') as rd:
        while True:
            line = rd.readline()
            if not line:
                break
            words = line[:-1].split(sep)
            if is_reverse:
                d[int(words[1])] = int(words[0])
            else:
                d[int(words[0])] = int(words[1])
    return d

def get_hash_value(key, node_hash, hash_count):
    if key not in node_hash:
        node_hash[key] = hash_count
        hash_count += 1 
    return node_hash[key], hash_count


def load_item_profile(filepath):
    def map_cate(x, cate2idx):
        cates = []
        x = x.split('|')
        for i in x:
            cates.append(cate2idx[i])
        return cates

    item_profile = pd.read_csv(filepath, sep='::', names=['item_id', 'movie_title', 'genre'], header=None)
    # print('origin item_profile', item_profile)
    item_profile = item_profile[item_profile['genre'] != '(no genres listed)']

    all_items = item_profile['item_id'].values
    all_cates = item_profile['genre'].values
    cates_token = []
    for cates_list in all_cates:
        cates = cates_list.split('|')
        for i in cates:
            cates_token.append(i) 
    cates_token = list(set(cates_token))
    cates_token.sort()
    catetoken2idx = {x[1]: x[0]+1 for x in enumerate(cates_token)}

    item_profile['genre'] = item_profile['genre'].apply(lambda x: map_cate(x, catetoken2idx))
    item2cate = item_profile[['item_id', 'genre']]
    item2cate.columns = ['item_id', 'cate_id']
    item2cate.to_csv(origin_data_path+'item2cate.tsv', sep='\t', index=False)
    datasaver.save_dict(catetoken2idx, origin_data_path+'genre2idx.tsv')

    return item2cate

def load_inter_file(item2cate, filepath):
    inter_data = pd.read_csv(filepath, sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'], header=None)
    # print('origin inter_data\n', inter_data)
    inter_data = pd.merge(inter_data, item2cate, how='inner', on=['item_id'])
    inter_data = inter_data[['user_id', 'item_id', 'cate_id', 'timestamp']]

    return inter_data