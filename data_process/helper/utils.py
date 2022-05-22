# -*- coding: utf-8 -*-
# @Time   : 2022/05/20
# @Author : Xinyan Fan

import os
import sys
import pandas as pd
import numpy as np
import copy
import random
import time
import argparse
from datetime import datetime
from tqdm import *
import copy
import wget
import zipfile
import requests

def my_parser():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--dataset', '-d', type=str, default='Taobao')
    parser.add_argument('--max_seq_len', '-msl', type=int, default=50)
    parser.add_argument('--neg_items_num', '-nin', type=int, default=19)
    parser.add_argument('--min_user_num', '-mun', type=int, default=10)
    parser.add_argument('--min_item_num', '-min', type=int, default=10)
    parser.add_argument('--target_items_used', '-tiu', type=int, default=10)
    parser.add_argument('--sample_mode', type=str, default='random-2-8')
    parser.add_argument('--drop_duplicates', type=bool, default=True)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--origin_file_path', type=str)
    parser.add_argument('--work_path', type=str)
    parser.add_argument('--size', type=int)
    parser.add_argument('--item_num', type=int)
    parser.add_argument('--sample_size', type=int, default=1000)

    args = parser.parse_args()

    config_dict = {
                    'dataset': args.dataset,\
                    'max_seq_len': args.max_seq_len, \
                    'neg_items_num': args.neg_items_num, \
                    'min_user_num': args.min_user_num, \
                    'min_item_num': args.min_item_num, \
                    'target_items_used': args.target_items_used, \
                    'sample_mode': args.sample_mode, \
                    'drop_duplicates': args.drop_duplicates, \
                    'origin_file_path': args.origin_file_path, \
                    'output_path': args.output_path, \
                    'work_path': args.work_path, \
                    'size': args.size, \
                    'item_num': args.item_num, \
                    'sample_size': args.sample_size
                  }
    return config_dict

def ensure_dir(dir_path):
    r"""Make sure the directory exists, if it does not exist, create it

    Args:
        dir_path (str): directory path

    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# load
def hash(token_list):
    """
    remap ids of token list from 1 to len(set(token_list)).

    Inputs:
    token_list: series
    """
    new_id_list = []
    token_hash = {}
    
    hash_count = 1
    for token in token_list:
        if token not in token_hash.keys():
            token_hash[token]  = hash_count
            hash_count += 1
        new_id_list.append(token_hash[token])
    return np.array(new_id_list), token_hash

def filter(data, column_name, min_num):
    print('data', data)
    column_num = data.groupby([column_name], as_index=False).size().reset_index()
    choosed_items = copy.deepcopy(column_num[(column_num['size'] >= min_num)])
    choosed_items = list(choosed_items[column_name])
    return data[data[column_name].isin(choosed_items)]

def get_item2pop(data):
    # data_groupby = data.groupby(['item_id'], as_index=False).size().reset_index()
    # item2pop = copy.deepcopy(data_groupby[['item_id', 'size']]).sort_values(['size'], ascending=False)
    # length = len(item2pop)
    items = copy.deepcopy(data['item_id']).values
    return list(items)

def sample(target_item, cate_id, item_seq, sample_type, domain2items, item_num, neg_items_num, domain2popitems, pop_item_list):
    """
    should avoid target_item !!!
    """
    pr_1 = int(sample_type.split('-')[1])
    pr_2 = int(sample_type.split('-')[2])
    pr = (pr_2 / (pr_1 + pr_2)) * 100

    seed = random.sample(range(0, 100), 1)[0]
    if 'random' in sample_type:
        if seed < pr:
            domain_data_list = domain2items[cate_id]
        else:
            domain_data_list = range(1, item_num+1)

    elif 'pop' in sample_type:
        if seed < pr:
            domain_data_list = domain2popitems[cate_id]
        else:
            domain_data_list = pop_item_list

    sampled_items = []
    while len(sampled_items) < neg_items_num:
        sampled_item = random.sample(domain_data_list, 1)
        while (sampled_item == target_item) or (sampled_item in item_seq) or (sampled_item in sampled_items):
            sampled_item = random.sample(domain_data_list, 1)
        sampled_items.append(sampled_item[0])
            
    return sampled_items


def transfer(x, dtype, max_seq_len=50):
    if dtype == 'user_id' or dtype == 'item_id' or dtype == 'item_seq_len' or dtype == 'domain_class':
        x = int(x)
    elif dtype == 'item_seq':
        x = x[1:-1].split(', ')
        x = list(map(int, x))
        len_seq = len(x)
        x = [0]*(max_seq_len-len_seq) + x
    elif dtype == 'sampled_items':
        x = x[1:-1].split(', ')
        x = list(map(int, x))
    return x

def str2list(x):
    x = x[1:-1].split(', ')
    new_x = [] if x[0] == '' else [int(i) for i in x] 
    return new_x

def download_with_bar(url, filepath):
    start = time.time()
    response = requests.get(url, stream=True)
    size = 0
    chunk_size = 1024
    content_size = int(response.headers['content-length'])  # the size of the whole file
    try:
        if response.status_code == 200:
            print('Start download, [File size]:{size:.2f} MB'.format(size = content_size / chunk_size /1024))
            with open(filepath, 'wb') as file:
                for data in response.iter_content(chunk_size = chunk_size):
                    file.write(data)
                    size += len(data)
                    print('\r'+'[Download progress]:%s%.2f%%' % ('>'*int(size*50/ content_size), float(size / content_size * 100)), end=' ')
        end = time.time()
        print('Download completed!,times: %.2fs' % (end - start))
    except:
        print('Error!')