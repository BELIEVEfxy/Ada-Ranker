# -*- coding: utf-8 -*-
# @Time : 2022/05/20
"""
This is the code for processing original ML10M dataset for Ada-Ranker.
The code includes:
    (1) downloading the original ML10M dataset
    (2) transferring the original ML10M dataset to the input format of out pipeline, be like:
        user_id item_id cate_id timestamp
        1       122     [5, 15] 838985046
        139     122     [5, 15] 974302621
        149     122     [5, 15] 1112342322
        182     122     [5, 15] 943458784
        215     122     [5, 15] 1102493547
        217     122     [5, 15] 844429650
"""

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
print('PROJECT_ROOT', PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from data_process.helper import datasaver, utils, dataprocess, sample, word2vec
from data_process.helper.CONST_VALUE import *
from tqdm import *
import wget
import zipfile


if __name__ == '__main__':
    ## download dataset:
        # download dataset from https://grouplens.org/datasets/movielens/10m/
        # You can also download ml-10m.zip from the browser and put it into `origin_data_path'
    filepath = origin_data_path + wget.filename_from_url(url)

    if not os.path.exists(filepath):
        print('\nThe original dataset does not exist. Downloading it from ', url)
        wget.download(url, out=filepath)

    if not os.path.exists(origin_data_path+'/ml-10M100K/'):
        with zipfile.ZipFile(filepath, mode="a") as f:
            print('\n>> Extracting dataset...')
            f.extractall(origin_data_path)
        f.close()

    ## process ML10M dataset
    dataprocess.process_origin_data()

    print('\nFinish preparing ML10M dataset!')
    
