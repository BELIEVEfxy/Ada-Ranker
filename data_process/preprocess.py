# -*- coding: utf-8 -*-
# @Time : 2022/05/20
"""
Pre-process data for Ada-Ranker.
This is the code for processing original ML10M dataset for Ada-Ranker.
The code includes:
    (1) processing the original dataset (filtering users whose #interactions is lower than 10 and remapping all ids)
    (2) sampling negative items by our proposed distritbuion-mixer sampling
    (3) transferring DataFrame to pickle files
    (4) pretraining item embeddings by word2vec algorithms.
"""

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
print('PROJECT_ROOT', PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

from data_process.helper import dataloader, datasaver, dataprocess, sample, word2vec
from data_process.helper.CONST_VALUE import *
from tqdm import *
import wget
import zipfile


if __name__ == '__main__':
    filepath = origin_data_path + dataset_name + '.tsv'
    """
    The input data should include 4 fields: 'user_id', 'item_id', 'cate_id', 'timestamp', and each element in `cate_id' is a list containing several categories of target item.

    The data format in `filepath' should be like:
        user_id item_id cate_id timestamp
        1       122     [5, 15] 838985046
        139     122     [5, 15] 974302621
        149     122     [5, 15] 1112342322
        182     122     [5, 15] 943458784
        215     122     [5, 15] 1102493547
        217     122     [5, 15] 844429650
    """

    ## process input dataset
    origin_data = dataloader.load_input_data(filepath, sep=SEP)
    item2cate = dataprocess.merge_category(origin_data)
    processed_data = dataprocess.filter_remap_sort(origin_data, item2cate)

    ## add item sequence for each interaction
    data_with_itemseq = dataprocess.add_item_seq(processed_data)

    ## distritbuion-mixer sampling
    cate2items = dataprocess.get_new_cate2items()
    data_with_negitems = sample.distritbuion_mixer_sampling(data_with_itemseq, cate2items)

    ## save train/valid/test data
    datasaver.save_data(data_with_negitems)

    ## pretrain item embeddings by word2vec
    word2vec.pretrain_word2vec()

    print('\nAll jobs done!')
    
