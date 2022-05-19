# -*- coding: utf-8 -*-
# @Time : 2022/05/20
"""
Pre-process data for Ada-Ranker.
This is the code for processing original ML10M dataset for Ada-Ranker.
The code includes download dataset, preprocess dataset, filter dataset, distribution-mixer sampling.
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
        # download dataset from 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
        # You can also download ml-10m.zip from the browser or https://pan.baidu.com/s/10kyIQvfsU-HvKG-dlEiHag?pwd=hn99 and put it into `origin_data_path'
    filepath = origin_data_path + wget.filename_from_url(url)

    if not os.path.exists(filepath):
        print('The original dataset does not exist. Downloading it from ', url)
        # utils.download_dataset(url, filepath)
        wget.download(url, out=filepath)
    if not os.path.exists(origin_data_path+'/ml-10M100K/'):
        with zipfile.ZipFile(filepath, mode="a") as f:
            print('Extracting dataset...')
            f.extractall(origin_data_path)
        f.close()

    ## process ML10M dataset
    origin_data = dataprocess.process_origin_data()
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

