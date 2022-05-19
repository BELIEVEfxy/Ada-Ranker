# -*- coding: utf-8 -*-
# @Time   : 2021/07/07
# @Author : Xinyan Fan

# @Time : 2021/07/12
# @Updata : 2021/08/28
# @Updata : 2022/01/05
"""
Use Word2Vec to pre-train item embedding
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import *

from data_process.helper import utils
from data_process.helper.CONST_VALUE import *

from gensim.models import Word2Vec


def load_item_seq(data):
    item_num = data['item_id'].values.max()
    # print('item_num', item_num)
    corpus = []
    by_userid_group = data.groupby('user_id')['item_id']
    for userid, group_frame in tqdm(by_userid_group, desc='load item_seq'):
        item_seq = group_frame.values.tolist()
        seq_ = []
        for token in item_seq:
            seq_.append(str(token))
        corpus.append(seq_)

    return corpus, item_num

def emb2str(embedding):
    str_emb = ''
    for element in embedding:
        str_emb += (str('%.6f' % element) + ',')
    return str_emb[:-1]

def pretrain_word2vec():
    print('\n>> Training word2vec...')
    # load data
    infile = processed_data_path + '/user_item_cate_time.tsv'
    outfile_path = processed_data_path
    utils.ensure_dir(outfile_path)

    user2item2cate2time = pd.read_csv(infile, sep='\t')

    corpus, item_num = load_item_seq(user2item2cate2time)
    
    w2v_model = Word2Vec(corpus, vector_size=PRETRAIN_VECTOR_SIZE, window=10, min_count=3, workers=4)
    w2v_model.save(outfile_path+"word2vec.model")

    fp = open(outfile_path+'item_emb_'+str(PRETRAIN_VECTOR_SIZE)+'.txt', 'w')
    for i in tqdm(range(1, item_num+1), desc='output Item Emb'):
        try:
            item_emb = w2v_model.wv[str(i)]
        except:
            item_emb = np.zeros((PRETRAIN_VECTOR_SIZE,), dtype=np.float)
        str_emb = emb2str(item_emb)
        fp.write('{0}{1}{2}\n'.format(i, '\t', str_emb))
    fp.close()
