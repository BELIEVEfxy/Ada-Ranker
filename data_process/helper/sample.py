# -*- coding: utf-8 -*-
# @Time   : 2022/05/20
# @Author : Xinyan Fan

# This is for Distritbuion-mixer Sampling

import os
import sys
import pandas as pd
import numpy as np
import copy
from tqdm import *
from data_process.helper import utils
from data_process.helper.CONST_VALUE import *
import random

def sample_while(candi_items, sampled_items_num, target_item, item_seq):
    cur_sampled_items = []
    while len(cur_sampled_items) < sampled_items_num:
        sampled_item = random.sample(candi_items, 1)[0]
        while (sampled_item == target_item) or (sampled_item in item_seq) or (sampled_item in cur_sampled_items):
            sampled_item = random.sample(candi_items, 1)[0]
        cur_sampled_items.append(sampled_item)
        
    return cur_sampled_items

def distritbuion_mixer_sampling(data, cate2items):
    user2item2cate2time = pd.read_csv(processed_data_path+'user_item_cate_time.tsv', sep='\t')
    user2item2cate2time[CATE_ID] = user2item2cate2time[CATE_ID].apply(lambda x: utils.str2list(x))

    users = data[USER_ID].values
    target_items = data[ITEM_ID].values
    item_seqs = data[ITEM_SEQ].values
    item_seq_lens = data[ITEM_SEQ_LEN].values
    stages = data[STAGE].values
    cates = data[CATE_ID].values
    cate_num = len(cate2items)
    print('the number of cates', cate_num)

    tmp_neg_file = processed_data_path + 'tmp_neg.tsv'
    f = open(tmp_neg_file, 'w')
    f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n'.format(USER_ID, ITEM_ID, CATE_ID, ITEM_SEQ, ITEM_SEQ_LEN, NEG_ITEMS, STAGE))

    cate2item_uni = copy.deepcopy(cate2items)
    for k, v in cate2item_uni.items():
        cate2item_uni[k] = list(set(v))

    for i in tqdm(range(len(target_items)), desc='distribution-mixer sampling'):
        user = users[i]
        target_item = target_items[i]
        item_seq = item_seqs[i]
        item_seq_len = item_seq_lens[i]
        stage = stages[i]

        for cate_ in cates[i]:
            # cate_ = cates[i][0] if len(cates[i]) == 1 else cates[i][random.sample([0,1], 1)[0]]
            sampled_cate = [cate_]
            multicate_num = random.sample([0, 1, 2], 1)[0]
            sampled_cate += random.sample(range(1, cate_num+1), multicate_num)

            sample_ratio = np.ones(len(sampled_cate)) / len(sampled_cate)
            sample_num = np.random.multinomial(NEG_ITEMS_NUM, sample_ratio, size=1)[0]

            sampled_items = []

            seed = random.sample(range(0, 100), 1)[0]
            for idx in range(len(sample_num)):
                sampled_items_num = sample_num[idx]
                if sampled_items_num == 0:
                    continue
                cate = sampled_cate[idx]
                # 50% universal 50% pop
                if seed < 50: # universal
                    candi_items = cate2item_uni[cate]
                else: # pop
                    candi_items = cate2items[cate]

                cur_sampled_items = sample_while(candi_items, sampled_items_num, target_item, item_seq)
                sampled_items += cur_sampled_items
            f.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\n'.format(user, target_item, cate_, item_seq, item_seq_len, sampled_items, stage))

        
    f.close()
    
    data = pd.read_csv(tmp_neg_file, sep='\t')
    data[ITEM_SEQ] = data[ITEM_SEQ].apply(lambda x: utils.str2list(x))
    data['neg_items'] = data['neg_items'].apply(lambda x: utils.str2list(x))

    print('sampled_data', data)
    os.remove(tmp_neg_file)

    return data