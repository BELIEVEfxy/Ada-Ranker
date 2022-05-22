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
from data_process.helper import utils, datasaver, dataloader
from data_process.helper.CONST_VALUE import *

def process_origin_data():
    """
    Input files: ratings.dat and movies.dat
        (1) ratings.dat: UserID::MovieID::Rating::Timestamp
                    1::122::5::838985046
                    1::185::5::838983525
                    1::231::5::838983392
                    1::292::5::838983421
                    1::316::5::838983392

        (2) movies.dat: MovieID::Title::Genres
                    1::Toy Story (1995)::Adventure|Animation|Children|Comedy|Fantasy
                    2::Jumanji (1995)::Adventure|Children|Fantasy
                    3::Grumpier Old Men (1995)::Comedy|Romance
                    4::Waiting to Exhale (1995)::Comedy|Drama|Romance
                    5::Father of the Bride Part II (1995)::Comedy

    Output files: ML10M.tsv
        (1) ML10M.tsv: user_id\titem_id\tcate_id\ttimestamp
                    user_id item_id cate_id timestamp
            0       1       122     [5, 15] 838985046
            1       139     122     [5, 15] 974302621
            2       149     122     [5, 15] 1112342322
            3       182     122     [5, 15] 943458784
        (2) cate2idx (cate = Genres)
        (3) item2cate
    """
    print('\n>> Processing original dataset...')
    # origin_data's shape as ['user_id', 'item_id', 'cate_id', 'timestamp']
    origin_data_file = origin_data_path + dataset_name + '.tsv'
    if not os.path.exists(origin_data_file):
        item_file_path = origin_data_path+'ml-10M100K/movies.dat'
        inter_file_path = origin_data_path+'ml-10M100K/ratings.dat'

        item2cate = dataloader.load_item_profile(item_file_path)
        origin_data = dataloader.load_inter_file(item2cate, inter_file_path)
        origin_data.to_csv(origin_data_file, sep='\t', index=False)
    else:
        origin_data = pd.read_csv(origin_data_file, sep='\t')
        origin_data[CATE_ID] = origin_data[CATE_ID].apply(lambda x: utils.str2list(x))

    return origin_data

def get_new_cate2items():
    user2item2cate2time = pd.read_csv(processed_data_path+'user_item_cate_time.tsv', sep='\t')
    user2item2cate2time[CATE_ID] = user2item2cate2time[CATE_ID].apply(lambda x: utils.str2list(x))

    item2cate = {}
    cate2item = {}
    items = user2item2cate2time[ITEM_ID].values
    cates = user2item2cate2time[CATE_ID].values
    for idx in tqdm(range(len(items)), desc='get_item2cate'):
        cate = cates[idx]
        item = items[idx]
        item2cate[item] = cate
        for c in cate:
            if c not in cate2item.keys():
                cate2item[c] = []
            cate2item[c].append(item)

    datasaver.save_dict(item2cate, processed_data_path+'item2cate.tsv')

    return cate2item

def merge_category(data):
    # Get cate2items and item2cate
    # Merge categories containing a small number of items (lower than 200) into one category
    print('\n>> Merging categories...')
    cate2items = {}
    item2cate = {}

    cates = data[CATE_ID].values
    items = data[ITEM_ID].values
    for idx in tqdm(range(len(cates)), desc='get cate2items'):
        cate = cates[idx]
        item = items[idx]
        for c in cate:
            if c not in cate2items.keys():
                cate2items[c] = set([])
            cate2items[c].add(item)
    large_cate = []
    small_cate = []
    for cate, items in cate2items.items():
        # print(cate, len(items))
        if len(items) <= MIN_ITEM_IN_CATE:
            small_cate.append(cate)
        else:
            large_cate.append(cate)
    cate2idx = {x[1]: x[0]+1 for x in enumerate(large_cate)}
    for sc in small_cate:
        cate2idx[sc] = len(large_cate) + 1
    datasaver.save_dict(cate2idx, processed_data_path+'merged_cate2idx.tsv')

    cates = data[CATE_ID].values
    items = data[ITEM_ID].values
    for idx in tqdm(range(len(cates)), desc='get item2cate'):
        cate = cates[idx]
        item = items[idx]
        new_cate = []
        for c in cate:
            new_cate.append(cate2idx[c])
        item2cate[item] = new_cate
    # datasaver.save_dict(item2cate, processed_data_path+'item2cate.tsv')

    return item2cate

def filter_users(data): # filter users
    user_size = data.groupby([USER_ID]).size()
    user_size = user_size.reset_index()
    user_size.columns = [USER_ID, 'size']
    # print('#user before filtering', len(user_size))
    user_list = set(user_size[user_size['size'] > MIN_USER_NUM][USER_ID].values)
    # print('#user after filtering', len(user_list))
    
    return list(user_list)


def remap(data, item2cate): # remap all ID and sort
    user2idx = {x[1]: x[0]+1 for x in enumerate(data[USER_ID].unique())}
    item2idx = {x[1]: x[0]+1 for x in enumerate(data[ITEM_ID].unique())}

    data[USER_ID] = data[USER_ID].apply(lambda x: user2idx[x])
    data[CATE_ID] = data[ITEM_ID].apply(lambda x: list(item2cate[x]))
    data[ITEM_ID] = data[ITEM_ID].apply(lambda x: item2idx[x])

    datasaver.save_dict(user2idx, processed_data_path+'user2idx.tsv')
    datasaver.save_dict(item2idx, processed_data_path+'item2idx.tsv')

    return data


def filter_remap_sort(origin_data, item2cate):
    print('\n>> Filtering illegal interactions and remapping ids...')
    data = origin_data.sort_values(by=[USER_ID, TIMESTAMP], ascending=[True, True])
    users = filter_users(data)
    pd_users = pd.DataFrame({USER_ID: users})
    data = pd.merge(data, pd_users, on=USER_ID, how='inner')

    data = data.drop_duplicates(subset=[USER_ID, ITEM_ID], keep='last')
    print('\ndrop_duplicates done!')
    data = data.sort_values(by=[USER_ID, TIMESTAMP], ascending=[True, True])
    print('\nsort_values done!')
    data = remap(data, item2cate)
    print('\nremap done!')
    
    # print(data)
    data = data.sort_values(by=[USER_ID, TIMESTAMP], ascending=[True, True])
    data[[USER_ID, ITEM_ID, CATE_ID, TIMESTAMP]].to_csv(processed_data_path + 'user_item_cate_time.tsv', sep='\t', index=False)

    return data


def add_item_seq(data): # add item sequence
    print('\n>> Adding item sequences')
    users = set(data[USER_ID].values)

    user2itemseq = {user: [] for user in users}
    by_userid_group = data.groupby(USER_ID)[ITEM_ID]
    for userid, group_frame in tqdm(by_userid_group, desc='load item_seq'):
        seq = group_frame.values.tolist()
        user2itemseq[userid] = seq

    tmp_file_name = processed_data_path + 'tmp.tsv'
    tmp_file = open(tmp_file_name, 'w')
    tmp_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(USER_ID, ITEM_ID, CATE_ID, ITEM_SEQ, ITEM_SEQ_LEN, STAGE))

    user_size = data.groupby([USER_ID]).size()
    user_size = user_size.reset_index()
    user_size.columns = [USER_ID, 'size']
    avg_seq_len = user_size['size'].values.mean()
    print('\naverage sequence length: ', int(avg_seq_len)) # 143

    cumsum_v = [0] + list(user_size['size'].values.cumsum())

    all_users = data[USER_ID].values
    all_items = data[ITEM_ID].values
    all_cates = data[CATE_ID].values

    for i in tqdm(range(len(cumsum_v)-1), desc='add item_seq'):
        st_idx = cumsum_v[i]
        ed_idx = cumsum_v[i+1]
        leng = ed_idx - st_idx
        if leng > TARGET_ITEM_USED:
            st_idx = ed_idx - TARGET_ITEM_USED
        user = all_users[st_idx]

        for j in range(st_idx, ed_idx-2):
            tmp_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(user, all_items[j], all_cates[j], user2itemseq[user][:j-cumsum_v[i]], j-cumsum_v[i], 'train'))
        tmp_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(user, all_items[ed_idx-2], all_cates[ed_idx-2], user2itemseq[user][:ed_idx-2-cumsum_v[i]], ed_idx-2-cumsum_v[i], 'valid'))
        tmp_file.write('{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(user, all_items[ed_idx-1], all_cates[ed_idx-1], user2itemseq[user][:ed_idx-1-cumsum_v[i]], ed_idx-1-cumsum_v[i], 'test'))

    tmp_file.close()

    # [user_id, item_id, cate_id, timestamp, item_seq, item_seq_len, stage]
    data = pd.read_csv(tmp_file_name, sep='\t')
    data[CATE_ID] = data[CATE_ID].apply(lambda x: utils.str2list(x))
    data[ITEM_SEQ] = data[ITEM_SEQ].apply(lambda x: utils.str2list(x))
    os.remove(tmp_file_name)

    return data
        

        