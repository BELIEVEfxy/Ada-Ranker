# -*- coding: utf-8 -*-
# @Time   : 2021/07/08
# @Author : Xinyan Fan

# @Updata : 2021/07/12

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import *
import pickle as pkl
from data_process.helper import utils
from data_process.helper.CONST_VALUE import *

def save_tsv_obj(data, outfile, names, sep='\t'):
    print('\nsaving {0} at {1}'.format(os.path.basename(outfile),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    data.to_csv(outfile, columns=names, sep=sep, index=False)
    print('finish saving {0} at {1}'.format(os.path.basename(outfile),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

def save_pkl_obj(v, filename):
    print('\nsaving {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    with open(filename, 'wb') as f:
        # pkl.dump(v, f)
        ##--  also try this:  pickletools.optimize()
        ##--  https://towardsdatascience.com/the-power-of-pickletools-handling-large-model-pickle-files-7f9037b9086b
        p = pkl.Pickler(f)
        p.fast = True
        p.dump(v)
    print('finish saving {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))


def save_embedding(data, outfile, sep = ' '):
    fp = open(outfile, 'w', encoding='utf-8')
    fp.writelines(str(len(data)) + sep + str(len(data[0])) + '\n')
    for i in range(len(data)):
        fp.write(str(i))
        for j in range(len(data[i])):
            fp.write(sep + str(data[i][j]))
        fp.write('\n')
    fp.close()
    print('write file done!\n')

def save_dict(data, outfile='data.tsv', sep = '\t'):
    data = sorted(data.items(), key=lambda x: x[0], reverse=False)
    fp = open(outfile, 'w', encoding='utf-8')
    for item in data:
        fp.writelines(str(item[0]) + sep + str(item[1]) + '\n')
    fp.close()
    print('write file done!\n')

def split_train_valid_test(data, name):
    train_data = data[data[STAGE] == 'train'][name]
    valid_data = data[data[STAGE] == 'valid'][name]
    test_data = data[data[STAGE] == 'test'][name]

    return train_data, valid_data, test_data

def save_data(data):
    sample_mode = 'distribution-mixer'
    name = [USER_ID, ITEM_ID, CATE_ID, ITEM_SEQ, ITEM_SEQ_LEN, NEG_ITEMS]

    train_data, valid_data, test_data = split_train_valid_test(data, name)

    new_outfile_path = processed_data_path + sample_mode + '/'
    utils.ensure_dir(new_outfile_path)

    save_tsv_obj(train_data, new_outfile_path+'train.tsv', names=name, sep='\t')
    save_pkl_obj(train_data, new_outfile_path+'train.pkl')

    save_tsv_obj(valid_data, new_outfile_path+'valid.tsv', names=name, sep='\t')
    save_pkl_obj(valid_data, new_outfile_path+'valid.pkl')

    save_tsv_obj(test_data, new_outfile_path+'test.tsv', names=name, sep='\t')
    save_pkl_obj(test_data, new_outfile_path+'test.pkl')