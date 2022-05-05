from operator import pos
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import *
from collections import Counter
from Utils import utils


class DataAnalyzer():
    def __init__(self, config, train_data, valid_data, test_data):
        self.ITEM_NUM = config['item_num']
        self.sep = '\t'
        self.output_path = config['output_path']
        utils.ensure_dir(self.output_path)
        self.image_path = self.output_path + "image/"
        utils.ensure_dir(self.image_path)

        self.train_data = train_data.dataset
        self.valid_data = valid_data.dataset
        self.test_data = test_data.dataset

        self.dataset = pd.concat([self.train_data, self.valid_data, self.test_data], ignore_index=True)

        # self.get_seq_len(self.dataset['item_seq_len'])

        # pos_items_array = np.array(self.dataset['item_id'])
        # neg_items_array = self.get_items_array(self.dataset['sampled_items'])
        # self.get_ctr_score(pos_items_array, neg_items_array)
        instance_pos_ctr, instance_neg_ctr = self.get_instance_ctr()
        self.plt_hist(instance_pos_ctr, 'pos')
        self.plt_hist(instance_neg_ctr, 'neg')

    def get_seq_len(self, item_seq_len):
        max_len = item_seq_len.max()
        min_len = item_seq_len.min()
        print('max item_seq_len: ', max_len)
        print('min_item_seq_len: ', min_len)

    def get_items_array(self, items_pd):
        """
        merge a series to a np.array.
        each element in the series is a list.
        """
        items = []
        for i in tqdm(items_pd):
            items += i
        return np.array(items)

    def get_ctr_score(self, pos_items_array, neg_items_array):
        """
        calculate ctr score of each pos/neg items.
        """
        print('calculating CTR scores of each item at {0}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        pos_cnt = dict(Counter(pos_items_array))
        neg_cnt = dict(Counter(neg_items_array))

        item2ctr = {}
        fp_ctr = open(self.output_path+'ctr_score.tsv', 'w')

        for i in tqdm(range(1, self.ITEM_NUM+1)):
            p, n = 0, 0
            if i in pos_cnt.keys():
                p = pos_cnt[i]
            if i in neg_cnt.keys():
                n = neg_cnt[i]
            try:
                item2ctr[i] = p / (p+n)
            except:
                item2ctr[i] = 0
            
            fp_ctr.write('{0}{1}{2}\n'.format(i, self.sep, item2ctr[i]))
            
        fp_ctr.close()

    def get_instance_ctr(self):
        """
        calculate pos item's CTR score of each instance.
        calculate neg items' avg.CTR score of each instance.
        """
        print('calculating CTR scores of pos/neg items of each instance at {0}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        ctr_score = pd.read_csv(self.output_path+'ctr_score.tsv', names=['item_id', 'ctr'], sep='\t')
        print(len(ctr_score), self.ITEM_NUM)
        ctr_score = ctr_score.set_index('item_id')
        print('CTR score of each item: \n', ctr_score)
        ctr_score = torch.cat([torch.tensor([0]), torch.tensor(ctr_score['ctr'].values)], dim=-1)
        print(len(ctr_score))

        instance_pos_ctr, instance_neg_ctr = [], []

        pos_items = self.dataset['item_id'].values
        neg_items = []
        for ni in tqdm(self.dataset['sampled_items'].values):
            for n in ni:
                neg_items.append(n)
        neg_items = np.array(neg_items)
        print(pos_items, neg_items)
        pos_item_ctr = ctr_score[pos_items]
        neg_item_ctr = ctr_score[neg_items]
        instance_pos_ctr = pos_item_ctr
        instance_neg_ctr = neg_item_ctr

        # fp = open(self.output_path+'instance_ctr.tsv', 'w')
        # for idx in tqdm(range(len(self.dataset))):
        #     pos_item = self.dataset.loc[idx, 'item_id']
        #     neg_item = self.dataset.loc[idx, 'sampled_items']

        #     pos_item_ctr = ctr_score[pos_item]
        #     neg_item_ctr = ctr_score[neg_item].numpy()
             
        #     instance_pos_ctr.append(pos_item_ctr)
        #     instance_neg_ctr.append(neg_item_ctr)
            
        #     fp.write('{}{}{}\n'.format(pos_item_ctr, '\t', neg_item_ctr))
        instance_pos_ctr = np.array(instance_pos_ctr)
        instance_neg_ctr = np.array(instance_neg_ctr).flatten()

        print(instance_pos_ctr.mean())
        print(instance_neg_ctr.mean())

        # fp.close()
        print(instance_pos_ctr, instance_neg_ctr)
        return instance_pos_ctr, instance_neg_ctr

    def plt_hist(self, x, column='pos'):
        """
        # item ctr distribution
        """
        print('plt hist at {0}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

        import matplotlib.pyplot as plt

        n, bins, patches = plt.hist(x, 50, density=False, facecolor='g', alpha=0.75)

        plt.xlabel('CTR')
        plt.ylabel('Count')
        plt.title(column)
        #plt.xlim(0,1)
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(self.image_path+column+'_ctr_hist.png')
        plt.show()
