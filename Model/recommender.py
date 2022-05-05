import math
from this import d
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, xavier_normal_

import Model.modules as modules
from Utils import utils

class AdaRecommender(nn.Module):
    
    def __init__(self, config, pre_item_emb):
        super(AdaRecommender, self).__init__()

        self.config = config
        self.train_type = config['train_type']
        # load parameters info
        self.max_seq_length = config['MAX_SEQ_LEN']
        self.n_users = config['user_num']
        self.n_items = config['item_num']
        self.n_cates = config['cate_num']
        self.device = config['device']
        self.NEG_ITEMS_NUM = config['neg_items_num']

        self.item_embedding_size = config['item_embedding_size']
        self.cate_embedding_size = config['cate_embedding_size']
        self.embedding_size = self.item_embedding_size + self.cate_embedding_size

        self.dropout_prob = config['dropout_prob']
        self.use_pre_item_emb = config['use_pre_item_emb']
        self.use_bce_loss = config['use_bce_loss']
        
        self.dnn_input_size = self.embedding_size * 2 #if config['use_extractor'] == True else self.embedding_size * 2
        self.dnn_inner_size = self.embedding_size

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, sparse=False)
        
        if self.use_pre_item_emb:
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, sparse=False)
            pre_item_emb = torch.from_numpy(pre_item_emb)
            pad_emb = torch.zeros([1, self.embedding_size])
            pre_item_emb = torch.cat([pad_emb, pre_item_emb], dim=-2).to(torch.float32)
            print('self.n_items', self.n_items)
            print('pre_item_emb', pre_item_emb)
            print('pre_item_emb', pre_item_emb.size())
            self.item_embedding.weight = nn.Parameter(pre_item_emb)
        else:
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, sparse=False, padding_idx=0)

        # model layers
        self._define_model_layers()

        if self.train_type == 'Ada-Ranker':
            # extract bias layers
            if self.config['extract_bias_type'] == 'avg':
                self.extract_bias_layer = nn.AdaptiveAvgPool2d((1, self.embedding_size))
            elif self.config['extract_bias_type'] == 'np':
                self.extract_bias_layer = modules.NeuProcessEncoder(self.embedding_size, self.embedding_size, self.embedding_size, self.dropout_prob, self.device, encode_type='np', cate_num=self.n_cates)
                self.all_items = set(range(1, self.n_items))
            else:
                self.extract_bias_layer = nn.Linear(self.embedding_size, self.embedding_size)

            # add bias layers
            if self.config['add_bias_type'] == 'film':
                self.film_affine_emb_scale = nn.Linear(self.embedding_size, 1)
                self.film_affine_emb_bias = nn.Linear(self.embedding_size, 1)
            elif self.config['add_bias_type'] == 'add_w':
                self.linear_layer = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

            # predict_layer
            if self.config['change_para_type'] == 'mem_net':
                self.mem_w1 = modules.MemoryUnit(self.dnn_input_size, self.dnn_inner_size, self.embedding_size)
                self.mem_b1 = modules.MemoryUnit(1, self.dnn_inner_size, self.embedding_size)
                self.mem_w2 = modules.MemoryUnit(self.dnn_inner_size, 1, self.embedding_size)
                self.mem_b2 = modules.MemoryUnit(1, 1, self.embedding_size)
                seq = [
                    nn.Dropout(p=self.dropout_prob), 
                    modules.AdaptLinear(self.dnn_input_size, self.dnn_inner_size),
                    nn.Tanh(),
                    modules.AdaptLinear(self.dnn_inner_size, 1),
                ]
            else:
                seq = [
                        nn.Dropout(p=self.dropout_prob), 
                        nn.Linear(self.dnn_input_size, self.dnn_inner_size),
                        nn.Tanh(),
                        nn.Linear(self.dnn_inner_size, 1),
                    ]
        else: # for base model's prediction layer
            seq = [
                        nn.Dropout(p=self.dropout_prob), 
                        nn.Linear(self.dnn_input_size, self.dnn_inner_size),
                        nn.Tanh(),
                        nn.Linear(self.dnn_inner_size, 1),
                    ]

        if self.use_bce_loss:
            seq.append(nn.Sigmoid())
        self.mlp_layers = torch.nn.Sequential(*seq).to(self.device)

        # loss
        if self.use_bce_loss:
            self.loss_fct = nn.BCELoss()

    def _define_model_layers(self):
        pass

    def forward(self, interaction):
        pass

    def _predict_layer(self, seq_emb, test_items_emb):
        if seq_emb.shape != test_items_emb.shape:
            seq_emb = torch.repeat_interleave(
                seq_emb, test_items_emb.shape[-2], dim=-2
            )
            seq_emb = seq_emb.reshape(test_items_emb.shape)

        seq_emb = torch.cat([seq_emb, test_items_emb], -1)

        if self.train_type == 'Ada-Ranker':
            if self.config['change_para_type'] == 'mem_net':
                domain_bias = self.domain_bias.squeeze(1)
                wei_1, wei_2 = self.mem_w1(domain_bias), self.mem_w2(domain_bias)
                bias_1, bias_2 = self.mem_b1(domain_bias), self.mem_b2(domain_bias)
                self.mlp_layers[1].memory_parameters(wei_1, bias_1)
                self.mlp_layers[3].memory_parameters(wei_2, bias_2)

        scores = self.mlp_layers(seq_emb).view(test_items_emb.shape[0], -1)

        return scores # [2048, 5]

    def _get_labels(self, B, true_num=1, false_num=19):
        true_label = torch.ones([true_num], dtype=torch.float) #.to(self.device)
        false_label = torch.zeros([false_num], dtype=torch.float) #.to(self.device)
        labels = torch.cat([true_label, false_label], -1)
        labels = labels.unsqueeze(0).expand(B)

        return labels.to(self.device)
    
    def _get_test_emb(self, interaction):
        pos_items = interaction['item_id']
        sampled_items = interaction['sampled_items']
        self.test_items = torch.cat([pos_items.reshape((-1, 1)), sampled_items], -1)
        test_items_emb = self.item_embedding(self.test_items) # [2048, 5, 64]

        return pos_items, test_items_emb

    def _bias_etractor(self, test_items_emb):
        domain_distri_emb = test_items_emb.reshape((-1))
        domain_distri_emb = domain_distri_emb.reshape((-1, self.NEG_ITEMS_NUM+1, self.embedding_size)) # [batch_size, 20, 64]

        domain_distri_emb = self.extract_bias_layer(domain_distri_emb) # [batch_size, 1, 64]
        if len(domain_distri_emb.size()) == 2:
            domain_distri_emb = domain_distri_emb.unsqueeze(1)

        return domain_distri_emb#.reshape((-1, self.embedding_size))
    
    def _add_domain_bias(self, input_tensor, domain_bias):
        if len(input_tensor.size()) == 2:
            input_tensor = input_tensor.unsqueeze(1) # [batch_size, 1, 64]
        if self.config['add_bias_type'] == 'add_w':
            output_tensor = self.linear_layer(input_tensor) + domain_bias # [2048, 1, 64]
        elif self.config['add_bias_type'] == 'film':
            gamma = self.film_affine_emb_scale(domain_bias) # [batch_size, 1, 1]
            beta = self.film_affine_emb_bias(domain_bias) # [batch_size, 1, 1]
            output_tensor = gamma * input_tensor + beta
        else:
            output_tensor = input_tensor

        return output_tensor # [2048, 1, 64]

    def _cal_loss(self, user_emb, test_items_emb, pos_items):
        # # BCE
        if self.use_bce_loss:
            logits = self._predict_layer(user_emb, test_items_emb)
            labels = self._get_labels(logits.shape, 1, self.NEG_ITEMS_NUM)
            loss = self.loss_fct(logits, labels)
        else:# BPR: sample 1 neg from global
            neg_items = torch.randint(0, self.n_items, user_emb.shape[0]).to(self.device)
            pos_item_emb = self.item_embedding(pos_items)
            neg_item_emb = self.item_embedding(neg_items)
            pos_score = self._predict_layer(user_emb, pos_item_emb)
            neg_score = self._predict_layer(user_emb, neg_item_emb)
            loss = modules.bpr_loss(pos_score, neg_score)

        return loss

    def calculate_loss(self, interaction):
        pos_items, test_items_emb = self._get_test_emb(interaction)

        if self.train_type == 'Ada-Ranker':
            self.domain_bias = self._bias_etractor(test_items_emb) # [batch_size, embedding_size]

        user_emb = self.forward(interaction) #[2048, 64]
        loss = self._cal_loss(user_emb, test_items_emb, pos_items)

        return loss

    def predict(self, interaction):
        _, test_items_emb = self._get_test_emb(interaction)

        if self.train_type == 'Ada-Ranker':
            self.domain_bias = self._bias_etractor(test_items_emb) # [batch_size, embedding_size]

        user_emb = self.forward(interaction) #[2048, 64]
        
        scores = self._predict_layer(user_emb, test_items_emb).detach().cpu().numpy()

        return scores

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters' + f': {params}'

