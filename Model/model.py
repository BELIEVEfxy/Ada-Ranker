# -*- coding: utf-8 -*-
# @Time   : 2021/08/25

import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import xavier_uniform_, xavier_normal_, normal_, uniform_

import Model.modules as modules
from Model.recommender import AdaRecommender
from Utils import utils

class MF(AdaRecommender):
    
    def __init__(self, config, pre_item_emb):
        super(MF, self).__init__(config, pre_item_emb)

    def forward(self, interaction):
        user = interaction['user_id']
        user_e = self.user_embedding(user)
        return user_e


class GRU4Rec(AdaRecommender):
    
    def __init__(self, config, pre_item_emb):
        super(GRU4Rec, self).__init__(config, pre_item_emb)

    def _define_model_layers(self):
        # gru
        self.hidden_size = self.embedding_size * 2
        self.num_layers = 1
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)  

    def forward(self, interaction):
        item_seq_emb = self.item_embedding(interaction['item_seq']) # [2048, 100, 64]
        if self.train_type == 'Ada-Ranker':
            item_seq_emb = self._add_domain_bias(item_seq_emb, self.domain_bias)
        item_seq_emb = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb)
        gru_output = self.dense(gru_output)

        seq_output = gru_output[:, -1]

        return seq_output

    def forward_origin(self, interaction):
        item_seq_emb = self.item_embedding(interaction['item_seq']) # [2048, 100, 64]
        item_seq_emb = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb)
        gru_output = self.dense(gru_output)

        seq_output = gru_output[:, -1]

        return seq_output

class SASRec(AdaRecommender):
    
    def __init__(self, config, pre_item_emb):
        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']
        super(SASRec, self).__init__(config, pre_item_emb)
        
    def _define_model_layers(self):
        # multi-head attention
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = modules.TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)  

    def _get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward_origin(self, interaction):
        item_seq = interaction['item_seq']
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self._get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = output[:, -1, :]
        return output  # [B H]
        
    def forward(self, interaction):
        item_seq = interaction['item_seq']
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        if self.train_type == 'Ada-Ranker':
            item_emb = self._add_domain_bias(item_emb, self.domain_bias)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self._get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = output[:, -1, :]
        return output  # [B H]


class NARM(AdaRecommender):

    def __init__(self, config, pre_item_emb):
        super(NARM, self).__init__(config, pre_item_emb)

    def _define_model_layers(self):
        # gru
        self.hidden_size = self.config['hidden_size']
        self.num_layers = self.config['num_layers']
        # define layers and loss
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, self.num_layers, bias=False, batch_first=True)
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(self.dropout_prob)
        self.b = nn.Linear(2 * self.hidden_size, self.embedding_size, bias=False)

    def forward(self, interaction):
        item_seq = interaction['item_seq']
        item_seq_len = interaction['item_seq_len']
        item_seq_emb = self.item_embedding(item_seq)
        if self.train_type == 'Ada-Ranker':
            item_seq_emb = self._add_domain_bias(item_seq_emb, self.domain_bias)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_out, _ = self.gru(item_seq_emb_dropout)

        # fetch the last hidden state of last timestamp
        c_global = ht = gru_out[:, -1] # self.gather_indexes(gru_out, item_seq_len - 1)
        # avoid the influence of padding
        mask = item_seq.gt(0).unsqueeze(2).expand_as(gru_out)
        q1 = self.a_1(gru_out)
        q2 = self.a_2(ht)
        q2_expand = q2.unsqueeze(1).expand_as(q1)
        # calculate weighted factors α
        h = nn.Softmax(-1)(q1 + q2_expand)
        # print('h', h.size())
        alpha = self.v_t(mask * h)
        c_local = torch.sum(alpha.expand_as(gru_out) * gru_out, 1)
        c_t = torch.cat([c_local, c_global], 1)
        c_t = self.ct_dropout(c_t)
        seq_output = self.b(c_t)

        return seq_output


class NextItNet(AdaRecommender):

    def __init__(self, config, pre_item_emb):
        super(NextItNet, self).__init__(config, pre_item_emb)

    def _define_model_layers(self):
        # load parameters info
        self.residual_channels = self.embedding_size
        self.block_num = self.config['block_num']
        self.dilations = self.config['dilations'] * self.block_num
        self.kernel_size = self.config['kernel_size']
        # self.reg_weight = self.config['reg_weight']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0)

        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        rb = [
            modules.ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        self.final_layer = nn.Linear(self.residual_channels, self.embedding_size)

    def forward(self, interaction):
        item_seq_emb = self.item_embedding(interaction['item_seq'])  # [batch_size, seq_len, embed_size]
        if self.train_type == 'Ada-Ranker':
            item_seq_emb = self._add_domain_bias(item_seq_emb, self.domain_bias)
        # Residual locks
        dilate_outputs = self.residual_blocks(item_seq_emb)
        hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels)  # [batch_size, embed_size]
        seq_output = self.final_layer(hidden)  # [batch_size, embedding_size]
        return seq_output


class SHAN(AdaRecommender):
    r"""
    SHAN exploit the Hierarchical Attention Network to get the long-short term preference
    first get the long term purpose and then fuse the long-term with recent items to get long-short term purpose
    """
    def __init__(self, config, pre_item_emb):
        super(SHAN, self).__init__(config, pre_item_emb)

    def _define_model_layers(self):
        # load the parameter information
        self.short_item_length = self.config["short_item_length"]  # the length of the short session items
        assert self.short_item_length <= self.max_seq_length, "short_item_length can't longer than the max_seq_length"

        # define layers and loss
        self.long_w = nn.Linear(self.embedding_size, self.embedding_size)
        self.long_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.embedding_size),
                a=-np.sqrt(3 / self.embedding_size),
                b=np.sqrt(3 / self.embedding_size)
            ),
            requires_grad=True
        )
        self.long_short_w = nn.Linear(self.embedding_size, self.embedding_size)
        self.long_short_b = nn.Parameter(
            uniform_(
                tensor=torch.zeros(self.embedding_size),
                a=-np.sqrt(3 / self.embedding_size),
                b=np.sqrt(3 / self.embedding_size)
            ),
            requires_grad=True
        )

        self.relu = nn.ReLU()

    def long_and_short_term_attention_based_pooling_layer(self, long_short_item_embedding, user_embedding, mask=None):
        """
        fusing the long term purpose with the short-term preference
        """
        long_short_item_embedding_value = long_short_item_embedding

        long_short_item_embedding = self.relu(self.long_short_w(long_short_item_embedding) + self.long_short_b)
        long_short_item_embedding = torch.matmul(long_short_item_embedding, user_embedding.unsqueeze(2)).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            long_short_item_embedding.masked_fill_(mask, -1e9)
        long_short_item_embedding = nn.Softmax(dim=-1)(long_short_item_embedding)
        long_short_item_embedding = torch.mul(long_short_item_embedding_value,
                                              long_short_item_embedding.unsqueeze(2)).sum(dim=1)

        return long_short_item_embedding

    def long_term_attention_based_pooling_layer(self, item_seq_embedding, user_embedding, mask=None):
        """
        get the long term purpose of user
        """
        item_seq_embedding_value = item_seq_embedding

        item_seq_embedding = self.relu(self.long_w(item_seq_embedding) + self.long_b)
        user_item_embedding = torch.matmul(item_seq_embedding, user_embedding.unsqueeze(2)).squeeze(-1)
        # batch_size * seq_len
        if mask is not None:
            user_item_embedding.masked_fill_(mask, -1e9)
        user_item_embedding = nn.Softmax(dim=1)(user_item_embedding)
        user_item_embedding = torch.mul(item_seq_embedding_value,
                                        user_item_embedding.unsqueeze(2)).sum(dim=1, keepdim=True)
        # batch_size * 1 * embedding_size

        return user_item_embedding

    def forward(self, interaction):
        user = interaction['user_id']
        item_seq = interaction['item_seq']

        item_seq_embedding = self.item_embedding(item_seq)
        if self.train_type == 'Ada-Ranker':
            item_seq_embedding = self._add_domain_bias(item_seq_embedding, self.domain_bias)
        user_embedding = self.user_embedding(user)

        # get the mask
        mask = item_seq.data.eq(0)
        long_term_attention_based_pooling_layer = self.long_term_attention_based_pooling_layer(
            item_seq_embedding, user_embedding, mask
        )
        # batch_size * 1 * embedding_size

        short_item_embedding = item_seq_embedding[:, -self.short_item_length:, :]
        mask_long_short = mask[:, -self.short_item_length:]
        batch_size = mask_long_short.size(0)
        x = torch.zeros(size=(batch_size, 1)).eq(1).to(self.device)
        mask_long_short = torch.cat([x, mask_long_short], dim=1)
        # batch_size * short_item_length * embedding_size
        long_short_item_embedding = torch.cat([long_term_attention_based_pooling_layer, short_item_embedding], dim=1)
        # batch_size * 1_plus_short_item_length * embedding_size

        long_short_item_embedding = self.long_and_short_term_attention_based_pooling_layer(
            long_short_item_embedding, user_embedding, mask_long_short
        )
        # batch_size * embedding_size

        return long_short_item_embedding


class SRGNN(AdaRecommender):
    r"""SRGNN regards the conversation history as a directed graph.
    In addition to considering the connection between the item and the adjacent item,
    it also considers the connection with other interactive items.
    Such as: A example of a session sequence(eg:item1, item2, item3, item2, item4) and the connection matrix A
    Outgoing edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     1     0     0
         2    0     0    1/2   1/2
         3    0     1     0     0
         4    0     0     0     0
        === ===== ===== ===== =====
    Incoming edges:
        === ===== ===== ===== =====
         \    1     2     3     4
        === ===== ===== ===== =====
         1    0     0     0     0
         2   1/2    0    1/2    0
         3    0     1     0     0
         4    0     1     0     0
        === ===== ===== ===== =====
    """
    def __init__(self, config, pre_item_emb):
        super(SRGNN, self).__init__(config, pre_item_emb)

    def _define_model_layers(self):
        # load parameters info
        self.step = self.config['step']

        # define layers and loss
        self.gnn = modules.GNN(self.embedding_size, self.step)
        self.linear_one = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_two = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_three = nn.Linear(self.embedding_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.embedding_size * 2, self.embedding_size, bias=True)

    def forward(self, interaction):
        item_seq = interaction['item_seq']
        item_seq_len = interaction['item_seq_len']
        alias_inputs = interaction['alias_item_seq']
        items = interaction['uni_item_seq']
        A = interaction['A']

        # alias_inputs, A, items, mask = self._get_slice(item_seq)
        hidden = self.item_embedding(items) # [B, L, D]
        if self.train_type == 'Ada-Ranker':
            hidden = self._add_domain_bias(hidden, self.domain_bias)
        hidden = self.gnn(A, hidden) # [B, L, D]
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.embedding_size) # [B, L, D]
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs) # [B, L, D]
        # fetch the last hidden state of last timestamp
        ht = seq_hidden[:, -1] # [B, D]
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        mask = item_seq.gt(0)
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        seq_output = self.linear_transform(torch.cat([a, ht], dim=1))
        return seq_output
