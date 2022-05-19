import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import Model.modules as modules

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

        self.embedding_size = config['embedding_size']

        self.dropout_prob = config['dropout_prob']
        self.use_pre_item_emb = config['use_pre_item_emb']
        self.use_bce_loss = config['use_bce_loss']
        
        self.dnn_input_size = self.embedding_size * 2
        self.dnn_inner_size = self.embedding_size

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size, sparse=False)
        
        if self.use_pre_item_emb:
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, sparse=False)
            pre_item_emb = torch.from_numpy(pre_item_emb)
            pad_emb = torch.zeros([1, self.embedding_size])
            pre_item_emb = torch.cat([pad_emb, pre_item_emb], dim=-2).to(torch.float32)
            print('pretrained item embedding table: \n', pre_item_emb)
            self.item_embedding.weight = nn.Parameter(pre_item_emb)
        else:
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, sparse=False, padding_idx=0)

        # model layers
        self._define_model_layers()

        if self.train_type == 'Ada-Ranker':
            # extract bias layers
            self.extract_distribution_layer = modules.NeuProcessEncoder(
                input_size = self.embedding_size, 
                hidden_size = self.embedding_size, 
                output_size = self.embedding_size, 
                dropout_prob = self.dropout_prob,
                device = self.device)

            # input modulation layers
            self.film_affine_emb_scale = nn.Linear(self.embedding_size, 1)
            self.film_affine_emb_bias = nn.Linear(self.embedding_size, 1)

            # parameter modulation layers
            self.mem_w1 = modules.MemoryUnit(self.dnn_input_size, self.dnn_inner_size, self.embedding_size)
            self.mem_b1 = modules.MemoryUnit(1, self.dnn_inner_size, self.embedding_size)
            self.mem_w2 = modules.MemoryUnit(self.dnn_inner_size, 1, self.embedding_size)
            self.mem_b2 = modules.MemoryUnit(1, 1, self.embedding_size)
            seq = [
                nn.Dropout(p=self.dropout_prob), 
                modules.AdaLinear(self.dnn_input_size, self.dnn_inner_size),
                nn.Tanh(),
                modules.AdaLinear(self.dnn_inner_size, 1),
            ]
        else: # for base model's prediction layer
            seq = [
                    nn.Dropout(p=self.dropout_prob), 
                    nn.Linear(self.dnn_input_size, self.dnn_inner_size),
                    nn.Tanh(),
                    nn.Linear(self.dnn_inner_size, 1),
                ]

        seq.append(nn.Sigmoid())
        self.mlp_layers = torch.nn.Sequential(*seq).to(self.device)

        # loss
        self.loss_fct = nn.BCELoss()

    def _define_model_layers(self):
        pass

    def forward(self, interaction):
        pass

    def _distribution_vector_etractor(self, candidate_items_emb):
        distribution_vector = candidate_items_emb.reshape((-1))
        distribution_vector = distribution_vector.reshape((-1, self.NEG_ITEMS_NUM+1, self.embedding_size))

        distribution_vector = self.extract_distribution_layer(distribution_vector) # [B, 1, dim]
        if len(distribution_vector.size()) == 2:
            distribution_vector = distribution_vector.unsqueeze(1)

        return distribution_vector
    
    def _input_modulation(self, input_tensor, domain_bias):
        if len(input_tensor.size()) == 2:
            input_tensor = input_tensor.unsqueeze(1) # [B, 1, dim]
        
        gamma = self.film_affine_emb_scale(domain_bias) # [B, 1, 1]
        beta = self.film_affine_emb_bias(domain_bias) # [B, 1, 1]
        output_tensor = gamma * input_tensor + beta

        return output_tensor # [B, 1, dim]

    def _parameter_modulation(self):
        _distribution_vec = self.distribution_vec.squeeze(1)
        wei_1, wei_2 = self.mem_w1(_distribution_vec), self.mem_w2(_distribution_vec)
        bias_1, bias_2 = self.mem_b1(_distribution_vec), self.mem_b2(_distribution_vec)
        self.mlp_layers[1].memory_parameters(wei_1, bias_1)
        self.mlp_layers[3].memory_parameters(wei_2, bias_2)

    def _predict_layer(self, seq_emb, candidate_items_emb):
        if seq_emb.shape != candidate_items_emb.shape:
            seq_emb = torch.repeat_interleave(
                seq_emb, candidate_items_emb.shape[-2], dim=-2
            )
            seq_emb = seq_emb.reshape(candidate_items_emb.shape)

        seq_emb = torch.cat([seq_emb, candidate_items_emb], -1)

        if self.train_type == 'Ada-Ranker': # parameter modulation
            self._parameter_modulation()

        scores = self.mlp_layers(seq_emb).view(candidate_items_emb.shape[0], -1)

        return scores # [B, NEG_ITEMS_NUM+1]

    def _get_labels(self, B, true_num=1, false_num=19):
        true_label = torch.ones([true_num], dtype=torch.float)
        false_label = torch.zeros([false_num], dtype=torch.float)
        labels = torch.cat([true_label, false_label], -1)
        labels = labels.unsqueeze(0).expand(B)

        return labels.to(self.device)
    
    def _get_candidates_emb(self, interaction):
        pos_items = interaction['item_id']
        neg_items = interaction['neg_items']
        self.test_items = torch.cat([pos_items.reshape((-1, 1)), neg_items], -1)
        candidate_items_emb = self.item_embedding(self.test_items) # [B, NEG_ITEMS_NUM+1, dim]

        return candidate_items_emb


    def _cal_loss(self, user_emb, candidate_items_emb):
        # # BCE loss
        logits = self._predict_layer(user_emb, candidate_items_emb)
        labels = self._get_labels(logits.shape, 1, self.NEG_ITEMS_NUM)
        loss = self.loss_fct(logits, labels)

        return loss

    def calculate_loss(self, interaction):
        candidate_items_emb = self._get_candidates_emb(interaction)

        if self.train_type == 'Ada-Ranker':
            self.distribution_vec = self._distribution_vector_etractor(candidate_items_emb) # [B, dim]

        user_emb = self.forward(interaction) # [B, dim]
        loss = self._cal_loss(user_emb, candidate_items_emb)

        return loss

    def predict(self, interaction):
        candidate_items_emb = self._get_candidates_emb(interaction)

        if self.train_type == 'Ada-Ranker':
            self.distribution_vec = self._distribution_vector_etractor(candidate_items_emb) # [B, dim]

        user_emb = self.forward(interaction) # [B, dim]
        scores = self._predict_layer(user_emb, candidate_items_emb).detach().cpu().numpy()

        return scores

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters' + f': {params}'

