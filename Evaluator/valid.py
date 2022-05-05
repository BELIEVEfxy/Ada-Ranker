# @Time   : 2021/8/025
import os
from logging import getLogger
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm
import Utils.utils as utils

class Evaluator(object):
    
    def __init__(self, config):
        self.config = config
    
    def collect(self, scores):
        # scores = scores.detach().cpu().numpy()
        
        #print('scores',scores)
        labels = np.concatenate(([1, ], np.zeros(scores.shape[-1] - 1, dtype=int)))
        group_auc_score = self.group_auc(scores)

        # truth = np.argsort(scores)
        #print('truth', truth)
        # truth_idx = np.argwhere(truth==0)[:, -1]
        # re = truth_idx.sum()

        # hit
        truth = np.argsort(scores) == 0
        hit_ks = [1, 3, 10]
        hit_scores = np.array([truth[:, -k:].sum() for k in hit_ks], dtype=float)
        hit_scores /= len(scores)
        
        # ndcg
        truth = np.sum(truth, axis=-2)
        truth = truth[::-1]
        
        weight = self._get_ndcg_weights(len(truth))
        C = truth * weight
        
        ndcg_ks = [None, 1, 3, 10]
        ndcg_scores = np.array([C[:k].sum() for k in ndcg_ks])
        ndcg_scores /= len(scores)
        
        # mrr
        weight = self._get_mrr_weights(len(truth))
        C = truth * weight
        
        mean_mrr = C.sum() / len(scores)
        
        results = {
            "group_auc":    [group_auc_score],
            "ndcg":         [ndcg_scores[0]],
            "ndcg@1":       [ndcg_scores[1]],
            "ndcg@3":       [ndcg_scores[2]],
            "ndcg@10":       [ndcg_scores[3]],
            "hit@1":       [hit_scores[0]],
            "hit@3":        [hit_scores[1]],
            "hit@10":        [hit_scores[2]],
            "mrr":          [mean_mrr]
        }
        return results

    def append(self, all_eval_res, eval_res):
        if len(all_eval_res.keys()) == 0:
            return eval_res

        for metric in all_eval_res.keys():
            all_eval_res[metric] += eval_res[metric]
        return all_eval_res

    def evaluate(self, all_eval_res):
        result_dict = {}
        for metric, values in all_eval_res.items():
            result_dict[metric] = np.array(values).mean(-1)

        return result_dict

    # def evaluate(self, all_eval_res):
    #     result_dict = {}
    #     batch_num = len(all_eval_res['group_auc'])
    #     a_batch = int(batch_num / 10)
    #     for metric, values in all_eval_res.items():
    #         # result_dict[metric] = np.array(values).mean(-1)
    #         tmp = []
    #         for i in range(9):
    #             tmp.append(np.array(values[a_batch*i: a_batch*(i+1)]).mean(-1))
    #         tmp.append(np.array(values[a_batch*9:]).mean(-1))
    #         result_dict[metric] = tmp

    #     return result_dict

    def auc_score(self, scores):
        """
        KDD 20: 'On Sampled Metrics for Item Recommendation'
        """
        truth = np.argsort(scores)[::-1]
        truth_idx = np.argwhere(truth==0)[:, -1] + 1 # 下标从1开始！！！
        n = scores.shape[-1]
        num_pos = 1
        num_neg = n-num_pos
        #auc = (truth_idx.sum() - 0.5 * num_pos * (num_pos+1)) / (num_pos*num_neg) # auc

        up = n-0.5*(num_pos-1)-truth_idx.sum()/num_pos
        down = num_neg

        # up = num_pos*(n+1)-0.5*(num_pos*(num_pos+1))-truth_idx.sum()
        # down = num_pos * num_neg
        auc = up / down
        return auc

    def group_auc(self, scores):
        labels = np.concatenate(([1, ], np.zeros(scores.shape[-1] - 1, dtype=int)))
        gauc = np.mean(np.apply_along_axis(self.auc_score, -1, scores))
        return gauc

    def _get_ndcg_weights(self, length):
        ndcg_weights = 1 / np.log2(np.arange(2, 2 + 100))
        if length > len(ndcg_weights):
            ndcg_weights = 1 / np.log2(np.arange(2, length + 2))
        return ndcg_weights[:length]

    def _get_mrr_weights(self, length):
        mrr_weights = 1 / np.arange(1, 1 + 100)
        if length > len(mrr_weights):
            mrr_weights = 1 / np.arange(2, length + 2)
        return mrr_weights[:length]