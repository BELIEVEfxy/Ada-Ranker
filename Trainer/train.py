# @Time   : 2021/8/25
import os
from logging import getLogger
from time import time

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm

from Utils import utils
from Evaluator.valid import Evaluator


class Trainer(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = getLogger()
        
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(1, self.epochs)
        self.stopping_step = config['stopping_step']
        # self.clip_grad_norm = {'max_norm': 10} # config['clip_grad_norm']
        self.valid_metric_bigger = True
        self.test_batch_size = config['batch_size']

        self.device = config['device']
        self.checkpoint_dir = config['output_path']+'saved/'
        utils.ensure_dir(self.checkpoint_dir)
        saved_model_file = '{}-{}.pth'.format(config['model'], utils.get_local_time())
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)
        self.weight_decay = config['weight_decay']

        self.start_epoch = 0
        self.cur_step = 1
        self.best_valid_score = -1
        self.best_valid_result = None
        self.valid_score_dict = dict()
        self.optimizer = self._build_optimizer(self.model.parameters())

        self.logger.info('###########################')
        self.logger.info('trainable parameters: ')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.logger.info("{0}\t{1}".format(name, param.data.size()))                     
        self.logger.info('###########################')

        self.evaluator = Evaluator(config)

    def _build_optimizer(self, params):

        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=self.learning_rate)
            if self.weight_decay > 0:
                self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(params, lr=self.learning_rate)
        return optimizer

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _save_checkpoint(self, epoch):
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_valid_score': self.best_valid_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file)

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        des = 4
        train_loss_output = ('epoch %d training' + ' [' + 'time' +
                             ': %.2fs, ') % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            des = ('train_loss%d' + ': %.' + str(des) + 'f')
            train_loss_output += ', '.join(des % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            des = '%.' + str(des) + 'f'
            train_loss_output += 'train loss' + ': ' + des % losses
        return train_loss_output + ']'

    def fit(self, train_data, valid_data=None, saved=True, load_best_model=False, model_file=None, show_progress=False):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the utils.early_stopping is invalid.
            saved (bool, optional): whether to save the model parameters, default: True
            show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            ### freeze base model's parameters
            if self.config['freeze']:
                for name, param in self.model.named_parameters():
                    if name in checkpoint['state_dict']:
                        param.requires_grad = False
            
            self.logger.info(message_output)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # valid
            if (epoch_idx + 1) % self.eval_step == 0:
                if epoch_idx == 0:
                   print('>> Valid before training...')
                valid_start_time = time()

                # # valid on train set
                # train_result = self.evaluate(train_data, load_best_model=False, show_progress=show_progress)
                # train_result_output = 'valid on train set: \n' + utils.dict2str(train_result)
                # self.logger.info(train_result_output)

                valid_result = self.evaluate(valid_data, load_best_model=False, show_progress=show_progress)
                valid_score = utils.calculate_valid_score(valid_result)
                # valid_score, valid_result = self._valid_epoch(valid_data, show_progress=show_progress)
                self.valid_score_dict[epoch_idx] = valid_score
                self.best_valid_score, self.cur_step, stop_flag, update_flag = utils.early_stopping(
                    valid_score, self.best_valid_score, self.cur_step,
                    max_step=self.stopping_step, bigger=self.valid_metric_bigger
                )
                valid_end_time = time()
                valid_score_output = ("epoch %d evaluating" + " [time"
                                    + ": %.2fs, " + "valid_score: %f]") % \
                                     (epoch_idx, valid_end_time - valid_start_time, valid_score)
                valid_result_output = 'valid on valid set: \n' + utils.dict2str(valid_result)

                self.logger.info(valid_score_output)
                self.logger.info(valid_result_output)
                if update_flag:
                    if saved:
                        self._save_checkpoint(epoch_idx)
                        update_output = 'Saving current best' + ': %s' % self.saved_model_file
                        self.logger.info(update_output)
                    self.best_valid_result = valid_result

                if stop_flag:
                    stop_output = 'Finished training, best eval result in epoch %d' % \
                                  (epoch_idx - self.cur_step * self.eval_step)
                    self.logger.info(stop_output)
                    break

            # train
            print("\n>> epoch {}".format(epoch_idx + 1))
            training_start_time = time()

            total_loss = None
            iter_data = (
                tqdm(
                    enumerate(train_data),
                    total=len(train_data),
                    desc="Train ",
                ) if show_progress else enumerate(train_data)
            )

            for batch_idx, inter_data in iter_data:
                if self.config['model'] != 'SRGNN':
                    interaction = {'user_id': inter_data[0].to(self.device, non_blocking=True),\
                                    'item_id': inter_data[1].to(self.device, non_blocking=True), \
                                    'item_seq': inter_data[2].to(self.device, non_blocking=True), \
                                    'sampled_items': inter_data[3].to(self.device, non_blocking=True), \
                                    'item_seq_len': inter_data[4].to(self.device, non_blocking=True), 
                                    }
                else:
                    interaction = {'user_id': inter_data[0].to(self.device, non_blocking=True),\
                                'item_id': inter_data[1].to(self.device, non_blocking=True), \
                                'item_seq': inter_data[2].to(self.device, non_blocking=True), \
                                'sampled_items': inter_data[3].to(self.device, non_blocking=True), \
                                'item_seq_len': inter_data[4].to(self.device, non_blocking=True), \
                                'uni_item_seq': inter_data[5].to(self.device, non_blocking=True), \
                                'alias_item_seq': inter_data[6].to(self.device, non_blocking=True), \
                                'A': inter_data[7].to(self.device, non_blocking=True),
                                }
                self.model.train()
                loss = self.model.calculate_loss(interaction)

                self.optimizer.zero_grad()
               
                total_loss = loss.item() if total_loss is None else total_loss + loss.item()
                self._check_nan(loss)
                loss.backward()

                # print('###########################')
                # print('parameter gradients: ')
                # print(self.model.item_embedding.weight.grad)
                # print(self.model.mem_b1.index.grad)
                # print(self.model.mlp_layers[1].weight.grad)
                # print(self.model.mlp_layers[1].bias.grad) 
                # print(self.model.mlp_layers[3].weight.grad)
                # print(self.model.mlp_layers[3].bias.grad)  
                # print(self.model.item_embedding.weight.grad)
                # print('###########################')

                self.optimizer.step()

            training_end_time = time()
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx+1, training_start_time, training_end_time, total_loss)

            self.logger.info(train_loss_output)      

        return self.best_valid_score, self.best_valid_result
    
    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, show_progress=False):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.
            show_progress (bool): Show the progress of evaluate epoch. Defaults to ``False``.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value.
        """
        if not eval_data:
            return

        if load_best_model:
            if model_file:
                checkpoint_file = model_file
            else:
                checkpoint_file = self.saved_model_file
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
            self.logger.info(message_output)

        self.model.eval()

        all_eval_res = {}
        iter_data = (
            tqdm(
                enumerate(eval_data),
                total=len(eval_data),
                desc="Evaluate   ",
            ) if show_progress else enumerate(eval_data)
        )
        for batch_idx, inter_data in iter_data:
            if self.config['model'] != 'SRGNN':
                interaction = {'user_id': inter_data[0].to(self.device, non_blocking=True),\
                                'item_id': inter_data[1].to(self.device, non_blocking=True), \
                                'item_seq': inter_data[2].to(self.device, non_blocking=True), \
                                'sampled_items': inter_data[3].to(self.device, non_blocking=True), \
                                'item_seq_len': inter_data[4].to(self.device, non_blocking=True), 
                                }
            else:
                interaction = {'user_id': inter_data[0].to(self.device, non_blocking=True),\
                            'item_id': inter_data[1].to(self.device, non_blocking=True), \
                            'item_seq': inter_data[2].to(self.device, non_blocking=True), \
                            'sampled_items': inter_data[3].to(self.device, non_blocking=True), \
                            'item_seq_len': inter_data[4].to(self.device, non_blocking=True), \
                            'uni_item_seq': inter_data[5].to(self.device, non_blocking=True), \
                            'alias_item_seq': inter_data[6].to(self.device, non_blocking=True), \
                            'A': inter_data[7].to(self.device, non_blocking=True),
                            }
            
            #batch_size = int(inter_data[0].size()[0])
            scores = self.model.predict(interaction)

            eval_res = self.evaluator.collect(scores)
            all_eval_res = self.evaluator.append(all_eval_res, eval_res)
            
        result = self.evaluator.evaluate(all_eval_res)
        return result
