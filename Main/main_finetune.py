# @Time   : 2021/8/30
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
print('PROJECT_ROOT', PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

import logging
from logging import getLogger
import time
from datetime import datetime 
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from Data.get_data import MyDataset
import Data.pretrain_loader as ptl
from Trainer.train import Trainer
from Utils import utils
from Utils.init_config import Config
import setproctitle

if __name__ == '__main__':
    setproctitle.setproctitle("AdaRanker-finetune")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Amazon', help='name of datasets')

    args, _ = parser.parse_known_args()

    config = Config(model=args.model, dataset=args.dataset)
    utils.init_logger(config)
    logger = getLogger()
    logger.info(config)

    utils.init_seed(config['seed'])
    saved = config['saved']
    model_name = config['model']
    dataset_name = config['dataset']
    
    output_path = config['output_path']
    utils.ensure_dir(output_path)
    result_file = output_path+'/results_finetune.tsv'

    sep = '\t'

    # device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])
    config['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################################################################################################################################
    # data
    pre_item_emb = ptl._load_pre_item_emb(config['dataset_path'], logger) if config['use_pre_item_emb'] == True else None
    
    fp = open(result_file, 'w')

    logger.info('Loading Pre-trained Model...')
    model = utils.get_model(model_name)(config, pre_item_emb).to(config['device'])
    logger.info(model)
    trainer = Trainer(config, model)
    saved_path = config['saved_model_path']

    train_file_path = config['train_dataset_path']

    print('loading dataset from {0} at {1}'.format(os.path.basename(train_file_path), datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    train_dataset = MyDataset(config, path=train_file_path, dataset_type='train')
    valid_dataset = MyDataset(config, path=train_file_path, dataset_type='valid')
    print('finish loading dataset at {}'.format(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    print(valid_dataset.dataset)

    train_data = DataLoader(dataset = train_dataset, batch_size = config['batch_size'], shuffle = True, pin_memory=False, num_workers=config['num_workers'], persistent_workers=True) 
    valid_data = DataLoader(dataset = valid_dataset, batch_size = config['batch_size'], shuffle = False, pin_memory=False, num_workers=config['num_workers'], persistent_workers=True)

    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, load_best_model=True, model_file=saved_path, show_progress=config['show_progress']
    )
    # model evaluation
    test_file_path = config['test_dataset_path']
    test_dataset = MyDataset(config, path=test_file_path, dataset_type='test')
    test_data = DataLoader(dataset = test_dataset, batch_size = config['batch_size'], shuffle = False, pin_memory=False, num_workers=config['num_workers'], persistent_workers=True)
    test_result = trainer.evaluate(test_data, load_best_model=saved, show_progress=config['show_progress'])

    logger.info('best valid ' + f': {best_valid_result}')
    logger.info('test result' + f': {test_result}')

    for metirc, result in test_result.items():
        fp.write(str(metirc)+sep+str(result)+'\n')
    fp.close()
    