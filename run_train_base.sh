#!/bin/bash
## Author: Xinyan Fan
## train base models in an end-to-end way

# root
MY_DIR="/home/xinyan_fan/xinyan/AdaRanker-backup/Ada-Ranker/"

# ALL_DATA_ROOT=$MY_DIR"dataset/"
ALL_DATA_ROOT="/home/xinyan_fan/xinyan/AdaRanker-backup/dataset/processed_data/"

# overall config
MODEL_NAME='GRU4Rec' # [MF, GRU4Rec, SASRec, NARM, NextItNet, SRGNN, SHAN]
DATASET_NAME="ML10M" # [Taobao, Xbox, ML10M]
train_type='Base' # ['Base', 'Ada-Ranker']

ALL_RESULTS_ROOT=$MY_DIR"result/Base/"$MODEL_NAME"_"$DATASET_NAME


SAMPLED_ITEMS_NUM=19
TRAIN_MODE='distribution-mixer' # distribution-mixer sampling
TEST_MODE='distribution-mixer' 
TRAIN_DATA_PATH=$ALL_DATA_ROOT"/"$DATASET_NAME"/"$TRAIN_MODE"/"
TEST_DATA_PATH=$ALL_DATA_ROOT"/"$DATASET_NAME"/"$TEST_MODE"/"

SAVED_MODEL_PATH=$ALL_RESULTS_ROOT"/"$MODEL_NAME"_"$DATASET_NAME"_train/saved/"$SAVED_MODEL

# train
saved=True
learning_rate=0.001
batch_size=1024
learner="adam" # [adam, sgd]
stopping_step=4 # early stopping step
# model
dropout_prob=0.4
item_embedding_size=32
cate_embedding_size=32
use_pre_item_emb=1 # 1 for loading pretrained emb to initialize emb_table
use_bce_loss=1 # 1 for using BCE loss; 0 for BPR loss
num_workers=4
freeze=0


# DATA_PATH="/home/v-xinyanfan/2021/WSDM/dataset/debug_dataset/"
### train ###################################
python Main/main_train.py \
    --MY_DIR=$MY_DIR \
    --model=$MODEL_NAME \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME"/" \
    --train_dataset_path=$TRAIN_DATA_PATH \
    --test_dataset_path=$TEST_DATA_PATH \
    --output_path=$ALL_RESULTS_ROOT"_train/" \
    --saved=$saved \
    --learning_rate=$learning_rate \
    --batch_size=$batch_size \
    --learner=$learner \
    --dropout_prob=$dropout_prob \
    --item_embedding_size=$item_embedding_size \
    --cate_embedding_size=$cate_embedding_size \
    --use_pre_item_emb=$use_pre_item_emb \
    --use_bce_loss=$use_bce_loss \
    --saved_model_path=$SAVED_MODEL_PATH \
    --neg_items_num=$SAMPLED_ITEMS_NUM \
    --stopping_step=$stopping_step \
    --num_workers=$num_workers \
    --train_type=$train_type \
    --freeze=$freeze \


