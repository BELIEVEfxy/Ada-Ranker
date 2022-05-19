#!/bin/bash
## Author: Xinyan Fan
## train base models in an end-to-end way

# path of project: need to change!
MY_DIR="/home/xinyan_fan/xinyan/AdaRanker-backup/Ada-Ranker/"

# path of processed_data: need to change!
ALL_DATA_ROOT="/home/xinyan_fan/xinyan/AdaRanker-backup/dataset/processed_data/"

# overall config
MODEL_NAME='NARM' # [MF, GRU4Rec, SASRec, NARM, NextItNet, SRGNN, SHAN]
DATASET_NAME="ML10M" # [Taobao, Xbox, ML10M]
train_type='Base' # ['Base', 'Ada-Ranker']

ALL_RESULTS_ROOT=$MY_DIR"result/"$train_type"/"$MODEL_NAME"_"$DATASET_NAME

TRAIN_MODE='distribution-mixer' # distribution-mixer sampling
DATA_PATH=$ALL_DATA_ROOT"/"$DATASET_NAME"/"$TRAIN_MODE"/"

SAVED_MODEL_PATH=$ALL_RESULTS_ROOT"/"$MODEL_NAME"_"$DATASET_NAME"_train/saved/"$SAVED_MODEL

# train
learning_rate=0.001
# model
dropout_prob=0.4
use_pre_item_emb=1 # 1 for loading pretrained emb to initialize emb_table
freeze=0


### train ###################################
python Main/main_train.py \
    --MY_DIR=$MY_DIR \
    --model=$MODEL_NAME \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME"/" \
    --train_dataset_path=$DATA_PATH \
    --test_dataset_path=$DATA_PATH \
    --saved_model_path=$SAVED_MODEL_PATH \
    --output_path=$ALL_RESULTS_ROOT"_train/" \
    --learning_rate=$learning_rate \
    --dropout_prob=$dropout_prob \
    --use_pre_item_emb=$use_pre_item_emb \
    --train_type=$train_type \
    --freeze=$freeze \


