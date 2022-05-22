#!/bin/bash
## Author: Xinyan Fan
## Using a trained Ada-Ranker/Base to infer on specific test set

# path of project: need to change!
MY_DIR="/home/xinyan_fan/SIGIR2022/Ada-Ranker/"

# path of processed_data: need to change!
ALL_DATA_ROOT="/home/xinyan_fan/SIGIR2022/dataset/processed_data/"

# overall config
MODEL_NAME='GRU4Rec' # [MF, GRU4Rec, SASRec, NARM, NextItNet, SRGNN, SHAN]
DATASET_NAME="ML10M" # [ML10M, Xbox, Taobao]
train_type='Ada-Ranker' # ['Base', 'Ada-Ranker']

ALL_RESULTS_ROOT=$MY_DIR"result/"$train_type"/"$MODEL_NAME"_"$DATASET_NAME


TRAIN_MODE='distribution-mixer' # distribution-mixer sampling
DATA_PATH=$ALL_DATA_ROOT"/"$DATASET_NAME"/"$TRAIN_MODE"/"

# train
learning_rate=0.001
batch_size=2048
stopping_step=10 # early stopping step
# model
dropout_prob=0.4
embedding_size=64
use_pre_item_emb=1 # 1 for loading pretrained emb to initialize emb_table
freeze=0


FILE_PATH=$ALL_RESULTS_ROOT"_finetune/saved/"
SAVED_MODEL="GRU4Rec-Apr-12-2022_09-50-20.pth"
SAVED_MODEL_PATH=$FILE_PATH$SAVED_MODEL


# ### finetune ###################################
python Main/main_inference.py \
    --MY_DIR=$MY_DIR \
    --model=$MODEL_NAME \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME"/" \
    --train_dataset_path=$DATA_PATH \
    --test_dataset_path=$DATA_PATH \
    --output_path=$ALL_RESULTS_ROOT"_inference/" \
    --learning_rate=$learning_rate \
    --batch_size=$batch_size \
    --stopping_step=$stopping_step \
    --dropout_prob=$dropout_prob \
    --embedding_size=$embedding_size \
    --use_pre_item_emb=$use_pre_item_emb \
    --saved_model_path=$SAVED_MODEL_PATH \
    --freeze=$freeze \

