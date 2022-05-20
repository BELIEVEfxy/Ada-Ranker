#!/bin/bash
## Author: Xinyan Fan
## train Ada-Ranker in two-stage way (Θ ⇒ Θ + Φ or Θ ⇒ Φ, controlled by `freeze`)

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
# model
dropout_prob=0.4
use_pre_item_emb=0 # 1 for loading pretrained emb to initialize emb_table
freeze=1 # 1 for only finetune adaptation parameters, 0 for finetune all parameters

# pre-train model's path
FILE_PATH=$MY_DIR"result/Base/"$MODEL_NAME"_"$DATASET_NAME"_train/saved/"
SAVED_MODEL="GRU4Rec-May-19-2022_22-22-54.pth" # the pretrained base model: need to change!
SAVED_MODEL_PATH=$FILE_PATH$SAVED_MODEL


# ### finetune ###################################
python Main/main_finetune.py \
    --MY_DIR=$MY_DIR \
    --model=$MODEL_NAME \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME"/" \
    --train_dataset_path=$DATA_PATH \
    --test_dataset_path=$DATA_PATH \
    --saved_model_path=$SAVED_MODEL_PATH \
    --output_path=$ALL_RESULTS_ROOT"_finetune/" \
    --learning_rate=$learning_rate \
    --dropout_prob=$dropout_prob \
    --use_pre_item_emb=$use_pre_item_emb \
    --train_type=$train_type \
    --freeze=$freeze \

