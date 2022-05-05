#!/bin/bash
## Author: Xinyan Fan

# root
MY_DIR="/home/v-xinyanfan/2021/KDD/xinyan_repo/AdaRanker_pipeline/"

# ALL_DATA_ROOT=$MY_DIR"dataset/"
ALL_DATA_ROOT="/home/v-xinyanfan/2021/KDD/dataset/processed_data/"

# overall config
MODEL_NAME='GRU4Rec' # [MF, GRU4Rec, SASRec, NARM, NextItNet, SRGNN, SHAN]
DATASET_NAME="ML10M_small" # [ML10M, Xbox, Taobao]
train_type='Ada-Ranker' # ['Base', 'Ada-Ranker']


SAMPLED_ITEMS_NUM=19
TRAIN_MODE='multicate' # multicate is distribution-mixer sampling
TEST_MODE='multicate' 
TRAIN_DATA_PATH=$ALL_DATA_ROOT"/"$DATASET_NAME"/"$TRAIN_MODE"/"
TEST_DATA_PATH=$ALL_DATA_ROOT"/"$DATASET_NAME"/"$TEST_MODE"/"

# train
saved=True
learning_rate=0.001
batch_size=2048
learner="adam" # [adam, sgd]
stopping_step=5 # early stopping step
# model
dropout_prob=0.4
item_embedding_size=32
cate_embedding_size=32
use_pre_item_emb=0 # 1 for loading pretrained emb to initialize emb_table
use_bce_loss=1 # 1 for using BCE loss; 0 for BPR loss
add_bias_type='film' # ['film', 'add_w']
extract_bias_type='np' # ['avg', 'np']
change_para_type='mem_net' # ['mem_net']
num_workers=4
freeze=0


ALL_RESULTS_ROOT=$MY_DIR"result/Ada-Ranker/result_"$extract_bias_type"_"$add_bias_type"_"$change_para_type"/"$MODEL_NAME"_"$DATASET_NAME

FILE_PATH=$ALL_RESULTS_ROOT"_finetune/saved/"
SAVED_MODEL="GRU4Rec-Apr-12-2022_09-50-20.pth"
SAVED_MODEL_PATH=$FILE_PATH$SAVED_MODEL


# ### finetune ###################################
python Main/main_inference.py \
    --MY_DIR=$MY_DIR \
    --model=$MODEL_NAME \
    --dataset=$DATASET_NAME \
    --dataset_path=$ALL_DATA_ROOT"/"$DATASET_NAME"/" \
    --train_dataset_path=$TRAIN_DATA_PATH \
    --test_dataset_path=$TEST_DATA_PATH \
    --output_path=$ALL_RESULTS_ROOT"_inference/" \
    --saved=$saved \
    --learning_rate=$learning_rate \
    --batch_size=$batch_size \
    --learner=$learner \
    --stopping_step=$stopping_step \
    --dropout_prob=$dropout_prob \
    --item_embedding_size=$item_embedding_size \
    --cate_embedding_size=$cate_embedding_size \
    --use_pre_item_emb=$use_pre_item_emb \
    --use_bce_loss=$use_bce_loss \
    --saved_model_path=$SAVED_MODEL_PATH \
    --neg_items_num=$SAMPLED_ITEMS_NUM \
    --add_bias_type=$add_bias_type \
    --extract_bias_type=$extract_bias_type \
    --change_para_type=$change_para_type \
    --num_workers=$num_workers \
    --freeze=$freeze \

