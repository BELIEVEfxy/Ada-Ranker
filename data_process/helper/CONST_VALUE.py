import os
from data_process.helper import utils

USER_ID = 'user_id'
ITEM_ID = 'item_id'
CATE_ID = 'cate_id'
TIMESTAMP = 'timestamp'
ITEM_SEQ = 'item_seq'
ITEM_SEQ_LEN = 'item_seq_len'
STAGE = 'stage'
NEG_ITEMS = 'neg_items'

TARGET_ITEM_USED = 10
MIN_USER_NUM = 10 # the number of interactions of a user must be higher than min_user_num
NEG_ITEMS_NUM = 19 # the number of negative samples
MIN_ITEM_IN_CATE = 200

PRETRAIN_VECTOR_SIZE = 64

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

dataset_name = 'ML10M'
url = 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'
origin_file_path = PROJECT_ROOT+"/dataset/"


origin_data_path = origin_file_path + "origin_data/" + dataset_name + '/'
processed_data_path = origin_file_path + "processed_data/" + dataset_name + '/'
utils.ensure_dir(origin_data_path)
utils.ensure_dir(processed_data_path)
