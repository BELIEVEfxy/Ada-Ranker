import numpy as np
import pandas as pd

def _transfer_emb(x):
    x = x.split(',')
    new_x = [float(x_) for x_ in x]
    return new_x

def _load_pre_item_emb(dataset_path, logger):
    logger.info('loading pretrained item embeddings...')

    item_emb_file_path = dataset_path + 'item_emb_64.txt'
    item_emb_data = pd.read_csv(item_emb_file_path, names=['old_id', 'emb'], sep='\t')

    item_emb_data['emb'] = item_emb_data['emb'].apply(lambda x: _transfer_emb(x))

    item_emb_ = item_emb_data['emb'].values
    if 0 in item_emb_data['old_id'].values:
        item_emb_ = item_emb_[1:]
    item_emb = []
    for ie in item_emb_:
        item_emb.append(ie)
    item_emb = np.array(item_emb)
    print('item_emb', item_emb)
    print('item_emb.size', item_emb.shape)

    return item_emb

# Beauty: [52205, 57290]
# Taobao1w_smallmap: [6862, 225948, 4693]
# Taobao1w_nomap: [1017976, 389758, 5443]
# Taobao100w_nomap: [1018012, 1807787, 7902]
# Amazon_pop: [182333, 864297, 11]