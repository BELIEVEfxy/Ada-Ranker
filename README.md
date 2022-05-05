# Ada-Ranker

This is our Pytorch implementation for our SIGIR 2022 long paper:

Xinyan Fan, Jianxun Lian, Wayne Xin Zhao, Zheng Liu, Chaozhuo Li and Xing Xie(2022). "Ada-Ranker: A Data Distribution Adaptive Ranking Paradigm for
Sequential Recommendation." In SIGIR 2022. 

# Quick Start

## train base model
You can use the shell command to train the model (only need to change `MY_DIR` and `ALL_DATA_ROOT`)
```
sh run_train_base.sh
```
## train Ada-Ranker

train Ada-Ranker in an end-to-end way.
```
sh run_train_adaranker.sh
```

## finetune Ada-Ranker

load pre-trained base model, and finetune all parameters in Ada-Ranker (`freeze=0`, set `SAVED_MODEL_PATH` to the path of pre-trained base model):

```
sh run_finetune.sh 
```

load pre-trained base model, and only finetune adaptation parameters in Ada-Ranker (`freeze=1`, set `SAVED_MODEL_PATH` to the path of pre-trained base model):

```
sh run_finetune.sh 
```

## inference
provide a trained model, and infer on test set.
```
sh run_inference.sh 
```

See more details of main files in `Main/`.

# Output
output path like:
```
AdaRanker/result/
    - Ada-Ranker/
        - result_np_film_mem_net/
            - GRU4Rec_ML10M_train/
                - saved/
                timestamp.log
            - GRU4Rec_ML10M_finetune/
                - saved/
                timestamp.log
    - Base/
        - GRU4Rec_ML10M_train/
                - saved/
                timestamp.log
```

# Dataset
> Data/get_data.py

Datasets are in folder: \\\\msralab\ProjectData\xReco\xinyan\AdaRanker_SIGIR_data\processed_data\. 

(1) `user_item_cate_time.tsv` is the user-item interaction file with item's category and action timestamp (after hashing). This file can be used to pre-train item embeddings by word2vec.

(2) `item_emb_64.txt` is optional to initialize item embedding table from a pre-trained embedding table (by word2vec, see more detail in *xinyan_repo/DataPipeline/word2vec.py*)

(3) `train.pkl`, `valid.pkl` and `test.pkl` are needed to train the model and their structure are the same. Each pkl file is transferred from the DataFrame in the tsv file.
For example, in `train.pkl` the data contains 6 fields which are processed previously: ['user_id', 'item_id', 'cate_id', 'item_seq', 'item_seq_len', 'neg_items'], and the DataFrame is like:
```
user_id	item_id	cate_id	item_seq	item_seq_len	neg_items
2	36	[19]	[32, 33, 34, 35]	4	[13816, 30633, 29780, 39149, 20546, 46865, 13353, 45664, 49311, 14805, 28765, 7435, 6579, 33844, 43311, 30097, 42826, 23042, 1624]
2	37	[19]	[32, 33, 34, 35, 36]	5	[41, 12854, 13815, 20934, 3494, 21349, 17290, 12898, 26532, 1942, 3544, 7712, 26479, 1740, 46791, 13696, 3316, 15662, 30455]
2	38	[19]	[32, 33, 34, 35, 36, 37]	6	[1360, 39105, 29735, 15763, 7595, 2777, 48139, 5405, 5317, 33184, 11442, 13402, 8480, 9657, 15475, 24955, 4643, 7752, 19465]
```




# Configuration
> config/

All configuration files are in *config/*. 'overall.yaml' contains basic training settings. Parameter settings of all models are in *config/model_config/*. In *config/dataset_config/*, you need to set 'user_num' and 'item_num' in corresponding yaml files.

You can also set parameters directly in the command line, such as:
```
python Main/main_train.py --batch_size=1024
```
See more details in `Utils/init_config.py` to know how to load all configurations.

# Train
> Trainer/train.py

A batch data is organized in a dictionary - `interaction`. When you change the fields in dataset, you also need to change this part.

# Model
> Model/model.py

This framework includes 7 basic sequential recommender models: MF, GRU4Rec, SASRec, NARM, NextItNet, SRGNN, SHAN. Loss type includes BCE and BPR. Prediction layer contains using a 2-layer MLP to predict scores. 

To implement a new model, you need to complete functions `_define_model_layers()` and `forward()` in each class.

# Evaluation
> Evaluator/valid.py

Including basic ranking metrics: group_auc, ndcg, hit, mean_mrr.



If you have any questions for our paper or codes, please send an email to xinyanruc@126.com.
