# Ada-Ranker

This is our PyTorch implementation for our SIGIR 2022 long paper:

> Xinyan Fan, Jianxun Lian, Wayne Xin Zhao, Zheng Liu, Chaozhuo Li and Xing Xie (2022). "Ada-Ranker: A Data Distribution Adaptive Ranking Paradigm for
Sequential Recommendation." In SIGIR 2022. [PDF](https://arxiv.org/pdf/2205.10775.pdf)

# Requirements
Environments:
```
python==3.8
pytorch==1.11.0
cudatoolkit==10.1
```
Install environments by:
```
pip install -r requirements.txt
```

# Dataset 
## Get our prepared dataset
You can download the processed ML10M data from this [link](https://pan.baidu.com/s/10kyIQvfsU-HvKG-dlEiHag?pwd=hn99), and put it in your dataset path.

## Process dataset
> data_process/

You can also use the pipeline in `data_process/` to generate the processed ML10M dataset automatically. This pipeline includes:
- downloading the original ML10M dataset
- processing the original dataset (filtering users whose #interactions is lower than 10 and remapping all ids)
- sampling negative items by our proposed distritbuion-mixer sampling
- transferring DataFrame to pickle files
- pretraining item embeddings by word2vec algorithms.

Quick start by:
```
sh run_dataprocess.sh
```

In general, there are two steps of preparing final data sets that Ada-Ranker uses.


### (1) Process the original dataset

In this project, we provide an example of processing the original ML10M dataset. See more details in *data_process/ml10m_prepare.py* 

For other datasets, you can use another script to get the input data, and its format should be like this:
```
user_id item_id cate_id timestamp
1       122     [5, 15] 838985046
139     122     [5, 15] 974302621
149     122     [5, 15] 1112342322
182     122     [5, 15] 943458784
215     122     [5, 15] 1102493547
217     122     [5, 15] 844429650
```
The input data should include 4 fields: 'user_id', 'item_id', 'cate_id', 'timestamp', and each element in `cate_id' is a list containing several categories of target item.

### (2) Obtain the training sets
You only need to provide a `.tsv` file containing the input data with the above 4 fields, and the programme will automatically process it to the final training sets that Ada-Ranker needs. See more details in *data_process/preprocess.py*. 

The main output files include:

-  `user_item_cate_time.tsv` is the user-item interaction file with item's category and action timestamp (after hashing). This file can be used to pre-train item embeddings by word2vec.

-  `item_emb_64.txt` is optional to initialize item embedding table from a pre-trained embedding table (by word2vec, see more detail in *Ada-Ranker/data_process/helper/word2vec.py*)

-  `train.pkl`, `valid.pkl` and `test.pkl` are needed to train the model and their structure are the same. Each pkl file is transferred from the DataFrame in the tsv file. See more in *Ada-Ranker/data_process/helper/datasaver.py* to know how to get '.pkl' files. 
    - For example, in `train.pkl` the data contains 6 fields which are processed previously: ['user_id', 'item_id', 'cate_id', 'item_seq', 'item_seq_len', 'neg_items'], and the DataFrame is like:
```
user_id	item_id	cate_id	item_seq	item_seq_len	neg_items
2	36	[19]	[32, 33, 34, 35]	4	[13816, 30633, 29780, 39149, 20546, 46865, 13353, 45664, 49311, 14805, 28765, 7435, 6579, 33844, 43311, 30097, 42826, 23042, 1624]
2	37	[19]	[32, 33, 34, 35, 36]	5	[41, 12854, 13815, 20934, 3494, 21349, 17290, 12898, 26532, 1942, 3544, 7712, 26479, 1740, 46791, 13696, 3316, 15662, 30455]
2	38	[19]	[32, 33, 34, 35, 36, 37]	6	[1360, 39105, 29735, 15763, 7595, 2777, 48139, 5405, 5317, 33184, 11442, 13402, 8480, 9657, 15475, 24955, 4643, 7752, 19465]
```


# Quick Start

## train base model
You can use the shell command to train the model (only need to change `MY_DIR` and `ALL_DATA_ROOT`)
```
sh run_train_base.sh
```
## train Ada-Ranker

Train Ada-Ranker in an end-to-end way.
```
sh run_train_adaranker.sh
```

## finetune Ada-Ranker

Load pre-trained base model, and finetune all parameters in Ada-Ranker (`freeze=0`, set `SAVED_MODEL_PATH` to the path of pre-trained base model):

```
sh run_finetune.sh 
```

load pre-trained base model, and only finetune adaptation parameters in Ada-Ranker (`freeze=1`, set `SAVED_MODEL_PATH` to the path of pre-trained base model):

```
sh run_finetune.sh 
```

## inference
provide a trained model, and infer on a specific test set.
```
sh run_inference.sh 
```

See more details of main files in `Main/`.

# Output
Output path will be like this:
```
AdaRanker/result/
    - Ada-Ranker/
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

# Details of Code
## Dataset
> Data/get_data.py

`train.pkl`, `valid.pkl` and `test.pkl` are needed to train the model and their structure are the same. Each pkl file is transferred from the DataFrame in the corresponding tsv file. (See more details in `data_process/` to know how to prepare them.)


## Configuration
> config/

All configuration files are in *config/*. 'overall.yaml' contains basic training settings. Parameter settings of all models are in *config/model_config/*. In *config/dataset_config/*, you need to set 'user_num' and 'item_num' in corresponding yaml files.

You can also set parameters directly in the command line, such as:
```
python Main/main_train.py --batch_size=1024
```
See more details in `Utils/init_config.py` to know how to load all configurations.

## Trainer
> Trainer/train.py

A batch data is organized in a dictionary - `interaction`. When you change the fields in dataset, you also need to change this part.

## Model
> Model/model.py

This framework includes 7 basic sequential recommender models: MF, GRU4Rec, SASRec, NARM, NextItNet, SRGNN, SHAN. Loss type is BCE loss. Prediction layer contains using a 2-layer MLP to predict scores. 

To implement a new model, you need to complete functions `_define_model_layers()` and `forward()` in each class.

## Evaluation
> Evaluator/valid.py

Including basic ranking metrics: group_auc, ndcg, hit, mean_mrr.



# Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
```
@inproceedings{Fan-SIGIR-2022,
    title  = {Ada-Ranker: A Data Distribution Adaptive Ranking Paradigm for Sequential Recommendation},
    author = {Xinyan Fan and
              Jianxun Lian and
              Wayne Xin Zhao and
              Zheng Liu and
              Chaozhuo Li and
              Xing Xie},
    booktitle = {{SIGIR} '22: The 45th International {ACM} {SIGIR} Conference on Research
               and Development in Information Retrieval, Madrid, Spain, July 11–15, 2022},
    year = {2022},
    publisher = {{ACM}},
    doi       = {10.1145/3477495.3531931}
}
```

If you have any questions for our paper or codes, please send an email to xinyanruc@126.com.
