# Basic idea

Compared to [OTTO – Multi-Objective Recommender System competition](https://www.kaggle.com/competitions/otto-recommender-system/data)

1. might only one action
2. only one item as target, might be possible to extend the training data


# Plans & TODO 

1. Structure 
    1. Candidate generation
    2. Rank
        1. [x] rule-based as first version
2. [ ] Text info (text and title info)
    1. [ ] text embedding as features
 
# Two Stages Approach

[Experiment tracking spreadsheet](https://docs.google.com/spreadsheets/d/1f9faO4stK0kIEKLOlKt0X2r3qUH_SsnBbEMjjOMw7MM/edit?usp=sharing)

## Candidate Generation

1. Metrics:
    1. Recall@100
2. Data explore
    1. [x] cold start status
    2. [ ] case study
        2. [x] save eval data and anlyze
        3. [ ] check several user sessions
2. CG
    1. [x] i2i
        1. [x] only keep specific country
        2. [x] train & test data both included 
        3. [ ] include the target in traing data
        3. [ ] fallback logics, unseen key in W2V
    2. [ ] u2i (ALS)
        1. [ ] efficiency
        2. [ ] re-run & save model
        2. [ ] eval recall@100 of model
        2. [ ] item similarity or sth?
    3. next item statistics
        1. [x] include weights in model 
        2. [x] popularity fallback -> low opportunity (less than 1% ) 
            1. [x] check opportunity
            2. [ ] filter by country
            2. [ ] filter by product cate
    4. Next Few Item (NFI)
    3. More co-visit statistics
    3. [ ] popular item in the same category
    
2. Rank
    1. ~~[ ] rule-based as first version~~
    2. Model based
        1. Features
            1. [x] source of candidate generationd
            2. [x] similarity between item sequ (last item) and target item (word2vector)
            2. [x] scores based on next_item_counter 
            3. [x] similarity score between viewed items and current item
        2. Downsampling
            1. [x] fraction -> 0.1 for now
            2. [ ] sample based on session or in all data??
3. Speed
    1. [ ] prev_item similarity -> increase running time by 4 times??
3. Others
    1. Text info
        2. [ ] Text info (text and title info)
        1. [ ] text embedding as features
        3. [x] a unique session ID for both train & test?
        4. [x] there are new items from next_item from train_df??df


## Rank

1. Metrics:
    1. MRR@100
    
# TODO 

1. [x] check previous trivago session-based competition



../data/rank_train_data_v2/train/part_1.parquet
Validating
(28566071, 11)
Memory: 2.23475506529212
0.005130737090165463
shape: (7, 4)
┌────────────┬──────────────┬───────────┬────────────┐
│ describe   ┆ session_id   ┆ rec_cnt   ┆ target_num │
│ ---        ┆ ---          ┆ ---       ┆ ---        │
│ str        ┆ f64          ┆ f64       ┆ f64        │
╞════════════╪══════════════╪═══════════╪════════════╡
│ count      ┆ 146565.0     ┆ 146565.0  ┆ 146565.0   │
│ null_count ┆ 0.0          ┆ 0.0       ┆ 0.0        │
│ mean       ┆ 81754.567209 ┆ 194.90377 ┆ 1.0        │
│ std        ┆ 47232.319874 ┆ 96.523984 ┆ 0.0        │
│ min        ┆ 0.0          ┆ 80.0      ┆ 1.0        │
│ max        ┆ 163580.0     ┆ 468.0     ┆ 1.0        │
│ median     ┆ 81763.0      ┆ 154.0     ┆ 1.0        │
└────────────┴──────────────┴───────────┴────────────┘
  5%|▌         | 1/20 [22:21<7:04:44, 1341.26s/it]

../data/rank_train_data_v2/train/part_2.parquet




# How to run it 

```
# init the shell
poetry shell

# freeze the package config

poetry lock

```