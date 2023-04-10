# Basic idea

Compared to [OTTO – Multi-Objective Recommender System competition](https://www.kaggle.com/competitions/otto-recommender-system/data)

1. might only one action
2. only one item as target, might be possible to extend the training data


# Plans & TODO 

1. Structure 
    1. Candidate generation
    2. Rank
        1. [ ] rule-based as first version
2. [ ] Text info (text and title info)
    1. [ ] text embedding as features
 
# Two Stages Approach

[Experiment tracking spreadsheet](https://docs.google.com/spreadsheets/d/1f9faO4stK0kIEKLOlKt0X2r3qUH_SsnBbEMjjOMw7MM/edit?usp=sharing)

## Candidate Generation

1. Metrics:
    1. Recall@100
2. Plans & steps
    1. [x] i2i
        1. [x] only keep specific country
        2. [x] train & test data both included 
        3. [ ] include the target in traing data
        3. [ ] fallback logics
    2. [x] u2i (ALS)
        1. [ ] re-run & save model
        2. [ ] eval recall@100 of model
        2. [ ] item similarity or sth?
    3. next item statistics
        1. [ ] include weights in model 
        2. [ ] popularity fallback -> low opportunity (less than 1% )
            1. [ ] check opportunity
            2. [ ] filter by country
            2. [ ] filter by product cate
    3. [ ] popular item in the same category
    
2. Rank
    1. [ ] rule-based as first version
    2. Model based
        1. Features
            1. [x] source of candidate generationd
            2. [ ] similarity between item sequ (last item) and target item (word2vector)
            2. [ ] scores based on next_item_counter 
3. Others
    1. Text info
        2. [ ] Text info (text and title info)
        1. [ ] text embedding as features
        3. [ ] a unique session ID for both train & test?
        4. [ ] there are new items from next_item from train_df??df


## Rank

1. Metrics:
    1. MRR@100
    
# TODO 

1. [ ] check previous trivago session-based competition

