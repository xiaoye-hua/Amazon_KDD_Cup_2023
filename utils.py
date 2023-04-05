from functools import lru_cache
import pandas as pd
import os

# Cache loading of data for multiple calls

@lru_cache(maxsize=1)
def read_product_data(train_data_dir):
    return pd.read_csv(os.path.join(train_data_dir, 'products_train.csv'))

@lru_cache(maxsize=1)
def read_train_data(train_data_dir):
    return pd.read_csv(os.path.join(train_data_dir, 'sessions_train.csv'))

@lru_cache(maxsize=3)
def read_test_data(task, test_data_dir):
    return pd.read_csv(os.path.join(test_data_dir, f'sessions_test_{task}.csv'))


def process_item_lst(row):
    prev_items = row['prev_items']
    res = [ele.replace('[', '').replace(']', '').replace('\n', '').replace("'", '').replace(' ', '') for ele in prev_items.split(' ')]
    return res