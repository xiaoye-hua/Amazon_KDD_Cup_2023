from functools import lru_cache
import pandas as pd
import numpy as np
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



def check_predictions(predictions,test_sessions, check_products=False, product_df=None):
    """
    These tests need to pass as they will also be applied on the evaluator
    """
    test_locale_names = test_sessions['locale'].unique()
    for locale in test_locale_names:
        sess_test = test_sessions.query(f'locale == "{locale}"')
        preds_locale =  predictions[predictions['locale'] == sess_test['locale'].iloc[0]]
        assert sorted(preds_locale.index.values) == sorted(sess_test.index.values), f"Session ids of {locale} doesn't match"

        if check_products:
            # This check is not done on the evaluator
            # but you can run it to verify there is no mixing of products between locales
            # Since the ground truth next item will always belong to the same locale
            # Warning - This can be slow to run
            products = product_df.query(f'locale == "{locale}"')
            predicted_products = np.unique( np.array(list(preds_locale["next_item_prediction"].values)) )
            assert np.all( np.isin(predicted_products, products['id']) ), f"Invalid products in {locale} predictions"