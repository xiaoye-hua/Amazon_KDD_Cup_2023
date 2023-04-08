import polars as pl


def get_recall_at_k(train_pl):
    result_pl = (
         train_pl
            .with_columns(pl.col('next_item').is_in(pl.col('next_item_prediction')).alias('recall@100'))
            # .with_columns(pl.col('next_item_prediction').arr
            #               # .contains(pl.col('next_item'))
            #              )
    )
    recallat100 = round(result_pl['recall@100'].mean(), 3)
    return recallat100

# 0   prev_items            100 non-null    object
#  1   next_item             100 non-null    object
#  2   locale                100 non-null    object
#  3   last_item             100 non-null    object
#  4   next_item_prediction  100 non-null    objec

def pd_get_recall_at_k(row):
    next_item_lst = row['next_item_prediction']
    next_item = row['next_item']
    length = len(next_item_lst)
    recallat20 = next_item in next_item_lst[:min(length, 20)]
    recallat100 = next_item in next_item_lst[:min(length, 100)]
    return length, recallat20, recallat100