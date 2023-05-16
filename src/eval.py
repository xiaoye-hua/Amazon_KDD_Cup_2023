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




def model_eval(target_df):
    eval_final = (
            target_df
            .lazy()
            .with_columns(
                pl.col('next_item_prediction').cast(pl.List(pl.Utf8))
            )
            .with_columns(
                pl.concat_list([pl.col('next_item'), pl.col('next_item_prediction')]).alias('mrr')
            )
            .with_columns(
                pl.col('mrr').arr.eval(
                    pl.arg_where(pl.element()==pl.element().first())
                )
            ).with_columns(
                pl.col('mrr').arr.eval(
                    pl.when(pl.element()==0).then(0).otherwise(1/pl.element())
                )
            ).with_columns(
                pl.col('mrr').arr.sum()
                , pl.col('next_item_prediction').arr.head(20).arr.contains(pl.col('next_item')).mean().alias('recall@20')
                , pl.col('next_item_prediction').arr.head(100).arr.contains(pl.col('next_item')).mean().alias('recall@100')

            )
    )
    final_res = eval_final.select(
            pl.count().alias('total_sessions')
            , pl.col('mrr').mean().round(4)
            , pl.col('recall@20').mean().round(4)
            , pl.col('recall@100').mean().round(4)

        ).collect()
    return final_res