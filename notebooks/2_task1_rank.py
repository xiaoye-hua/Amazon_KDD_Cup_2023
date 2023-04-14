#!/usr/bin/env python
# coding: utf-8

# # Packages 

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')
import sys
import logging
base_dir = '../'
sys.path.append(base_dir)
import os
from utils import *

import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.similarities.annoy import AnnoyIndexer


from annoy import AnnoyIndex
import polars as pl
import implicit
import scipy.sparse as sps
from utils import str2list
from src.config import raw_data_session_id_dir, candidate_file_name
from lightgbm import LGBMRanker
from utils import *


# # Config 

# In[2]:


candidate_file_name


# In[3]:


debug = False

if debug:
    SAMPLE_NUM = 100
else:
    SAMPLE_NUM = None

candidate_path = '../data/candidates/'
model_dir = '../model_training'
# train_data_dir = '.'
# test_data_dir = '.'
task = 'task1'
w2v_model = 'w2v_v3'
nic_model = 'nic'
rank_model_version = 'rank_lgbm'

rank_model_dir = os.path.join(model_dir, rank_model_version)
model_for_eval = True
w2v_topn=100
nic_topn=100
# PREDS_PER_SESSION = 100

# num_tree = 100



# # target locales: locales needed for task1
target_locals = ["DE", 'JP', 'UK']

# submit_file = f'submission_{task}_ALS.parquet'
num_tree = 100
w2v_model_dir = os.path.join(model_dir, w2v_model)
w2v_model_file = os.path.join(w2v_model_dir, f"{model_for_eval}.model")
annoy_index_file = os.path.join(w2v_model_dir, f"{str(num_tree)}_{model_for_eval}.index")


# In[4]:


# get_ipython().system(' mkdir {rank_model_dir}')


# In[5]:


rank_model_dir


# In[6]:


# model_dir


# # Function 

# In[7]:


# get_ipython().system(' ls ../data/candidates/')


# # Original data 

# In[8]:


eval_pl = pl.scan_parquet(os.path.join(base_dir, raw_data_session_id_dir, 'sessions_eval.parquet'), n_rows=SAMPLE_NUM).filter(pl.col('locale').is_in(target_locals)).with_columns(pl.col('prev_items').apply(str2list))

# df_sess.head(3).collect()
test_pl = pl.scan_parquet(os.path.join(base_dir, raw_data_session_id_dir, 'sessions_test_task1.parquet'), n_rows=SAMPLE_NUM).with_columns(pl.col('prev_items').apply(str2list))


# # Eval candidate Generation 

# In[9]:


# w2v_file = os.path.join(candidate_path, f'task1_{data_type}_w2v_top100.parquet')
# w2v_pl = pl.scan_parquet(w2v_file, n_rows=SAMPLE_NUM)#.with_columns(pl.col('prev_items').apply(str2list))


# In[10]:


candidate_file_name


# In[ ]:





# In[11]:


def get_all_candidates(data_type, task=task, 
                       w2v_model=w2v_model, 
                       nic_model=nic_model
                      ,model_for_eval=model_for_eval,
                      w2v_topn=w2v_topn
                      , nic_topn=nic_topn):
    
    w2v_file = os.path.join(candidate_path, 
                           candidate_file_name.format(
                    task=task
                , data_type=data_type
                , model_version=w2v_model
                , model_for_eval=model_for_eval
                , topn=w2v_topn
                           ))
    nic_file = os.path.join(candidate_path,
                candidate_file_name.format(
                    task=task
                    , data_type=data_type
                    , model_version=nic_model
                    , model_for_eval=model_for_eval
                    , topn=nic_topn
                           ))
    if data_type == 'test':
        original_file_name = f"sessions_{data_type}_{task}.parquet"
    else:
        original_file_name = f"sessions_{data_type}.parquet"
    original_pl = pl.scan_parquet(os.path.join(base_dir, raw_data_session_id_dir, original_file_name), n_rows=SAMPLE_NUM).filter(pl.col('locale').is_in(target_locals)).with_columns(pl.col('prev_items').apply(str2list))
    w2v_pl = pl.scan_parquet(w2v_file, n_rows=SAMPLE_NUM)#.with_columns(pl.col('prev_items').apply(str2list))
    nic_pl = pl.scan_parquet(nic_file, n_rows=SAMPLE_NUM)#.with_columns(pl.col('prev_items').apply(str2list))
    
    # get w2v weight
    w2v_pl = w2v_pl.with_columns(
        pl.lit([list(range(w2v_topn, 0, -1))]).alias('w2v_weight')
    )
    get_w2v_weight = pl.element().rank()*0
    nic_pl = nic_pl.with_columns(
        pl.col('next_item_prediction').arr.eval(get_w2v_weight, parallel=True).alias('w2v_weight').cast(pl.List(pl.Int64))
    )
    cols = ['session_id', 'next_item_prediction', 'w2v_weight']
    # combined_pl = (
    #     w2v_pl.select(cols).join(nic_pl.select(cols), how='left', on='session_id', suffix='_nic')
    # )
    combined_pl = (
            pl.concat([w2v_pl.select(cols), nic_pl.select(cols)], how='vertical')
                .groupby('session_id')
                .agg(
                    pl.col('next_item_prediction').flatten()
                    , pl.col('w2v_weight').flatten()
                )
    )
    combined_pl = (
        combined_pl.join(original_pl, how='left', on='session_id')
            # .with_columns(
            #     pl.col('prev_items')
            # )
    )
    return combined_pl


# In[12]:


train_cg_pl = get_all_candidates(data_type='train')
eval_cg_pl = get_all_candidates(data_type='eval')
test_cg_pl = get_all_candidates(data_type='test')


# In[ ]:


print('Train.....')
print(train_cg_pl.collect().shape)


# In[ ]:

print('Eval.....')

print(eval_cg_pl.collect().shape)


# In[ ]:


# print(f"{train_cg_pl.collect().shape}; {eval_cg_pl.collect().shape}; {test_cg_pl.collect().shape}")


# In[ ]:


# train_cg_pl.collect().head()


# In[ ]:


# eval_cg_pl.collect()


# In[ ]:





# # In[ ]:


# # get_w2v_weight = pl.element().rank()*0

# # train_pl.with_columns(
# #     # pl.lit([list(range(100, 0, -1))]).alias('w2v_weight')
# #     pl.col('next_item_prediction').arr.lengths().alias('length')
# #     , pl.col('next_item_prediction').arr.eval(get_w2v_weight, parallel=True).alias('w2v_weight')
# #     # , pl.lit([0 for ele in range(pl.col('length'))])

# # ).head().collect()


# # In[ ]:


# # dir(pl.element())


# # In[ ]:





# # In[ ]:


# # w2v_pl.head().collect()


# # In[ ]:


# # def get_all_candidates(data_type):
# #     w2v_file = os.path.join(candidate_path, f'task1_{data_type}_w2v_top100.parquet')
# #     nic_file = os.path.join(candidate_path, f'task1_{data_type}_nic_top100.parquet')
# #     w2v_pl = pl.scan_parquet(w2v_file, n_rows=SAMPLE_NUM)#.with_columns(pl.col('prev_items').apply(str2list))
# #     nic_pl = pl.scan_parquet(nic_file, n_rows=SAMPLE_NUM)#.with_columns(pl.col('prev_items').apply(str2list))

# #     cols_to_keep = ['session_id', 'next_item_prediction', 'cg_source']

# #     w2v_pl = w2v_pl.with_columns(pl.lit('word2vec').alias('cg_source')).select(cols_to_keep).explode('next_item_prediction')
# #     nic_pl = nic_pl.with_columns(pl.lit('next_item_counter').alias('cg_source')).select(cols_to_keep).explode('next_item_prediction')
# #     # length = w2v_pl.collect().shape[0]
# #     # w2v_pl = w2v_pl.with_columns(
# #     #     pl.Series(values=[list(range(100, 0, -1))]*length, name='w2v_weight')
# #     # )
# #     cg_pl = (
# #             pl.concat([w2v_pl, nic_pl], how='vertical')
# #                 .groupby(['session_id', 'next_item_prediction'])
# #                 .agg(
# #                     pl.col('cg_source')
# #                 )
# #                 .with_columns(
# #                     pl.when(pl.col('cg_source').arr.contains(pl.lit('word2vec'))).then(1).otherwise(0).alias('whether_w2v')
# #                     , pl.when(pl.col('cg_source').arr.contains(pl.lit('next_item_counter'))).then(1).otherwise(0).alias('whether_nic')
# #                 )
# #                 .select(
# #                     pl.all().exclude('cg_source')
# #                 )
# #     )
# #     return cg_pl


# # ## Eval candidate generation 

# # In[ ]:


# eval_topn = 300

# col = f"recall@{eval_topn}"

# eval_final = (
#         eval_cg_pl
#         .lazy()
#         .with_columns(
#             pl.col('next_item_prediction').cast(pl.List(pl.Utf8))
#         )
#         .with_columns(
#             pl.concat_list([pl.col('next_item'), pl.col('next_item_prediction')]).alias('mrr')
#         )
#         ).with_columns(
#             pl.col('next_item_prediction').arr.head(eval_topn).arr.contains(pl.col('next_item')).mean().alias(col)

#         )
# final_res = eval_final.select(
#         pl.count().alias('total_sessions')
#         , pl.col(col).mean()

#     ).collect()
# final_res


# # In[ ]:


# # eval_cg_pl.head().collect()


# # # Feature Process 

# # ## Load Model 

# # In[ ]:


# nic_model = (
#     pl.scan_parquet('../model_training/next_item_counter_v2/nic_model.parquet')
#         .explode(['next_item_prediction', 'next_item_weight'])
#         .select(
#             pl.all().exclude('item')
#             , pl.col('item').alias('last_prev_item')
#         )
#             )
# print(nic_model.schema)



# # w2v_model_file = '../model_training/v2/w2v.model'
# w2vec = Word2Vec.load(w2v_model_file)
# annoy_index = AnnoyIndexer()
# annoy_index.load(annoy_index_file)


# # In[ ]:


# w2v_model_file


# # In[ ]:


# # len(w2vec.wv)


# # In[ ]:


# # train_cg_pl.schema


# # In[ ]:


# # w2vec.wv.similarity('2', '3')


# # In[ ]:


# # target_df = train_candidates
# # data_type = 'train'

# def get_w2v_simi(x):
#     try:
#         simi = w2vec.wv.similarity(x['next_item_prediction'],
#                                                   x['last_prev_item']
#                                                  )
#     except:
#         simi = 0
#     return simi

# def get_feature(target_df, data_type):
#     # if data_type == 'train':
#     #     basic_info_pl = pl.scan_parquet(os.path.join(base_dir, raw_data_session_id_dir, 'sessions_train.parquet'))
#     # else:
#     #     basic_info_pl = pl.scan_parquet(os.path.join(base_dir, raw_data_session_id_dir, 'sessions_test_task1.parquet'))
#     # basic_info_pl = basic_info_pl.with_columns(
#     #         pl.col('prev_items').apply(str2list)
#     #     )
#     # print(f"basci_info_pl:")
#     # print(basic_info_pl.schema)
#     # nic model 

#     target_df = (
#         target_df.explode(['next_item_prediction', 'w2v_weight'])
#         .with_column(
#                 pl.col('prev_items').arr.get(-1).alias('last_prev_item')
#             )
#             .join(nic_model, how='left', on=['last_prev_item', 'next_item_prediction'])
#             .with_columns(
#                 pl.when(pl.col('next_item_weight').is_null()).then(0).otherwise(pl.col('next_item_weight')).alias('next_item_weight')
#                 , pl.struct(["next_item_prediction", "last_prev_item"]).apply(
#                     lambda x: get_w2v_simi(x)).alias('last_item_similarity')
#                 , pl.when(pl.col('locale')=='DE').then(1).when(pl.col('locale')=='DE')
#                     .then(2)
#                     .otherwise(3).alias('locale')
#             ).sort('session_id')
#     )
#     if data_type != 'test':
#         target_df = (
#             target_df
#                 .with_columns(
#                     pl.when(pl.col('next_item')==pl.col('next_item_prediction')).then(1).otherwise(0).alias('target')
#                 )
#         )
#     return target_df
# # target_df.head(3).collect()


# # In[ ]:


# train_candidates = get_feature(target_df=train_cg_pl, data_type='train')


# # In[ ]:


# # train_candidates.head().collect()


# # In[ ]:


# eval_candidates = get_feature(target_df=eval_cg_pl, data_type='eval')


# # In[ ]:


# # eval_candidates.collect()


# # In[ ]:





# # In[ ]:


# # target_df = eval_cg_pl
# # data_type = 'eval'
# # (
# #     target_df.explode(['next_item_prediction', 'w2v_weight'])
# #     .with_column(
# #             pl.col('prev_items').arr.get(-1).alias('last_prev_item')
# #         )
# #         .join(nic_model, how='left', on=['last_prev_item', 'next_item_prediction'])
# #         .with_columns(
# #             pl.when(pl.col('next_item_weight').is_null()).then(0).otherwise(pl.col('next_item_weight')).alias('next_item_weight')
# #             # , pl.struct(["next_item_prediction", "last_prev_item"]).apply(
# #             #     lambda x: w2vec.wv.similarity(x['next_item_prediction'],
# #             #                                   x['last_prev_item']
# #             #                                  )).alias('last_item_similarity')
# #             , pl.when(pl.col('locale')=='DE').then(1).when(pl.col('locale')=='DE')
# #                 .then(2)
# #                 .otherwise(3).alias('locale')
# #         ).sort('session_id')
# # ).head().collect()
# # # if data_type != 'test':
# # #     target_df = (
# # #         target_df
# # #             .with_columns(
# # #                 pl.when(pl.col('next_item')==pl.col('next_item_prediction')).then(1).otherwise(0).alias('target')
# # #             )
# # #     )
# # # target_df.head().collect()


# # In[ ]:


# # eval_candidates.head().collect()


# # In[ ]:


# # test_candidates.collect()


# # In[ ]:


# # test_candidates.head().collect()


# # In[ ]:





# # # Ranker 

# # ## Config 

# # In[ ]:


# estimator = 10
# if debug:
#     estimator = 3
    
# target = 'target'
# feature_cols = [
#     # 'whether_w2v'
#     # , 'whether_nic'
#     'next_item_weight'
#     , 'locale'
#     , 'w2v_weight'
#     , 'last_item_similarity'
# ]

# categorical_feature=['locale']


# # In[ ]:


# # train_candidates.head().collect()[feature_cols + [target]]


# # ## Train model

# # In[ ]:


# # train_candidates = train_candidates.sort('session_id')


# # .head(100).collect()


# # In[ ]:


# ranker = LGBMRanker(
#     objective="lambdarank",
#     metric="ndcg",
#     boosting_type="dart",
#     n_estimators=estimator, 
#     importance_type='gain',
#     eval_at=[5]
# )


# # In[ ]:


# train_candidates = train_candidates.collect()
# train_candidates.shape


# # In[ ]:


# print(train_candidates.select('target').mean())


# # In[41]:


# eval_candidates = eval_candidates.collect()
# eval_candidates.shape


# # In[42]:


# print(eval_candidates.select('target').mean())


# # In[43]:


# session_lengths_train = train_candidates.groupby('session_id').agg(pl.count()).select('count').to_pandas()['count'].values.tolist()
# session_lengths_eval = eval_candidates.groupby('session_id').agg(pl.count()).select('count').to_pandas()['count'].values.tolist()


# # In[44]:


# ranker = ranker.fit(
#     X=train_candidates[feature_cols].to_pandas(),
#     y=train_candidates[target].to_pandas(),
#     group=session_lengths_train,
#     eval_set=[(train_candidates[feature_cols].to_pandas(), train_candidates[target].to_pandas()),
#            (eval_candidates[feature_cols].to_pandas(), eval_candidates[target].to_pandas())
#              ],
#     eval_group=[session_lengths_train,
#                 session_lengths_eval
#                ]
#     , categorical_feature=categorical_feature
#     # , early_stopping_rounds=
# )


# # ## Save model 

# # In[45]:


# import joblib


# # In[46]:


# joblib.dump(
#         value=ranker,
#         filename=os.path.join(rank_model_dir, 'model.pkl')
# )


# # ## Load Model 

# # In[47]:


# del ranker


# # In[48]:


# ranker = joblib.load(os.path.join(rank_model_dir, 'model.pkl'))
# ranker


# # ### Importance 

# # In[49]:


# impotant_df = pd.DataFrame(
#     {
#         'features': ranker.feature_name_,
#         'importance': ranker.feature_importances_
#     }
# ).sort_values('importance', ascending=False)
# impotant_df


# # # Test inference 

# # In[50]:


# test_candidates = get_feature(target_df=test_cg_pl, data_type='test')


# # In[51]:


# test_candidates = test_candidates.collect()
# test_candidates.shape


# # In[52]:


# inference = ranker.predict(test_candidates[feature_cols].to_pandas())


# # In[53]:


# test_result = (test_candidates
#      .lazy()
#      .with_columns(
#          pl.Series(name='predict', values=inference)
#      )
#      .with_columns(
#         pl.col('predict').rank(method='ordinal', descending=True).over('session_id').alias('rank')
#      )
#      .sort(['session_id', 'rank'])
#      .filter(pl.col('rank')<=100)
#      .groupby(['session_id'])
#      .agg(
#          pl.col('next_item_prediction')
#      )
# )
# # test_result.head().collect()


# # In[55]:


# test_pl.schema


# # In[56]:


# predictions = test_pl.join(test_result, how='left', on='session_id').collect()[['locale', 'next_item_prediction']].to_pandas()


# # In[57]:


# predictions.shape


# # In[58]:


# # original_test.head()


# # In[60]:


# check_predictions(predictions, test_sessions=test_pl.collect().to_pandas(), 
#                   # check_products=True, product_df=products
#                  )
# # Its important that the parquet file you submit is saved with pyarrow backend
# predictions.to_parquet('../data/sub_files/rank_v3.parque', engine='pyarrow')


# # In[46]:


# # # You can submit with aicrowd-cli, or upload manually on the challenge page.
# # !aicrowd submission create -c task-1-next-product-recommendation -f "../data/sub_files/rank_v3.parque"


# # In[ ]:




