import os

data_dir = 'data/'
eval_data_dir = os.path.join(data_dir, 'eval_data')

# if True, the label of eval data will not be included in training data
model_for_eval = True 

raw_data_dir = os.path.join(data_dir, 'raw_data')
raw_data_session_id_dir = os.path.join(data_dir, 'raw_data_session_id')
candidate_dir = os.path.join(data_dir, 'candidates')


candidate_file_name = "{task}_{data_type}_{model_version}_{model_for_eval}_top{topn}.parquet"
submit_file_name = "submit_{task}_{model_version}_{model_for_eval}_top{topn}.parquet"


# # word2vector model
# w2v_model_file_name = '{model_version}_{model_for_eval}.model'
# w2v_index_file_name = '{model_version}_{num_tree}_{model_for_eval}.index'