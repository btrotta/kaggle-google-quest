import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from transformers import *
import lightgbm as lgb

# the BERT code is based on the following script:
# https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer

# OPTIONS
short = False  # use only a small sample of train and test set
# END OPTIONS

# read data
train_data = pd.read_csv('/kaggle/input/google-quest-challenge/train.csv')
test_data = pd.read_csv('/kaggle/input/google-quest-challenge/test.csv')
if short:
    train_data = train_data.sample(frac=0.01)
    test_data = test_data.sample(frac=0.01)
train_data.sort_values('question_title', inplace=True)

# preprocessing
max_len = 450
tokenizer = BertTokenizer.from_pretrained('/kaggle/input/download-bert/tokenizer/',
                                          additional_special_tokens=['<END_TITLE>'])


def encode_string(df):
    q_title, q, a = df['question_title'], df['question_body'], df['answer']
    q_enc_dict = tokenizer.encode_plus(q_title + ' <END_TITLE> ' + q, None, max_length=max_len, pad_to_max_length=True,
                                       add_special_tokens=True)
    a_enc_dict = tokenizer.encode_plus(a, None, max_length=max_len, pad_to_max_length=True, add_special_tokens=True)
    return pd.Series([q_enc_dict['input_ids'], q_enc_dict['attention_mask'], q_enc_dict['token_type_ids'],
                      a_enc_dict['input_ids'], a_enc_dict['attention_mask'], a_enc_dict['token_type_ids']])


train_tokens = train_data[['qa_id']].copy()
train_tokens[['q_enc', 'q_mask', 'q_type_ids', 'a_enc', 'a_mask', 'a_type_ids']] = train_data.apply(encode_string,
                                                                                                    axis=1)
test_tokens = test_data[['qa_id']].copy()
test_tokens[['q_enc', 'q_mask', 'q_type_ids', 'a_enc', 'a_mask', 'a_type_ids']] = test_data.apply(encode_string, axis=1)

# prediction arrays
test_pred = pd.DataFrame(index=test_data.index, columns=['qa_id'] + list(train_data.columns[11:41]))
test_pred['qa_id'] = test_data['qa_id'].copy()
test_pred.iloc[:, 1:31] = 0
train_pred = pd.concat([train_data[['qa_id']].copy(), train_data.iloc[:, 11:41].copy()], axis=1)
train_pred['qa_id'] = train_pred['qa_id'].copy()
train_pred.iloc[:, 1:31] = 0

# predict from saved model
for i in range(5):
    K.clear_session()
    config = BertConfig()
    config.output_hidden_states = False
    q_bert_model = TFBertModel.from_pretrained('/kaggle/input/download-bert/model/', config=config)
    q_enc = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    q_mask = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    q_type_ids = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    q_bert = q_bert_model(q_enc, attention_mask=q_mask, token_type_ids=q_type_ids)[0]
    q_bert_summary = tf.keras.layers.Flatten()(tf.keras.layers.AveragePooling1D(max_len)(q_bert))
    a_bert_model = TFBertModel.from_pretrained('/kaggle/input/download-bert/model/')
    a_enc = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    a_mask = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    a_type_ids = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    a_bert = a_bert_model(a_enc, attention_mask=a_mask, token_type_ids=a_type_ids)[0]
    a_bert_summary = tf.keras.layers.Flatten()(tf.keras.layers.AveragePooling1D(max_len)(a_bert))
    comb_bert_summary = tf.keras.layers.Concatenate()([q_bert_summary, a_bert_summary])
    dropout = tf.keras.layers.Dropout(0.2)(comb_bert_summary)
    output = tf.keras.layers.Dense(30, activation='sigmoid')(dropout)
    model = tf.keras.models.Model(inputs=[q_enc, q_mask, q_type_ids, a_enc, a_mask, a_type_ids], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-5), loss='binary_crossentropy')
    model.load_weights('/kaggle/input/bert-5-fold-3-epoch/model_{}/model.h5'.format(i))
    test_pred.iloc[:, 1:31] += model.predict([np.array(list(test_tokens[c].values)) for c in
                                              ['q_enc', 'q_mask', 'q_type_ids', 'a_enc', 'a_mask', 'a_type_ids']])
    train_pred.iloc[:, 1:31] += model.predict([np.array(list(train_tokens[c].values)) for c in
                                               ['q_enc', 'q_mask', 'q_type_ids', 'a_enc', 'a_mask', 'a_type_ids']])

test_pred.iloc[:, 1:31] /= 5
train_pred.iloc[:, 1:31] /= 5

# lgb models to limit number of different values predicted, as discussed here:
# https://www.kaggle.com/c/google-quest-challenge/discussion/118724
submission = pd.DataFrame(index=test_data.index, columns=['qa_id'] + list(train_data.columns[11:41]))
submission['qa_id'] = test_data['qa_id'].copy()
score_cols = train_data.columns[11:41]
for s in score_cols:
    lgb_train = lgb.Dataset(train_pred[[s]], label=train_data[s])
    params = {'objective': 'xentropy', 'num_leaves': 3, 'learning_rate': 0.01, 'seed': 0, 'min_data_in_leaf': 20}
    est = lgb.train(params, lgb_train, num_boost_round=50)
    submission[s] = est.predict(test_pred[[s]])
    # check that column is not all the same
    if submission[s].max() - submission[s].min() < 0.000001:
        max_ind = submission[s].idxmax()
        submission.loc[max_ind, s] = min(0.99999999, submission.loc[max_ind, s] + 0.00001)
        max_ind = submission[s].idxmin()
        submission.loc[max_ind, s] = max(0.00000001, submission.loc[max_ind, s] - 0.00001)

# save submission
submission.to_csv('submission.csv', index=False)
