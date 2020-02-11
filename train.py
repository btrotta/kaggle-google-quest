import os
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import *
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import tensorflow.keras.backend as K

# the BERT code is based on the following script:
# https://www.kaggle.com/akensert/bert-base-tf2-0-now-huggingface-transformer

# OPTIONS
short = False  # use only a small sample of train and test set
# END OPTIONS

# read data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
if short:
    train_data = train_data.sample(frac=0.01)
    test_data = test_data.sample(frac=0.01)

# preprocessing
max_len = 450
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
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

# identify question and answer cols
score_cols = train_data.columns[11:41]
q_scores = score_cols[:21]
a_scores = score_cols[21:]

# model
cv = GroupKFold(5)
for i, (train_ind, test_ind) in enumerate(cv.split(train_tokens, groups=train_data['question_title'])):
    train_bool = train_tokens.index.isin(train_tokens.iloc[train_ind].index)
    K.clear_session()
    config = BertConfig()
    config.output_hidden_states = False
    q_bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
    q_enc = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    q_mask = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    q_type_ids = tf.keras.layers.Input((max_len,), dtype=tf.int32)
    q_bert = q_bert_model(q_enc, attention_mask=q_mask, token_type_ids=q_type_ids)[0]
    q_bert_summary = tf.keras.layers.Flatten()(tf.keras.layers.AveragePooling1D(max_len)(q_bert))
    a_bert_model = TFBertModel.from_pretrained('bert-base-uncased', config=config)
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
    model.fit([np.array(list(train_tokens.loc[train_bool, c].values)) for c in
               ['q_enc', 'q_mask', 'q_type_ids', 'a_enc', 'a_mask', 'a_type_ids']],
              train_data.loc[train_bool, score_cols].values, epochs=3, verbose=2,  batch_size=6)
    if not os.path.exists('model_{}'.format(i)):
        os.mkdir('model_{}'.format(i))
    model.save_weights(os.path.join('model_{}'.format(i), 'model.h5'))

