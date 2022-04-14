from main_transfer import pre_proc
import pandas as pd
from embedding import BERTEmbedding
import dask.dataframe as dd
import tensorflow as tf
from transferlearningexperimental import TL
from numpy.random import seed
from tensorflow_addons.metrics.f_scores import F1Score
import gc
import numpy as np
import utils

n_data = []
accuracy = []
f1_scores = []


def sample(data_path: str, size_sample: int):
    if size_sample % 3 != 0:
        raise ValueError('Balanced data not possible. Please provide a multiple of 3.')
    size = int(size_sample / 3)
    ddf = dd.read_csv(data_path).compute()[['text', 'polarity']].fillna('')
    pos = ddf.loc[ddf['polarity'] == 1]
    neu = ddf.loc[ddf['polarity'] == 0]
    neg = ddf.loc[ddf['polarity'] == -1]

    frac_pos = size / len(pos) + 1e-5
    frac_neu = size / len(neu) + 1e-5
    frac_neg = size / len(neg) + 1e-5

    sample_pos = pos.sample(frac=frac_pos, replace=frac_pos > 1, random_state=0)
    sample_neu = neu.sample(frac=frac_neu, replace=frac_neu > 1, random_state=0)
    sample_neg = neg.sample(frac=frac_neg, replace=frac_neg > 1, random_state=0)

    sample_tot = dd.concat([sample_pos, sample_neu, sample_neg]).sample(frac=1, random_state=0).compute()
    return {'doc': sample_tot['text'].values}, {'doc': tf.one_hot(sample_tot['polarity'].values + 1, 3, dtype='int32')}

####### results ##########
# 30 000: 80.5 - 60.09
# 27 000: 80.0 - 58.12
# 24 000: 0.8067 - val_f1_score: 0.6119
# 21 000: 80.8 - 57.9
# 18 000: 80.0 - 59.19
# 15 000: 80.00 - 57.20
# 12 000: 81.00 - 61.54
#  9 000: 80.17 - 54.20
#  6 000: 79.67 - 56.75
#  3 000: 79.67 - 57.19
#  0: 79.5
f1 = F1Score(num_classes=3, average='macro')
i = 6
number_data = 3000 + i * 3000
print('Fetching data for PRET stage, corpus size = {}'.format(number_data))
# x_doc, y_doc = sample('ExternalData/yelp/train-*.csv', size_sample=number_data)
# x_val, y_val = pre_proc(doc_path='ExternalData/yelp/test-*.csv')
print('Starting PRET stage')
tf.random.set_seed(10)
seed(10)
model = TL(hierarchy=(False, True),
               hidden_units=400,
               embedding_layer=BERTEmbedding(),
               drop_1=0.3,
               drop_2=0.4,
               hop=3,
               regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1])
# model.fit(x_doc, y_doc, validation_data=(x_val, y_val), epochs=100, batch_size=4, callbacks=[callback])
# del x_doc, y_doc, x_val, y_val, callback
# gc.collect()
# tf.keras.backend.clear_session()
#
# print('Fetching data for FT stage.')
# x_train, y_train = pre_proc(asp_path='ExternalData/sem_train_2015.csv')
# x_val, y_val = pre_proc(asp_path='ExternalData/sem_test_2015.csv')
#
# print('Finetuning the pretrained model.')
# callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', f1])
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=4, callbacks=[callback])
# del x_train, y_train
# gc.collect()
# tf.keras.backend.clear_session()
# result = model.evaluate(x_val, y_val, batch_size=2)
# del x_val, y_val, model
# gc.collect()
# tf.keras.backend.clear_session()
# print('Accuracy: ', result[1])
# print('f1: ', result[2])
# print('PRET corpus size: ', number_data)
# del result
# gc.collect()
# tf.keras.backend.clear_session()
