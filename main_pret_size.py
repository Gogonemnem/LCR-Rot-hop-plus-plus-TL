from models.layers.embedding import BERTEmbedding
from models.transferlearning import TL
from utils.data_loader import load_data
from utils.sampler import sample
import dask.dataframe as dd

import tensorflow as tf
from tensorflow_addons.metrics.f_scores import F1Score


pret_train_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\yelp\pret\train-*.csv'
pret_test_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\yelp\pret\test-*.csv'

ft_train_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\semeval_2015\restaurant\ft\train.csv'
ft_test_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\semeval_2015\restaurant\ft\test.csv'


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

results = []

for size in range(3_000, 33_000, 3_000):
    model = TL(hierarchy=(False, True),
               hidden_units=400,
               embedding_layer=BERTEmbedding(),
               drop_1=0.3,
               drop_2=0.4,
               hop=3,
               regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4))
    batch_size = 16

    f1 = F1Score(num_classes=3, average='macro')
    loss = 'categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # PRET
    x_train, y_train = sample(pret_train_path, size)
    x_val, y_val = load_data(doc_path=pret_test_path)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', f1])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=batch_size, callbacks=[callback])

    # FT
    x_train, y_train = load_data(asp_path=ft_train_path)
    x_val, y_val = load_data(asp_path=ft_test_path)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', f1])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=batch_size, callbacks=[callback])

    result = model.evaluate(x_val, y_val, batch_size=batch_size)
    results.append(result[1:3])

print(results)
