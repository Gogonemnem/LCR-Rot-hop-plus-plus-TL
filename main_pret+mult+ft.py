from models.layers.embedding import BERTEmbedding
from models.transferlearning import TL
from utils.data_loader import load_data

import tensorflow as tf
from tensorflow_addons.metrics.f_scores import F1Score

tf.random.set_seed(0)


pret_train_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\yelp\pret\train-*.csv'
pret_test_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\yelp\pret\test-*.csv'


mult_doc_train_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\yelp\mult\2015\train-*.csv'
mult_doc_test_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\yelp\mult\2015\test-*.csv'

mult_asp_train_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\semeval_2015\restaurant\mult\train-*.csv'
mult_asp_test_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\semeval_2015\restaurant\mult\test-*.csv'


ft_train_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\semeval_2015\restaurant\ft\train.csv'
ft_test_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\semeval_2015\restaurant\ft\test.csv'


model = TL(hierarchy=(False, True),
           hidden_units=300,
           embedding_layer=BERTEmbedding(),
           drop_1=0.2,
           drop_2=0.5,
           hop=3,
           regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)
)

batch_size = 16
lamb = 0.25

f1 = F1Score(num_classes=3, average='macro')
loss = 'categorical_crossentropy'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# PRET
x_train, y_train = load_data(doc_path=pret_train_path)
x_val, y_val = load_data(doc_path=pret_test_path)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', f1])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=batch_size, callbacks=[callback])

# MULT
x_train, y_train = load_data(doc_path=mult_doc_train_path, asp_path=mult_asp_train_path)
x_val, y_val = load_data(doc_path=mult_doc_test_path, asp_path=mult_asp_test_path)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.compile(optimizer=optimizer, loss=loss, loss_weights={'asp': 1, 'doc': lamb},
                      metrics=['accuracy', f1])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=batch_size, callbacks=[callback])

# FT
x_train, y_train = load_data(asp_path=ft_train_path)
x_val, y_val = load_data(asp_path=ft_test_path)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', f1])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=batch_size, callbacks=[callback])