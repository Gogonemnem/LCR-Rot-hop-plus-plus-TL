from models.layers.embedding import BERTEmbedding
import tensorflow as tf
import keras_tuner as kt
from models.transferlearning import TL
from tensorflow_addons.metrics import F1Score
from utils.data_loader import load_data

def build_mult(hp):
    tf.random.set_seed(0)
    emb = BERTEmbedding()

    # Tune regularizers rate for L1 regularizer with values from 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08 or 1e-09
    hp_l1_rates = hp.Choice("l1_regularizer", values=[10**-i for i in range(3, 10)])

    # Tune regularizers rate for L2 regularizer with values from 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08 or 1e-09
    hp_l2_rates = hp.Choice("l2_regularizer", values=[10**-i for i in range(3, 10)])

    regularizer = tf.keras.regularizers.L1L2(l1=hp_l1_rates, l2=hp_l2_rates)


    # Tune learning rate for Adam optimizer with values from 0.01, 0.001 & 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[10**-i for i in range(2, 5)])

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    # Tune dropout layers with values from 0.2 - 0.7 with stepsize of 0.1.
    drop_rate_1 = hp.Float("dropout_1", 0.2, 0.6, step=0.1)
    drop_rate_2 = hp.Float("dropout_2", 0.2, 0.6, step=0.1)

    # Tune number of hidden layers for the BiLSTMs
    hidden_units = hp.Int("hidden_units", min_value=200, max_value=400, step=50)

    # Tune lambda
    doc_weight = hp.Float("lambda", 0, 1, step=0.25)

    f1 = F1Score(num_classes=3, average='macro', name='f1')

    model = TL(embedding_layer=emb, hop=3, hierarchy=(False, True), drop_1=drop_rate_1, drop_2=drop_rate_2, hidden_units=hidden_units, regularizer=regularizer)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', loss_weights={'asp': 1, 'doc': doc_weight}, metrics=['acc', f1])

    return model


mult_doc_train_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\yelp\mult\2015\train-*.csv'
mult_doc_test_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\yelp\mult\2015\test-*.csv'

mult_asp_train_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\semeval_2015\restaurant\mult\train-*.csv'
mult_asp_test_path = r'C:\Users\gonem\CodeProjects\seminar-ba-qm\ExternalData\semeval_2015\restaurant\mult\test-*.csv'

x_train, y_train = load_data(doc_path=mult_doc_train_path, asp_path=mult_asp_train_path)
x_val, y_val = load_data(doc_path=mult_doc_test_path, asp_path=mult_asp_test_path)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Instantiate the tuner
tuner = kt.Hyperband(build_mult,
                    objective=kt.Objective("val_asp_acc", direction="max"),
                    max_epochs=10,
                    factor=3,
                    hyperband_iterations=2,
                    directory="logs/hp",
                    project_name="mult",)

tuner.search(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, callbacks=[stop_early], verbose=1)

models = tuner.get_best_models(num_models=1)
best_model = models[0]
best_model.evaluate(x_val, y_val)
# This model could be saved and used for further testing
