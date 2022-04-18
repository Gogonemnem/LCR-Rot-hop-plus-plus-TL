from embedding import BERTEmbedding
import tensorflow as tf
import keras_tuner as kt
from transferlearningexperimental import TL
from tensorflow_addons.metrics import F1Score
import utils

def build_mult(hp):
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


def main():
    yelp_mult_train_path = 'ExternalData/yelp/mult/train-*.csv'
    yelp_mult_test_path = 'ExternalData/yelp/mult/test-*.csv'

    sem_rest_mult_train_path = 'ExternalData/semeval_2016/restaurant/mult/train-*.csv'
    sem_rest_mult_test_path = 'ExternalData/semeval_2016/restaurant/mult/test-*.csv'

    doc_train_x, doc_train_y = utils.csv_to_input(yelp_mult_train_path, ['text', 'polarity'])
    doc_test_x, doc_test_y = utils.csv_to_input(yelp_mult_test_path, ['text', 'polarity'])
    
    *asp_train_x, asp_train_y = utils.csv_to_input(sem_rest_mult_train_path, ['context_left', 'target', 'context_right', 'polarity'])
    *asp_test_x, asp_test_y = utils.csv_to_input(sem_rest_mult_test_path, ['context_left', 'target', 'context_right', 'polarity'])

    x_train = {'asp': asp_train_x, 'doc': doc_train_x}
    y_train = {'asp': tf.one_hot(asp_train_y+1, 3, dtype='int32'), 'doc': tf.one_hot(doc_train_y+1, 3, dtype='int32')}

    x_test = {'asp': asp_test_x, 'doc': doc_test_x}
    y_test = {'asp': tf.one_hot(asp_test_y+1, 3, dtype='int32'), 'doc': tf.one_hot(doc_test_y+1, 3, dtype='int32')}

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # Instantiate the tuner
    tuner = kt.Hyperband(build_mult,
                        objective=kt.Objective("val_asp_acc", direction="max"),
                        max_epochs=10,
                        factor=3,
                        hyperband_iterations=2,
                        directory="logs/hp/mult",
                        project_name="kt_hyperband",)
    
    tuner.search(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, callbacks=[stop_early], verbose=1)

    models = tuner.get_best_models(num_models=1)
    best_model = models[0]
    print(best_model)
    best_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=16, epochs=20, callbacks=[stop_early], verbose=1)
    best_model.save("trained_mult")


if __name__ == "__main__":
    main()