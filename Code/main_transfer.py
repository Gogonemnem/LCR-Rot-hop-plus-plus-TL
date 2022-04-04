import sys

import tensorflow as tf

from embedding import BERTEmbedding
import utils
from transferlearningexperimental import TL
from tensorflow_addons.metrics.f_scores import F1Score
from test import sample
import dask.dataframe as dd
from numpy.random import seed


def pre_proc(doc_path: str = None, asp_path: str = None):
    """
    Function that takes in a path to the document data and to the aspect data and returns a dictionary
    suitable for fitting the model.

    Args:
        doc_path: path to the document dataset
        asp_path: path to the aspect dataset

    Returns:
        x,y: dictionaries suitable for training and validating models

    """
    if doc_path is not None and asp_path is not None:
        doc_x, doc_y = utils.csv_to_input(doc_path, ['text', 'polarity'])
        *asp_x, asp_y = utils.csv_to_input(asp_path, ['context_left', 'target', 'context_right', 'polarity'])
        x = {'asp': asp_x, 'doc': doc_x}
        y = {'asp': tf.one_hot(asp_y + 1, 3, dtype='int32'), 'doc': tf.one_hot(doc_y + 1, 3, dtype='int32')}
        return x, y

    if doc_path is not None and asp_path is None:
        doc_x, doc_y = utils.csv_to_input(doc_path, ['text', 'polarity'])
        x = {'doc': doc_x}
        y = {'doc': tf.one_hot(doc_y + 1, 3, dtype='int32')}
        return x, y

    if doc_path is None and asp_path is not None:
        *asp_x, asp_y = utils.csv_to_input(asp_path, ['context_left', 'target', 'context_right', 'polarity'])
        x = {'asp': asp_x}
        y = {'asp': tf.one_hot(asp_y + 1, 3, dtype='int32')}
        return x, y


def make_model(settings: list,
               pret_train_path: str, pret_test_path: str,
               mult_asp_train_path: str, mult_asp_test_path: str,
               mult_doc_train_path: str, mult_doc_test_path: str,
               asp_train_path: str, asp_test_path: str,
               h: dict, pret_model_path: str=None, mult_model_path: str=None):
    """
    Makes a LCR-Rot-Hop++ model with indicated Transfer Learning settings

    Arguments:
        settings: list [PRET, MULT, FT] and can be set to True or False for each option
        data_sizes: list containing the amount of data we want to use for PRET and MULT respectively
        optimizer: the optimizer used in compiling the model
        loss: loss used for compiling the model
    """
    print("Initializing a Transfer Learning Model...")
    ### FT ONLY ###
    # 0 => 0.8580
    # 1 => 0.8641
    # 2 => 0.8672
    # 3 => 0.8580
    # 4 => 0.8702
    # 5 => 0.8718
    # 6 => 0.8595
    # 7 => 0.8656
    # 8 => 0.8855 (epoch 12)
    tf.random.set_seed(10)
    seed(10)
    bs = 16
    loss = 'categorical_crossentropy'
    f1 = F1Score(num_classes=3, average='macro')
    if h is not None:
        optimizer = tf.keras.optimizers.Adam(learning_rate=h['lr'])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    if h is None:
        model = TL(hierarchy=(False, True),
                   hidden_units=300,
                   embedding_layer=BERTEmbedding(),
                   drop_1=0.2,
                   drop_2=0.5,
                   hop=3,
                   regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4))
    else:
        model = TL(hierarchy=(False, True),
                   hidden_units=h['hidden_units'],
                   embedding_layer=BERTEmbedding(),
                   drop_1=h['drop_1'],
                   drop_2=h['drop_2'],
                   hop=3,
                   regularizer=h['regularizer'])
    if mult_model_path is not None:
        print('MULT model given. Transfering weights and continuing from MULT stage...')
        model.load_weights(mult_model_path)
    elif pret_model_path is not None:
        print('PRET model given. Transfering weights and continuing from PRET stage...')
        model.load_weights(pret_model_path)


    # Pretraining Stage
    if settings[0] and pret_model_path is None:
        print("PRET stage started, fetching data...")
        x_train, y_train = pre_proc(doc_path=pret_train_path)
        x_val, y_val = pre_proc(doc_path=pret_test_path)

        print("Fitting Model for PRET...")
        if h is not None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=h['lr'])
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', f1])
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=bs, callbacks=[callback])
        tf.keras.backend.clear_session()
        del x_train, y_train, x_val, y_val, callback
        print("PRET stage completed")

    # Multitask Stage
    if settings[1] and mult_model_path is None:
        print("MULT stage started, fetching data...")
        x_train, y_train = pre_proc(doc_path=mult_doc_train_path, asp_path=mult_asp_train_path)
        x_val, y_val = pre_proc(doc_path=mult_doc_test_path, asp_path=mult_asp_test_path)

        print("Fitting model for MULT...")
        if h is not None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=h['lr'])
            lamb = h['lambda']
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            lamb = 0.5
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.compile(optimizer=optimizer, loss=loss, loss_weights={'asp': 1, 'doc': lamb},
                      metrics=['accuracy', f1])
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=bs, callbacks=[callback])
        tf.keras.backend.clear_session()
        del x_train, y_train, x_val, y_val, callback
        print("MULT stage completed")

    # Finetuning Stage
    if settings[2]:
        print("FT stage started, fetching data...")
        x_train, y_train = pre_proc(asp_path=asp_train_path)
        x_val, y_val = pre_proc(asp_path=asp_test_path)

        print("Fitting model for FT...")
        if h is not None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=h['lr'])
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy', f1])
        model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=200, batch_size=bs, callbacks=[callback])
        tf.keras.backend.clear_session()
        del x_train, y_train, x_val, y_val, callback
        print("FT stage completed")
    print("Model completed and returned. The following stages have been executed: ")
    if settings[0]: print("PRET")
    if settings[1]: print("MULT")
    if settings[2]: print("FT")

    return model


def main():
    # hyper_param_pret = {'regularizer': tf.keras.regularizers.L1L2(l1=1e-08, l2=1e-05),
    #                     'lr': 0.001,
    #                     'drop_1': 0.2,
    #                     'drop_2': 0.3,
    #                     'hidden_units': 350,
    #                     'lambda': 0.3}
    # pret_for_ft_model = make_model(settings=[True, False, False],
    #                                pret_train_path='ExternalData/yelp/train-*.csv',
    #                                pret_test_path='ExternalData/yelp/test-*.csv',
    #                                mult_asp_train_path='ExternalData/rest-mult/train-*.csv',
    #                                mult_asp_test_path='ExternalData/rest-mult/test-*.csv',
    #                                mult_doc_train_path='ExternalData/yelp-mult/train-*.csv',
    #                                mult_doc_test_path='ExternalData/yelp-mult/test-*.csv',
    #                                asp_train_path='ExternalData/sem_train_2016.csv',
    #                                asp_test_path='ExternalData/sem_test_2016.csv',
    #                                h=hyper_param_pret)
    # pret_for_ft_model.save('pret_for_ft')
    # del pret_for_ft_model
    # tf.keras.backend.clear_session()

    # print("Building the PRET+FT model...")
    # pret_ft_model = make_model(settings=[True, False, True],
    #                            pret_train_path='ExternalData/yelp/train-*.csv',
    #                            pret_test_path='ExternalData/yelp/test-*.csv',
    #                            mult_asp_train_path='ExternalData/rest-mult/train-*.csv',
    #                            mult_asp_test_path='ExternalData/rest-mult/test-*.csv',
    #                            mult_doc_train_path='ExternalData/yelp-mult/train-*.csv',
    #                            mult_doc_test_path='ExternalData/yelp-mult/test-*.csv',
    #                            asp_train_path='ExternalData/sem_train_2016.csv',
    #                            asp_test_path='ExternalData/sem_test_2016.csv',
    #                            h=hyper_param_pret,
    #                            pret_model_path='weights/pret_for_ft')
    # pret_ft_model.save_weights('weights_pf/pret_ft-ft')
    # del pret_ft_model
    # tf.keras.backend.clear_session()

    # print("Building the FT model...")
    # ft_model = make_model(settings=[False, False, True],
    #                            pret_train_path='ExternalData/yelp/train-*.csv',
    #                            pret_test_path='ExternalData/yelp/test-*.csv',
    #                            mult_asp_train_path='ExternalData/rest-mult/train-*.csv',
    #                            mult_asp_test_path='ExternalData/rest-mult/test-*.csv',
    #                            mult_doc_train_path='ExternalData/yelp-mult/train-*.csv',
    #                            mult_doc_test_path='ExternalData/yelp-mult/test-*.csv',
    #                            asp_train_path='ExternalData/sem_train_2016.csv',
    #                            asp_test_path='ExternalData/sem_test_2016.csv',
    #                            h=hyper_param_pret)
    # ft_model.save_weights('weights_ft/ft-ft')
    # del ft_model
    # tf.keras.backend.clear_session()

    # print("Building the PRET+MULT+FT model...")
    # pret_mult_ft_model = make_model(settings=[True, True, True],
    #                                 pret_train_path='ExternalData/yelp/train-*.csv',
    #                                 pret_test_path='ExternalData/yelp/test-*.csv',
    #                                 mult_asp_train_path='ExternalData/semeval_2016/restaurant/mult/train-*.csv',
    #                                 mult_asp_test_path='ExternalData/semeval_2016/restaurant/mult/test-*.csv',
    #                                 mult_doc_train_path='ExternalData/yelp/mult/2016/train-*.csv',
    #                                 mult_doc_test_path='ExternalData/yelp/mult/2016/test-*.csv',
    #                                 asp_train_path='ExternalData/semeval_2016/restaurant/ft/train.csv',
    #                                 asp_test_path='ExternalData/semeval_2016/restaurant/ft/test.csv',
    #                                 h=hyper_param_pret,
    #                                 pret_model_path='Weights/weights/pret_for_ft')
    # pret_mult_ft_model.save_weights('weights_pmf/pret_mult_ft-ft')
    # del pret_mult_ft_model
    # tf.keras.backend.clear_session()

    # print("Building the MULT+FT model...")
    # mult_ft_model = make_model(settings=[False, True, True],
    #                                 pret_train_path='ExternalData/yelp/train-*.csv',
    #                                 pret_test_path='ExternalData/yelp/test-*.csv',
    #                                 mult_asp_train_path='ExternalData/semeval_2016/restaurant/mult/train-*.csv',
    #                                 mult_asp_test_path='ExternalData/semeval_2016/restaurant/mult/test-*.csv',
    #                                 mult_doc_train_path='ExternalData/yelp/mult/2016/train-*.csv',
    #                                 mult_doc_test_path='ExternalData/yelp/mult/2016/test-*.csv',
    #                                 asp_train_path='ExternalData/semeval_2016/restaurant/ft/train.csv',
    #                                 asp_test_path='ExternalData/semeval_2016/restaurant/ft/test.csv',
    #                                 h=hyper_param_pret
    #                                 )
    # mult_ft_model.save_weights('weights_mf/mult_ft-ft')
    # del pret_mult_ft_model
    # tf.keras.backend.clear_session()

    print("Making MULT-based models. Getting best PRET...")
    hyper_param_mult = {'regularizer': tf.keras.regularizers.L1L2(l1=1e-05, l2=1e-06),
                        'lr': 0.001,
                        'drop_1': 0.6,
                        'drop_2': 0.5,
                        'hidden_units': 400,
                        'lambda': 0.5}
    # pret_for_mult_model = make_model(settings=[True, False, False],
    #                                  pret_train_path='ExternalData/yelp/train-*.csv',
    #                                  pret_test_path='ExternalData/yelp/test-*.csv',
    #                                  mult_asp_train_path='ExternalData/rest-mult/train-*.csv',
    #                                  mult_asp_test_path='ExternalData/rest-mult/test-*.csv',
    #                                  mult_doc_train_path='ExternalData/yelp-mult/train-*.csv',
    #                                  mult_doc_test_path='ExternalData/yelp-mult/test-*.csv',
    #                                  asp_train_path='ExternalData/sem_train_2016.csv',
    #                                  asp_test_path='ExternalData/sem_test_2016.csv',
    #                                  h=hyper_param_mult)
    # pret_for_mult_model.save_weights('weights_pret/pret_for_mult')
    # del pret_for_mult_model
    #
    print("Building PRET+MULT model...")
    pret_mult_mult_model = make_model(settings=[True, True, False],
                                      pret_train_path='ExternalData/yelp/train-*.csv',
                                      pret_test_path='ExternalData/yelp/test-*.csv',
                                      mult_asp_train_path='ExternalData/semeval_2016/restaurant/mult/train-*.csv',
                                      mult_asp_test_path='ExternalData/semeval_2016/restaurant/mult/test-*.csv',
                                      mult_doc_train_path='ExternalData/yelp/mult/2016/train-*.csv',
                                      mult_doc_test_path='ExternalData/yelp/mult/2016/test-*.csv',
                                      asp_train_path='ExternalData/semeval_2016/restaurant/ft/train.csv',
                                      asp_test_path='ExternalData/semeval_2016/restaurant/ft/test.csv',
                                      h=hyper_param_mult,
                                      pret_model_path='Weights/weights/pret_for_ft')
    pret_mult_mult_model.save_weights('weights_pm-mult/pm-mult')
    del pret_mult_mult_model
    tf.keras.backend.clear_session()
    
    print("Building PRET+MULT+FT model...")
    pret_mult_ft_mult_model = make_model(settings=[True, True, True],
                                         pret_train_path='ExternalData/yelp/train-*.csv',
                                         pret_test_path='ExternalData/yelp/test-*.csv',
                                         mult_asp_train_path='ExternalData/semeval_2016/restaurant/mult/train-*.csv',
                                         mult_asp_test_path='ExternalData/semeval_2016/restaurant/mult/test-*.csv',
                                         mult_doc_train_path='ExternalData/yelp/mult/2016/train-*.csv',
                                         mult_doc_test_path='ExternalData/yelp/mult/2016/test-*.csv',
                                         asp_train_path='ExternalData/semeval_2016/restaurant/ft/train.csv',
                                         asp_test_path='ExternalData/semeval_2016/restaurant/ft/test.csv',
                                         h=hyper_param_mult,
                                         mult_model_path='weights_pm-mult/pm-mult')
    pret_mult_ft_mult_model.save_weights('weights_pmf-mult/pmf-mult')
    del pret_mult_ft_mult_model
    tf.keras.backend.clear_session()

    print("Building MULT model...")
    mult_model = make_model(settings=[False, True, False],
                                         pret_train_path='ExternalData/yelp/train-*.csv',
                                         pret_test_path='ExternalData/yelp/test-*.csv',
                                         mult_asp_train_path='ExternalData/semeval_2016/restaurant/mult/train-*.csv',
                                         mult_asp_test_path='ExternalData/semeval_2016/restaurant/mult/test-*.csv',
                                         mult_doc_train_path='ExternalData/yelp/mult/2016/train-*.csv',
                                         mult_doc_test_path='ExternalData/yelp/mult/2016/test-*.csv',
                                         asp_train_path='ExternalData/semeval_2016/restaurant/ft/train.csv',
                                         asp_test_path='ExternalData/semeval_2016/restaurant/ft/test.csv',
                                         h=hyper_param_mult)
    mult_model.save_weights('weights_mult/mult')
    del mult_model
    tf.keras.backend.clear_session()

    print("Building MULT+FT model...")
    mult_ft_mult_model = make_model(settings=[False, True, True],
                                         pret_train_path='ExternalData/yelp/train-*.csv',
                                         pret_test_path='ExternalData/yelp/test-*.csv',
                                         mult_asp_train_path='ExternalData/semeval_2016/restaurant/mult/train-*.csv',
                                         mult_asp_test_path='ExternalData/semeval_2016/restaurant/mult/test-*.csv',
                                         mult_doc_train_path='ExternalData/yelp/mult/2016/train-*.csv',
                                         mult_doc_test_path='ExternalData/yelp/mult/2016/test-*.csv',
                                         asp_train_path='ExternalData/semeval_2016/restaurant/ft/train.csv',
                                         asp_test_path='ExternalData/semeval_2016/restaurant/ft/test.csv',
                                         h=hyper_param_mult,
                            mult_model_path='weights_mult/mult')
    mult_ft_mult_model.save_weights('weights_mf/mf-mult')
    del mult_ft_mult_model
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
