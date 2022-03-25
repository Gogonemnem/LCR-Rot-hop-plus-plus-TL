import tensorflow as tf

import embedding
import utils
from transferlearningexperimental import TL
from test import sample


def main():
    # Data Locations
    training_path = "ExternalData/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml"
    validation_path = "ExternalData/ABSA15_Restaurants_Test.xml"
    doc_path = ""

    train_data_path = "ExternalData/sem_train_2015.csv"
    test_data_path = "ExternalData/sem_test_2015.csv"

    emb = embedding.BERTEmbedding()

    # Turns the XML files to the CSV files
    utils.semeval_to_csv(training_path, train_data_path)
    utils.semeval_to_csv(validation_path, test_data_path)

    *sentences, polarity = utils.semeval_data(train_data_path)
    asp_x_train = {'asp': sentences}
    asp_y_train = {'asp': tf.one_hot(polarity+1, 3)}
    
    *sentences, polarity = utils.semeval_data(test_data_path)
    asp_x_test = {'asp': sentences}
    asp_y_test = {'asp': tf.one_hot(polarity+1, 3)}

    import dask.dataframe as dd
    _, x, y = dd.read_csv('ExternalData/yelp/*.csv').compute()[:3000].T.values
    doc_x_train = {'doc': x}
    doc_y_train = {'doc': tf.one_hot(y+1, 3)}
    doc_x_test = {'doc': 0}
    doc_y_test = {'doc': 0}

    ##### Make it so you can input different combination of data?
    # so for example pret -> only {'asp': ...} and ft -> only {'doc': ...}

    # empty_asp_train = np.full(tf.shape(asp_y_train), np.nan)
    # empty_doc_train = np.full(tf.shape(doc_y_train), np.nan)
    # empty_asp_test = np.full(tf.shape(asp_y_test), np.nan)
    # empy_doc_test = np.full(tf.shape(doc_y_test), np.nan)

    # pret_x_train = {'asp': empty_asp_train, 'doc': doc_x_train}
    # pret_x_test = {'asp': empty_asp_test, 'doc': doc_x_test}
    # pret_y_train = {'asp': empty_asp_train, 'doc': doc_y_train}
    # pret_y_test = {'asp': empty_asp_test, 'doc': doc_y_test}

    
    import dask.dataframe as dd
    df_asp = dd.read_csv('ExternalData\sem_train_2015.csv', dtype={'polarity': 'object'})
    *asp_x, asp_y = sample(df_asp, size=3000, balance=False).compute().fillna('').T.values

    df_doc: dd.DataFrame = dd.read_csv('ExternalData/yelp/*.csv', dtype={'polarity': 'object'})
    _, doc_x, doc_y = sample(df_doc, size=1000, balance=True).compute().T.values
    mult_x_train = {'asp': asp_x, 'doc': doc_x}
    # mult_x_test = 
    mult_y_train = {'asp': tf.one_hot(asp_y.astype(int)+1, 3), 'doc': tf.one_hot(doc_y.astype(int)+1, 3)}
    # mult_y_test = asp_y_test | doc_y_test

    # ft_x_train = {'asp': asp_x_train, 'doc': empty_doc_train}
    # ft_x_test = {'asp': asp_x_test, 'doc': empy_doc_test}
    # ft_y_train = {'asp': asp_y_train, 'doc': empty_doc_train}
    # ft_y_test = {'asp': asp_y_test, 'doc': empy_doc_test}

    # initialize informally, 
    # TODO: improve compilation
    model = TL(hidden_units=300, embedding_layer=emb)
    
    # compile at each stage because input changes
    # pret
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(doc_x_train, doc_y_train, epochs=1, batch_size=32)
    # TODO: save best model

    # TODO: load best previous model if exists
    # # mult
    model.compile(optimizer='adam', loss='categorical_crossentropy', loss_weights = {'asp': 1, 'doc': 0.5}, metrics=['accuracy'])
    model.fit(mult_x_train, mult_y_train, epochs=5, batch_size=32)

    # ft
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(asp_x_train, asp_y_train, validation_data=(asp_x_test, asp_y_test), epochs=200, batch_size=32)


if __name__ == '__main__':
    main()