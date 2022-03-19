import datetime
import pandas as pd

import tensorflow as tf

from haabsamodel import HAABSA
import utils


def main():
    embedding_path = "ExternalData/glove.6B.300d.txt"
    training_path = "ExternalData/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml"
    validation_path = "ExternalData/ABSA15_Restaurants_Test.xml"

    train_data_path = "ExternalData/sem_train_2015.csv"
    test_data_path = "ExternalData/sem_test_2015.csv"

    # histogram_freq needs to be zero when working with GloVe embeddings!
    # is a bug in keras https://github.com/tensorflow/tensorflow/issues/41244
    log_dir = "Code/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0, update_freq='batch')
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=log_dir,
                                                     save_weights_only=True,
                                                     verbose=1)

    utils.semeval_to_csv(training_path, train_data_path)
    utils.semeval_to_csv(validation_path, test_data_path)

    # Data processing
    left, target, right, polarity = utils.semeval_data(train_data_path)
    x_train = [left, target, right]
    y_train = tf.one_hot(polarity+1, 3, dtype='int64')

    left, target, right, polarity = utils.semeval_data(test_data_path)
    x_test = [left, target, right]
    y_test = tf.one_hot(polarity+1, 3, dtype='int64')

    # Specify loss function
    # loss='categorical_crossentropy' does not work properly, not properly scaled to regulizers
    cce = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM)

    # Model call
    haabsa = HAABSA([training_path, validation_path],
                    embedding_path, hop=1, hierarchy=None, regularizer=tf.keras.regularizers.L2(
                        l2=0.00001))

    # adam is a optimizer just like Stochastic Gradient Descent
    haabsa.compile('adam',  # tf.keras.optimizers.SGD(learning_rate=0.07, momentum=0.95),
                   loss=cce, metrics=[
                       'categorical_accuracy'], run_eagerly=False)  # TODO:run_eagerly off when done!

    # pretrained or not -> Loads the weights
    # haabsa.load_weights(checkpoint_path)

    haabsa.fit(x_train, y_train, validation_data=(
        x_test, y_test), epochs=10, batch_size=32,
        callbacks=[tensorboard_callback, cp_callback])
    # print(haabsa.summary())

    # just for us to debug the predictions
    predictions = haabsa.predict(x_test)
    print(predictions)
    pd.DataFrame(predictions).to_csv('Code/logs/predictions.csv')


if __name__ == '__main__':
    # import logging

    # # Create and configure logger
    # logging.basicConfig(filename="newfile.log",
    #                     format='%(asctime)s %(message)s',
    #                     filemode='w')

    # # Creating an object
    # logger = logging.getLogger()

    # # Setting the threshold of logger to DEBUG
    # logger.setLevel(logging.DEBUG)

    main()
