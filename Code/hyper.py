import tensorflow as tf
import kerastuner as kt
from haabsamodel import HAABSA
import utils

# useful links
# https://towardsdatascience.com/hyperparameter-tuning-with-kerastuner-and-tensorflow-c4a4d690b31a
# https://medium.com/swlh/hyperparameter-tuning-in-keras-tensorflow-2-with-keras-tuner-randomsearch-hyperband-3e212647778f


def build_model(hp):

    # Tune regularizers rate for L1 regularizer with values from 0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08 or 1e-09
    hp_l1_rates = hp.Choice("l1_regularizer", values=[10**-i for i in range(1, 10)])

    # Tune regularizers rate for L2 regularizer with values from 0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08 or 1e-09
    hp_l2_rates = hp.Choice("l2_regularizer", values=[10**-i for i in range(1, 10)])

    regularizer = tf.keras.regularizers.L1L2(l1=hp_l1_rates, l2=hp_l2_rates)


    # Tune learning rate for Adam optimizer with values from 0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08 or 1e-09
    hp_learning_rate = hp.Choice("learning_rate", values=[10**-i for i in range(1, 10)])

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    # Tune dropout layers with values from 0 - 0.7 with stepsize of 0.1.
    drop_rate_1 = hp.Float("dropout_1", 0, 0.7, step=0.1)
    drop_rate_2 = hp.Float("dropout_2", 0, 0.7, step=0.1)

    # Tune number of hidden layers for the BiLSTMs
    hidden_units = hp.Int("hidden_units", min_value=200, max_value=400, step=50)

    # Initialize model.
    model = HAABSA([training_path, validation_path],
                    embedding_path, hop=1, hierarchy=None, drop_1=drop_rate_1, drop_2=drop_rate_2, hidden_units=hidden_units, regularizer=regularizer)

    # loss='categorical_crossentropy' works here, bc hyperparameter tuning
    # adam is a optimizer just like Stochastic Gradient Descent
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # Data processing
    left, target, right, polarity = utils.semeval_data(train_data_path)
    x_train = [left, target, right]
    y_train = tf.one_hot(polarity+1, 3, dtype='int64')

    left, target, right, polarity = utils.semeval_data(test_data_path)
    x_test = [left, target, right]
    y_test = tf.one_hot(polarity+1, 3, dtype='int64')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    # Instantiate the tuner
    tuner = kt.Hyperband(build_model,
                        objective="val_accuracy",
                        max_epochs=20,
                        factor=3,
                        hyperband_iterations=10,
                        directory="logs/fit",
                        project_name="kt_hyperband",)

    tuner.search(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=64, callbacks=[stop_early], verbose=1)
    
    # Get the optimal hyperparameters from the results
    best_hps=tuner.get_best_hyperparameters()[0]
    print(best_hps)

if __name__ == '__main__':
    # Implement some way afterwards
    # these are global variables now
    embedding_path = "ExternalData/glove.6B.300d.txt"
    training_path = "ExternalData/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml"
    validation_path = "ExternalData/ABSA15_Restaurants_Test.xml"

    train_data_path = "ExternalData/sem_train_2015.csv"
    test_data_path = "ExternalData/sem_test_2015.csv"

    main()

