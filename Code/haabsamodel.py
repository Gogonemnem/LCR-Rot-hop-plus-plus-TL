import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Softmax, Dense, Activation
from tensorflow_addons.layers import AdaptiveAveragePooling1D

import utils
from embedding import GloveEmbedding, BERTEmbedding
from attention import BilinearAttention, HierarchicalAttention


class HAABSA(tf.keras.Model):
    # hierarchy is tuple of size 2: 1st dim determines combining, 2nd dim determines iterative
    def __init__(self, training_path, test_path, embedding_path, invert=False, hop=1, hierarchy: tuple = None):
        super().__init__()

        if hop < 0:
            raise ValueError(f"The variable may not be lower than 0: {hop=}")

        self.invert = invert
        self.hop = hop
        self.hierarchy = hierarchy

        # self.embedding = self.create_embedding_layer(training_path, test_path, embedding_path)
        self.embedding = GloveEmbedding(embedding_path, [training_path, test_path])
        # self.embedding = BERTEmbedding()
        self.embedding_dim = self.embedding.embedding_dim

        hidden_units = self.embedding_dim  # OMG I THOUGHT HEE JUST RANDOMLY SET IT TO 300
        self.left_bilstm = Bidirectional(
            LSTM(hidden_units, return_sequences=True))
        self.target_bilstm = Bidirectional(
            LSTM(hidden_units, return_sequences=True))
        self.right_bilstm = Bidirectional(
            LSTM(hidden_units, return_sequences=True))

        self.average_pooling = AdaptiveAveragePooling1D(1)

        self.attention_left = BilinearAttention(2*self.embedding_dim)
        self.attention_right = BilinearAttention(2*self.embedding_dim)
        self.attention_target_left = BilinearAttention(2*self.embedding_dim)
        self.attention_target_right = BilinearAttention(2*self.embedding_dim)

        if hierarchy is not None:
            if hierarchy[0]:  # combine all
                self.hierarchical = HierarchicalAttention(2*self.embedding_dim)
            else:  # separate inner & outer
                self.hierarchical_inner = HierarchicalAttention(
                    2*self.embedding_dim)
                self.hierarchical_outer = HierarchicalAttention(
                    2*self.embedding_dim)

        self.probabilities = Dense(3, Activation('softmax'))

    def build(self, inputs_shape):
        # TODO: modify tweaking params!!!! for example regulizer = l2
        # Commented out, check self.prediction comment below in `call()`
        # self.weight_matrix = self.add_weight(name="weight", shape=(8*self.embedding_dim, 3),
        #                    initializer="glorot_uniform", trainable=True)
        # self.bias = self.add_weight(name="bias", shape=(3, ),
        #                    initializer="glorot_uniform", trainable=True)

        super().build(inputs_shape)

    def call(self, inputs):
        # embedding is not trainable, thus you can use it again
        input_left, input_target, input_right = inputs[0], inputs[1], inputs[2]

        embedded_left = self.embedding(input_left)
        left_bilstm = self.left_bilstm(embedded_left)

        embedded_target = self.embedding(input_target)
        target_bilstm = self.target_bilstm(embedded_target)

        embedded_right = self.embedding(input_right)
        right_bilstm = self.right_bilstm(embedded_right)

        # shape: [batch, 2*embed_dim], squeeze otherwise [batch, 1, 2*embed_dim]
        representation_left = tf.squeeze(self.average_pooling(left_bilstm))
        representation_right = tf.squeeze(self.average_pooling(right_bilstm))
        representation_target_left = representation_target_right = tf.squeeze(
            self.average_pooling(target_bilstm))

        # for hop == 0, this loop is skipped -> LCR model (no attention, no rot)
        # implementing all combinations of inverse, hop & 4 implementations of hierachical attention made this part kind of unreadable
        for i in range(self.hop):
            if self.invert:
                representation_target_left = self.attention_target_left(
                    [target_bilstm, representation_left])
                representation_target_right = self.attention_target_right(
                    [target_bilstm, representation_right])

                representation_left = self.attention_left(
                    [left_bilstm, representation_target_left])
                representation_right = self.attention_right(
                    [right_bilstm, representation_target_right])

            else:
                representation_left = self.attention_left(
                    [left_bilstm, representation_target_left])
                representation_right = self.attention_right(
                    [right_bilstm, representation_target_right])

                representation_target_left = self.attention_target_left(
                    [target_bilstm, representation_left])
                representation_target_right = self.attention_target_right(
                    [target_bilstm, representation_right])

            if self.hierarchy is not None and self.hierarchy[1]:  # iterate
                if self.hierarchy[0]:  # combine all
                    representations = tf.stack(
                        [representation_left, representation_target_left, representation_target_right, representation_right], axis=1)
                    representation_left, representation_target_left, representation_target_right, representation_right = tf.unstack(
                        self.hierarchical(representations), axis=1)
                else:  # separate inner & outer
                    representations = tf.stack(
                        [representation_left, representation_right], axis=1)
                    representation_left, representation_right = tf.unstack(
                        self.hierarchical_outer(representations), axis=1)

                    representations = tf.stack(
                        [representation_target_left, representation_target_right], axis=1)
                    representation_target_left, representation_target_right = tf.unstack(
                        self.hierarchical_inner(representations), axis=1)

        # don´t iterate
        if self.hierarchy is not None and not self.hierarchy[1]:
            if self.hierarchy[0]:  # combine all
                representations = tf.stack(
                    [representation_left, representation_target_left, representation_target_right, representation_right], axis=1)
                representation_left, representation_target_left, representation_target_right, representation_right = tf.unstack(
                    self.hierarchical(representations), axis=1)
            else:  # separate inner & outer
                representations = tf.stack(
                    [representation_left, representation_right], axis=1)
                representation_left, representation_right = tf.unstack(
                    self.hierarchical_outer(representations), axis=1)

                representations = tf.stack(
                    [representation_target_left, representation_target_right], axis=1)
                representation_target_left, representation_target_right = tf.unstack(
                    self.hierarchical_inner(representations), axis=1)

        v = tf.concat([representation_left, representation_target_left,
                      representation_target_right, representation_right], axis=1)

        pred = self.probabilities(v)
        # pred = self.output_softmax(v @ self.weight_matrix + self.bias) # Not sure if the previous line is equal
        return pred


def main():
    embedding_path = "ExternalData/glove.6B.300d.txt"
    training_path = "ExternalData/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml"
    validation_path = "ExternalData/ABSA15_Restaurants_Test.xml"

    train_data_path = "ExternalData/sem_train_2015.csv"
    test_data_path = "ExternalData/sem_test_2015.csv"
    utils.semeval_to_csv(training_path, train_data_path)
    utils.semeval_to_csv(validation_path, test_data_path)

    haabsa = HAABSA(training_path, validation_path,
                    embedding_path, hierarchy=(0, 0))
    haabsa.compile(tf.keras.optimizers.SGD(),  loss='categorical_crossentropy', metrics=[
                   'acc'], run_eagerly=True)  # TODO:run_eagerly off when done!

    left, target, right, polarity = utils.semeval_data(train_data_path)
    x_train = [left, target, right]
    y_train = tf.one_hot(polarity.astype('int64'), 3)

    left, target, right, polarity = utils.semeval_data(test_data_path)
    x_test = [left, target, right]
    y_test = tf.one_hot(polarity.astype('int64'), 3)

    haabsa.fit(x_train, y_train, validation_data=(
        x_test, y_test), epochs=1)  # , batch_size=5)
    # print(haabsa.summary())

    predictions = haabsa.predict(x_test)
    print(predictions)


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
