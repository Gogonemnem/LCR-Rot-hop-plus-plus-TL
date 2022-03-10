import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Softmax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow_addons.layers import AdaptiveAveragePooling1D

import utils


class HAABSA(tf.keras.Model):
    # hierarchy is tuple of size 2: 1st dim determines combining, 2nd dim determines iterative
    def __init__(self, training_path, test_path, embedding_path, embedding_dim, invert=False, hop=1, hierarchy: tuple=None):
        super().__init__()

        if hop < 0:
            raise ValueError(f"The variable may not be lower than 0: {hop=}")

        self.invert = invert
        self.hop = hop
        self.hierarchy = hierarchy

        self.embedding_dim = embedding_dim 
        self.embedding = self.create_embedding_layer(training_path, test_path, embedding_path)
        
        hidden_units = embedding_dim # OMG I THOUGHT HEE JUST RANDOMLY SET IT TO 300
        self.left_bilstm = Bidirectional(LSTM(hidden_units, return_sequences=True))
        self.target_bilstm = Bidirectional(LSTM(hidden_units, return_sequences=True))
        self.right_bilstm = Bidirectional(LSTM(hidden_units, return_sequences=True))
        
        self.average_pooling = AdaptiveAveragePooling1D(1)
        
        self.attention_left = BilinearAttention(2*self.embedding_dim)
        self.attention_right = BilinearAttention(2*self.embedding_dim)
        self.attention_target_left = BilinearAttention(2*self.embedding_dim)
        self.attention_target_right = BilinearAttention(2*self.embedding_dim)

        if hierarchy is not None:
            if hierarchy[0]: # combine all
                self.hierarchical = HierarchicalAttention(2*self.embedding_dim)
            else: # separate inner & outer
                self.hierarchical_inner = HierarchicalAttention(2*self.embedding_dim)
                self.hierarchical_outer = HierarchicalAttention(2*self.embedding_dim)


        self.output_softmax = Softmax()

    def build(self, inputs_shape):
        # TODO: modify tweaking params!!!! for example regulizer = l2
        self.weight_matrix = self.add_weight(name="weight", shape=(8*self.embedding_dim, 3),
                           initializer="glorot_uniform", trainable=True)
        self.bias = self.add_weight(name="bias", shape=(3, ),
                           initializer="glorot_uniform", trainable=True)

    def call(self, inputs):
        # embedding is not trainable, thus you can use it again
        input_left, input_target, input_right = inputs[0], inputs[1], inputs[2]

        vector_left = self.vectorizer(input_left)
        vector_target = self.vectorizer(input_target)
        vector_right = self.vectorizer(input_right)

        embedded_left = self.embedding(vector_left)
        embedded_target = self.embedding(vector_target)
        embedded_right = self.embedding(vector_right)

        left_bilstm = self.left_bilstm(embedded_left)
        target_bilstm = self.target_bilstm(embedded_target)
        right_bilstm = self.right_bilstm(embedded_right)

        # shape: [batch, 2*embed_dim], squeeze otherwise [batch, 1, 2*embed_dim]
        representation_left = tf.squeeze(self.average_pooling(left_bilstm))
        representation_right = tf.squeeze(self.average_pooling(right_bilstm))
        representation_target_left = representation_target_right = tf.squeeze(self.average_pooling(target_bilstm))
        
        # for hop == 0, this loop is skipped -> LCR model (no attention, no rot)
        # implementing all combinations of inverse, hop & 4 implementations of hierachical attention made this part kind of unreadable
        for _ in range(self.hop):
            if self.invert:
                representation_target_left = self.attention_target_left([target_bilstm, representation_left])
                representation_target_right = self.attention_target_right([target_bilstm, representation_right])

                representation_left = self.attention_left([left_bilstm, representation_target_left])
                representation_right = self.attention_right([right_bilstm, representation_target_right])

                
            else:
                representation_left = self.attention_left([left_bilstm, representation_target_left])
                representation_right = self.attention_right([right_bilstm, representation_target_right])

                representation_target_left = self.attention_target_left([target_bilstm, representation_left])
                representation_target_right = self.attention_target_right([target_bilstm, representation_right])
            
            if self.hierarchy is not None and self.hierarchy[1]: # iterate
                if self.hierarchy[0]: # combine all
                    representations = tf.stack([representation_left, representation_target_left, representation_target_right, representation_right], axis=1)
                    representation_left, representation_target_left, representation_target_right, representation_right = tf.unstack(self.hierarchical(representations), axis=1)
                else: # separate inner & outer
                    representations = tf.stack([representation_left, representation_right], axis=1)
                    representation_left, representation_right = tf.unstack(self.hierarchical_outer(representations), axis=1)

                    representations = tf.stack([representation_target_left, representation_target_right], axis=1)
                    representation_target_left, representation_target_right = tf.unstack(self.hierarchical_inner(representations), axis=1)

        if self.hierarchy is not None and not self.hierarchy[1]: # don´t iterate
            if self.hierarchy[0]: # combine all
                representations = tf.stack([representation_left, representation_target_left, representation_target_right, representation_right], axis=1)
                representation_left, representation_target_left, representation_target_right, representation_right = tf.unstack(self.hierarchical(representations), axis=1)
            else: # separate inner & outer
                representations = tf.stack([representation_left, representation_right], axis=1)
                representation_left, representation_right = tf.unstack(self.hierarchical_outer(representations), axis=1)

                representations = tf.stack([representation_target_left, representation_target_right], axis=1)
                representation_target_left, representation_target_right = tf.unstack(self.hierarchical_inner(representations), axis=1)


        v = tf.concat([representation_left, representation_target_left, representation_target_right, representation_right], axis=1)
        pred = self.output_softmax(v @ self.weight_matrix + self.bias)
        return pred


    # only training path or also validation/test path?
    def create_embedding_layer(self, training_path, test_path, embeddings_path):
        vectorizer, _ = utils.vocabulary_index(training_path)
        self.vectorizer, word_index = utils.vocabulary_index(test_path, vectorizer)
        embeddings_index = utils.load_pretrained_embeddings(embeddings_path)
        
        num_tokens = len(self.vectorizer.get_vocabulary()) + 2
        

        embedding_matrix = np.zeros((num_tokens, self.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"

                # padding when sentence (or target phrase) is too short
                # no idea what oov is though
                embedding_matrix[i] = embedding_vector

        embedding_layer = Embedding(num_tokens, self.embedding_dim, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), trainable=False)
        return embedding_layer

class BilinearAttention(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs) # seed, scale can be given
        self.dim = dim

    def build(self, input_shape):
        # TODO: modify tweaking params!!!! for example regulizer = l2
        self.weight_matrix = self.add_weight(name="att_weight", shape=(self.dim, self.dim),
                           initializer="glorot_uniform", trainable=True)
        self.bias = self.add_weight(name="att_bias", shape=(1, ),
                           initializer="glorot_uniform", trainable=True)

        return super().build(input_shape)

    def call(self, inputs):
        hidden, pool_target = inputs[0], inputs[1]
        length = tf.shape(hidden)[1]

        # sizes: batch x L x 1 = batch x L x 2d @ 2d x 2d @ batch x 2d x 1
        # first_term = hidden @ self.weight_matrix @ pool_target
        first_term = tf.einsum('bik, bk -> bi', hidden @ self.weight_matrix, pool_target)
        # sizes: batch x L x 1 = 1 x 1 * batch x L x 1
        second_term = self.bias * tf.ones([length, ])

        func = tf.keras.activations.tanh(first_term + second_term)
        alpha = tf.keras.activations.softmax(func)

        # basically hidden^T @ alpha, but these are 3 dimensional tensors
        r = tf.einsum('bki,bk->bi', hidden, alpha)
        return r

    def get_config(self):
        # config = {'use_scale': self.use_scale}
        base_config = super().get_config()
        return base_config
        # return dict(list(base_config.items()) + list(config.items()))

class HierarchicalAttention(tf.keras.layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs) 
        self.dim = dim

    def build(self, input_shape):
        # TODO: modify tweaking params!!!! for example regulizer = l2
        self.weight_matrix = self.add_weight(name="att_weight", shape=(self.dim, 1),
                           initializer="glorot_uniform", trainable=True)
        self.bias = self.add_weight(name="att_bias", shape=(1, ),
                           initializer="glorot_uniform", trainable=True)

        return super().build(input_shape)

    def call(self, representations):
        length = tf.shape(representations)[1]

        first_term = representations @ self.weight_matrix
        second_term = self.bias * tf.ones([length, 1]) # , 1 needed bc tensorflow weird

        func = tf.keras.activations.tanh(first_term + second_term)
        alpha = tf.keras.activations.softmax(func)

        representations *= alpha
        return representations

    def get_config(self):
        # config = {'use_scale': self.use_scale}
        base_config = super().get_config()
        return base_config
        # return dict(list(base_config.items()) + list(config.items()))


def main():
    embedding_path = "ExternalData/glove.6B.300d.txt"
    training_path = "ExternalData/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml"
    validation_path = "ExternalData/ABSA15_Restaurants_Test.xml"
    
    embedding_dim = 300 # dependent on embedding data

    train_data_path = "ExternalData/sem_train_2015.csv"
    test_data_path = "ExternalData/sem_test_2015.csv"
    utils.semeval_to_csv(training_path, train_data_path)
    utils.semeval_to_csv(validation_path, test_data_path)

    
    haabsa = HAABSA(training_path, validation_path, embedding_path, embedding_dim, hierarchy=(0, 0))
    haabsa.compile(tf.keras.optimizers.SGD(),  loss='categorical_crossentropy', metrics=['acc'], run_eagerly=True) # TODO:run_eagerly off when done!
    
    left, target, right, polarity = utils.semeval_data(train_data_path)
    x_train = [left, target, right]
    y_train = tf.one_hot(polarity.astype('int64'), 3)

    left, target, right, polarity = utils.semeval_data(test_data_path)
    x_test = [left, target, right]
    y_test = tf.one_hot(polarity.astype('int64'), 3)
    # y_test = polarity.astype('int64')
    
    haabsa.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1)
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