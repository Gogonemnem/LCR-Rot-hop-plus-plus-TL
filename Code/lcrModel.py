import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Softmax
from tensorflow_addons.layers import AdaptiveAveragePooling1D

import utils


class LeftCenterRight(tf.keras.Model):
    def __init__(self, training_path, embedding_path, embedding_dim):
        super().__init__()

        self.embedding_dim = embedding_dim 
        self.embedding = self.create_embedding_layer(training_path, embedding_path)
        
        hidden_units = embedding_dim # OMG I THOUGHT HEE JUST RANDOMLY SET IT TO 300
        self.left_bilstm = Bidirectional(LSTM(hidden_units, return_sequences=True))
        self.target_bilstm = Bidirectional(LSTM(hidden_units, return_sequences=True))
        self.right_bilstm = Bidirectional(LSTM(hidden_units, return_sequences=True))
        
        self.target_pooling = AdaptiveAveragePooling1D(1)
        
        self.attention_left = BilinearAttention()
        self.attention_right = BilinearAttention()
        self.attention_target_left = BilinearAttention()
        self.attention_target_right = BilinearAttention()

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

        # pool_target shape: [batch, 2*embed_dim], squeeze otherwise [batch, 1, 2*embed_dim]
        pool_target = tf.squeeze(self.target_pooling(target_bilstm))

        representation_left = self.attention_left([left_bilstm, pool_target])
        representation_right = self.attention_right([right_bilstm, pool_target])

        representation_target_left = self.attention_target_left([left_bilstm, representation_left])
        representation_target_right = self.attention_target_right([right_bilstm, representation_right])

        v = tf.concat([representation_left, representation_target_left, representation_target_right, representation_right], axis=1)
        pred = self.output_softmax(v @ self.weight_matrix + self.bias)
        return pred


    # only training path or also validation/test path?
    def create_embedding_layer(self, training_path, embeddings_path):
        self.vectorizer, word_index = utils.vocabulary_index(training_path)
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs) # seed, scale can be given

    def build(self, input_shape):
        # TODO: modify tweaking params!!!! for example regulizer = l2
        self.weight_matrix = self.add_weight(name="att_weight", shape=(input_shape[-1][-1], input_shape[-1][-1]),
                           initializer="glorot_uniform", trainable=True)
        self.bias = self.add_weight(name="att_bias", shape=(1, ),
                           initializer="glorot_uniform", trainable=True)

        return super().build(input_shape)

    def call(self, inputs):
        hidden, pool_target = inputs[0], inputs[1]
        length = tf.shape(hidden)[1]

        # sizes: batch x L x 1 = batch x L x 2d @ 2d x 2d @ 2d x 1
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


def main():
    embedding_path = "C:/Users/Gonem/CodeProjects/seminar-ba-qm/Wallaart-HAABSA/data/externalData/glove.6B.50d.txt"
    training_path = "C:/Users/Gonem/CodeProjects/seminar-ba-qm/Wallaart-HAABSA/data/externalData/absa-2015_restaurants_trial.xml"
    data_path = "sem_trial_2015.csv"
    embedding_dim = 50 # dependent on embedding data
    
    lcr = LeftCenterRight(training_path, embedding_path, embedding_dim)
    lcr.compile(tf.keras.optimizers.SGD(), loss='categorical_crossentropy', run_eagerly=True) # TODO:run_eagerly off when done!
    left, target, right, polarity = utils.semeval_data(data_path)
    x_train = [left, target, right]
    y_train = polarity.astype('int64')

    y_train = tf.one_hot(y_train, 3)
    
    lcr.fit(x_train, y_train)
    print(lcr.summary())

    
    

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