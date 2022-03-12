
import tensorflow as tf


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