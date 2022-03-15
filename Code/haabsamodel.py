import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, Dropout
from tensorflow_addons.layers import AdaptiveAveragePooling1D
from embedding import GloveEmbedding, BERTEmbedding
from attention import BilinearAttention, HierarchicalAttention


class HAABSA(tf.keras.Model):
    # hierarchy is tuple of size 2: 1st dim determines combining, 2nd dim determines iterative
    def __init__(self, training_path, test_path, embedding_path, invert=False, hop=1, hierarchy: tuple = None, regularizer=None):
        super().__init__()

        if hop < 0:
            raise ValueError(f"The variable may not be lower than 0: {hop=}")

        self.invert = invert
        self.hop = hop
        self.hierarchy = hierarchy

        # according to https://arxiv.org/pdf/1207.0580.pdf
        self.drop_input = Dropout(0.2)
        self.drop_output = Dropout(0.5)

        self.embedding = GloveEmbedding(
            embedding_path, [training_path, test_path])
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

        self.attention_left = BilinearAttention(
            2*self.embedding_dim, regularizer)
        self.attention_right = BilinearAttention(
            2*self.embedding_dim, regularizer)
        self.attention_target_left = BilinearAttention(
            2*self.embedding_dim, regularizer)
        self.attention_target_right = BilinearAttention(
            2*self.embedding_dim, regularizer)

        if hierarchy is not None:
            if hierarchy[0]:  # combine all
                self.hierarchical = HierarchicalAttention(
                    2*self.embedding_dim, regularizer)
            else:  # separate inner & outer
                self.hierarchical_inner = HierarchicalAttention(
                    2*self.embedding_dim, regularizer)
                self.hierarchical_outer = HierarchicalAttention(
                    2*self.embedding_dim, regularizer)

        self.probabilities = Dense(3, Activation(
            'softmax'), bias_initializer='zeros', kernel_regularizer=regularizer, bias_regularizer=regularizer)

    def build(self, inputs_shape):
        # TODO: modify tweaking params!!!! for example regulizer = l2
        # Commented out, check self.prediction comment below in `call()`
        # self.weight_matrix = self.add_weight(name="weight", shape=(8*self.embedding_dim, 3),
        #                    initializer="glorot_uniform", trainable=True)
        # self.bias = self.add_weight(name="bias", shape=(3, ),
        #                    initializer="glorot_uniform", trainable=True)

        # super().build(inputs_shape)
        pass

    def call(self, inputs):
        input_left, input_target, input_right = inputs[0], inputs[1], inputs[2]

        ##### Embedding & BiLSTMs
        # embedding is not trainable, thus you can use it again
        embedded_left = self.embedding(input_left)
        embedded_left = self.drop_input(embedded_left)
        left_bilstm = self.left_bilstm(embedded_left)

        embedded_target = self.embedding(input_target)
        embedded_target = self.drop_input(embedded_target)
        target_bilstm = self.target_bilstm(embedded_target)

        embedded_right = self.embedding(input_right)
        embedded_right = self.drop_input(embedded_right)
        right_bilstm = self.right_bilstm(embedded_right)

        # Representations
        # shape: [batch, 2*embed_dim], squeeze otherwise [batch, 1, 2*embed_dim]
        # squeeze doesn't work for run_eagerly=False, spent too much time on this...
        # maybe change code to work with shape: [batch, 1, 2*embed_dim]?
        representation_target_left = representation_target_right = self.average_pooling(
            target_bilstm)[:, 0, :]

        ##############################################################################################
        ########## I took some creative liberty here and changed this part of the model     ##########
        ########## It may not improve the model, but it is easier to program the iterations ##########
        ########## Drawing the diagram though, not so much                                  ##########
        ##############################################################################################
        representation_left = self.average_pooling(left_bilstm)[:, 0, :]
        representation_right = self.average_pooling(right_bilstm)[:, 0, :]

        # for hop == 0, this loop is skipped -> LCR model (no attention, no rot)
        for _ in range(self.hop):
            if self.hierarchy is not None and self.hierarchy[1]:
                representation_left, representation_target_left, representation_target_right, representation_right = self._apply_hierarchical_attention(
                    representation_left, representation_target_left, representation_target_right, representation_right)

            representation_left, representation_target_left, representation_target_right, representation_right = self._apply_bilinear_attention(
                left_bilstm, target_bilstm, right_bilstm, representation_left, representation_target_left, representation_target_right, representation_right)

        # if this similar line is UNDER `_apply_bilinear_attention()`, then add the CHECK for [0], otherwise doubly applied
        # if this similar line is ABOVE `_apply_bilinear_attention()`, the model of trusca is slightly changed, but DON´T CHECK for [0]!
        if self.hierarchy is not None:  # and not self.hierarchy[0]
            representation_left, representation_target_left, representation_target_right, representation_right = self._apply_hierarchical_attention(
                representation_left, representation_target_left, representation_target_right, representation_right)

        # MLP, why is it called MLP btw? I don´t think it should be.
        v = tf.concat([representation_left, representation_target_left,
                      representation_target_right, representation_right], axis=1)
        v = self.drop_output(v)

        pred = self.probabilities(v)
        # Not sure if the previous line is equal
        # pred = self.output_softmax(v @ self.weight_matrix + self.bias)
        return pred

    def _apply_bilinear_attention(self, left_bilstm, target_bilstm, right_bilstm, representation_left, representation_target_left, representation_target_right, representation_right):
        if self.invert:
            representation_target_left = self.attention_target_left(
                [target_bilstm, representation_left])
            representation_target_right = self.attention_target_right(
                [target_bilstm, representation_right])

        representation_left = self.attention_left(
            [left_bilstm, representation_target_left])
        representation_right = self.attention_right(
            [right_bilstm, representation_target_right])

        if not self.invert:
            representation_target_left = self.attention_target_left(
                [target_bilstm, representation_left])
            representation_target_right = self.attention_target_right(
                [target_bilstm, representation_right])

        return representation_left, representation_target_left, representation_target_right, representation_right

    def _apply_hierarchical_attention(self, representation_left, representation_target_left, representation_target_right, representation_right):
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
        return representation_left, representation_target_left, representation_target_right, representation_right
