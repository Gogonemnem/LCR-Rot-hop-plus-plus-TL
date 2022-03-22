import numpy as np
from Code.embedding import BERTEmbedding
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, Dropout
from tensorflow_addons.layers import AdaptiveAveragePooling1D


class Document(tf.keras.Model):
    def __init__(self, embedding_layer=BERTEmbedding(), drop_1: float = 0.2, drop_2: float = 0.5, hidden_units: int = None, regularizer=None):
        """Creates a new LCR model for the document level.

        Args:
            drop_1 (float, optional): Float between 0 and 1. Fraction of the input units to drop. Defaults to 0.2.
            drop_2 (float, optional): Float between 0 and 1. Fraction of the output neurons to drop. Defaults to 0.5.
            hidden_units (int, optional): Number of hidden units in the LSTM layer. Thus half the number of hidden units in the BiLSTM layer. If None, number is equal to the embedding dimension. Defaults to None.
            regularizer (_type_, optional): A tensorflow/keras regularizer. E.g., L1, L2 or L1 & L2. Defaults to no regularizer.

        Make sure to use sensible inputs, no checks are made.
        """
        super().__init__()

        # According to https://arxiv.org/pdf/1207.0580.pdf
        self.drop_input = Dropout(drop_1)
        self.drop_output = Dropout(drop_2)

        self.embedding_layer = embedding_layer
        # Check which embedding to use
        self.embedding_dim = self.embedding_layer.embedding_dim

        # BiLSTM layers
        self.hidden_units = hidden_units if hidden_units else self.embedding_dim
        self.left_bilstm = Bidirectional(
            LSTM(self.hidden_units, return_sequences=True))
        self.center_bilstm = Bidirectional(
            LSTM(self.hidden_units, return_sequences=True))
        self.right_bilstm = Bidirectional(
            LSTM(self.hidden_units, return_sequences=True))

        # 'Attention layer'
        self.average_pooling = AdaptiveAveragePooling1D(1)

        # MLP layer, why is it called MLP btw? I don´t think it should be.
        self.probabilities = Dense(3, Activation(
            'softmax'), bias_initializer='zeros', kernel_regularizer=regularizer, bias_regularizer=regularizer)

    def call(self, input):
        """Describes the model by relating the layers

        Args:
            input: The document

        Returns: Probabilities per class
        """
        # BiLSTMs
        input_left = self.embedding_layer(input)
        input_left = self.drop_input(input_left)
        left_bilstm = self.left_bilstm(input_left)

        input_center = self.embedding_layer(input)
        input_center = self.drop_input(input_center)
        center_bilstm = self.center_bilstm(input_center)

        input_right = self.embedding_layer(input)
        input_right = self.drop_input(input_right)
        right_bilstm = self.right_bilstm(input_right)

        # Representations
        representation_left = self.average_pooling(left_bilstm)[:, 0, :]
        representation_center = self.average_pooling(center_bilstm)[:, 0, :]
        representation_right = self.average_pooling(right_bilstm)[:, 0, :]

        # MLP
        v = tf.concat([representation_left, representation_center, representation_right], axis=1)
        v = self.drop_output(v)

        pred = self.probabilities(v)
        return pred
