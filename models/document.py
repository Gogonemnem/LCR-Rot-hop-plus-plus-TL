from models.layers.embedding import BERTEmbedding
import tensorflow as tf
from tensorflow_addons.layers import AdaptiveAveragePooling1D


class Document(tf.keras.Model):
    def __init__(self, embedding_layer=BERTEmbedding(), drop_1: float=0.2, drop_2: float=0.5, hidden_units: int = None, regularizer=None):
        """Creates a new LCR model for the document level.

        Args:
            drop_1 (float, optional): Float between 0 and 1. Fraction of the input units to drop. Defaults to 0.2.
            drop_2 (float, optional): Float between 0 and 1. Fraction of the output neurons to drop. Defaults to 0.5.
            hidden_units (int, optional): Number of hidden units in the LSTM layer. Thus half the number of hidden units in the BiLSTM layer. If None, number is equal to the embedding dimension. Defaults to None.
            regularizer (_type_, optional): A tensorflow/keras regularizer. E.g., L1, L2 or L1 & L2. Defaults to no regularizer.

        Make sure to use sensible inputs, no checks are made.
        """
        super().__init__()

        self.drop_input = tf.keras.layers.Dropout(drop_1)
        self.drop_output = tf.keras.layers.Dropout(drop_2)

        self.embedding_layer = embedding_layer
        self.hidden_units = hidden_units if hidden_units else self.embedding_layer.embedding_dim

        # BiLSTM layers
        self.left_bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
        )
        self.center_bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
        )
        self.right_bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_units, return_sequences=True)
        )

        # 'Attention layer'
        self.average_pooling = AdaptiveAveragePooling1D(1)

        # 'MLP' layer
        self.probabilities = tf.keras.layers.Dense(3, 
            tf.keras.layers.Activation('softmax'),
            bias_initializer='zeros',
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer
        )

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
