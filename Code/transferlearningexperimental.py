import numpy as np
import tensorflow as tf

import document
import lcrrothopplusplus
from embedding import BERTEmbedding


class TL(tf.keras.Model):
    def __init__(self, share_bilstms=True, share_mlp=False, embedding_layer=BERTEmbedding(), invert: bool = False, hop: int = 1, hierarchy: tuple = None, drop_1: float = 0.2, drop_2: float = 0.5, hidden_units: int = None, regularizer=None):
        """Creates a new model for transfer learning.

        Args:
            drop_1 (float, optional): Float between 0 and 1. Fraction of the input units to drop. Defaults to 0.2.
            drop_2 (float, optional): Float between 0 and 1. Fraction of the output neurons to drop. Defaults to 0.5.
            hidden_units (int, optional): Number of hidden units in the LSTM layer. Thus half the number of hidden units in the BiLSTM layer. If None, number is equal to the embedding dimension. Defaults to None.
            regularizer (_type_, optional): A tensorflow/keras regularizer. E.g., L1, L2 or L1 & L2. Defaults to no regularizer.

        Make sure to use sensible inputs, no checks are made.
        """
        super().__init__()

        self.doc_model = document.Document(
            embedding_layer, drop_1, drop_2, hidden_units, regularizer)
        self.asp_level = lcrrothopplusplus.LCRRothopPP(
            embedding_layer, invert, hop, hierarchy, drop_1, drop_2, hidden_units, regularizer)

        # These references are set at initialization -> pretraining 'shared' does not allow you to separate them again at mult or ft, unless you manually separate them 
        if share_bilstms:
            self.asp_level.left_bilstm = self.doc_model.left_bilstm
            self.asp_level.target_bilstm = self.doc_model.center_bilstm
            self.asp_level.right_bilstm = self.doc_model.right_bilstm

        if share_mlp:
            self.asp_level.probabilities = self.doc_model.probabilities

    def call(self, inputs):
        """Describes the model by relating the layers

        Args:
            input: The Left, center, right & document data

        Returns: Probabilities per class
        """
        *input_asp, input_doc = inputs[0], inputs[1], inputs[2], inputs[3]

        # TODO: Make formal check if pret, mult or ft
        # Should not be necessary if loss weights are set correctly (I think it can handle nans)
        # but it reduces computations
        # if tf.math.reduce_any(tf.math.is_nan(inputs[3])):
        doc_pred = self.doc_model(input_doc)
        asp_pred = self.asp_level(input_asp)

        return {'asp': asp_pred, 'doc': doc_pred}
