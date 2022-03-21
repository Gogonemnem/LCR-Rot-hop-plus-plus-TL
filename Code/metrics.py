from pydoc import doc
from random import sample
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import Loss, CategoricalCrossentropy
from tensorflow.keras.metrics import Metric, CategoricalAccuracy


class CombinedCrossEntropy(Loss):
    def __init__(self, doc_weight=1, reduction=tf.keras.losses.Reduction.AUTO, name='combined_cross_entropy'):
        super().__init__(reduction, name)
        self.doc_weight = doc_weight

    def call(self, y_true, y_pred):        
        asp_true, doc_true = y_true[0], y_true[1]
        asp_pred, doc_pred = y_pred[0], y_pred[1]

        if not tf.reduce_any(tf.math.is_nan(asp_true)):
            asp_loss = CategoricalCrossentropy(reduction=self.reduction, name='asp_loss')(asp_true, asp_pred)
        else:
            asp_loss = 0.0
        
        if not tf.reduce_any(tf.math.is_nan(doc_true)):
            doc_loss = CategoricalCrossentropy(reduction=self.reduction, name='doc_loss')(doc_true, doc_pred)
        else:
            doc_loss = 0.0
        return asp_loss + self.doc_weight * doc_loss
    
class CombinedAccuracy(Metric):
    def __init__(self, name='accuracy', **kwargs):
        super().__init__(name, **kwargs)
        # self.accuracy = self.add_weight(name='acc', initializer='zeros')
        self.accuracy = CategoricalAccuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        asp_true, doc_true = y_true[0], y_true[1]
        asp_pred, doc_pred = y_pred[0], y_pred[1]

        if tf.reduce_any(tf.math.is_nan(asp_true)):
            self.accuracy.update_state(asp_true, asp_pred, sample_weight=sample_weight)
        else:
            self.accuracy.update_state(doc_true, doc_pred, sample_weight=sample_weight)
        
        # if tf.reduce_any(tf.math.is_nan(asp_true)):
        #     values = tf.equal(tf.compat.v1.argmax(doc_true, axis=-1), tf.compat.v1.argmax(doc_pred, axis=-1))
        # else: 
        #     values = tf.equal(tf.compat.v1.argmax(asp_true, axis=-1), tf.compat.v1.argmax(asp_pred, axis=-1))
        
        # y_true = tf.cast(y_true, tf.bool)
        # y_pred = tf.cast(y_pred, tf.bool)

        # values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        # values = tf.cast(values, self.dtype)
        # if sample_weight is not None:
        #     sample_weight = tf.cast(sample_weight, self.dtype)
        #     sample_weight = tf.broadcast_to(sample_weight, values.shape)
        #     values = tf.multiply(values, sample_weight)
        # self.accuracy.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.accuracy.result()

