from __future__ import absolute_import
import keras.backend as K
import tensorflow as tf

def round_through(x):
    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)

def _hard_sigmoid(x):
    x = (0.5 * x) + 0.50001
    return tf.clip_by_value(x, 0, 1)

def binary_tanh(x): 
    return round_through(_hard_sigmoid(x))

def binarize(W):
    return 2 * round_through(_hard_sigmoid(W)) - 1

