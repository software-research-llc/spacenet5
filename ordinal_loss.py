"""https://github.com/JHart96/keras_ordinal_categorical_crossentropy"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import losses

def sparse_loss(y_true, y_pred):
    """
    Ordinal sparse categorical crossentropy
    """
    weights = K.cast(K.abs(K.argmax(y_true, axis=3) - K.argmax(y_pred, axis=3)), dtype='float32')
    return K.clip((1.0 + weights) * losses.sparse_categorical_crossentropy(y_true, y_pred), 0,1)

def loss(y_true, y_pred):
    """
    Ordinal categorical crossentropy
    """
    weights = K.cast(K.abs(K.argmax(y_true, axis=3) - K.argmax(y_pred, axis=3)), dtype='float32')
    return K.clip((1.0 + weights) * losses.binary_crossentropy(y_true, y_pred), 0., 1.)
