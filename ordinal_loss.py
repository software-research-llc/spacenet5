"""https://github.com/JHart96/keras_ordinal_categorical_crossentropy"""

from tf.keras import backend as K
from tf.keras import losses

def sparse_loss(y_true, y_pred):
    """
    Ordinal sparse categorical crossentropy
    """
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.sparse_categorical_crossentropy(y_true, y_pred)

def loss(y_true, y_pred):
    """
    Ordinal categorical crossentropy
    """
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)
