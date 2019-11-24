from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
from settings import *

class ChannelCompression(Layer):
    """
    Turn an image of N,W,H,C into a single N,W,H image, using the channel values as
    the probability of a pixel at that coordinate representing a class.

    i.e. The net outputs one channel per class, and we take the max of it's predictions
    and output that from this layer.
    """
    def compute_output_shape(self, input_shape):
        return input_shape[0:-1]#(input_shape[0], input_shape[1], input_shape[2])#(None,) + tuple(MASKSHAPE)#(None,) + tuple(MASKSHAPE)
    
    def call(self, x):
        # N, W, H, C
        return tf.argmax(x, axis=3)
