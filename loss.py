import keras.backend as K
import keras
import snflow as flow
import numpy as np
import tensorflow as tf
from functools import partial

def dice_xent_loss(y_true, y_pred, weight_map=1):
    """Adaptation of https://arxiv.org/pdf/1809.10486.pdf for multilabel 
    classification with overlapping pixels between classes. Dec 2018.
    """
    loss_dice = weighted_dice(y_true, y_pred, weight_map)
    loss_xent = weighted_binary_crossentropy(y_true, y_pred, weight_map)

    return loss_dice + loss_xent

def weighted_binary_crossentropy(y_true, y_pred, weight_map):
    return K.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean((K.binary_crossentropy(y_true, 
                                                 y_pred)*weight_map)) / (tf.reduce_sum(weight_map) + K.epsilon())

def weighted_dice(y_true, y_pred, weight_map):

    if weight_map is None:
        raise ValueError("Weight map cannot be None")
    if not isinstance(weight_map, int) and y_true.shape != weight_map.shape:
        raise ValueError("Weight map must be the same size as target vector: {} != {}".format(y_true.shape, weight_map.shape))
    
    dice_numerator = 2.0 * K.sum(y_pred * y_true * weight_map, axis=[1,2,3])
    dice_denominator = K.sum(weight_map * y_true, axis=[1,2,3]) + \
                                                             K.sum(y_pred * weight_map, axis=[1,2,3])
    loss_dice = (dice_numerator) / (dice_denominator + K.epsilon())
    h1=tf.square(tf.minimum(0.1,loss_dice)*10-1)
    h2=tf.square(tf.minimum(0.01,loss_dice)*100-1)
    return 1.0 - tf.reduce_mean(loss_dice) + \
            tf.reduce_mean(h1)*10 + \
            tf.reduce_mean(h2)*10

@tf.function
def dice_loss(y_true, y_pred):
    ys_sum = K.sum(y_true) + K.sum(y_pred)
    if tf.math.equal(ys_sum, tf.constant(0.0)):
        return tf.constant(0.001)
    intersection = tf.math.logical_and(tf.cast(y_true, tf.bool), tf.cast(y_pred, tf.bool))
    return tf.constant(2.0) * K.sum(tf.cast(intersection, tf.float32)) / ys_sum

def pixelwise_loss(y_true, y_pred):
#    equal = tf.cast(tf.math.equal(y_true, y_pred), tf.float32)
#    one = tf.constant(1, dtype=tf.float32)
#    eqsum = tf.cast(K.sum(equal) + 1e-5, dtype=tf.float32)
#    denom = 20 * 256 * 256 * 1
#    frac = 5 / denom
    def grad(*x):
        return x * 5
    return K.ones_like(y_true), grad

def old(y_true, y_pred):
    return tf.math.abs(tf.math.reduce_sum(y_true - y_pred))
    return tf.math.abs(tf.compat.v2.norm(y_true) - tf.compat.v2.norm(y_pred))
    return tf.reduce_sum(tf.image.total_variation(tf.concat([y_true, y_pred], 0)))
    equal = K.equal(y_true, y_pred)
    return tf.constant(256 * 256) - tf.cast(K.sum(tf.cast(equal, tf.int32)), tf.int32)

def ssim_metric(y_true, y_pred):
    # source: https://gist.github.com/Dref360/a48feaecfdb9e0609c6a02590fd1f91b

    y_true = tf.expand_dims(y_true, -1)
    y_pred = tf.expand_dims(y_pred, -1)
    y_true = tf.transpose(y_true, [0, 2, 3, 1])
    y_pred = tf.transpose(y_pred, [0, 2, 3, 1])
    patches_true = tf.compat.v1.image.extract_image_patches(y_true, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    patches_pred = tf.compat.v1.image.extract_image_patches(y_pred, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

    u_true = K.mean(patches_true, axis=3)
    u_pred = K.mean(patches_pred, axis=3)
    var_true = K.var(patches_true, axis=3)
    var_pred = K.var(patches_pred, axis=3)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom
    ssim = tf.where(tf.math.is_nan(ssim), K.zeros_like(ssim), ssim)
    return ssim

class Dummy:
    pass

def segment_loss(y_true, y_pred):
        EPSILON = 1.0
        mode = 0
        self = Dummy()

        self.targets = y_true
        self.pre_outputs = y_pred

        self.loss_numerator1 = tf.reduce_sum(self.pre_outputs * self.targets) + EPSILON
        self.loss_denominator1 = tf.reduce_sum(self.pre_outputs + self.targets - self.pre_outputs[:, :, :] * self.targets) + EPSILON
        return 0 - self.loss_numerator1 / self.loss_denominator1

        loss_numerator2 = tf.reduce_sum(self.pre_outputs[:, :, :, 1:2] * (1 - self.targets)) + EPSILON
        loss_denominator2 = tf.reduce_sum(self.pre_outputs[:, :, :, 1:2] + (1 - self.targets) - self.pre_outputs[:, :, :, 1:2] * (1 - self.targets)) + EPSILON

        if mode == 0:
            self.loss = -(self.loss_numerator1 / self.loss_denominator1 + self.loss_numerator2 / self.loss_denominator2 / 13)
        elif mode == 1:
            self.loss = -(self.loss_numerator1 / self.loss_denominator1 + self.loss_numerator2 / self.loss_denominator2)
        elif mode == 2:
            self.loss = -(self.loss_numerator1 / self.loss_denominator1)
        elif mode == 3:
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat([self.targets, 1 - self.targets], axis=3), logits=self.initial_outputs))
        elif mode == 4:
            self.loss = -(self.loss_numerator1 / self.loss_denominator1 + self.loss_numerator2 / self.loss_denominator2 / 2)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

keras.losses.dice_xent_loss = dice_xent_loss
keras.losses.segment_loss = segment_loss
keras.losses.dice_loss = dice_loss
keras.losses.pixelwise_loss = pixelwise_loss
keras.losses.ssim_metric = ssim_metric
keras.losses.old = old
