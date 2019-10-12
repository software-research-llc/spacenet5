import keras.backend as K
import keras
import numpy as np
import tensorflow as tf
from functools import partial


"""
Define our custom loss function.
"""
from keras import backend as K
import tensorflow as tf

import dill


def binary_focal_loss(gamma=2., alpha=.25):
    import snflow as flow
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    import snflow as flow
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


def discriminator_loss(disc_real_output, disc_generated_output):
  import snflow as flow
#  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
#def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = flow.binary_crossentropy(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target):
#  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
#def generator_loss(target, gen_output):
  import snflow as flow
  gan_loss = flow.binary_crossentropy(tf.ones_like(target), target)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (flow.LAMBDA * l1_loss)
#  total_gen_loss = gan_loss + l1_loss

  return total_gen_loss

def image_absolute_error(y_true, y_pred):
    c1_mask = tf.boolean_mask(y_true, y_pred[:,:,0])
    c2_mask = tf.boolean_mask(y_true, y_pred[:,:,1])
    c3_mask = tf.boolean_mask(y_true, y_pred[:,:,2])

def segmentation_loss(seg_gt, seg_logits):
    mask = seg_gt == seg_gt# <= flow.N_CLASSES
    seg_logits = tf.boolean_mask(seg_logits, mask)
    seg_gt = tf.boolean_mask(seg_gt, mask)
    seg_predictions = tf.argmax(seg_logits, axis=0)

    seg_loss_local = tf.nn.sigmoid_cross_entropy_with_logits(logits=seg_logits,
                                                                    labels=seg_gt)
    seg_loss = tf.reduce_mean(seg_loss_local)
#    tf.summary.scalar('loss/segmentation', seg_loss)

#    mean_iou, update_mean_iou = tf.compat.v2.streaming_mean_iou(seg_predictions, seg_gt,
#                                                   flow.N_CLASSES)
#    tf.summary.scalar('accuracy/mean_iou', mean_iou)
    return seg_loss#, mean_iou, update_mean_iou

def jaccard_distance(y_true, y_pred, smooth=100):
    """ Calculates mean of Jaccard distance as a loss function """
    intersection = tf.reduce_sum((y_true) * y_pred, axis=(1,2,3))
    sum_ = tf.reduce_sum((y_true) + y_pred, axis=(1,2,3))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    jd =  (1 - jac) * smooth
    return tf.reduce_mean(jd)

def jaccard_distance_abs(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


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

#    y_true = tf.expand_dims(y_true, -1)
#    y_pred = tf.expand_dims(y_pred, -1)
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
        mode = 1
        self = Dummy()

        self.targets = y_true
        self.pre_outputs = y_pred

        self.loss_numerator1 = tf.reduce_sum(self.pre_outputs * self.targets) + EPSILON
        self.loss_denominator1 = tf.reduce_sum(self.pre_outputs + self.targets - self.pre_outputs[:, :, :] * self.targets) + EPSILON
#        return 0 - self.loss_numerator1 / self.loss_denominator1

        self.loss_numerator2 = tf.reduce_sum(self.pre_outputs[:, :, :, 1:2] * (1 - self.targets)) + EPSILON
        self.loss_denominator2 = tf.reduce_sum(self.pre_outputs[:, :, :, 1:2] + (1 - self.targets) - self.pre_outputs[:, :, :, 1:2] * (1 - self.targets)) + EPSILON

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
        
        return self.loss
#        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
