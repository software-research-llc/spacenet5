import settings as S
import tensorflow as tf
import numpy as np
import flow
import pickle
import tqdm
import sklearn.metrics
import logging
import keras.backend as K
import test

logger = logging.getLogger(__name__)



def _f1_stats(tp, fp, fn):
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)

    return f1, prec, rec


def f1_score(y_true, y_pred):
    """
    sklearn.metrics.f1_score (only for numpy.ndarray objects).
    """
    return sklearn.metrics.f1_score(y_true.ravel(),
                                    y_pred.ravel(),
                                    average='micro',
                                    labels=[i for i in range(1,S.N_CLASSES)])


@tf.function
def remove_background(y_pred):
    """
    Takes a one-hot encoded prediction and removes the background pixels (which are at index 0),
    returning the rounded predictions.
    """
    background = tf.constant(([1] + [0] * (S.MASKSHAPE[-1] - 1)) * S.MASKSHAPE[0] * S.MASKSHAPE[1] * S.BATCH_SIZE, dtype=tf.int64)
    bg = tf.reshape(background, [S.BATCH_SIZE, S.MASKSHAPE[0] * S.MASKSHAPE[1], -1])

    pr = tf.cast(K.round(y_pred), tf.int64)
    pr = tf.clip_by_value(pr - bg, 0, 1)

    return pr


@tf.function
def get_gt_pr(y_true, y_pred):
    """
    Removes background pixels and returns input tensors cast to a boolean data type.
    """
#    y_true = tf.reshape(y_true, [S.BATCH_SIZE, 1024 * 1024, S.N_CLASSES])
#    y_pred = tf.reshape(y_pred, [S.BATCH_SIZE, 1024 * 1024, S.N_CLASSES])
    gt = tf.cast(remove_background(y_true), tf.bool)
    pr = tf.cast(remove_background(y_pred), tf.bool)

    return gt, pr


@tf.function
def iou_score(y_true, y_pred):
    """
    Return an intersection over union score.
    """
    gt, pr = get_gt_pr(y_true, y_pred)

    intersection = tf.cast(tf.logical_and(gt, pr), tf.int64)
    intersection = tf.reduce_sum(intersection)
    union = tf.cast(tf.logical_or(gt, pr), tf.int64)
    union = tf.reduce_sum(union)

    if union > 0:
        score = intersection / union
    else:
        score = tf.constant(0.0, dtype=tf.float64)
    return score


@tf.function
def recall(y_true, y_pred):
    """
    Returns recall score: true positives / (true positives + false negatives),
    i.e. the percentage of pixels that didn't go unnoticed.
    """
    gt, pr = get_gt_pr(y_true, y_pred)

    intersection = tf.reduce_sum(tf.cast(tf.logical_and(gt, pr), tf.int64))
    total = tf.reduce_sum(tf.cast(gt, tf.int64))
    if total > 0:
        return intersection / total
    else:
        return tf.constant(0.0, dtype=tf.float64)


@tf.function
def num_correct(y_true, y_pred):
    """
    Returns the raw number of correct pixel predictions (true positives).
    """
    gt, pr = get_gt_pr(y_true, y_pred)

    intersection = tf.reduce_sum(tf.cast(tf.logical_and(gt, pr), tf.int64))
    return intersection


@tf.function
def tensor_f1_score(y_true, y_pred):
    """
    Global F1-score, excluding background pixels.  Treats all other classes as a single class.
    """
    gt, pr = get_gt_pr(y_true, y_pred)

    tp = tf.reduce_sum(tf.cast(tf.logical_and(gt, pr), tf.int64))
    fp = tf.reduce_sum(tf.clip_by_value(tf.cast(pr, tf.int64) - tf.cast(gt, tf.int64), 0, 1))
    fn = tf.reduce_sum(tf.clip_by_value(tf.cast(gt, tf.int64) - tf.cast(pr, tf.int64), 0, 1))

    if (tp + fp) > 0 and (tp + fn) > 0:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if (prec + rec) > 0:
            score = 2 * tf.math.multiply_no_nan(prec, rec) / (prec + rec)
        else:
            score = tf.constant(0.0, tf.float64)
    else:
        score = tf.constant(0.0, tf.float64)

    return score


# df1 = 4 / sum((f1+epsilon)**-1 for f1 in [no_damage_f1, minor_damage_f1, major_damage_f1, destroyed_f1]), where epsilon = 1e-6
@tf.function
def damage_f1_score(y_true, y_pred):
    gt, pr = y_true, y_pred

    tp = tf.reduce_sum(gt * pr)
    fp = tf.reduce_sum(tf.clip_by_value(pr - gt, 0, 1))
    fn = tf.reduce_sum(tf.clip_by_value(gt - pr, 0, 1))

    if (tp + fp) > 0 and (tp + fn) > 0:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if (prec + rec) > 0:
            score = 2 * tf.math.multiply_no_nan(prec, rec) / (prec + rec)
            score = tf.cast(score, tf.float64)
        else:
            score = tf.constant(0.0, tf.float64)
    else:
        score = tf.constant(0.0, tf.float64)

    return score


if __name__ == '__main__':
    import infer
    import math
    import train
    model = train.build_model()
    S.BATCH_SIZE = 1
    model = train.load_weights(model)
    df = flow.Dataflow(files=flow.get_validation_files(), batch_size=1, shuffle=False)
    totals = 0#[0,0,0]
    num = 1
    pbar = tqdm.tqdm(df, desc="Scoring")
    for x,y_ in pbar:
        pred = model.predict(x)#np.expand_dims(x[j], axis=0))
        y_pred = infer.convert_prediction(pred)
        y_true = infer.convert_prediction(y_)
        scores = f1_score(y_true, y_pred)#sklearn.metrics.f1_score(y_true.astype(int), y_pred.astype(int), average='macro')
        totals += scores
        pbar.set_description("%f" % (totals / num))
        num += 1
