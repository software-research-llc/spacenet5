import settings as S
import tensorflow as tf
import numpy as np
import flow
import pickle
import tqdm
#import segmentation_models as sm
import sklearn.metrics
import logging
import keras.backend as K

logger = logging.getLogger(__name__)


def _f1_stats(tp, fp, fn):
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)

    return f1, prec, rec


def f1_score(y_true, y_pred):
        return sklearn.metrics.f1_score(y_true.ravel().astype(int),
                                        y_pred.ravel().astype(int),
                                        average='micro',
                                        labels=[i for i in range(1,S.N_CLASSES)])


@tf.function
def damage_f1_score(f1_scores:list):
    f1 = len(f1_scores) / sum([1 / (x + 1e-5) for x in f1_scores])
    return f1


@tf.function
def remove_background(y_pred):
    background = tf.constant(([1] + [0] * (S.MASKSHAPE[-1] - 1) * S.MASKSHAPE[0] * S.MASKSHAPE[1] * 16)
    bg = tf.reshape(background, [-1, S.MASKSHAPE[0] * S.MASKSHAPE[1], S.N_CLASSES])

    pr = tf.cast(K.round(y_pred), tf.int32)
    pr = tf.clip_by_value(pr - bg, 0, 1)

    return pr


@tf.function
def iou_score(y_true, y_pred):
    gt = tf.cast(remove_background(y_true), tf.bool)
    pr = tf.cast(remove_background(y_pred), tf.bool)

    intersection = tf.cast(tf.logical_and(gt, pr), tf.int32)
    intersection = tf.reduce_sum(intersection)
    union = tf.cast(tf.logical_or(gt, pr), tf.int32)
    union = tf.reduce_sum(union)

    if union > 0:
        score = intersection / union
    else:
        score = tf.constant(0.0, dtype=tf.float64)
    return score


@tf.function
def pct_correct(y_true, y_pred):
    gt = tf.cast(remove_background(y_true), tf.bool)
    pr = tf.cast(remove_background(y_pred), tf.bool)

    intersection = tf.reduce_sum(tf.cast(tf.logical_and(gt, pr), tf.int32))
    total = tf.reduce_sum(tf.cast(gt, tf.int32))
    if total > 0:
        return intersection / total
    else:
        return tf.constant(0.0, dtype=tf.float64)


@tf.function
def num_correct(y_true, y_pred):
    gt = tf.cast(remove_background(y_true), tf.bool)
    pr = tf.cast(remove_background(y_pred), tf.bool)

    intersection = tf.reduce_sum(tf.cast(tf.logical_and(gt, pr), tf.int32))
    return intersection


@tf.function
def tensor_f1_score(y_true, y_pred):
    gt = tf.cast(remove_background(y_true), tf.bool)
    pr = tf.cast(remove_background(y_pred), tf.bool)

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
        y_pred = infer.weave_pred(pred)
        y_true = infer.weave_pred(y_)
        scores = f1_score(y_true, y_pred)#sklearn.metrics.f1_score(y_true.astype(int), y_pred.astype(int), average='macro')
        totals += scores
        pbar.set_description("%f" % (totals / num))
        num += 1
