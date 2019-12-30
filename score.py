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


# Used for running_damage_f1_score
global_tp = [0.] * S.N_CLASSES
global_fp = [0.] * S.N_CLASSES
global_fn = [0.] * S.N_CLASSES
NUM_SAMPLES = 0
SAMPLES_SEEN = 0
# also used for running_damage_f1_score
CLASS_1 = [0.] * S.N_CLASSES
CLASS_1[1] = 1.
CLASS_1 = CLASS_1 * S.MASKSHAPE[0] * S.MASKSHAPE[1] * S.BATCH_SIZE
CLASS_1 = tf.constant(tf.reshape(CLASS_1, [S.BATCH_SIZE, S.MASKSHAPE[0] * S.MASKSHAPE[1], S.N_CLASSES]))
CLASS_2 = [0.] * S.N_CLASSES
CLASS_2[2] = 1.
CLASS_2 = CLASS_2 * S.MASKSHAPE[0] * S.MASKSHAPE[1] * S.BATCH_SIZE
CLASS_2 = tf.constant(tf.reshape(CLASS_2, [S.BATCH_SIZE, S.MASKSHAPE[0] * S.MASKSHAPE[1], S.N_CLASSES]))
CLASS_3 = [0.] * S.N_CLASSES
CLASS_3[3] = 1.
CLASS_3 = CLASS_3 * S.MASKSHAPE[0] * S.MASKSHAPE[1] * S.BATCH_SIZE
CLASS_3 = tf.constant(tf.reshape(CLASS_3, [S.BATCH_SIZE, S.MASKSHAPE[0] * S.MASKSHAPE[1], S.N_CLASSES]))
CLASS_4 = [0.] * S.N_CLASSES
CLASS_4[4] = 1.
CLASS_4 = CLASS_4 * S.MASKSHAPE[0] * S.MASKSHAPE[1] * S.BATCH_SIZE
CLASS_4 = tf.constant(tf.reshape(CLASS_4, [S.BATCH_SIZE, S.MASKSHAPE[0] * S.MASKSHAPE[1], S.N_CLASSES]))


def initialize_f1(num_samples: "Number of samples in dataset (e.g. `len(dataflow)`"):
    """
    To get around the fact that metrics are only passed a BATCH_SIZE number of samples
    per call (and to avoid RAM issues), we keep a running total of true positives, false
    positives, and false negatives that we update every time running_damage_f1_score() is
    called.  When that function is called as many times as there are samples in the dataset,
    it resets the tallies.

    This function initializes/resets those numbers.
    """
    global SAMPLES_SEEN
    global NUM_SAMPLES
    global global_tp, global_fp, global_fn
    NUM_SAMPLES = num_samples
    SAMPLES_SEEN = 0
    for i in range(1,5):
        global_tp[i] = 0
        global_fp[i] = 0
        global_fn[i] = 0


def _f1_stats(tp, fp, fn):
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)

    return f1, prec, rec


def sklearn_f1_score(y_true, y_pred):
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
    i.e. the percentage of building pixels that didn't go unnoticed.
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


@tf.function
def nonlogical_f1_score(y_true, y_pred):
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


@tf.function
def _isolate_class_1(y_pred):
    pr = y_pred * CLASS_1

    return pr

@tf.function
def _isolate_class_2(y_pred):
    pr = y_pred * CLASS_2 

    return pr

@tf.function
def _isolate_class_3(y_pred):
    pr = y_pred * CLASS_3

    return pr

@tf.function
def _isolate_class_4(y_pred):
    pr = y_pred * CLASS_4

    return pr

# df1 = 4 / sum((f1+epsilon)**-1 for f1 in [no_damage_f1, minor_damage_f1, major_damage_f1, destroyed_f1]), where epsilon = 1e-6
def running_damage_f1_score(y_true, y_pred):
    """
    To get around the fact that metrics are only passed a BATCH_SIZE number of samples
    per call (and to avoid RAM issues), we keep a running total of true positives, false
    positives, and false negatives that we update every time running_damage_f1_score() is
    called.  When this function is called as many times as there are samples in the dataset,
    it resets the tallies.

    Used to validate damage classification scores when this file is run as a stand alone script.
    """
    global global_tp, global_fp, global_fn, SAMPLES_SEEN, NUM_SAMPLES
    y_true = tf.cast(y_true, tf.float32)
    scores = {}

    for i in range(1, 5):
        # zero out everything but the current iteration's class of interest
        if i == 1:
            pr = _isolate_class_1(y_pred)
            gt = _isolate_class_1(y_true)
        elif i == 2:
            pr = _isolate_class_2(y_pred)
            gt = _isolate_class_2(y_true)
        elif i == 3:
            pr = _isolate_class_3(y_pred)
            gt = _isolate_class_3(y_true)
        elif i == 4:
            pr = _isolate_class_4(y_pred)
            gt = _isolate_class_4(y_true)

        # standard f1-score variables
        tp = tf.reduce_sum(gt * pr)
        fp = tf.reduce_sum(tf.clip_by_value(pr - gt, 0, 1))
        fn = tf.reduce_sum(tf.clip_by_value(gt - pr, 0, 1))

        # update global tally of this damage class' counts
        global_tp[i] += tp
        global_fp[i] += fp
        global_fn[i] += fn
       
        # use the global counts to calculate this damage class' F1-score
        prec = global_tp[i] / (global_tp[i] + global_fp[i] + 1e-8)
        rec = global_tp[i] / (global_tp[i] + global_fn[i] + 1e-8)

        score = 2 * prec * rec / (prec + rec + 1e-8)
        scores[i] = score

    # reset values when we've been called as many times as there are samples
    SAMPLES_SEEN += 1
    if SAMPLES_SEEN >= NUM_SAMPLES:
        initialize_f1(NUM_SAMPLES)

    # xView2 contest formula (harmonic mean of all 4 damage class F1-scores)
    df1 = 4 / np.sum([1 / (scores[i] + 1e-8) for i in range(1,5)])
    return df1


if __name__ == '__main__':
    import infer
    import math
    import train
    model = train.build_deeplab_model(classes=6, damage=True, train=False)
    S.BATCH_SIZE = 1
    model = train.load_weights(model, S.MODELSTRING)
    df = flow.DamagedDataflow(files=flow.get_validation_files(), batch_size=1, shuffle=True, return_postmask=True, return_stacked=True)
    totals = 0#[0,0,0]
    num = 1
    pbar = tqdm.tqdm(df, desc="Scoring")
    initialize_f1(len(df) * 2)

    for x, y_ in pbar:
        pred = model.predict(x)#np.expand_dims(x[j], axis=0))
        #y_pred = infer.convert_prediction(pred)
        #y_true = infer.convert_prediction(y_)
        num += 1
        scores = running_damage_f1_score(y_, pred)#sklearn.metrics.f1_score(y_true.astype(int), y_pred.astype(int), average='macro')
        pbar.set_description("%f" % (scores))#/ (num / 50)))
    print("{:7s}{:10s}{:10s}{:10s}".format("Class", "True Pos", "False Pos", "False Neg"))
    for i in range(1,5):
        print("{:7s}{:10.10s}{:10.10s}{:10.10s}".format(str(i), str(global_tp[i].numpy()), str(global_fp[i].numpy()), str(global_fn[i].numpy())))
