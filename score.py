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


TP = 0
FP = 0
FN = 0
TN = 0

def _f1_stats(tp, fp, fn):
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * prec * rec / (prec + rec)

    return f1, prec, rec


def f1score(y_true, y_pred, threshold=0.1):
    y_true = y_true.squeeze()#.astype(np.uint8)
    y_pred = y_pred.squeeze()#.astype(np.uint8)
    truth = y_true#y_true[y_true == 1]
    pred = y_pred#y_pred[y_pred > threshold]
    
    tp = np.sum(np.clip(truth * pred, 0, 1))
    fp = np.sum(np.clip(pred - truth, 0, 1))
    fn = np.sum(np.clip(truth - pred, 0, 1))

    return _f1_stats(tp, fp, fn)


def tf1score(y_true, y_pred):
    tp = tf.reduce_sum(tf.clip_by_value(y_true * y_pred, 0, 1))
    fp = tf.reduce_sum(tf.clip_by_value(y_pred - y_true, 0, 1))
    fn = tf.reduce_sum(tf.clip_by_value(y_true - y_pred, 0, 1))

    return _f1_stats(tp, fp, fn)


def f1_loss(y_true, y_pred):
    return 1 - tf1score(tf.cast(y_true, tf.float32), y_pred)[0]


def f1_score(y_true, y_pred):
        return sklearn.metrics.f1_score(y_true.ravel().astype(int),
                                        y_pred.ravel().astype(int),
                                        average='micro',
                                        labels=[1,2,3,4,5])


def iou_score(y_true, y_pred):
    gt = tf.cast(y_true, tf.bool)
    pr = tf.cast(y_pred, tf.bool)

    intersection = K.sum(gt * pr, axis=3)
    union = K.sum(gt + pr, axis=3) - intersection

    if union > 0:
        score = intersection / union
    else:
        score = 0
    return score


def my_f1score(y_true, y_pred):
    global TP, FP, FN, TN
    tp,fp,tn,fn = 0,0,0,0

    true = y_true.ravel()
    pred = y_pred.ravel()
    for i in range(len(true)):
        if true[i] == pred[i] and true[i] > 0:
            tp += 1
        elif true[i] == pred[i]:
            tn += 1
        else:
            fp += 1

    TP += tp
    FP += fp
    TN += tn
    FN += fn
    if tp == 0:
        if fp == 0 and fn == 0:
            return 1,1,1
        return 0,0,0
    prec = TP / (TP + FP)
    rec = TP / (TP + FN)
    return 2 * prec * rec / (prec + rec), prec, rec

"""
def f1_score(actual, predicted):
    actual = actual.astype(bool).astype(np.uint8)
    predicted = predicted.astype(bool).astype(np.uint8)
    tp = tf.compat.v1.count_nonzero(predicted * actual)
    tn = tf.compat.v1.count_nonzero((predicted - 1) * (actual - 1))
    fp = tf.compat.v1.count_nonzero(predicted * (actual - 1))
    fn = tf.compat.v1.count_nonzero((predicted - 1) * actual)
    
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)

    return f1


def f1score(model:sm.Unet, picklefile:str="validationflow.pickle"):
    with open(picklefile, "rb") as f:
        valid_df = pickle.load(f)
    y_true = []
    y_pred = []
    i = 0
    for x,y in tqdm.tqdm(valid_df):
        y_true.append(y)
        x_ = infer.infer(model, x.squeeze(), compress=True)
        y_pred.append(x_)
        i += 1
        if i % MAXINMEM == 0:
            print("F1 Score: %f" % f1_score(np.array(y_true), np.array(y_pred)))
            y_true.clear()
            y_pred.clear()

    score = f1_score(y_true, y_pred)
    return score
"""

if __name__ == '__main__':
    import infer
    import math
    import train
    model = train.build_model()
    S.BATCH_SIZE = 1
    model = train.load_weights(model)
    df = flow.Dataflow(files=flow.get_validation_files(), batch_size=1, shuffle=False)
    totals = 0#[0,0,0]
    i = 1
    pbar = tqdm.tqdm(df, desc="Scoring")
    for x,y in pbar:
        pred = model.predict(x)
        y_true = infer.convert_prediction(y)
        y_pred = infer.convert_prediction(pred)
        y_true[y_true > 1] = 1
        y_pred[y_pred > 1] = 1
        # skip images with no buildings
        if len(y_true[y_true != 0]) == 0:
            if len(y_pred[y_pred != 0]) == 0:
                scores = 1.
            else:
                logger.debug("Skipping image with no buildings")
                continue
        else:
            scores = f1_score(y_true, y_pred)#sklearn.metrics.f1_score(y_true.astype(int), y_pred.astype(int), average='macro')
        totals += scores
        #for i in range(len(scores)):
        #    totals[i] += scores[i]
        pbar.set_description("%f" % (totals / i))
        #pbar.set_description("%f (%f/%f)" % (totals[0],totals[1],totals[2]))
        i += 1
