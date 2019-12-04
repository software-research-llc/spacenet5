import settings as S
import tensorflow as tf
import numpy as np
import flow
import pickle
import tqdm
import segmentation_models as sm
import infer
import train
import sklearn.metrics

MAXINMEM = 10

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
    model = train.build_model()
    S.BATCH_SIZE = 1
    model = train.load_weights(model)
    df = flow.Dataflow(files=flow.get_validation_files(), batch_size=1)
    total = 0
    i = 1
    pbar = tqdm.tqdm(df, desc="Scoring")
    for x,y in pbar:
        pred = model.predict(x)
        scores = sklearn.metrics.f1_score(np.round(y).astype(int).squeeze(), np.round(pred).astype(int).squeeze(), average='micro')
        total += scores
        pbar.set_description("%f" % (total / i))
        i += 1
