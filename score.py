import tensorflow as tf
import numpy as np
import flow
import pickle
import tqdm
import segmentation_models as sm
import infer
import train

MAXINMEM = 10

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

if __name__ == '__main__':
    model = train.build_model()
    model = train.load_weights(model)
    score = f1score(model)
