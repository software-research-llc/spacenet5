import numpy as np
import flow
import pickle
import sklearn
import tqdm
import segmentation_models as sm
import infer
import train


def f1score(model:sm.Unet, picklefile:str="validationflow.pickle"):
    with open(picklefile, "rb") as f:
        valid_df = pickle.load(f)
    y_true = []
    y_pred = []
    for x,y in tqdm.tqdm(valid_df):
        y_true.append(y)

        x_ = infer.infer(model, x.squeeze(), compress=True)
        
        y_pred.append(x_)

    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    return sklearn.metrics.f1_score(y_true, y_pred)


if __name__ == '__main__':
    model = train.build_model()
    model = train.load_weights(model)
    score = f1score(model)
    print("F1 Score: %f" % score)
