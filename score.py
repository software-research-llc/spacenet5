import flow
import pickle
import sklearn
import tqdm
import segmentation_models as sm
import infer
import train


def f1score(model:sm.Unet, picklefile:str="validationflow.pickle"):
    valid_df = pickle.load(picklefile)
    y_true = []
    y_pred = []
    for x,y in tqdm.tqdm(valid_df):
        y_true.append(x)
        y_true.append(y)

        x_, y_ = infer.infer(model, pre=x, post=y, compress=True)
        
        y_pred.append(x_)
        y_pred.append(y_)

    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    return sklearn.metrics.f1_score(y_true, y_pred)


if __name__ == '__main__':
    model = train.build_model()
    model = train.load_weights()
    score = f1score(model)
    print("F1 Score: %f" % score)
