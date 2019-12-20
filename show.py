import test
import settings as S
import flow
import train
import infer
import matplotlib.pyplot as plt
import numpy as np
import plac
import time
import test


def display_images(images, names=None):
    fig = plt.figure()
    sqrt = int(np.ceil(np.sqrt(len(images))))

    for i in range(len(images)):
        fig.add_subplot(sqrt, sqrt, i+1)
        plt.imshow(images[i].squeeze())
        if names is not None:
            plt.title(names[i])

    plt.show()


def predict_and_show(df, argmax=True):
    model = train.build_model()
    model = train.load_weights(model)

    for (pre,post),mask in df:
        mask = infer.convert_prediction(mask)
        pred = model.predict((pre,post))

        maxed = infer.convert_prediction(pred, argmax=True)
        pred1,pred2 = infer.convert_prediction(pred, argmax=False)

        display_images([pre, post, maxed, pred1, pred2, mask], ["Pre", "Post", "Argmax", "Pred1", "Pred2", "Ground Truth"])


def predict_and_show_no_argmax(df):
    return predict_and_show(df, argmax=False)


def show(df, channels):
    i = 0
    for (x1,x2),y in df:
        pre = infer.weave(x1)
        post = infer.weave(x2)
        mask = infer.weave_pred(y)

        prename = df.samples[i][0].img_name
        postname = df.samples[i][1].img_name

        premask = df.samples[i][0].multichannelchipmask()
        premask = infer.weave_pred(premask)

        display_images([pre, post, premask, mask], [prename, postname, "Pre mask", "Post mask"])
        i += 1
        time.sleep(0.5)


def main(predict: ("Do prediction", "flag", "p"),
         argmax: ("Don't argmax() over the channel axis", "flag", "a"),
         image: ("Show this specific image", "option", "i")=""):

    df = flow.Dataflow(files=flow.get_validation_files(), shuffle=True, batch_size=1, buildings_only=True, return_stacked=False)
    if image:
        for i in range(len(df.samples)):
            if image in df.samples[i][0].img_name or image in df.samples[i][1].img_name:
                df.samples = [df.samples[i]]

    if predict:
        if argmax:
            predict_and_show_no_argmax(df)
        else:
            predict_and_show(df)
    else:
        show(df)

if __name__ == '__main__':
    S.BATCH_SIZE = 1
    plac.call(main)
