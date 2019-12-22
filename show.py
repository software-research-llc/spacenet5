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

    for imgs, mask in df:
        pre = imgs[...,:3]
        post = imgs[...,3:]

        pred = model.predict(imgs)
        mask = infer.convert_prediction(mask)

        maxed = infer.convert_prediction(pred, argmax=True)
        pred, _ = infer.convert_prediction(pred, argmax=False)
        pred1 = pred[...,0]
        pred2 = pred[...,1]

        try:
            display_images([pre, post, maxed, pred1, pred2, mask], ["Pre", "Post", "Argmax", "Pred1", "Pred2", "Ground Truth"])
        except Exception as exc:
            [print(x.shape) for x in [pre,post,maxed,pred1,pred2,mask]]
            raise exc

def predict_and_show_no_argmax(df):
    return predict_and_show(df, argmax=False)


def show(df):
    i = 0
    for imgs, masks in df:
        if len(imgs) == 2:
            pre = imgs[0]
            post = imgs[1]
        else:
            pre = imgs
            post = imgs

        mask = infer.convert_prediction(y)

        prename = df.samples[i][0].img_name
        postname = df.samples[i][1].img_name

        display_images([pre, post, summed, averaged, meaned, mask], [prename, postname, "Sum", "Average", "Avg - mean(avg)", "Mask"])
        i += 1


def main(predict: ("Do prediction", "flag", "p"),
         image: ("Show this specific image", "option", "i")=""):

    df = flow.Dataflow(files=flow.get_validation_files(), shuffle=True, batch_size=1, buildings_only=True, return_stacked=True, transform=0.5, return_average=False)
    if image:
        for i in range(len(df.samples)):
            if image in df.samples[i][0].img_name or image in df.samples[i][1].img_name:
                df.samples = [df.samples[i]]

    if predict:
        predict_and_show(df)
    else:
        show(df)

if __name__ == '__main__':
    S.BATCH_SIZE = 1
    plac.call(main)
