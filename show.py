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
        plt.imshow(images[i])
        if names is not None:
            plt.title(names[i])

    plt.show()


def predict_and_show(df):
    model = train.build_model()
    model = train.load_weights(model)

    for x,y in df:
        pred = model.predict(x)

        pred = infer.weave_pred(pred)
        pred = test.randomize_damage(pred)
        mask = infer.weave_pred(y)
        pre = infer.weave(x[0])
        post = infer.weave(x[1])

        display_images([pre, post, mask, pred], ["Pre", "Post", "Ground Truth", "Prediction"])


def show(df):
    i = 0
    for (x1,x2),y in df:
        pre = infer.weave(x1)
        post = infer.weave(x2)
        mask = infer.weave_pred(y)

        prename = df.samples[i][0].img_name
        postname = df.samples[i][1].img_name

        premask = df.samples[i][0].multichannelchipmask()
        premask = infer.weave_pred(premask)
        premask = test.randomize_damage(premask)

        display_images([pre, post, premask, mask], [prename, postname, "Pre mask", "Post mask"])
        i += 1
        time.sleep(0.5)


def main(predict: ("Do prediction", "flag", "p")):
    df = flow.Dataflow(files=flow.get_validation_files(), shuffle=True)
    if predict:
        predict_and_show(df)
    else:
        show(df)

if __name__ == '__main__':
    plac.call(main)
