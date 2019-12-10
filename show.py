import settings as S
import flow
import train
import infer
import matplotlib.pyplot as plt
import numpy as np


def display_images(images, names=None):
    fig = plt.figure()
    sqrt = int(np.ceil(np.sqrt(len(images))))

    for i in range(len(images)):
        fig.add_subplot(sqrt, sqrt, i+1)
        plt.imshow(images[i])
        if names is not None:
            plt.title(names[i])

    plt.show()


if __name__ == '__main__':
    df = flow.Dataflow(files=flow.get_validation_files(), shuffle=True)
    model = train.build_model()
    model = train.load_weights(model)

    for x,y in df:
        pred = model.predict(x)

        pred = infer.weave_pred(pred.astype(int))
        mask = infer.weave_pred(y.astype(int))
        pre = infer.weave(x[0])
        post = infer.weave(x[1])

        display_images([pre, post, mask, pred], ["Pre", "Post", "Ground Truth", "Prediction"])
