import time
import random
import tensorflow as tf
import numpy as np
import os
import sys
import skimage
import train
from flow import Dataflow, BuildingDataflow, Target, get_training_files, get_validation_files, get_test_files
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
import settings as S
import score
import logging
from scipy.ndimage import label, find_objects


logger = logging.getLogger(__name__)


def convert_prediction(pred, argmax=True, threshold=None, focus_upper=False):
    """
    Turn a model's prediction output into a grayscale segmentation mask.

    Takes a one hot encoded (batch_size, width * height, num_classes)
    array and converts.
    """
    x = pred.squeeze().reshape(S.MASKSHAPE[:2] + [-1])

    if focus_upper is True:
        # double the predicted probability of high-channel (damaged building) pixels
        one = x[...,0:3]
        two = x[...,3:] * 2
        x = np.dstack([one,two])

    if argmax is True:
        if isinstance(threshold, float):
            x[x<threshold] = 0
            x[:,:,0] = 0
        return np.argmax(x, axis=2)
    else:
        return x[...,0:3], x[...,3:]


def weave_pred(pred):
    img = []
    for p in pred:
        x = convert_prediction(p)
        img.append(x)
    return Target.weave(img)


def weave_pred_no_argmax(pred):
    img = []
    for p in pred:
        x, _ = convert_prediction(p, argmax=False)
        img.append(x)
    return Target.weave(img)[...,1]


def weave(chips):
    """
    Stitch together 1/16th size squares of an image back to the original image.
    """
    return Target.weave(chips)


def bounding_rectangles(img, diagonals=True):
    """
    Return bounding boxes for contiguous blobs within an image.
    """
    if diagonals is True:
        # consider diagonally connected pixels as a single shape
        struct = [[1,1,1],[1,1,1],[1,1,1]]
    else:
        struct = None
    rects = label(img, structure=struct)
    objs = find_objects(rects[0])#, max_label=4)
    return objs


def show_random(model, df):
    """
    Choose a random test image pair and display the results of inference.

    Returns None.
    """
    threshold = 0.50
    idx = random.randint(0, len(df) - 1)
    img, y_true = df.__getitem__(idx, preprocess=False)
    net_img, _ = df.__getitem__(idx, preprocess=True)
    mask_img1, mask_img2 = convert_prediction(model.predict(net_img), argmax=False)

    fig = plt.figure()
    fig.add_subplot(1,3,1)
    plt.imshow(img.squeeze())
    plt.title(df.samples[idx].img_name)

    fig.add_subplot(1,3,2)
    plt.imshow(mask_img1)
    plt.title("Mask channels 0:3")

    fig.add_subplot(1,3,3)
    plt.imshow(mask_img2)
    plt.title("Mask channels 3:")

    scores = score.f1_score(convert_prediction(y_true), np.argmax(np.dstack([mask_img1, mask_img2]), axis=2))
    gt = convert_prediction(y_true, argmax=True)
    if len(gt[gt != 0]) == 0:
        plt.close(fig)
        logger.info("Skipping image with no buildings")
        return show_random(model, df)
    print("F1-Score: {}\n".format(scores))
    plt.show()
    return


if __name__ == "__main__":
    # Testing and inspection
    model = train.build_model()
    model = train.load_weights(model)
    S.BATCH_SIZE = 1
    df=Dataflow(files=get_validation_files(), batch_size=1, shuffle=True)
    while True:
        show_random(model, df)
        time.sleep(1)
