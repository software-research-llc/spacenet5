import time
import random
import tensorflow as tf
import numpy as np
import os
import sys
import skimage
import train
import flow
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize
import settings as S
import score
import logging

logger = logging.getLogger(__name__)


def convert_prediction(pred, argmax=True):
    """
    Turn a model's prediction output into a grayscale segmentation mask.
    """
    x = pred.squeeze().reshape(S.MASKSHAPE)
    if argmax is True:
        return np.argmax(x, axis=2)
    else:
        return x[...,0:3], x[...,3:]


def weave_pred(pred):
    img = []
    for p in pred:
        x = convert_prediction(p)
        img.append(x)
    return flow.Target.weave(img)


def compress_channels_old(mask:np.ndarray):
    """
    Compress an N-dimensional image, i.e. WxHxN, where N is the number of channels.

    Returns a grayscale (single channel) image with each pixel value being an integer
    that represents the class of the original input image at that pixel location.
    """
    assert mask.shape == tuple(S.MASKSHAPE), f"expected shape {MASKSHAPE}, got {mask.shape}"
    cutoff = 0.50
    cmpr = np.zeros(S.TARGETSHAPE, dtype=np.uint8)

    for i in range(1, mask.shape[-1]):
        cmpr[mask[:,:,i] > cutoff] = i

    return cmpr.squeeze()


def compress_channels(img):
    assert img.shape == [1,S.MASKSHAPE[0],S.MASKSHAPE[1],S.N_CLASSES], str(img.shape)
    return np.argmax(img, axis=2)


def decompress_channels(img:np.ndarray):
    """
    Decompress an image that was compressed by compress_channels().

    Returns one grayscale image per object class (see settings.py).

    Note: use np.stack(decompress_channels(img), axis=2) to get an N-channel image.
    """
    assert img.shape[:2] == tuple(S.TARGETSHAPE[:2]), f"expected shape {TARGETSHAPE[:2]}, got {img.shape}"
    chans = [np.zeros(S.TARGETSHAPE[:2], dtype=np.uint8) for _ in CLASSES]

    for i, chan in enumerate(chans):
        chan[img == i] = 1

    return chans


def infer(model, pre:np.ndarray, post:np.ndarray=None, compress:bool=True):
    """
    Do prediction.  If compress is True, calls compress_channels() on the return images.

    Returns a (mask_image, damage_image) tuple.
    """
    assert pre.ndim == 3, f"expected 3 dimensions, got {pre.shape}"
    if post is not None:
        assert post.ndim == 3, f"expected 3 dimensions, got {post.shape}"

    premask = model.predict(np.expand_dims(pre, axis=0))
    if post is None:
        return premask.squeeze()

    postmask = model.predict(np.expand_dims(post, axis=0))
    if compress:
        return compress_channels(premask.squeeze()), compress_channels(postmask.squeeze())
    else:
        return premask.squeeze(), postmask.squeeze()


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
    df=flow.Dataflow(files=flow.get_validation_files(), batch_size=1, shuffle=True)
    while True:
        show_random(model, df)
        time.sleep(1)
