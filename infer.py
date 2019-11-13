import time
import random
import tensorflow as tf
import segmentation_models as sm
import numpy as np
import os
import sys
import skimage
import train
import flow
import cv2
import matplotlib.pyplot as plt
from settings import *


def compress_channels(mask):
    assert mask.shape == tuple(MASKSHAPE), f"expected {MASKSHAPE}, got {mask.shape}"
    cutoff = 0.03
    cmpr = np.zeros(TARGETSHAPE, dtype=np.uint8)

    for i in range(mask.shape[-1] - 1):
        cmpr[mask[:,:,i] > cutoff] = i

    return cmpr.squeeze()

def decompress_channels(img):
    assert [img.shape[0], img.shape[1]] == [TARGETSHAPE[0], TARGETSHAPE[1]], f"expected {TARGETSHAPE[:2]}, got {img.shape[:2]}"
    chans = [np.zeros(TARGETSHAPE[:2], dtype=np.uint8) for _ in CLASSES]

    for i, chan in enumerate(chans):
        chan[img == i] = 1

    return chans

def infer(model, pre, post=None):
    """Do prediction, returning (mask_image, damage_image)"""
    assert pre.ndim == 3, f"expected 3 dimensions, got {pre.shape}"
    premask = model.predict(np.expand_dims(pre, axis=0))

    if post is None:
        return premask.squeeze()
    postmask = model.predict(np.expand_dims(post, axis=0))

    return compress_channels(premask.squeeze()), compress_channels(postmask.squeeze())


def show_random(model):
    """Choose a random test image and display the results of inference"""
    files = list(flow.get_test_files())
    idx = random.randint(0,len(files)-1)
    preimg = skimage.io.imread(files[idx][0])
    postimg = skimage.io.imread(files[idx][1])

    mask,dmg = infer(model, preimg, postimg)

    fig = plt.figure()
    fig.add_subplot(2,5,1)
    plt.imshow(preimg)
    plt.title(files[idx][0])

    fig.add_subplot(2,5,2)
    plt.imshow(postimg)
    plt.title(files[idx][1])

    fig.add_subplot(2,5,3)
    plt.imshow(mask,cmap='gray')
    plt.title("Pre disaster")

    fig.add_subplot(2,5,4)
    plt.imshow(dmg.squeeze(),cmap='gray')
    plt.title("Post disaster")

    for i, chan in enumerate(decompress_channels(dmg.squeeze())):
        fig.add_subplot(2,5,5+i)
        plt.imshow(chan, cmap='gray')
        plt.title("Damage - {}".format(CLASSES[i]))

    plt.show()

if __name__ == "__main__":
    # Testing and inspection
    model = train.build_model()
    model = train.load_weights(model)
    while True:
        show_random(model)
        time.sleep(1)
