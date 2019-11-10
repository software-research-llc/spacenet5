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


def infer(model, pre, post=None):
    """Do prediction, returning (mask_image, damage_image)"""
    assert pre.ndim == 3, f"expected 3 dimensions, got {pre.shape}"
    premask = model.predict(np.expand_dims(pre, axis=0))
    if post is None:
        return premask.squeeze()

    postmask = model.predict(np.expand_dims(post, axis=0))
    return premask.squeeze(), postmask.squeeze()


def show_random(model):
    """Choose a random test image and display the results of inference"""
    files = flow.get_test_files()
    idx = random.randint(0,len(files)-1)
    img = skimage.io.imread(files[idx])

    mask,dmg = infer(model, img)
    mask2 = np.zeros(SAMPLESHAPE)
    mask2[:,:,0:2] = mask[:,:,3:]

    fig = plt.figure()
    fig.add_subplot(2,2,1)
    plt.imshow(img)
    plt.title("Input")

    fig.add_subplot(2,2,2)
    plt.imshow(mask[:,:,:3])
    plt.title("Mask channels 0-2")

    fig.add_subplot(2,2,3)
    plt.imshow(mask2)
    plt.title("Mask channels 3-4")

    fig.add_subplot(2,2,4)
    plt.imshow(dmg.squeeze())
    plt.title("Damage")

    plt.show()

if __name__ == "__main__":
    # Testing and inspection
    model = train.build_model()
    model = train.load_weights(model)
    while True:
        show_random(model)
        time.sleep(1)
