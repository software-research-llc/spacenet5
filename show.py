#!/usr/bin/env python3
from skimage import io
import matplotlib.pyplot as plt
import sys
import glob
import os
import plac
import random
import imageio
import numpy as np
from inspect import getsourcefile

DATADIR = "%s/data/train/AOI_7_Moscow/" % os.path.abspath(os.path.dirname(getsourcefile(lambda : 0)))

def get_npy(filename=None, dataset="PS-RGB"):
    filename = get_file(filename=filename, dataset=dataset)
    return np.asarray(imageio.imread(filename))

def get_image(filename=None, dataset="PS-RGB"):
    filename = get_file(filename=filename, dataset=dataset)
    return io.imread(filename)

def get_file(filename=None, dataset="PS-RGB"):
    datadir = os.path.join(DATADIR, dataset.upper())
    if not filename:
        files = os.listdir(datadir)
        filename = os.path.join(datadir, files[random.randint(0,len(files)-1)])
    if not os.path.exists(filename):
        filename = glob.glob("%s/*%s*" % (datadir, filename))
        if filename:
            return filename[0]
    return filename

def cli(dataset: ("One of MS, PAN, PS-RGB, or PS-MS", "option", "d")="PS-RGB",
        *filename: "The name of the file, or a substring within the name"):
    """Leave filename blank to choose a random image"""
    for f in filename:
        fpath = get_file(filename=f, dataset=dataset)
        print("Displaying %s" % fpath)
        im = io.imread(fpath)
        plt.imshow(im)
    if not filename:
        fpath = get_file(dataset=dataset)
        print("Displaying %s" % fpath)
        im = io.imread(fpath)
        plt.imshow(im)
    plt.show()

if __name__ == '__main__':
    plac.call(cli)
