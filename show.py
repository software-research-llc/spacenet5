#!/usr/bin/env python3
from skimage import io
import matplotlib.pyplot as plt
import sys
import glob
import os
import plac
import random
from inspect import getsourcefile

DATADIR = "%s/data/train/AOI_7_Moscow/" % os.path.abspath(os.path.dirname(getsourcefile(lambda : 0)))

def get_file(dataset="PS-RGB", filename = None):
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
        *filename: "The name of the file, or the first characters of it"):
    for f in filename:
        fpath = get_file(dataset, f)
        print("Displaying %s" % fpath)
        im = io.imread(fpath)
        plt.imshow(im)
    if not filename:
        fpath = get_file(dataset)
        print("Displaying %s" % fpath)
        im = io.imread(fpath)
        plt.imshow(im)
    plt.show()

if __name__ == '__main__':
    plac.call(cli)
