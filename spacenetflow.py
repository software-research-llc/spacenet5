#!/usr/bin/env python3
import sys
import glob
import os
import re
import random
import math
from collections import namedtuple
from inspect import getsourcefile
import plac
import scipy
import imageio
from skimage import io
from skimage.io import imread
from skimage.transform import resize
import networkx
import matplotlib.pyplot as plt
import numpy as np
import keras
import pandas as pd
import mpmath
import cv2
import tqdm
from keras.applications import xception
import interactatscope
import random
import preprocess as pp

BATCH_SIZE = 32

# Accuracy of the multiprecision floating point arithmetic library
mpmath.dps = 100
# The size of unmodified satellite images
CHIP_CANVAS_SIZE = [1300,1300,3]
# The target image size for input to the network
IMSHAPE = [256,256,3]
# The image size for skeletonized path networks (training TARGET images, not input samples)
TARGET_IMSHAPE = [256,256,1]
# The shape of the decoder output
DECODER_OUTPUT_SHAPE = IMSHAPE

# The file containing descriptions of road networks (linestrings) for training
TARGETFILE = "train_AOI_7_Moscow_geojson_roads_speed_wkt_weighted_simp.csv"
# The dataset we use (PS-RGB is panchromatic sharpened red-green-blue data)
DATASET = "PS-RGB"
# Where the satellite images are
DATADIR = "%s/data/train/AOI_7_Moscow/" % \
          os.path.abspath(os.path.dirname(getsourcefile(lambda:0)))


class SpacenetSequence(keras.utils.Sequence):
    def __init__(self, x_set: "List of paths to images",
                 y_set: "Associated targets", batch_size, transform=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.transform = transform

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        x, y = np.array([resize(get_image(file_name), IMSHAPE) for file_name in batch_x]), \
               np.array([self.y[Target.expand_imageid(imageid)].image() for imageid in batch_x])
        for idx in range(len(x)):
            x[idx] = pp.subtract_image_mean(x[idx])
            if self.transform:
                if random.random() > self.transform:
                    x[idx] = x[idx][::-1]
        return x,y
    @staticmethod
    def all(batch_size = BATCH_SIZE):
        imageids = get_filenames()
        return SpacenetSequence(imageids, TargetBundle(), batch_size)

def get_npy(filename=None, dataset="PS-RGB"):
    filename = get_file(filename=filename, dataset=dataset)
    return np.asarray(imageio.imread(filename))

def get_image(filename=None, dataset="PS-RGB"):
    filename = get_file(filename=filename, dataset=dataset)
    return io.imread(filename)

def get_file(filename=None, dataset="PS-RGB"):
    datadir = os.path.join(DATADIR, dataset.upper())
    if filename is None:
        files = os.listdir(datadir)
        filename = os.path.join(datadir, files[random.randint(0,len(files)-1)])
    if not os.path.exists(filename):
        filename = glob.glob("%s/*%s" % (datadir, filename))
        if filename:
            return filename[0]
    return filename

def get_filenames(filename=None, dataset="PS-RGB"):
    datadir = os.path.join(DATADIR, dataset.upper())
    if filename is None:
        return os.listdir(datadir)
    else:
        return glob.glob("%s/*%s*" % (datadir, filename))

def get_imageids():
    paths = get_filenames()
    return [Target.expand_imageid(path.replace(DATASET + "_", "")) for path in paths]

class TargetBundle:
    def __init__(self, transform=False):
        self.targets = {}
        i = 0
        imageids = get_imageids()
        for imageid in imageids:
            self.targets[imageid] = Target(imageid)
            i += 1
        self.add_df(Target.df)

    def add_df(self, df):
        for idx,linestring in enumerate(Target.df['WKT_Pix']):
            if linestring.lower().find("empty") != -1:
                continue
            imageid = Target.expand_imageid(Target.df['ImageId'][idx])
            try:
                weight = mpmath.mpf(Target.df['length_m'][idx]) / mpmath.mpf(Target.df['travel_time_s'][idx])
            except ZeroDivisionError:
                print("ZeroDivisionError: %s, %s, length = %s, time = %s" % (imageid,
                      linestring,
                      Target.df['length_m'][idx],
                      Target.df['travel_time_s'][idx]))
                weight = 0
            self.targets[imageid].add_linestring(linestring, weight)

    def __getitem__(self, idx):
        try:
            return self.targets[idx]
        except KeyError:
            try:
                return self.targets[Target.expand_imageid(idx)]
            except KeyError as exc:
                raise KeyError("cannot find `%s' or `%s' in target dict" %
                               (str(idx), Target.expand_imageid(idx)))

    def __len__(self, idx):
        return len(self.targets)

    def __getnext__(self, idx):
        for key in self.targets:
            yield self.targets[key]

class Target:
    regex = re.compile("[\d\.]+ [\d\.]+")
    df = pd.read_csv(TARGETFILE)
    idbase = os.listdir(DATADIR)[1].replace("PS-RGB_", "")

    def __init__(self, imageid):
        self.graph = networkx.Graph()
        self.imageid = imageid

    @staticmethod
    def expand_imageid(imageid):
        imageid = imageid.replace(DATASET + "_", "")
        regex = re.compile("chip(\d+)(.tif)?")
        m = re.search(regex, imageid)
        mstr = m.string[m.start():m.end()]
        mnum = m.groups()[0]
        if len(mnum) >= 5:
            ret = imageid
        else:
            num = "{:>5.5s}".format(mnum)
            ret = imageid.replace(mstr, "") + "chip" + num.replace(" ", "0")
        return ret

    def add_linestring(self, string, weight):
        if string.lower().find("empty") != -1:
            return
        edges = re.findall(Target.regex, string)
        for i in range(len(edges) - 1):
            x1,y1 = edges[i].split(" ")
            x2,y2 = edges[i+1].split(" ")
            x1,y1 = float(x1), float(y1)
            x2,y2 = float(x2), float(y2)
            self.graph.add_edge((x1,y1), (x2,y2), weight=weight)
        return self

    def image(self):
        img = np.zeros(CHIP_CANVAS_SIZE, dtype=np.int32)
        for edge in self.graph.edges():
            origin_x, origin_y = 0, 0
            x1,y1 = edge[0]
            x2,y2 = edge[1]
            x1,y1 = round(x1), round(y1)
            x2,y2 = round(x2), round(y2)
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 15)
#            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 10)
#        kernel = np.ones((1, 75))
#        img = cv2.dilate(img, kernel, iterations=1)
#        img = cv2.erode(img, kernel, iterations=1)
        img = img / 255
        img = resize(img, IMSHAPE)
        status, ret = cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY)
        return np.array(cv2.cvtColor(np.cast['float32'](ret), cv2.COLOR_RGB2GRAY)).reshape(TARGET_IMSHAPE)
