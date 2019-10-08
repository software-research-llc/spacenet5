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
import preprocess

BATCH_SIZE = 45

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
TARGETFILES = [ "train_AOI_7_Moscow_geojson_roads_speed_wkt_weighted_simp.csv",
                "train_AOI_8_Mumbai_geojson_roads_speed_wkt_weighted_simp.csv" ]
# The dataset we use (PS-RGB is panchromatic sharpened red-green-blue data)
DATASET = "PS-RGB"
# The directory of this file
MYDIR = os.path.abspath(os.path.dirname(getsourcefile(lambda:0)))
CITIES = [ "AOI_7_Moscow", "AOI_8_Mumbai" ]#, "AOI_9_San_Juan" ]

# Where the satellite images are
BASEDIR = "%s/data/train/" % MYDIR


class SpacenetSequence(keras.utils.Sequence):
    def __init__(self, x_set: "List of paths to images",
                 y_set: "Associated targets", batch_size, transform=False):
        self.x, self.y = x_set, y_set
        random.shuffle(self.x)
        self.batch_size = batch_size
        self.transform = transform

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = []
        for ex in batch_x:
            for city in CITIES:
                try:
                    file_name = os.path.join(BASEDIR, city, DATASET, ex)
                    image = get_image(file_name)
                    x.append(resize(image, IMSHAPE))
                except Exception as exc:
                    pass

#        x, y = np.array([resize(get_image(file_name), IMSHAPE) for file_name in batch_x]), \
#               np.array([self.y[Target.expand_imageid(imageid)].image() for imageid in batch_x])
        x = np.array(x)
        y = np.array([self.y[Target.expand_imageid(imageid)].image() for imageid in batch_x])
        for idx in range(len(x)):
            if self.transform:
                if random.random() > self.transform:
                    x[idx] = preprocess.transform(x[idx])
        return x,y

    @staticmethod
    def all(batch_size = BATCH_SIZE, transform = False):
        imageids = []
        for i in range(len(CITIES)):
            imageids += get_filenames(datadir=BASEDIR + CITIES[i])
        return SpacenetSequence(imageids, TargetBundle(), batch_size=batch_size, transform=transform)

def get_npy(filename=None, dataset="PS-RGB"):
    filename = get_file(filename=filename, dataset=dataset)
    return np.asarray(imageio.imread(filename))

def get_image(filename=None, dataset="PS-RGB"):
    if not os.path.exists(str(filename)):
        for city in CITIES:
            trying = get_file(filename=filename, dataset=dataset, datadir=BASEDIR + city)
            if trying:
                filename = trying
                break
    if not os.path.exists(str(filename)):
        raise Exception("File not found: %s" % filename)
    return io.imread(filename)

def get_file(filename=None, dataset="PS-RGB", datadir=BASEDIR + CITIES[0]):
    if os.path.exists(str(filename)):
        return filename
    datadir = os.path.join(datadir, dataset.upper())
    if filename is None:
        files = os.listdir(datadir)
        filename = os.path.join(datadir, files[random.randint(0,len(files)-1)])
    if not os.path.exists(str(filename)):
        trying = glob.glob("%s/*%s.tif" % (datadir, filename))
        if trying:
            return trying[0]
        else:
            trying = re.search("chip[0]*(\d+)$", str(filename))
            if trying:
                trying = trying.groups()[0]
                trying = glob.glob("%s/*%s*.tif" % (datadir, trying))
                if trying:
                    filename = trying[0]
                else:
                    return None
    if not filename:
        return None
    return filename

def get_filenames(filename=None, dataset="PS-RGB", datadir=BASEDIR + CITIES[0]):
    datadir = os.path.join(datadir, dataset.upper())
    if filename is None:
        return os.listdir(datadir)
    else:
        return glob.glob("%s/*%s*" % (datadir, filename))

def get_imageids(datadir=BASEDIR + CITIES[0], dataset="PS-RGB"):
    paths = get_filenames(datadir=datadir, dataset=dataset)
    return [Target.expand_imageid(path.replace(datadir + "_", "")) for path in paths]

class TargetBundle:
    def __init__(self, transform=False):
        self.targets = {}
        i = 0
        imageids = []
        for city in CITIES:
            imageids += get_imageids(datadir=BASEDIR + city)
        for imageid in imageids:
            imageid = Target.expand_imageid(imageid)
            self.targets[imageid] = Target(imageid)
            i += 1
        self.add_df(Target.df)

    def add_df(self, df):
        for idx,linestring in enumerate(df['WKT_Pix']):
            if linestring.lower().find("empty") != -1:
                continue
            imageid = Target.expand_imageid(df['ImageId'][idx])
            try:
                weight = mpmath.mpf(df['length_m'][idx]) / mpmath.mpf(df['travel_time_s'][idx])
            except ZeroDivisionError:
                print("ZeroDivisionError: %s, %s, length = %s, time = %s" % (imageid,
                      linestring,
                      df['length_m'][idx],
                      df['travel_time_s'][idx]))
                weight = 0
            if imageid not in self.targets:
                imageid = imageid.replace("_chip", "_PS-RGB_chip")
            self.targets[imageid].add_linestring(linestring, weight)

    def __getitem__(self, idx):
        ret = self.targets.get(idx, None)
        if not ret:
            expanded = Target.expand_imageid(idx)
            ret = self.targets.get(expanded, None)
            if ret:
                return ret
            for key in self.targets.keys():
                if re.search(expanded, key):
                    return self.targets[key]
            for key in self.targets.keys():
                if re.search(idx, key):
                    return self.targets[key]
        else:
            return ret

    def __len__(self, idx):
        return len(self.targets)

    def __getnext__(self, idx):
        for key in self.targets:
            yield self.targets[key]

class Target:
    regex = re.compile("[\d\.]+ [\d\.]+")
    df = pd.read_csv(TARGETFILES[0])
    for targetfile in TARGETFILES[1:]:
        df.append(pd.read_csv(targetfile))

    def __init__(self, imageid):
        self.graph = networkx.Graph()
        self.imageid = imageid

    @staticmethod
    def expand_imageid(imageid):
        regex = re.compile("chip(\d+)(.tif)?")
        m = re.search(regex, imageid)
        mstr = m.string[m.start():m.end()]
        mnum = m.groups()[0]
        ret = imageid.replace(mstr, "") + "chip" + mnum
        return ret
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

    def chip(self):
        return re.search("_(chip\d+)", self.imageid).groups()[0]

    def image(self):
        img = np.zeros(CHIP_CANVAS_SIZE, dtype=np.int32)
        for edge in self.graph.edges():
            origin_x, origin_y = 0, 0
            x1,y1 = edge[0]
            x2,y2 = edge[1]
            x1,y1 = round(x1), round(y1)
            x2,y2 = round(x2), round(y2)
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 25)
#            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 10)
#        kernel = np.ones((1, 75))
#        img = cv2.dilate(img, kernel, iterations=1)
#        img = cv2.erode(img, kernel, iterations=1)
        img = img / 255
        img = resize(img, IMSHAPE)
        status, ret = cv2.threshold(img, 0.25, 1, cv2.THRESH_BINARY)
        return np.array(cv2.cvtColor(np.cast['float32'](ret), cv2.COLOR_RGB2GRAY)).reshape(TARGET_IMSHAPE)
