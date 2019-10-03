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
mpmath.dps = 100

IMSHAPE = (1300,1300,3)
TARGETFILE = "train_AOI_7_Moscow_geojson_roads_speed_wkt_weighted_simp.csv"
DATADIR = "%s/data/train/AOI_7_Moscow/" % \
          os.path.abspath(os.path.dirname(getsourcefile(lambda:0)))


class SpacenetSequence(keras.utils.Sequence):
    def __init__(self, x_set: "List of paths to images",
                 y_set: "Associated classes", batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([resize(get_image(file_name), IMSHAPE) for file_name in batch_x]), batch_y
            
    @staticmethod
    def all():
        imageids = ['chip%d' % i for i in range(1,2000)]
        return SpacenetSequence(imageids, imageids, 32)

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
        filename = glob.glob("%s/*%s*" % (datadir, filename))
        if filename:
            return filename[0]
    return filename

def get_filenames(filename=None, dataset="PS-RGB"):
    datadir = os.path.join(DATADIR, dataset.upper())
    if filename is None:
        return os.listdir(datadir)
    else:
        return glob.glob("%s/*%s*" % (datadir, filename))

class Target:
    regex = re.compile("[\d\.]+ [\d\.]+")
    df = pd.read_csv(TARGETFILE)

    def __init__(self, imageid):
        self.graph = networkx.Graph()
        self.imageid = imageid
        self.add_df(df=Target.df)

    @staticmethod
    def expand_imageid(imageid):
        regex = re.compile("chip(\d+)")
        m = re.search(regex, imageid)
        mstr = m.string[m.start():m.end()]
        mnum = m.groups()[0]
        if len(mnum) >= 5:
            ret = imageid
        else:
            num = "{:>5.5s}".format(mnum)
            ret = imageid.strip(mstr) + "chip" + num.replace(" ", "0")
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
#        for edge in edges:
#            p1,p2 = edge.split(',')
#            x1,y1 = p1.strip().split(" ")
#            x2,y2 = p2.strip().split(" ")
#            x1,y1 = float(x1), float(y1)
#            x2,y2 = float(x2), float(y2)
#            self.graph.add_edge((x1,y1), (x2, y2), weight=weight)#(float(x1), float(x2)), (float(y1), float(y2)), weight=weight)
        return self

    def add_df(self, df):
        for idx,linestring in enumerate(Target.df['WKT_Pix']):
            if Target.expand_imageid(Target.df['ImageId'][idx]).find(Target.expand_imageid(self.imageid)) != -1:
                weight = mpmath.mpf(Target.df['length_m'][idx]) / mpmath.mpf(Target.df['travel_time_s'][idx])
                self.add_linestring(linestring, weight)

    def image(self):
        img = np.zeros(IMSHAPE)
#        (IMSHAPE[0], IMSHAPE[1]))
        for edge in self.graph.edges:
            origin_x, origin_y = 0, 0
            x1,y1 = edge[0]
            x2,y2 = edge[1]
            x1,y1 = round(x1), round(y1)
            x2,y2 = round(x2), round(y2)
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 20)
#        kernel = np.ones((1, 75))
#        img = cv2.dilate(img, kernel, iterations=1)
#        img = cv2.erode(img, kernel, iterations=1)
        img = img / 255
#        plt.imshow(img)
#        fig = plt.figure()
#        plt.imshow(get_image(self.imageid))
#        plt.show()
        return img
