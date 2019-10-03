#!/usr/bin/env python3
import sys
import glob
import os
import re
import random
from collections import namedtuple
from inspect import getsourcefile
import plac
import imageio
from skimage import io
from skimage.io import imread
from skimage.transform import resize
import networkx
import matplotlib.pyplot as plt
import numpy as np
import keras
import pandas as pd

TARGETFILE = "train_AOI_7_Moscow_geojson_roads_speed_wkt_weighted_raw.csv"
DATADIR = "%s/data/train/AOI_7_Moscow/" % \
          os.path.abspath(os.path.dirname(getsourcefile(lambda : 0)))


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

        return np.array([
            resize(imread(file_name), (299, 299))
               for file_name in batch_x]), np.array(batch_y)

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
    regex = re.compile("[\d\.]+ [\d]+\.?[\d]*")
    df = pd.read_csv(TARGETFILE)

    def __init__(self, imageid=None):
        self.image = np.zeros((299, 299, 1))
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

    def add_linestring(self, string):
        if string.lower().find("empty") != -1:
            return
        line = string.replace(", ", ",").strip('"').strip("LINESTRING").strip(")").strip("(")
        edges = re.findall(Target.regex, line)
        for edge in edges:
            x,y = edge.split(" ")
            self.graph.add_edge(float(x), float(y))
        return self

    def add_df(self, df):
        for idx,linestring in enumerate(Target.df['WKT_Pix']):
            if Target.expand_imageid(Target.df['ImageId'][idx]).find(self.imageid) != -1:
                self.add_linestring(linestring)

