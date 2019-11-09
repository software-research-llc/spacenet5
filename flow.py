import threading
import random
import os
import cv2
import tensorflow as tf
import pandas as pd
import shapely
import json
import numpy as np
from settings import *
import matplotlib.pyplot as plt
import skimage
from keras.preprocessing.image import ImageDataGenerator
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Cacher(threading.Thread):
    """Cache a tiny amount of input data (global interpreter lock
       is released during blocking IO)"""
    def __init__(self, df):
        threading.Thread.__init__(self, daemon=True)
        self.cache = {}
        self.df = df
        self.idx = 0
        self.lock = threading.Lock()
        self.batch_size = df.batch_size
        self.samples = df.samples

    def run(self):
        while True:
            try:
                idx = self.idx
                for i in range(2):
                    if not self.cache.get(idx+i, None):
                        self.cache[idx] = self.df.get_idx(idx+i)
            except RuntimeError:
                pass

class Dataflow(tf.keras.utils.Sequence):
    def __init__(self, batch_size=1, samples=None, transform=False, shuffle=False, validation_set=False):
        self.transform = transform
        self.shuffle = shuffle
        self.batch_size = batch_size

        if transform:
            data_gen_args = dict(rotation_range=25,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.2,
                                     shear_range=0.2)
        else:
            data_gen_args = {}
        self.image_datagen = ImageDataGenerator(**data_gen_args)
        self.mask_datagen = ImageDataGenerator(**data_gen_args)

        if samples:
            self.samples = samples
        else:
            files = []
            for directory in LABELDIRS:
                files += [os.path.join(directory, f) for f in os.listdir(directory)]
            length = len(files)
            if validation_set:
                files = files[int(length*0.9):]
            else:
                files = files[:int(length*0.9)]
            self.samples = [Target.from_file(f) for f in files]
        
        self.cacher = Cacher(self)
        self.cacher.start()

    def __len__(self):
        """Length of this dataflow in units of batch_size"""
        length = int(np.ceil(len(self.samples) / float(self.batch_size)))
        return length

    def __getitem__(self, idx):
        """Return images,masks := numpy arrays of size batch_size"""
        while not self.cacher.cache.get(idx, None):
            self.cacher.idx = idx
        ret = self.cacher.cache[idx]
        del self.cacher.cache[idx]
        return ret

    def get_idx(self, idx):
        x = np.array([ex.image() for ex in self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]])
        y = np.array([tgt.mask() for tgt in self.samples[idx * self.batch_size:(idx + 1) * self.batch_size]])
        """"
        trans_dict = { 'theta': 90, 'shear': 0.1 }
        for i in range(len(x)):
            if random.random() < float(self.transform):
                x[i] = self.image_datagen.apply_transform(x[i], trans_dict)
                y[i] = self.image_datagen.apply_transform(y[i], trans_dict)
        """
        return x,y

class Building:
    def __init__(self):
        self.wkt = None

    def coords(self):
        wkt = self.wkt
        pairs = []
        for pair in re.findall(r"\-?\d+\.?\d+ \-?\d+\.?\d+", wkt):
            xy = pair.split(" ")
            x,y = round(float(xy[0])), round(float(xy[1]))
            pairs.append(np.array([x,y]))
        return np.array(pairs)

    def color(self):
        return CLASSES.index(self.klass)

class Target:
    def __init__(self, text):
        self.buildings = []
        self.parse_json(text)

    def parse_json(self, text):
        data = json.loads(text)
        self.img_name = data['metadata']['img_name']
        self.metadata = data['metadata']

        for feature in data['features']['xy']:
            prop = feature['properties']
            if prop['feature_type'] != 'building':
                continue
            b = Building()
            try:
                b.klass = prop['subtype']
            except:
                logger.debug("no subtype for building {} in {}".format(prop['uid'], self.img_name))
                b.klass = 'no-damage'

            if b.klass not in CLASSES:
                CLASSES.append(b.klass)
                CLASSES.sort()
            b.wkt = feature['wkt']
            b.uid = prop['uid']
            self.buildings.append(b)

    def mask(self):
        img = np.zeros(TARGETSHAPE)
        for b in self.buildings:
            if len(b.coords() > 1):
                cv2.fillConvexPoly(img, b.coords(), b.color())
        return img

    def image(self):
        for path in IMAGEDIRS:
            fullpath = os.path.join(path, self.img_name)
            try:
                return skimage.io.imread(fullpath)
            except OSError as exc:
                continue
        raise exc

    @staticmethod
    def from_file(filename):
        with open(filename) as f:
            return Target(f.read())

if __name__ == '__main__':
    df = Dataflow()
    fig = plt.figure()

    fig.add_subplot(1,2,1)
    plt.imshow(df.samples[0].image())
    plt.title("image")

    fig.add_subplot(1,2,2)
    plt.imshow(df.samples[0].mask().squeeze(), cmap='plasma')
    plt.title("mask")

    plt.show()
