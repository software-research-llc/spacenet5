import cv2
import tensorflow as tf
import pandas as pd
import shapely
import json
import numpy as np
from settings import *

class Dataflow(keras.utils.Sequence):
    def __init__(self, xset, yset, batch_size, transform=False, shuffle=False):
        self.transform = transform
        self.shuffle = shuffle
        self.batchsize = batch_size
        self.x = xset
        self.y = yset
        if transform:
            self.datagen_args = dict(rotation_range=25,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.2,
                                     shear_range=0.2)
        else:
            self.datagen_args = {}
        self.image_datagen = ImageDataGenerator(**data_gen_args)
        self.mask_datagen = ImageDataGenerator(**data_gen_args)

    def __len__(self):
        length = int(np.ceil(len(self.x) / float(self.batch_size)))
        return length

    def __getitem__(self, idx):
        x = np.array(self.x[idx])
        y = np.array(self.y[idx])
        transform_dict = { 'theta': 90, 'shear': 0.1 }
        if random.random() < float(self.transform):
            x = self.image_datagen.apply_transform(x, trans_dict)
            y = self.image_datagen.apply_transform(y, trans_dict)
        return x,y

class Building:
    def coords(self):
        wkt = self.wkt
        pairs = []
        for pair in re.findall(r"\-?\d+\.?\d+ \-?\d+\.?\d+", wkt):
            xy = pair.split(" ")
            x,y = float(xy[0]), float(xy[1])
            pairs.append((x,y))
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
            for prop in feature['properties']:
                if prop['feature_type'] != 'building':
                    continue
                b = Building()
                b.klass = prop['subtype']
                if b.klass not in CLASSES:
                    CLASSES.append(b.klass)
                    CLASSES.sort()
                b.wkt = prop['wkt']
                b.uid = prop['uid']
                self.buildings.append(b)

    def image(self):
        img = np.zeros(TARGETSHAPE)
        for b in self.buildings:
            cv2.fillConvexPoly(img, b.coords(), b.color())
        return img

class TargetBundle(list):
    def __init__(self):
        self.aref = self.__getitem__
        self.__getitem__ = self.getitem

    def getitem(self, idx):
        data = (self.aref(idx))
        return data

