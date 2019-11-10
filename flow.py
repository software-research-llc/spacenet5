import glob
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


class Dataflow(tf.keras.utils.Sequence):
    """A keras.utils.Sequence subclass to feed data to the model"""
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

    def __len__(self):
        """Length of this dataflow in units of batch_size"""
        length = int(np.ceil(len(self.samples) / float(self.batch_size)))
        return length

    def __getitem__(self, idx):
        """Return images,masks := numpy arrays of size batch_size"""
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
    """Carries the data for a single building"""
    def __init__(self):
        self.wkt = None
        self._coords = None

    def coords(self):
        """Parses the WKT data and caches it for subsequent calls"""
        if self._coords is not None:
            return self._coords
        wkt = self.wkt
        pairs = []
        for pair in re.findall(r"\-?\d+\.?\d+ \-?\d+\.?\d+", wkt):
            xy = pair.split(" ")
            x,y = round(float(xy[0])), round(float(xy[1]))
            pairs.append(np.array([x,y]))
        self._coords = np.array(pairs)
        return self._coords

    def color(self, scale=False):
        """Get the color value for a building subtype (i.e. index into CLASSES)"""
        ret = CLASSES.index(self.klass)
        if scale:
            ret = ret / N_CLASSES
        return ret


class Target:
    """Target objects provide filenames, metadata, input images, and masks for training"""
    def __init__(self, text):
        self.buildings = []
        self.parse_json(text)

    def parse_json(self, text):
        """Parse a JSON formatted string and assign instance variables from it"""
        data = json.loads(text)
        self.img_name = data['metadata']['img_name']
        self.metadata = data['metadata']

        for feature in data['features']['xy']:
            prop = feature['properties']
            if prop['feature_type'] != 'building':
                continue
            b = Building()
            b.klass = prop.get('subtype', "no-damage")
           
            if b.klass not in CLASSES:
                logger.error(f"Unrecognized building subtype: {b.klass}")

            b.wkt = feature.get('wkt', None)
            b.uid = prop['uid']
            self.buildings.append(b)

    def mask(self, img=None):
        """Get the target mask for supervised training of the model"""
        if img is None:
            img = np.zeros(TARGETSHAPE)
        for b in self.buildings:
            coords = b.coords()
            if len(coords > 1):
                cv2.fillConvexPoly(img, coords, b.color())
        return img

    def image(self):
        """Get the input image (i.e. satellite chip) to feed to the model"""
        for path in IMAGEDIRS:
            fullpath = os.path.join(path, self.img_name)
            try:
                return skimage.io.imread(fullpath)
            except OSError as exc:
                continue
        raise exc

    @staticmethod
    def from_file(filename):
        """Create a Target object from the path of a .JSON file"""
        with open(filename) as f:
            return Target(f.read())


def get_test_files():
    """Return a list of the paths of images belonging to the test set"""
    prefiles = []
    postfiles = []
    sortfunc = lambda x: os.path.basename(x)
    for d in TESTDIRS:
        prefiles += glob.glob(os.path.join(d, "*pre*"))
        postfiles += glob.glob(os.path.join(d, "*post*"))
    return zip(sorted(prefiles, key=sortfunc), sorted(postfiles, key=sortfunc))


if __name__ == '__main__':
    # Testing and data inspection
    import time
    df = Dataflow(batch_size=1)
    while True:
        idx = random.randint(0,len(df) - 1)
        fig = plt.figure()

        fig.add_subplot(1,3,1)
        plt.imshow(df.samples[idx].image())
        plt.title(df.samples[idx].img_name)

        fig.add_subplot(1,3,2)
        plt.imshow(df.samples[idx].image())
        colormap = {0: 'red', 1: 'blue', 2: 'green', 3: 'purple', 4: 'orange', 5: 'yellow', 6: 'brown' }
        for b in df.samples[idx].buildings:
            plt.plot(b.coords()[:,0], b.coords()[:,1], color=colormap[b.color()])
        plt.title("image overlaid with mask")

        fig.add_subplot(1,3,3)
        plt.imshow(df.samples[idx].mask().squeeze(), cmap='gray')
        plt.title("mask")

        plt.show()
        time.sleep(1)
