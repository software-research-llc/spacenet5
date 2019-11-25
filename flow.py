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
import pickle
from skimage.transform import resize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Dataflow(tf.keras.utils.Sequence):
    """A tf.keras.utils.Sequence subclass to feed data to the model"""
    def __init__(self, batch_size=1, samples=None, transform=None, shuffle=False, validation_set=False):
        self.transform = transform
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.preproc = sm.get_preprocessing(BACKBONE)
        """"
        if transform:
            data_gen_args = dict(rotation_range=25,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.2,
                                     shear_range=0.2)
        else:
            data_gen_args = {}
        """
        self.image_datagen = ImageDataGenerator()

        if samples is not None:
            self.samples = samples
        else:
            if validation_set:
                files = get_validation_files()
            else:
                files = get_training_files()
            self.samples = [(Target.from_file(pre), Target.from_file(post)) for (pre,post) in files]

    def __len__(self):
        """Length of this dataflow in units of batch_size"""
        length = int(np.ceil(len(self.samples) / float(self.batch_size)))
        return length

    def __getitem__(self, idx):
        """
        pre_image and post_image are the pre-disaster and post-disaster samples.
        premask is the uint8, single channel localization target we're training to predict.
        """
#        x = [(resize(pre.image(), TARGETSHAPE), resize(post.image(), TARGETSHAPE)) for (pre, post) in self.samples[idx*self.batch_size:(idx+1)*self.batch_size]]
#        y = [(resize(pre.mask(), MASKSHAPE), resize(post.mask(), MASKSHAPE)) for (pre, post) in self.samples[idx*self.batch_size:(idx+1)*self.batch_size]]
        x = []
        y = []
        trans_dict = { 'theta': 90, 'shear': 0.1 }
        for (pre, post) in self.samples[idx*self.batch_size:(idx+1)*self.batch_size]:
            premask = pre.mask()
            pre = resize(pre.image(), TARGETSHAPE)
            if isinstance(self.transform, float) and random.random() < float(self.transform):
                pre = self.image_datagen.apply_transform(pre, trans_dict)
                premask = self.image_datagen.apply_transform(premask, trans_dict)

            pre = self.preproc(pre)
            x.append(pre)
            y.append(premask)

        return np.array(x), np.array(y).astype(np.uint8)

    @staticmethod
    def from_pickle(picklefile:str=PICKLED_TRAINSET):
        with open(picklefile, "rb") as f:
            return pickle.load(f)

    def to_pickle(self, picklefile:str=PICKLED_TRAINSET):
        with open(picklefile, "wb") as f:
            return pickle.dump(self, f)


class Building:
    """Carries the data for a single building; multiple Buildings are
       owned by a single Target"""
    def __init__(self, pre=None):
        self.wkt = None
        self._coords = None
        # If specified, self.pre == True and self.post == False (both == None if not specified)
        self.pre = pre
        self.post = False if pre is not None else None

    def coords(self, downvert=False, **kwargs):
        """Parses the WKT data and caches it for subsequent calls"""
        if self._coords is not None:
            return self._coords
        wkt = self.wkt
        pairs = []
        for pair in re.findall(r"\-?\d+\.?\d+ \-?\d+\.?\d+", wkt):
            xy = pair.split(" ")
            x,y = float(xy[0]), float(xy[1])
            if downvert is True:
                x,y = self.downvert(x,y,**kwargs)
            else:
                x,y = round(x), round(y)
            pairs.append(np.array([x,y]))
        self._coords = np.array(pairs)
        return self._coords

    def color(self):
        """Get the color value for a building subtype (i.e. index into CLASSES)"""
        ret = CLASSES.index(self.klass)
        return ret

    def downvert(self, x, y,
                 orig_x=SAMPLESHAPE[0],
                 orig_y=SAMPLESHAPE[1],
                 new_x=TARGETSHAPE[0],
                 new_y=TARGETSHAPE[1]):
        x = x * (new_x / orig_x)
        y = y * (new_y / orig_y)
        return round(x), round(y)

    def upvert(self, x, y,
               orig_x=SAMPLESHAPE[0],
               orig_y=SAMPLESHAPE[1],
               new_x=TARGETSHAPE[0],
               new_y=TARGETSHAPE[1]):
        x = x / (new_x / orig_x)
        y = y / (new_y / orig_y)
        return round(x), round(y)



class Target:
    """Target objects provide filenames, metadata, input images, and masks for training.
       One target per input image."""
    def __init__(self, text:str):
        self.buildings = []
        self.parse_json(text)

    def parse_json(self, text:str):
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

    def mask(self, img:np.ndarray=None):
        """Get the Target's mask for supervised training of the model"""
        if img is None:
            img = np.zeros(MASKSHAPE)
        for b in self.buildings:
            coords = b.coords()#downvert=True, orig_x=1024, new_y=1024)#, new_x=256,new_y=256)
            if len(coords > 1):
                cv2.fillConvexPoly(img, coords, b.color())
        return img

    def image(self):
        """Get this Target's input image (i.e. satellite chip) to feed to the model"""
        for path in IMAGEDIRS:
            fullpath = os.path.join(path, self.img_name)
            try:
                return skimage.io.imread(fullpath)
            except OSError as exc:
                continue
        raise exc

    @staticmethod
    def from_file(filename:str):
        """Create a Target object from a path to a .JSON file"""
        with open(filename) as f:
            return Target(f.read())


def get_files(directories):
    prefiles = []
    postfiles = []
    sortfunc = lambda x: os.path.basename(x)
    for d in directories:
        prefiles += glob.glob(os.path.join(d, "*pre*"))
        postfiles += glob.glob(os.path.join(d, "*post*"))
    return list(zip(sorted(prefiles, key=sortfunc), sorted(postfiles, key=sortfunc)))


def get_test_files():
    """
    Return a list of the paths of images belonging to the test set as
    (preimage, postimage) tuples, e.g. ("socal-pre-004.png", "socal-post-004.png").
    """
    return get_files(TESTDIRS)


def get_training_files():
    """
    Return a list of the .json files describing the training images.
    """
    files = get_files(LABELDIRS)
    length = len(files)
    return files[:int(length*SPLITFACTOR)]


def get_validation_files():
    """
    Return a list of the .json files describing the validation set (holdout set).
    """
    files = get_files(LABELDIRS)
    length = len(files)
    return files[int(length*SPLITFACTOR):]


if __name__ == '__main__':
    # Testing and data inspection
    import time
    """
    if os.path.exists(PICKLED_TRAINSET):
        df = Dataflow.from_pickle(PICKLED_TRAINSET)
        logger.info("Loaded training dataflow from pickle file.")
    else:
        logger.warning("Generating dataflows")
        df = Dataflow(batch_size=1)
    """
    df = Dataflow()
    while True:
        idx = random.randint(0,len(df) - 1)
        (pre, post) = df.samples[idx]
        i = 1
        fig = plt.figure()
        for sample in (pre,post):

            fig.add_subplot(2,3,i)
            plt.imshow(sample.image())
            plt.title(sample.img_name)
            i += 1

            fig.add_subplot(2,3,i)
            plt.imshow(sample.image())
            #background, no-damage, minor-damage, major-damage, destroyed, un-classified
            #    0           1           2             3            4            5
            colormap = {0: 'k', 1: 'b', 2: 'g', 3: 'y', 4: 'r', 5: 'w'}
            polys = []
            for b in sample.buildings:
                coords = b.coords()#[b.upvert(x,y,1024,1024) for (x,y) in zip(b.coords()[:,0], b.coords()[:,1])]
                xs = np.array(coords)[:,0]
                ys = np.array(coords)[:,1]
                polys.append(plt.plot(xs, ys, colormap[b.color()], antialiased=True, lw=0.5))
            plt.title("building polygons")
            i += 1

            fig.add_subplot(2,3,i)
            plt.imshow(sample.mask().squeeze())#, cmap='gray')
            plt.title("target mask")
            i += 1

        plt.show()
        time.sleep(1)
