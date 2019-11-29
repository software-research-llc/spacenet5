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


def get_abs_path(filename):
    if os.path.exists(filename):
        return filename
    for path in IMAGEDIRS:
        fullpath = os.path.join(path, filename)
        if os.path.exists(fullpath):
            return os.path.abspath(fullpath)
    raise Exception("could not find {} in any directory".format(filename))

def get_files(directories):
    """
    Return a list of all files found in all directories we're given.  The files are
    sorted lexicographically by filename (not full path) and returned in (pre, post) pairs.

    Note that we assume the file list is well formed, i.e. that all pairs exist and match by
    name.
    """
    prefiles = []
    postfiles = []
    sortfunc = lambda x: os.path.basename(x)
    for d in directories:
        prefiles += glob.glob(os.path.join(d, "*pre*"))
        postfiles += glob.glob(os.path.join(d, "*post*"))
    assert len(prefiles) == len(postfiles), f"something is wrong, len(predisaster)" + \
                                             " != len(postdisaster): {} {}".format(len(prefiles), len(postfiles))
    return list(zip(sorted(prefiles, key=sortfunc), sorted(postfiles, key=sortfunc)))

def get_test_files():
    """
    Return a list of the paths of images belonging to the test set as
    (preimage, postimage) tuples, e.g. ("socal-pre-004.png", "socal-post-004.png").
    """
    return get_files(TESTDIRS)

def get_validation_files():
    """
    Return a list of the .json files describing the validation set (holdout set).

    See settings.py for definition of SPLITFACTOR (proportion of the training set heldout).
    """
    files = get_files(LABELDIRS)
    length = len(files)
    return files[int(length*SPLITFACTOR):]

def get_training_files():
    """
    Return a list of the .json files describing the training images.
    """
    files = get_files(LABELDIRS)
    length = len(files)
    return files[:int(length*SPLITFACTOR)]


class Dataflow(tf.keras.utils.Sequence):
    """
    A tf.keras.utils.Sequence subclass to feed data to the model.
    """
    def __init__(self, files=get_training_files(), batch_size=1, transform=None, shuffle=False):
        self.transform = transform
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.image_datagen = ImageDataGenerator()

        if ".pickle" in files:
            with open(files, "rb") as f:
                logger.info("Creating Targets from pickle file")
                self.samples = pickle.load(f)
        elif ".json" in files[0][0].lower():
            logger.info("Creating Targets from JSON format files")
            self.samples = [(Target.from_json(pre), Target.from_json(post)) for (pre,post) in files]
        elif ".png" in files[0][0].lower():
            logger.info("Creating Targets from a list of PNG files")
            self.samples = [(Target.from_png(pre), Target.from_png(post)) for (pre,post) in files]

        if shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        """Length of this dataflow in units of batch_size"""
        length = int(np.ceil(len(self.samples) / float(self.batch_size)))
        return length

    def __getitem__(self, idx):
        """
        pre_image and post_image are the pre-disaster and post-disaster samples.
        premask is the uint8, single channel localization target we're training to predict.
        """
        x = []
        y = []
        # Rotate 90-270 degrees, shear by 0.1-0.2 degrees
        trans_dict = { 'theta': 90 * random.randint(1, 3), 'shear': 0.1 * random.randint(1, 2) }
        for (pre, post) in self.samples[idx*self.batch_size:(idx+1)*self.batch_size]:
            premask = pre.mask()
            if INPUTSHAPE != SAMPLESHAPE:
                pre = resize(pre.image(), INPUTSHAPE)
            else:
                pre = pre.image()
            if isinstance(self.transform, float) and random.random() < float(self.transform):
                pre = self.image_datagen.apply_transform(pre, trans_dict)
                premask = self.image_datagen.apply_transform(premask, trans_dict)

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
    def __init__(self, target=None):
        self.wkt = None
        self._coords = None
        self.target = None

    def coords(self, downvert=True, **kwargs):
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
        # For pre-disaster images, the building subtype isn't specified, but we want
        # the value of those buildings in our masks to be 1 and nothing else
        if self.klass is None:
            return 1
        # post-disaster images include subtype information (see settings.py for CLASSES)
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
       One target per input image (i.e. two targets per pre-disaster, post-disaster set)."""
    def __init__(self, text:str=""):
        self.buildings = []
        if text:
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
            b = Building(target=self)
            b.klass = prop.get('subtype', None)
           
            if b.klass not in CLASSES:
                logger.error(f"Unrecognized building subtype: {b.klass}")

            b.wkt = feature.get('wkt', None)
            b.uid = prop['uid']
            self.buildings.append(b)

    def mask(self):
        """Get the Target's mask for supervised training of the model"""
        chan1 = np.ones(MASKSHAPE[:2], dtype=np.uint8)
        chan2 = np.zeros(MASKSHAPE[:2], dtype=np.uint8)
        img = np.stack([chan1,chan2], axis=2)
        for b in self.buildings:
            coords = b.coords()#downvert=True, orig_x=1024, new_y=1024)#, new_x=256,new_y=256)
            if len(coords) > 0:
                try:
                    cv2.fillPoly(img, np.array([coords]), [0, b.color()])
                except Exception as exc:
                    logger.warning("cv2.fillPoly(img, {}, {}) call failed: {}".format(str(coords), b.color(), exc))
                    cv2.fillConvexPoly(img, coords, (0, b.color()))
        return img.reshape((MASKSHAPE[0] * MASKSHAPE[1], -1))

    def rcnn_masks(self):
        masks = []
        for b in self.buildings:
            img = np.zeros(MASKSHAPE[:2], dtype=np.uint8)
            coords = b.coords()
            if len(coords) > 0:
                cv2.fillPoly(img, np.array([coords]), b.color())
                masks.append(img)
        if len(masks) == 0:
            return np.array(masks), np.ones([0], dtype=np.int32)
        #masks.append(np.zeros(MASKSHAPE[:2], dtype=np.uint8))
        return np.dstack(masks).astype(bool), np.ones([len(masks)], dtype=np.int32)

    def image(self):
        """Get this Target's input image (i.e. satellite chip) to feed to the model"""
        return skimage.io.imread(self.image_path())

    def image_path(self):
        return self.img_path
        return get_abs_path(self.img_name)

    @staticmethod
    def from_json(filename:str):
        """Create a Target object from a path to a .JSON file"""
        with open(filename) as f:
            t = Target(f.read())
            t.img_path = get_abs_path(t.img_name)
            return t

    @staticmethod
    def from_png(filename:str):
        """Create a Target object from a path to a .PNG file.

        Note: from_json() will only store the base filename, but this
              function expects the string passed to be the full (absolute)
              path to the .png file."""
        target = Target()
        target.img_name = filename
        target.img_path = filename
        target.metadata = dict()
        return target




if __name__ == '__main__':
    # Testing and data inspection
    import time
    df = Dataflow(transform=0.5)
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
            colormap = {0: 'w', 1: 'b', 2: 'g', 3: 'y', 4: 'r', 5: 'k'}
            polys = []
            colors = set()
            for b in sample.buildings:
                coords = [b.upvert(c[0], c[1]) for c in b.coords()]#[b.upvert(x,y,1024,1024) for (x,y) in zip(b.coords()[:,0], b.coords()[:,1])]
                try:
                    xs = np.array(coords)[:,0]
                    ys = np.array(coords)[:,1]
                except Exception as exc:
                    logger.error(f"{b.uid}: {exc}")
                    continue
                if b.klass is None:
                    if "no-damage" not in colors:
                        label = "no-damage"
                    else:
                        label = None
                elif CLASSES[b.color()] not in colors:
                    label = CLASSES[b.color()]
                else:
                    label = None
                colors.add(label)
                polys.append(plt.plot(xs, ys, colormap[b.color()], antialiased=True, lw=0.85, label=label))
            plt.title("building polygons")
            if len(sample.buildings) > 0:
                plt.legend()
            i += 1

            fig.add_subplot(2,3,i)
            plt.imshow(sample.mask().squeeze(), cmap='terrain')
            plt.title("target mask")
            i += 1

        plt.show()
        time.sleep(1)
