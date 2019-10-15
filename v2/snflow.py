#!/usr/bin/env python3
import sys
import glob
import os
import re
import random
import tqdm
from inspect import getsourcefile
from skimage import io
from skimage.transform import resize
import skimage
import networkx
import numpy as np
import keras
import pandas as pd
import mpmath
import cv2
import logging
import loss
import tensorflow as tf
log = logging.getLogger(__name__)


N_CLASSES = 5
# The directory of this file (don't change this)
MYDIR = os.path.abspath(os.path.dirname(getsourcefile(lambda:0)))
# The path to the satellite images (be careful when changing, very delicate)
BASEDIR = "%s/data/train/" % MYDIR
# Don't touch this
RUNNING_TESTS = False

class SpacenetSequence(keras.utils.Sequence):
    """A sequence object that feeds tuples of (x, y) data via __getitem__()
       and __len__()"""
    def __init__(self, x_set: "List of paths to images",
                 y_set: "Should be a TargetBundle object", batch_size,
                 transform=False,
                 test=None, shuffle=False,
                 model=None):
        self.x, self.y = x_set, y_set
        if shuffle:
            random.shuffle(self.x)
        self.batch_size = batch_size
        self.transform = transform
        self.test = test
        self.model = model

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        global RUNNING_TESTS
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        x = []

        for ex in batch_x:
            try:
                filename = get_file(ex)
                image = get_image(filename)
                x.append(image)
            except Exception as exc:
                log.error("{} on {}".format(str(exc), file_name))
                raise exc
        if not x:
            raise Exception("x is empty")

        x = np.array(x)
        y = np.array([self.y[Target.expand_imageid(imageid)].image() for imageid in batch_x])#, dtype=DATATYPE)

        for idx, imageid in enumerate(batch_x):
            if self.transform:
                r = random.randint(0,1)
                if self.transform < r:
                    x[idx] = skimage.transform.rotate(x[idx], 90)
                    y[idx] = skimage.transform.rotate(y[idx], 90)

        if RUNNING_TESTS:
            return x,y,batch_x
        else:
            return x,y

    @staticmethod
    def all(model=None, batch_size=BATCH_SIZE, transform=False, shuffle=False):
        imageids = get_imageids()
        return SpacenetSequence(imageids, TargetBundle(), batch_size=batch_size, shuffle=shuffle, transform=transform, model=model)

def Sequence(**kwargs):
    return SpacenetSequence.all(**kwargs)

def get_image(filename=None, dataset="PS-RGB"):
    """Return the satellite image corresponding to a given partial pathname or chipid.
       Returns a random (but existing) value if called w/ None."""
    requested = filename
    if not os.path.exists(str(filename)):
        for city in CITIES:
            trying = get_file(filename=filename, dataset=dataset, datadir=BASEDIR + city)
            if trying:
                filename = trying
                break
    if not os.path.exists(str(filename)):
        log.info("Returning contents of {} for requested file {}".format(filename, requested))
        raise Exception("File not found: %s" % filename)
    return resize(io.imread(filename), IMSHAPE, anti_aliasing=True)

def get_file(filename=None, datadir=None, dataset="PS-RGB"):
    """Return the path corresponding to a given partial chipid or path.
       Returns a random (but existing) value if called w/ None."""
    if os.path.exists(str(filename)):
        return filename
    elif os.path.exists(str(filename) + ".tif"):
        return filename + ".tif"
    elif filename is None:
        allfiles = get_filenames()
        i = random.randint(0, len(allfiles))
        return allfiles[i]
    for city in CITIES:
        trying = os.path.join(BASEDIR, city, dataset.upper())
        if os.path.exists(os.path.join(trying, filename)):
            return os.path.join(trying, filename)
        elif os.path.exists(os.path.join(trying, filename) + ".tif"):
            return os.path.join(trying, filename) + ".tif"
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

def get_filenames(dataset="PS-RGB"):
    """Return a list of every path for every image"""
    ret = []
    for city in CITIES:
        ret += os.listdir(os.path.join(BASEDIR, city, dataset))
    return ret

def get_imageids(dataset="PS-RGB"):
    """Return the ImageIDs of all images (as opposed to the file paths)"""
    paths = get_filenames()
    return [os.path.basename(path) for path in paths]#Target.expand_imageid(path.replace(datadir + "_", "")) for path in paths]

class TargetBundle:
    """A dict-like container of Target objects"""
    def __init__(self, transform=False):
        self.targets = {}
        self.max_speed = 0
        self.mean_speed = 0
        i = 0
        imageids = get_imageids()
        for imageid in imageids:
            imageid = Target.expand_imageid(imageid)
            self.targets[imageid] = Target(imageid, tb=self)
            i += 1
        self.add_df(Target.df)

    def add_df(self, df):
        count = 0
        for idx,linestring in enumerate(df['WKT_Pix']):
            if linestring.lower().find("empty") != -1:
                continue
            imageid = Target.expand_imageid(df['ImageId'][idx])
            try:
                weight = float(df['travel_time_s'][idx])
                distance = float(df['length_m'][idx])
            except ZeroDivisionError:
                log.error("ZeroDivisionError: %s, %s, length = %s, time = %s" % (imageid,
                      linestring,
                      df['length_m'][idx],
                      df['travel_time_s'][idx]))
                weight = 0
            if imageid not in self.targets:
                count += 1
                imageid = imageid.replace("_chip", "_PS-RGB_chip")
            self.targets[imageid].add_linestring(linestring, weight / distance)
        self.mean_speed /= count

    def __getitem__(self, idx):
        """Return a target corresponding to the (possibly partial)
           ImageID or path name given -- returns the first match w/o
           checking for uniqueness of partial matches, but will always
           return a unique match if the entire ID or file path is given."""
        ret = self.targets.get(idx, None)
        if not ret:
            expanded = Target.expand_imageid(idx)
            ret = self.targets.get(expanded, None)
            if ret:
                return ret
            raise IndexError("Cannot locate {}".format(idx))
        else:
            return ret

    def __len__(self):
        return len(self.targets)

class Target:
    """An object representing all the information about a training target,
       e.g. the target mask for the network output to match, the ImageID,
       the file path to the image file, the weights for paths in the chip
       (square satellite image), etc."""
    regex = re.compile("[\d\.]+ [\d\.]+")
    df = pd.read_csv(TARGETFILE)

    def __init__(self, imageid, tb: "The owning TargetBundle"):
        self.graph = networkx.Graph()
        self.imageid = imageid
        self.tb = tb
        self._img = None

    @staticmethod
    def expand_imageid(imageid):
        regex = re.compile("chip(\d+)(.tif)?")
        m = re.search(regex, imageid)
        mstr = m.string[m.start():m.end()]
        mnum = m.groups()[0]
        ret = imageid.replace(mstr, "") + "chip" + mnum
        return ret

    def add_linestring(self, string, weight):
        """Take a linestring + the weight for the edge it represents, and
           add that information to what's stored in this object."""
        if string.lower().find("empty") != -1:
            return
        elif weight > self.tb.max_speed:
            self.tb.max_speed = weight
        edges = re.findall(Target.regex, string)
        for i in range(len(edges) - 1):
            x1,y1 = edges[i].split(" ")
            x2,y2 = edges[i+1].split(" ")
            x1,y1 = float(x1), float(y1)
            x2,y2 = float(x2), float(y2)
            self.graph.add_edge((x1,y1), (x2,y2), weight=weight)
        self.tb.mean_speed += weight
        return self

    def pixel(self, weight):
        """Returns the tuple of a pixel for painting, e.g. (0, 255, 0, 0)"""
        n_classes = N_CLASSES
        if n_classes == 1 or n_classes == 2:
            return 1.0
        channel = (n_classes - 1) / (self.tb.max_speed / weight)
        return round(channel)

    def image(self):
        if self._img is not None:
            return self._img
        img = np.zeros([CHIP_CANVAS_SIZE[0], CHIP_CANVAS_SIZE[1]])
        for edge in self.graph.edges():
            weight = self.graph[edge[0]][edge[1]]['weight']
            pixel = self.pixel(weight)
            x1,y1 = edge[0]
            x2,y2 = edge[1]
            x1,y1 = round(x1), round(y1)
            x2,y2 = round(x2), round(y2)
            cv2.line(img, (x1, y1), (x2, y2), pixel, 10)#, cv2., 0)
        img = resize(img, TARGET_IMSHAPE, anti_aliasing=True)
        if DATATYPE != np.float32:
            img = img.astype(DATATYPE)
        img = img.reshape((-1,1)).astype(int)
        self._img = img
        return img
#        return np.cast['uint8'](img)#.reshape(TARGET_IMSHAPE)

if __name__ == '__main__':
    RUNNING_TESTS = True
    log.warning("Running all tests...")
    allfiles = get_filenames()
    for filename in allfiles:
        got = get_file(filename)
        if os.path.basename(got) != filename:
            log.error("{} != {}".format(got, filename))
    seq = SpacenetSequence.all(batch_size=1)
    tb = TargetBundle()
    iteration = 0
    for x,y,batch_x in tqdm.tqdm(seq):
        if len(x) != 1 or len(y) != 1:
            log.error("Batch size is not working, should be {}, but is {}".format(seq.batch_size, len(x)))
        img = get_image(allfiles[iteration])
        if np.sum(img != x[0]) != 0:
            log.error("get_file({}) != data given for file {}".format(allfiles[iteration], batch_x[0]))
            sys.exit()
        if allfiles[iteration].replace(".tif", "") != Target.expand_imageid(allfiles[iteration]):
            log.error("{} maps to {}".format(allfiles[iteration], Target.expand_imageid(allfiles[iteration])))
            sys.exit()
        img = tb[Target.expand_imageid(allfiles[iteration])].image()
        if img is None:
            log.error("cannot find tb[{}]".format(Target.expand_imageid(allfiles[iteration])))
            sys.exit()
        if np.sum(img != y[0]) != 0:
            log.error("tb[{}].image() != y for {}".format(Target.expand_imageid(allfiles[iteration]), batch_x[0]))
            sys.exit()
        iteration += 1
    log.warning("All tests passed!")