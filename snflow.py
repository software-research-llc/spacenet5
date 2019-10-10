#!/usr/bin/env python3
import glob
import os
import re
import random
from inspect import getsourcefile
from skimage import io
from skimage.transform import resize
import networkx
import numpy as np
import keras
import pandas as pd
import mpmath
import cv2
import logging
log = logging.getLogger(__name__)

# We represent the target masks as N-channel images, with each channel corresponding to
# the meaning of a pixel being "no road" (0), "slow road" (1), "midspeed road" (2),
# or "fast road" (3).  The N_ClASSES can be <= 4 because we use image processing libs
# to manipulate our data, and they require at most 4 channels (red, green, blue, alpha),
# but there will probably be a couple of functions that throw exceptions b/c I didn't try
# anything but 4.
#
# e.g. if there's a fast road drawn as a line from the middle of the top of a chip to the
# middle of the bottom of the chip (here chip means a square satellite image), i.e. right
# down its center, then the corresponding target image would have 255 in every x, y index
# position of the alpha channel corresponding to the x,y pixel coordinates of the road.

# batch size
BATCH_SIZE = 5

# see header above
N_CLASSES = 3

# Accuracy of the multiprecision floating point arithmetic library (doesn't matter in the end)
mpmath.dps = 100

# The shape of unmodified satellite images
CHIP_CANVAS_SIZE = [1300,1300,3]

# The shape of images that are fed to the neural network (scaled CHIP_CANVAS_SIZE)
IMSHAPE = [256,256,3]

# Target image shape, i.e. shape of the neural net output.  Note that our U-net
# doesn't like the aspect ratio being changed, so keep dims in the same proportion
TARGET_IMSHAPE = [IMSHAPE[0], IMSHAPE[1], N_CLASSES]

# The shape of the decoder output (this more or less has to equal TARGET_IMSHAPE)
DECODER_OUTPUT_SHAPE = TARGET_IMSHAPE

# The files containing road network linestrings for ground truth during training
TARGETFILES = [
           #     "train_AOI_4_Shanghai_geojson_roads_speed_wkt_weighted_simp.csv",
                "train_AOI_7_Moscow_geojson_roads_speed_wkt_weighted_simp.csv",
                "train_AOI_8_Mumbai_geojson_roads_speed_wkt_weighted_simp.csv"
              ]

# The subset of the image dataset we use (PS-RGB == panchromatic sharpened RGB)
DATASET = "PS-RGB"

CITIES = [
           #"AOI_4_Shanghai", 
           "AOI_7_Moscow",
           "AOI_8_Mumbai"
         ]#, "AOI_9_San_Juan" ]


# The directory of this file (don't change this)
MYDIR = os.path.abspath(os.path.dirname(getsourcefile(lambda:0)))

# The path to the satellite images (be careful when changing, very delicate)
BASEDIR = "%s/data/train/" % MYDIR


class SpacenetSequence(keras.utils.Sequence):
    """A sequence object that feeds tuples of (x, y) data via __getitem__()
       and __len__()"""
    def __init__(self, x_set: "List of paths to images",
                 y_set: "Should be a TargetBundle object", batch_size,
                 transform=False,
                 test=None, shuffle=True):
        self.x, self.y = x_set, y_set
        if shuffle:
            random.shuffle(self.x)
        self.batch_size = batch_size
        self.transform = transform
        self.test = test

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
                    x.append(image)
                except Exception as exc:
                    pass

#        x, y = np.array([resize(get_image(file_name), IMSHAPE) for file_name in batch_x]), \
#               np.array([self.y[Target.expand_imageid(imageid)].image() for imageid in batch_x])
        x = np.array(x)
        y = np.array([self.y[Target.expand_imageid(imageid)].image() for imageid in batch_x], dtype=np.uint8)
        return x,y

    @staticmethod
    def all(batch_size = BATCH_SIZE, transform = False):
        imageids = []
        for i in range(len(CITIES)):
            imageids += get_filenames(datadir=BASEDIR + CITIES[i])
        return SpacenetSequence(imageids, TargetBundle(), batch_size=batch_size, transform=transform)

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

def get_file(filename=None, dataset="PS-RGB", datadir=BASEDIR + CITIES[0]):
    """Return the path corresponding to a given partial chipid or path.
       Returns a random (but existing) value if called w/ None."""
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
    """Return a list of every path for every image"""
    datadir = os.path.join(datadir, dataset.upper())
    if filename is None:
        return os.listdir(datadir)
    else:
        return glob.glob("%s/*%s*" % (datadir, filename))

def get_imageids(datadir=BASEDIR + CITIES[0], dataset="PS-RGB"):
    """Return the ImageIDs of all images (as opposed to the file paths)"""
    paths = get_filenames(datadir=datadir, dataset=dataset)
    return [Target.expand_imageid(path.replace(datadir + "_", "")) for path in paths]

class TargetBundle:
    """A dict-like container of Target objects"""
    def __init__(self, transform=False):
        self.targets = {}
        self.max_speed = 0
        i = 0
        imageids = []
        for city in CITIES:
            imageids += get_imageids(datadir=BASEDIR + city)
        for imageid in imageids:
            imageid = Target.expand_imageid(imageid)
            self.targets[imageid] = Target(imageid, tb=self)
            i += 1
        self.add_df(Target.df)

    def add_df(self, df):
        for idx,linestring in enumerate(df['WKT_Pix']):
            if linestring.lower().find("empty") != -1:
                continue
            imageid = Target.expand_imageid(df['ImageId'][idx])
            try:
                weight = float(df['travel_time_s'][idx])
            except ZeroDivisionError:
                log.error("ZeroDivisionError: %s, %s, length = %s, time = %s" % (imageid,
                      linestring,
                      df['length_m'][idx],
                      df['travel_time_s'][idx]))
                weight = 0
            if imageid not in self.targets:
                imageid = imageid.replace("_chip", "_PS-RGB_chip")
            self.targets[imageid].add_linestring(linestring, weight)

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
            for key in self.targets.keys():
                if re.search(expanded, key):
                    return self.targets[key]
            for key in self.targets.keys():
                if re.search(idx, key):
                    return self.targets[key]
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
    df = pd.read_csv(TARGETFILES[0])
    for targetfile in TARGETFILES[1:]:
        df.append(pd.read_csv(targetfile))

    def __init__(self, imageid, tb: "The owning TargetBundle"):
        self.graph = networkx.Graph()
        self.imageid = imageid
        self.tb = tb

    @staticmethod
    def expand_imageid(imageid):
        regex = re.compile("chip(\d+)(.tif)?")
        m = re.search(regex, imageid)
        mstr = m.string[m.start():m.end()]
        mnum = m.groups()[0]
        ret = imageid.replace(mstr, "") + "chip" + mnum
        return ret
        """
        if len(mnum) >= 5:
            ret = imageid
        else:
            num = "{:>5.5s}".format(mnum)
            ret = imageid.replace(mstr, "") + "chip" + num.replace(" ", "0")
        return ret
        """

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
        return self

    def chip(self):
        """Return the chip number that this target corresponds to (only unique
           with respect to a given city, other cities will have identical chipIDs)"""
        return re.search("_(chip\d+)", self.imageid).groups()[0]

    def pixel(self, weight):
        """Returns the tuple of a pixel for painting, e.g. (0, 255, 0, 0)"""
        channel = N_CLASSES / self.tb.max_speed * weight
        channel = channel# * (4 / N_CLASSES)
        pixel = [0] * 4
        pixel[round(channel)] = 255
        return pixel

    def image(self):
        """Create the mask (output for the neural network to match) for this target.
        Format is CHANNELS LAST!

        We represent the target masks as N-channel images, with each channel corresponding to
        the meaning of a pixel being "no road" (0), "slow road" (1), "midspeed road" (2),
        or "fast road" (3).  The N_ClASSES can be <= 4 because we use image processing libs
        to manipulate our data, and they require at most 4 channels (red, green, blue, alpha),
        but there will probably be a couple of functions that throw exceptions b/c I didn't try
        anything but 1 and 4.
    
        e.g. if there's a fast road drawn as a line from the middle of the top of a chip to the
        middle of the bottom of the chip (here chip means a square satellite image), i.e. right
        down its center, then the corresponding target image would have 255 in every x, y index
        position of the alpha channel corresponding to the x,y pixel coordinates of the road.
        """
        img = np.zeros((CHIP_CANVAS_SIZE[0], CHIP_CANVAS_SIZE[1], N_CLASSES))
        for edge in self.graph.edges():
            weight = self.graph[edge[0]][edge[1]]['weight']
            pixel = self.pixel(weight)
            x1,y1 = edge[0]
            x2,y2 = edge[1]
            x1,y1 = round(x1), round(y1)
            x2,y2 = round(x2), round(y2)
            cv2.line(img, (x1, y1), (x2, y2), pixel, 10)
        img = resize(img, TARGET_IMSHAPE, anti_aliasing=True)
        return np.cast['uint8'](img).reshape(TARGET_IMSHAPE)
