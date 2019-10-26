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
import tensorflow.keras as keras
import pandas as pd
import mpmath
import cv2
import logging
import tensorflow as tf
import segmentation_models as sm
import keras
from keras.preprocessing.image import ImageDataGenerator
log = logging.getLogger(__name__)

CLASSES = [ 'background', 'slow', 'medium', 'fast' ]
BACKBONE = 'resnext101'
BATCH_SIZE = 3
N_CLASSES = len(CLASSES)
IMSHAPE = [512,512,3]
ORIG_IMSHAPE = [1300,1300,3]
ROAD_LINE_WIDTH = 3

# The directory of this file (don't change this)
MYDIR = os.path.abspath(os.path.dirname(getsourcefile(lambda:0)))
# The path to the satellite images (be careful when changing, very delicate)
BASEDIR = "%s/data/train/" % MYDIR
TESTDIR = "%s/data/test/" % MYDIR
# Don't touch this
RUNNING_TESTS = False
TARGETFILE = os.path.join(MYDIR, "targets.csv")
CITIES = ["AOI_2_Vegas",
          "AOI_3_Paris",
          "AOI_4_Shanghai",
          "AOI_5_Khartoum",
          "AOI_7_Moscow",
          "AOI_8_Mumbai"]
#          "AOI_9_San_Juan",
DATATYPE=int
TRANSFORM = False
keras.backend.set_image_data_format('channels_last')
channel_speed_dict = {}
MAX_SPEED = 29

def trup(x,y):
    x = x * (ORIG_IMSHAPE[0] / IMSHAPE[0])
    y = y * (ORIG_IMSHAPE[1] / IMSHAPE[1])
    return round(x), round(y)

def trdown(x,y):
    x = x * (IMSHAPE[0] / ORIG_IMSHAPE[0])
    y = y * (IMSHAPE[1] / ORIG_IMSHAPE[1])
    return round(x), round(y)

def preprocess(imgs):
    return sm.get_preprocessing(BACKBONE)(imgs)

class SpacenetSequence(keras.utils.Sequence):
    """A sequence object that feeds tuples of (x, y) data via __getitem__()
       and __len__()"""
    def __init__(self, x_set: "List of paths to images",
                 y_set: "Should be a TargetBundle object", batch_size,
                 transform=False,
                 test=None, shuffle=False,
                 model=None):
        global TRANSFORM
        self.test = test
        if self.test == False:
            self.x = x_set[:int(np.floor(len(x_set) * 0.95))]
            self.y = y_set
        elif self.test == True:
            self.x = x_set[int(round(len(x_set) * 0.95)):]
            self.y = y_set
        else:
            self.x = x_set
            self.y = y_set

        if shuffle:
            random.shuffle(self.x)
        self.batch_size = batch_size
        self.transform = transform
        TRANSFORM = transform
        self.model = model

        # we create two instances with the same arguments
        self.datagen_args = dict(rotation_range=25,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.2,
                                 shear_range=0.2,
                                 )
        self.image_datagen = ImageDataGenerator(**data_gen_args)
        self.mask_datagen = ImageDataGenerator(**data_gen_args)
        # Provide the same seed and keyword arguments to the fit and flow methods
        #mask_datagen.fit(masks, augment=True, seed=seed)

    def __len__(self):
        length = int(np.ceil(len(self.x) / float(self.batch_size)))
        return length

    def __getitem__(self, idx):
        global RUNNING_TESTS
        if idx >= len(self) or idx < 0:
            return
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        if len(batch_x) < 1:
            return
        while len(batch_x) < self.batch_size:
            batch_x.append(self.x[idx * self.batch_size])
        y = [self.y[Target.expand_imageid(imageid)].image() for imageid in batch_x]

        x = []
        for idx,ex in enumerate(batch_x):
            try:
                filename = get_file(ex)
                image = get_image(filename)
                if random.random() < self.transform:
                    trans_dict = { 'theta': 90, 'shear': 0.1 }
                    image = self.image_datagen.apply_transform(image, trans_dict)
                    y[idx] = self.image_datagen.apply_transform(y[idx], trans_dict)
                elif random.random() < self.transform:
                    image = np.fliplr(image)
                    y[idx] = np.fliplr(y[idx])
                elif random.random() < self.transform:
                    image = np.flipud(image)
                    y[idx] = np.flipud(y[idx])
#                elif random.randint(0,1) < self.transform:
#                    image = cv2.GaussianBlur(image, (3, 3), 0)
#                    y[idx] = cv2.GaussianBlur(image, (3, 3), 0)
#                elif random.randint(0,1) < self.transform:
#                    image = tf.image.rot90(image)
#                    y[idx] = tf.image.rot90(y[idx])
                procd_image = normalize(image)
                x.append(procd_image)
            except Exception as exc:
                log.error("{} on {}".format(str(exc), filename))
                import pdb; pdb.set_trace()
                raise exc

        x = np.array(x)
        y = np.array(y)
#        x = preprocess(x)
#        if len(y.shape) < 4:
#            y = y.reshape([1, y.shape[0], y.shape[1], y.shape[2]])

        if RUNNING_TESTS:
            return x,y,batch_x
        else:
            return x,y

    def get_generator(self):
        def _gen():
            while True:
                for i in range(len(self)):
                    yield self[i]
                raise StopIteration

        return _gen

    @staticmethod
    def all(model=None, batch_size=BATCH_SIZE, transform=False, shuffle=False, test=None):
        imageids = get_filenames()
        return SpacenetSequence(imageids, TargetBundle(), batch_size=batch_size,
                                shuffle=shuffle, transform=transform, model=model,
                                test=test)

def Sequence(**kwargs):
    return SpacenetSequence.all(**kwargs)

# we create two instances with the same arguments
data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
#image_datagen.fit(images, augment=True, seed=seed)
#mask_datagen.fit(masks, augment=True, seed=seed)
#image_generator = image_datagen.flow_from_directory(
#    'data/images',
#    class_mode=None,
#    seed=seed)
#mask_generator = mask_datagen.flow_from_directory(
#    'data/masks',
#    class_mode=None,
#    seed=seed)
# combine generators into one which yields image and masks
#train_generator = zip(image_generator, mask_generator)
#model.fit_generator(
#    train_generator,
#    steps_per_epoch=2000,
#    epochs=50)

def cvt_16bit_to_8bit(img):
    ch1 = img[:,:,0]
    ch2 = img[:,:,1]
    ch3 = img[:,:,2]
    stk = []
    for ch in [ch1,ch2,ch3]:
        ch = ch - np.min(ch)
        if np.max(ch) > 0:
            ch = ch / np.max(ch)
        stk.append(ch)
    return np.stack(stk, axis=2)

def normalize(img, max=1.0):
    if img.max() > 0:
        return img * (max/img.max())
    else:
        return img

def erosion(imagearg):
    from skimage.morphology import erosion, dilation, opening, closing, white_tophat
    from skimage.morphology import black_tophat, skeletonize, convex_hull_image
    from skimage.morphology import disk

    selem = disk(6)
    eroded = erosion(orig_phantom, selem)
    plot_comparison(orig_phantom, eroded, 'erosion')

def threshold(image, block_size = 15, off = 10):
    global_thresh = skimage.filters.threshold_otsu(image)
    binary_global = image > global_thresh
    adaptive_thresh = skimage.filters.threshold_local(binary_global, block_size, offset=off)
    return adaptive_thresh


def get_image(filename=None, dataset="PS-RGB"):
    """Return the satellite image corresponding to a given partial pathname or chipid.
       Returns a random (but existing) value if called w/ None."""
    requested = filename
    filename = get_file(requested)
    img = io.imread(filename)
    if img.dtype == np.uint16:
        img = cvt_16bit_to_8bit(img)
    """
    if random.random() < TRANSFORM / 2:
        antialias = False 
    else:
        antialias = True 
    """
    antialias = True
    return resize(img, IMSHAPE, anti_aliasing=antialias)
    #return resize(io.imread(filename), IMSHAPE, anti_aliasing=True)
    """
    if not os.path.exists(str(filename)):
        for city in CITIES:
            trying = get_file(filename=filename, dataset=dataset, datadir=BASEDIR + city)
            if trying:
                filename = trying
                break
    if not os.path.exists(str(filename)):
        log.info("Returning contents of {} for requested file {}".format(filename, requested))
        raise Exception("File not found: %s" % filename)
    """
    return resize(io.imread(filename), IMSHAPE, anti_aliasing=True)
#    return io.imread(filename)

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
    filename = filename.replace(".tif", "")
    allfiles = get_filenames()
    for f in allfiles:
        if filename + ".tif" in f:
            return f
    match = re.search("(AOI_\d_.+)_(PS-RGB|chip)", filename)
    if match:
        city = match.groups(0)[0]
        found = os.path.join(BASEDIR, city, "PS-RGB", filename)
        if os.path.exists(found):
            return found
    for city in CITIES:
        trying = os.path.join(BASEDIR, city, dataset.upper())
        if os.path.exists(os.path.join(trying, filename)):
            return os.path.join(trying, filename)
        elif os.path.exists(os.path.join(trying, filename) + ".tif"):
            return os.path.join(trying, filename) + ".tif"
    trying = glob.glob("%s/*/PS-RGB/*%s.tif" % (BASEDIR, filename))
    if trying:
        return trying[0]
    else:
        trying = re.search("chip(\d+)(.tif)?$", str(filename))
        if trying:
            trying = trying.groups()[0]
            trying = glob.glob("%s/*/PS-RGB/*%s*.tif" % (BASEDIR, trying))
            if trying:
                filename = trying[0]
            else:
                return None
        else:
            trying = glob.glob("%s/*/PS-RGB/*%s*.tif" % (BASEDIR, trying))
            if trying:
                return trying[0]
    if not filename:
        return None
    return filename

def get_filenames(dataset="PS-RGB"):
    """Return a list of every path for every image"""
    ret = []
    for city in CITIES:
        try:
            ret += [os.path.join(BASEDIR, city, dataset, f) for f in os.listdir(os.path.join(BASEDIR, city, dataset))]
        except Exception as exc:
            print(exc)
            import pdb; pdb.set_trace()
    return ret

def get_test_filenames(dataset="PS-RGB"):
    ret = []
    for city in os.listdir(TESTDIR):
        if "AOI_8" in city or "AOI_7" in city or "AOI_9" in city or "AOI_10" in city:
            ret += [os.path.join(TESTDIR, city, "PS-RGB", f) \
                    for f in os.listdir(os.path.join(TESTDIR, city, "PS-RGB"))]
    return ret
    ret = []
    for city in os.listdir(TESTDIR):
        ret += [os.path.join(TESTDIR, city, "PS-RGB", f) for \
                f in os.listdir(os.path.join(TESTDIR, city, "PS-RGB"))]
    return ret

def get_imageids(dataset="PS-RGB"):
    """Return the ImageIDs of all images (as opposed to the file paths)"""
    paths = get_filenames()
    return [Target.expand_imageid(os.path.basename(path)) for path in paths]#Target.expand_imageid(path.replace(datadir + "_", "")) for path in paths]

class TargetBundle:
    """A dict-like container of Target objects"""
    def __init__(self, transform=False):
        self.targets = {}
        self.targetsIdx = []
        self.max_speed = 0
        self.mean_speed = 0
        i = 0
        imageids = get_imageids()
        for imageid in imageids:
            imageid = Target.expand_imageid(imageid)
            self.targets[imageid] = Target(imageid, tb=self)
            self.targetsIdx.append(self.targets[imageid])
            i += 1
        self.add_df(Target.df)

    def add_df(self, df):
        for idx,linestring in enumerate(df['WKT_Pix']):
            if linestring.lower().find("empty") != -1:
                continue
            imageid = Target.expand_imageid(df['ImageId'][idx])
            try:
                travel_time_s = float(df['travel_time_s'][idx])
                length_m = float(df['length_m'][idx])
            except ZeroDivisionError:
                log.error("ZeroDivisionError: %s, %s, length = %s, time = %s" % (imageid,
                      linestring,
                      df['length_m'][idx],
                      df['travel_time_s'][idx]))
                weight = float(df['weight'])
            self.targets[imageid].add_linestring(linestring, travel_time_s, length_m)

    def __getitem__(self, idx):
        """Return a target corresponding to the (possibly partial)
           ImageID or path name given -- returns the first match w/o
           checking for uniqueness of partial matches, but will always
           return a unique match if the entire ID or file path is given."""
        if isinstance(idx, int):
            return self.targetsIdx[idx]
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
        imageid = os.path.basename(imageid)
        regex = re.compile("(chip|img)(\d+)(.tif)?")
        m = re.search(regex, imageid)
        mstr = m.string[m.start():m.end()]
        mnum = m.groups()[1]
        ret = imageid.replace(mstr, "") + m.groups()[0] + mnum
        ret = ret.replace("PS-RGB_", "").replace(".tif", "")
        return ret

    def add_linestring(self, string, travel_time_s, length_m):
        """Take a linestring + the weight for the edge it represents, and
           add that information to what's stored in this object."""
        if string.lower().find("empty") != -1:
            return
#        elif travel_time_s > self.tb.max_speed:
#            self.tb.max_speed = travel_time_s
        edges = re.findall(Target.regex, string)
        for i in range(len(edges) - 1):
            x1,y1 = edges[i].split(" ")
            x2,y2 = edges[i+1].split(" ")
            x1,y1 = float(x1), float(y1)
            x2,y2 = float(x2), float(y2)
            self.graph.add_edge((x1,y1), (x2,y2), travel_time_s=travel_time_s, length_m=length_m)
#            self.graph[(x1,y1)]['pts'] = [x1,y1]
#            self.graph[(x2,y2)]['pts'] = [x2,y2]
        self.tb.mean_speed += travel_time_s
        return self

    def get_speed_channel(self, weight):
        """ Total speed:  3619260.706599
            Mean speed:   11.570896
            Standard dev: 8.353577
            Mode speed:   11.000000
            Min speed:    7.000000
            Max speed:    29.000000
            Num roads:    312790
            Num bins:     20
            Speed bins:   [(23, 7), (19, 10), (24, 16), (17, 63), (14, 113), (27, 266), (15, 269), (22, 689), (12, 1086), (18, 1519), (29, 1944), (10, 6526), (25, 12595), (20, 17005), (13, 19079), (16, 32672), (7, 37082), (8, 44358), (9, 65153), (11, 72338)]
            Ordered bins: [(7, 37082), (8, 44358), (9, 65153), (10, 6526), (11, 72338), (12, 1086), (13, 19079), (14, 113), (15, 269), (16, 32672), (17, 63), (18, 1519), (19, 10), (20, 17005), (22, 689), (23, 7), (24, 16), (25, 12595), (27, 266), (29, 1944)]
        """
        weight = int(round(weight))
        if weight in range(10):
            ret = 1
            channel_speed_dict[1] = 9
        elif weight in range(10,15):
            ret = 2
            channel_speed_dict[2] = 13
        elif weight in range(15,99):
            ret = 3
            channel_speed_dict[3] = 23
        else:
            log.error("Unrecognized weight: {}".format(weight))
            ret = 1
        return ret
#        return int(round( (N_CLASSES - 2) / (sp_max - sp_min) * (weight - sp_min) )) + 1

    def points(self):
        seen = set()
        ary = np.zeros((IMSHAPE[0], IMSHAPE[1]), dtype=bool)
        for s,e in self.graph.edges():
            if (s,e) in seen:
                continue
            elif (e,s) in seen:
                continue
            else:
                seen.add((s,e))
                seen.add((e,s))

            x1,y1 = s
            x2,y2 = e
            x1,y1 = trdown(x1,y1)
            x2,y2 = trdown(x2,y2)
            ary[x1,y1] = True
            ary[x2,y2] = True
        skel = skimage.morphology.skeletonize(ary)
        return skel.reshape((-1,))

    def image(self):
        #if self._img is not None:
        #    return self._img
        img = np.zeros((IMSHAPE[0], IMSHAPE[1]), dtype=np.uint8)
        for edge in self.graph.edges():
            num = self.graph[edge[0]][edge[1]]['length_m']
            denom = self.graph[edge[0]][edge[1]]['travel_time_s']
            if denom > 0:
                weight = num / denom
            else:
                weight = num
                log.warning("zero travel_time_s for {}".format(self.imageid))
                import pdb; pdb.set_trace()
            klass = self.get_speed_channel(weight)
            x1,y1 = edge[0]
            x2,y2 = edge[1]
            x1,y1 = trdown(x1,y1)
            x2,y2 = trdown(x2,y2)
            # draw road line
            cv2.line(img, (x1, y1), (x2, y2), klass, ROAD_LINE_WIDTH)
        #img = resize(img, [IMSHAPE[0], IMSHAPE[1]], anti_aliasing=True)
        #img = img.reshape((IMSHAPE[0] * IMSHAPE[1], -1))
        #self._img = img
        return np.expand_dims(img, 2)
#        return np.cast['uint8'](img)#.reshape(TARGET_IMSHAPE)

def get_speed_channel(weight):
    return Target.get_speed_channel(__name__, weight)

def get_channel_speed(channel):
    if channel_speed_dict.get(channel, None):
        return channel_speed_dict[channel]
    for i in range(7,29):
        if get_speed_channel(i) == channel:
            return i
    return 1

if __name__ == '__main__':
    RUNNING_TESTS = True
    log.warning("Running all tests...")
    allfiles = get_filenames()
    for filename in allfiles:
        got = get_file(os.path.basename(filename))
        if got != filename:
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
        if os.path.basename(allfiles[iteration]).replace(".tif", "").replace("PS-RGB_","") != Target.expand_imageid(allfiles[iteration]):
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
