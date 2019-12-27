"""
Dataset (dataflow) code for feeding to the model, etc; contains enumerable objects and
functions to find files or apply transforms.
"""
import sys
import glob
import random
import os
import cv2
import tensorflow as tf
import pandas as pd
import shapely
import json
import numpy as np
import settings as S
import matplotlib.pyplot as plt
import skimage
from keras.preprocessing.image import ImageDataGenerator
import logging
import re
import pickle
from skimage.transform import resize
import cv2
import glob
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input
#from infer import convert_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_glob_path(filename):
    """
    Shell-style wildcard (*, ?, etc.) compatible path lookup.  Searches all directories
    in settings.py for a matching file from the dataset.
    """
    for path in S.IMAGEDIRS:
        matches = glob.glob(os.path.join(path, filename))
        if len(matches) != 0:
            return os.path.abspath(os.path.join(path, matches[-1]))

    raise FileNotFoundError(filename)


def get_abs_path(filename):
    """
    Search the dataset directories in settings.py for a matching filename.
    """
    if "*" in filename or "?" in filename:
        return get_glob_path(filename)
    elif os.path.exists(filename):
        return filename

    for path in S.IMAGEDIRS:
        fullpath = os.path.join(path, filename)
        if os.path.exists(fullpath):
            return os.path.abspath(fullpath)

    raise FileNotFoundError(filename)


def get_image(filename):
    """
    Get the full path of an image in the known dataset(s).  Accepts
    shell glob patterns.
    """
    return get_abs_path(filename)


def get_files(directories):
    """
    Return full paths for all files in all directories we're given.  The files are
    sorted lexicographically by filename (not full path) and returned in (pre, post) pairs.

    No checks are performed, and files are found with glob patterns `*pre*` and `*post*`,
    since the dataset for the Xview2 challenge names the pre-disaster and post-disaster images
    as such (e.g. `honolulu-hurricane-pre-00001.png`).
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
    return get_files(S.TESTDIRS)


def get_validation_files():
    """
    Return a list of the .json files describing the validation set (holdout set).

    See settings.py for definition of SPLITFACTOR (proportion of the training set heldout).
    """
    files = get_files(S.LABELDIRS)
    length = len(files)
    return files[int(length*S.SPLITFACTOR):]


def get_training_files():
    """
    Return a list of the .json files describing the training images.
    """
    files = get_files(S.LABELDIRS)
    length = len(files)
    return files[:int(length*S.SPLITFACTOR)]


def apply_gaussian_blur(img:np.ndarray, kernel=(5,5)):
    """
    Apply a Gaussian blur effect to an image; default kernel size is (5,5),
    and dimensions must both be odd.  Smaller numbers mean less blur.

    Blur is applied to channels individually.
    """
    chans = []
    for i in range(img.shape[-1]):
        chans.append(cv2.GaussianBlur(img[...,i].squeeze(), kernel, cv2.BORDER_DEFAULT))
    ret = np.dstack(chans)
    return ret.reshape(img.shape)


def convert_postmask_to_premask(postmask):
    """
    Take a post-disaster mask, and return a copy with values > 1 replaced by 1.  Used
    because from manual inspection, the post-disaster pixel coordinates appear more
    reliable.
    """
    chan1 = postmask[...,0].copy()
    chan2 = np.argmax(postmask, axis=2)
    chan2 = np.clip(chan2, 0, 1)
    return np.dstack([chan1,chan2])


def interlace(pre, post, step=3):
    """
    Interlay a pre- and post-disaster image set, returning an image composed of pixels
    from each image in an alternating fashion.
    """
    pre = pre.copy()
    pre[range(0, pre.shape[0], step),range(0,pre.shape[1],step)] = post[range(0,post.shape[0],step),range(0,post.shape[1],step)]

    return pre, post


def convert_prediction(pred, argmax=True, threshold=None):
    """
    Turn a model's prediction output (one-hot encoded) into a grayscale (segmented) mask.

    Parameters:
        pred: one-hot encoded array of dimensions (height*width, num_classes)
        argmax (optional): boolean, determines whether np.argmax() is used.
        threshold (optional): float, if specified determines the min value to consider true.

    Returns:
        a grayscale mask of (width, height) composed of integer values corresponding to pixel
        class at each (x,y) location.
    """
    x = pred.squeeze().reshape(S.MASKSHAPE[:2] + [-1])# + [pred.shape[-1]])
    if argmax is True:
        if isinstance(threshold, float):
            x[x<threshold] = 0
            x[:,:,0] = 0
            thresh = np.argmax(x, axis=2)
            return thresh
        else:
            return np.argmax(x, axis=2)
    else:
        return x[...,0:3], x[...,3:]


def eliminate_unclassified(pre:np.ndarray, post:np.ndarray, warped_mask:np.ndarray):
    """
    Takes a pre-disaster, post-disaster, and a mask.  Zeros-out the location of all
    "un-classified" buildings found in the mask provided by altering the input array
    values at those locations, then zeros out the mask values at those locations as
    well and returns all three.

    Since "un-classified" is not part of the test values, we basically want those
    buildings to effectively not exist, otherwise they'll throw off our training
    (there are ~11,000 of them in the training set).
    """
    # get a (width,height,channels) sized version of input mask with integer values
    mask = convert_prediction(warped_mask)
    mask = np.dstack([mask,mask,mask])

    # remove superfluous dimensions (caused by a batch size of 1)
    pre = pre.squeeze()
    post = post.squeeze()

    # zero out the pre-disaster and post-disaster image locations
    pre[mask==5] = 0
    post[mask==5] = 0

    # zero out the mask locations
    warped_mask[warped_mask==(0,0,0,0,0,1)] = 0

    return pre,post,warped_mask


class Dataflow(tf.keras.utils.Sequence):
    """
    A tf.keras.utils.Sequence subclass to feed data to the model.

    See comments in the source for parameter explanations and available options, etc.
    """
    def __init__(self, files=get_training_files(), batch_size=1, transform=None, shuffle=True, buildings_only=False, interlace=False, return_postmask=True, return_stacked=False, return_average=False, return_post_only=False, return_single_channel=False, segmentation_models=False):
        # boolean, return pre- and post-disaster images added together and divided by 2 (their average)
        self.return_average = return_average
        # boolean, return pre- and post- stacked channelwise
        self.return_stacked = return_stacked
        # boolean, return the post-disaster mask, not the pre-disaster mask (pre-disaster is the default)
        self.return_postmask = return_postmask
        # float, if provided the percent of samples that will be deformed for training
        self.transform = transform
        # boolean, whether to shuffle (randomize order of) the samples or not
        self.shuffle = shuffle
        # int, how many values to return per __getitem__() call
        self.batch_size = batch_size
        # boolean, whether to return an interlaced value, see interlace()
        self.interlace = interlace
        # return only the post-disaster image 
        self.return_post_only = return_post_only
        # should probably remove these
        self.return_single_channel = return_single_channel
        self.segmentation_models = segmentation_models

        # deformation object to apply to images (so sample and mask pixel locations correspond)
        self.image_datagen = ImageDataGenerator()
        
        # a list of tuples, each being composed of (pre-disaster Target object, post-disaster Target object)
        self.samples = []

        self._do_shuffle = False

        if ".json" in files[0][0].lower():
            logger.info("Creating Targets from JSON format files")
            # use from_json() as the Target object constructor below
            method = Target.from_json
        elif ".png" in files[0][0].lower():
            logger.info("Creating Targets from a list of PNG files")
            # use from_png() as the Target object constructor below
            method = Target.from_png
        else:
            raise RuntimeError("Files should be in PNG or JSON format")

        # for each (pre-disaster, post-disaster) pair of files, construct two Targets and store them
        for (pre,post) in files:
            t_pre = method(pre, df=self)
            t_post = method(post, df=self)
            # skip images with no buildings if that option is specified
            if buildings_only is True and len(t_pre.buildings) > 0:
                self.samples.append((t_pre, t_post))
            elif not buildings_only:
                self.samples.append((t_pre, t_post))

        if shuffle:
            random.shuffle(self.samples)

    def __len__(self):
        """
        Length of this dataflow in units of batch_size.
        """
        length = int(np.ceil(len(self.samples) / float(self.batch_size)))
        return length

    def __getitem__(self, idx, preprocess=False):
        """
        TODO: refactor now that we aren't adding individual options as we test things out
        during the challenge.

        `pre_image` and `post_image` are the pre-disaster and post-disaster samples.
        `premask` and `postmask` are the uint8, grayscale (single channel) masks for each.
        """
        if self.shuffle is True and self._do_shuffle is True:
            self._do_shuffle = False
            random.shuffle(self.samples)
        elif self.shuffle is True and idx == len(self)-1:
            # reshuffle samples next call
            self._do_shuffle = True

        x_avgs = []
        x_pres = []
        y_pres = []
        x_posts = []
        y_posts = []
        stacked = []
        #x_pre = np.empty(0)
        #x_post = np.empty(0)
        #y_pre = np.empty(0)
        #y_post = np.empty(0)
        return_postmask = self.return_postmask

        for (pre, post) in self.samples[idx*self.batch_size:(idx+1)*self.batch_size]:
            if 'test' in pre.img_name or 'test' in post.img_name:
                x_pre, x_post = pre.image(), post.image()
                if self.return_average is True:
                    return np.expand_dims( (x_pre.astype(np.uint32) + x_post.astype(np.uint32) / 2).astype(np.uint8), axis=0 ), pre.img_path
                elif self.return_stacked is True:
                    return np.expand_dims(np.dstack([x_pre, x_post]), axis=0), pre.img_path
                else:
                    return (x_pre, x_post), pre.img_path

            premask = pre.sm_multichannelmask() if self.segmentation_models else pre.multichannelmask()
            postmask = post.sm_multichannelmask() if self.segmentation_models else post.multichannelmask()
            if not return_postmask:
                premask = convert_postmask_to_premask(postmask.copy())

            preimg = pre.image()
            postimg = post.image()
            avgimg = ((preimg.astype(np.uint32) + postimg.astype(np.uint32)) / 2).astype(np.uint8)

            # un-classified screws training up (test set doesn't include that value), so we zero them all out
            preimg, postimg, postmask = eliminate_unclassified(preimg, postimg, postmask)

            if preprocess is True:
                if not self.return_average:
                    preimg = preprocess_input(preimg)
                    postimg = preprocess_input(postimg)
                else:
                    avgimg = preprocess_input(avgimg)

            # training deformations
            if isinstance(self.transform, float) and random.random() < float(self.transform):
                # Rotate 90-270 degrees
                rotate_dict = { 'theta': 90 * random.randint(1, 3),}

                # rotate the sample and mask by the same angle together
                if not self.return_average:
                    preimg = self.image_datagen.apply_transform(preimg, rotate_dict)
                    postimg = self.image_datagen.apply_transform(postimg, rotate_dict)
                else:
                    avgimg = self.image_datagen.apply_transform(avgimg, rotate_dict)

                premask = self.image_datagen.apply_transform(premask, rotate_dict)
                postmask = self.image_datagen.apply_transform(postmask, rotate_dict)

            if isinstance(self.transform, float) and random.random() < float(self.transform):
                # apply a Gaussian blur to the sample, but not the mask
                ksize = 3
                if not self.return_average:
                    preimg = apply_gaussian_blur(preimg, kernel=(ksize,ksize))
                    postimg = apply_gaussian_blur(postimg, kernel=(ksize,ksize))
                else:
                    avgimg = apply_gaussian_blur(avgimg, kernel=(ksize,ksize))

            if not self.return_average:
                x_pre = np.array(preimg, copy=False).astype(np.int32)
                x_post = np.array(postimg, copy=False).astype(np.int32)
            else:
                x_avg = np.array(avgimg, copy=False)

            if self.return_single_channel is True:
                # segmentation_models compatibility
                y_pre = np.array(premask.astype(int))
                y_post = np.array(postmask.astype(int))


                #x_pre = tf.compat.v1.image.resize(x_pre, (1008, 1008), align_corners=True)
                #x_post = tf.compat.v1.image.resize(x_post, (1008, 1008), align_corners=True)
            else:
                y_pre = np.array(premask.astype(int).reshape(S.MASKSHAPE[0]*S.MASKSHAPE[1], -1), copy=False)
                y_post = np.array(postmask.astype(int).reshape(S.MASKSHAPE[0]*S.MASKSHAPE[1], -1), copy=False)

            #y_pre = np.array(premask.astype(int), copy=False)
            #y_post = np.array(postmask.astype(int), copy=False)

            #y_pre = convert_prediction(y_pre)
            #y_post = convert_prediction(y_post)
            #y_pre = y_pre.reshape(1, 1024, 1024, 1)
            #y_post = y_pre.reshape(1, 1024, 1024, 1)

            if interlace is True:
                x_pre, x_post = interlace(x_pre, x_post)
            elif self.return_stacked is True:
                stacked.append(np.dstack([x_pre, x_post]))

            if not self.return_average:
                x_pres.append(x_pre)
                x_posts.append(x_post)
            else:
                x_avgs.append(x_avg)
            y_pres.append(y_pre)
            y_posts.append(y_post)

            #x_pre = np.array(pre.chips(preimg)).astype(np.int32)
            #x_post = np.array(post.chips(postimg)).astype(np.int32)

            #y_pre = np.array([chip.astype(int).reshape(S.MASKSHAPE[0]*S.MASKSHAPE[1], S.N_CLASSES) for chip in pre.chips(premask)])
            #y_post = np.array([chip.astype(int).reshape(S.MASKSHAPE[0]*S.MASKSHAPE[1], 6) for chip in post.chips(postmask)])

        if not self.return_average:
            x_pres = np.array(x_pres, copy=False)
            x_posts = np.array(x_posts, copy=False)
        else:
            x_avgs = np.array(x_avgs, copy=False)

        y_pres = np.array(y_pres, copy=False)
        y_posts = np.array(y_posts, copy=False)

        y_ret = y_posts if return_postmask is True else y_pres
        if self.return_average is True:
            return np.array(x_avgs, copy=False), y_ret
        elif self.return_post_only is True:
            return x_posts, y_ret
        elif self.return_stacked is True:
            return np.array(stacked, copy=False), y_ret
        else:
            return (x_pres, x_posts), y_ret

    def prune_to(self, filename):
        for samples in df.samples:
            if filename in samples[0].img_name or filename in samples[1].img_name:
                return samples
        return None

    @staticmethod
    def from_pickle(picklefile:str=S.PICKLED_TRAINSET):
        with open(picklefile, "rb") as f:
            return pickle.load(f)

    def to_pickle(self, picklefile:str=S.PICKLED_TRAINSET):
        with open(picklefile, "wb") as f:
            return pickle.dump(self, f)


class BuildingDataflow(Dataflow):
    """
    Dataflow object that returns individual building coordinates and the corresponding
    damage class.
    """
    def __init__(self, *args, **kwargs):
        super(BuildingDataflow, self).__init__(buildings_only=True, batch_size=1, *args, **kwargs)
        self.limit = 80

    def __getitem__(self, idx):
        boxes = []
        classes = []
        step = 24
        limit = self.limit

        for (pre, post) in self.samples[idx*self.batch_size:(idx+1)*self.batch_size]:
            preimg = pre.image()
            postimg = post.image()

            for bldg in post.buildings:
                # skip un-classified
                img, klass = bldg.extract_from_images_by_contours(preimg,postimg)
                if klass == 5:
                    klass = 4
                boxes.append(img)
                #classes.append(klass)
                """
                for i in range(0, len(img), step):
                    for j in range(0, len(img[i]), step):
                        boxes.append(img[i:i+step,j:j+step,...])
                        classes.append(klass)
                """

                # make all `un-classified` buildings the most common damage type
                #if klass == 5:
                #    klass = 5
                onehot = [0] * 4
                onehot[int(klass - 1)] = 1
                classes.append(onehot)

        shuffled = list(zip(boxes, classes))
        random.shuffle(shuffled)
        boxes, classes = zip(*shuffled)

        return np.array(boxes)[:limit], np.array(classes)[:limit]


class DamagedDataflow(Dataflow):
    """
    Dataflow object that only includes pre- and post-disaster images
    that have damaged buildings (buildings with a damage class of 2, 3, or 4).
    """
    def __init__(self, *args, **kwargs):
        Dataflow.__init__(self, *args, **kwargs)

        # discard any sample pairs with only undamaged (or unclassified) buildings
        samples_copy = self.samples.copy()
        self.samples.clear()
        for (pre,post) in samples_copy:
            for b in post.buildings:
                if b.color() > 1 and b.color() <= 4:
                    self.samples.append( (pre,post) )
                    break

class Building:
    """Carries the data for a single building; multiple Buildings are
       owned by a single Target object"""
    MAP = {}
    PRE = 1
    POST = 2

    def __init__(self, target=None):
        self.wkt = None
        self._coords = None
        self.target = None

    @staticmethod
    def get(uid:str, key=None):
        """
        Look up a building by UID.  Returns both PRE and POST image buildings if one
        isn't specified.
        """
        if key is None:
            return Building.MAP[(uid,Building.PRE)], Building.MAP[(uid,Building.POST)]
        return Building.MAP[(uid,key)]

    def __repr__(self):
        string = "<Building {} from {}: {}>".format(self.uid, self.target.img_name, str(self.coords()).replace("\n", ","))
        return string

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
        # For pre-disaster images, the building subtype isn't specified, but we want
        # the value of those buildings in our masks to be 1 and nothing else
        if self.klass is None:
            return 1
        # post-disaster images include subtype information (see settings.py for CLASSES)
        ret = S.CLASSES.index(self.klass)
        return ret

    def downvert(self, x, y,
                 orig_x=S.SAMPLESHAPE[0],
                 orig_y=S.SAMPLESHAPE[1],
                 new_x=S.TARGETSHAPE[0],
                 new_y=S.TARGETSHAPE[1]):
        x = x * (new_x / orig_x)
        y = y * (new_y / orig_y)
        return round(x), round(y)

    def upvert(self, x, y,
               orig_x=S.SAMPLESHAPE[0],
               orig_y=S.SAMPLESHAPE[1],
               new_x=S.TARGETSHAPE[0],
               new_y=S.TARGETSHAPE[1]):
        x = x / (new_x / orig_x)
        y = y / (new_y / orig_y)
        return round(x), round(y)

    @staticmethod
    def get_all_in(pre:np.ndarray, post:np.ndarray, mask:np.ndarray):
        """
        Return the portions of `pre` and `post` that contain buildings.

        Works by extracting contiguous regions in the `mask` provided, so
        this is imperfect.
        """
        if pre.shape[-1] > 3:
            post = pre[...,3:]
            pre = pre[...,:3]
        pre = pre.squeeze()#astype(np.uint8)
        post = post.squeeze()#astype(np.uint8)
        mask = mask.astype(np.uint8)
        boxes = []
        coords = []
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x,y,w,h = cv2.boundingRect(c)
            if w < 5 or h < 5:
                continue
            # extract buildings along with a 5 pixel boundary
            x_ = np.max([0,x])
            y_ = np.max([0,y])
            #preview = pre[y_:y+h,x_:x+w,:]
            postview = post[y_:y+h,x_:x+w,:]
            box = cv2.resize(postview, S.DMG_SAMPLESHAPE, interpolation=cv2.INTER_AREA)
            #tiny = postview#np.vstack([preview,postview])

            #x,y,z = tiny.shape
            #x,y = np.min([S.DMG_SAMPLESHAPE[0],x]), np.min([S.DMG_SAMPLESHAPE[1],y])
            #box = np.zeros(S.DMG_SAMPLESHAPE, dtype=np.uint8)
            #box[0:x,0:y,:] = tiny.astype(np.uint8)[:S.DMG_SAMPLESHAPE[0],:S.DMG_SAMPLESHAPE[1],:]

            boxes.append(box)
            coords.append((x,y,w,h))

        return np.array(boxes), np.array(coords)

    def extract_from_images(self, pre:np.ndarray, post:np.ndarray):
        """
        Extracts bounding boxes for the coordinates of this building from
        the passed `pre` and `post` images.
        """
        boxes = []
        classes = []
        coords = self.coords()
        x,y,w,h = cv2.boundingRect(np.array(coords))

        for img in [post]:
            box = img[y:y+h,x:x+w,...]
            boxes.append(box)
            classes.append(self.color())

        return np.hstack(boxes), np.array(classes, copy=False)

    def extract_from_images_by_contours(self, pre:np.ndarray, post:np.ndarray):
        """
        Slice out the patches of `pre` and `post` corresponding to the location of this building,
        then return them stacked on top of each other and zero-padded to S.DMG_SAMPLESHAPE.

        i.e. get individual pre-disaster and post-disaster buildings for training

        Returns: (box, subtype)

            box:     the stacked image.
            subtype: the damage class (subtype) of the buildings in the post-disaster image
        """
        boxes = []
        mask = np.zeros(pre.shape[:2], dtype=np.uint8)#S.SAMPLESHAPE)#[S.DMG_SAMPLESHAPE[0]//2,S.DMG_SAMPLESHAPE[1]//2], dtype=np.uint8)
        #ret = np.zeros(S.DMG_SAMPLESHAPE)

        cv2.fillPoly(mask,np.array([self.coords()]), 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        logger.debug(f"number of building contours: {len(contours)}")
        for c in contours:
            x,y,w,h = cv2.boundingRect(c)

            # extract buildings along with a 5 pixel boundary
            x_ = x#np.max([0,x-5])
            y_ = y#np.max([0,y-5])
            preview = pre[y_:y+h,x_:x+w,:]
            postview = post[y_:y+h,x_:x+w,:]
            #np.stack([preview,postview])
            tiny = postview

            #tiny = np.vstack([preview,postview])

            x,y,z = tiny.shape
            x,y = np.min([S.DMG_SAMPLESHAPE[0],x]), np.min([S.DMG_SAMPLESHAPE[1],y])
            box = np.zeros(S.DMG_SAMPLESHAPE, dtype=np.uint8)
            box[0:x,0:y,:] = tiny.astype(np.uint8)[:S.DMG_SAMPLESHAPE[0],:S.DMG_SAMPLESHAPE[1],:]


        return box, self.color()


class Target:
    """
    Target objects provide filenames, metadata, input images, and masks for training.

    One target per input image (i.e. two targets per pre-disaster, post-disaster set).
    """
    def __init__(self, text:str="", df:Dataflow=None):
        self._df = df
        self.buildings = []
        self.image_datagen = ImageDataGenerator()
        if text:
            self.parse_json(text)

    def parse_json(self, text:str):
        """
        Parse a JSON formatted string and assign instance variables to this target
        based on the contents; format is expected to comply with the xBD dataset.
        """
        data = json.loads(text)
        self.img_name = data['metadata']['img_name']
        self.metadata = data['metadata']

        for feature in data['features']['xy']:
            prop = feature['properties']
            if prop['feature_type'] != 'building':
                continue
            b = Building(target=self)
            b.klass = prop.get('subtype', None)
           
            if b.klass not in S.CLASSES:
                logger.error(f"Unrecognized building subtype: {b.klass}")

            b.wkt = feature.get('wkt', None)
            b.uid = prop['uid']
            b.target = self
            #if len(b.coords()) > 1:
            self.buildings.append(b)
            #else:
            #    continue

            if b.klass is None:
                key = "pre"
            else:
                key = "post"
            Building.MAP[(b.uid, key)] = b

    def mask(self):
        """
        Get this Target's mask for supervised training of the model (single channel).

        Returns: a grayscale image, each pixel value being the class for that pixel location.
        """
        img = np.zeros(S.SAMPLESHAPE, dtype=np.uint8)
        for b in self.buildings:
            coords = b.coords()#downvert=True, orig_x=1024, new_y=1024)#, new_x=256,new_y=256)
            if len(coords) > 0:
                try:
                    cv2.fillPoly(img, np.array([coords]), b.color())
                except Exception as exc:
                    logger.error("cv2.fillPoly(img, {}, {}) call failed: {}".format(str(coords), b.color(), exc))
        return img

    def multichannelchipmask(self):
        """
        Return a multichannel mask cut up into 1/16th sized squares.
        """
        mask = self.multichannelmask()
        return self.chips(image=mask)

    def multichannelmask(self):
        """
        Get the Target's mask for supervised training of the model.  Each channel
        corresponds to a class, and values are either 0 (background) or 1 (object at
        that location).

        When reshaped, this results in a target mask that's a one hot encoded list
        of pixels, e.g. [[0,0,1,0], [1,0,0,0], [1,0,0,0]] if there were 4 classes, 3 pixels,
        and 2 of those pixels were background with 1 pixel an object of class 2.
        Note that we assume channels come last, not first.

        Returns: 1024 x 1024 x N_CLASSES numpy.ndarray with dtype np.uint8.
        """
        # Each class gets one channel
        # Fill background channel with 1s
        chans = [np.ones(S.SAMPLESHAPE[:2], dtype=np.uint8)]
        top = 6 if "post" in self.img_name else S.N_CLASSES
        for i in range(1, top):
            chan = np.zeros(S.SAMPLESHAPE[:2], dtype=np.uint8)
            chans.append(chan)

        # For each building, set pixels according to (x,y) coordinates; they end up
        # being a one-hot-encoded vector corresponding to class for each (x,y) location
        for b in self.buildings:
            coords = b.coords()
            # "un-classified" buildings will cause an error during evaluation, so don't train
            # for them
            color = b.color()# if b.color() != 5 else 1
            if len(coords) > 0:
                # Set the pixels at coordinates in this class' channel to 1
                cv2.fillPoly(chans[color], np.array([coords]), 1)
                # Zero out the background pixels for the same coordinates
                cv2.fillPoly(chans[0], np.array([coords]), 0)
        img = np.dstack(chans)
        return img

    def mrcnn_image(self):
        """
        Get a Mask-RCNN compatible image.
        """
        pre = self.image()
        return pre

    def mrcnn_masks(self):
        """
        Get Mask-RCNN compatible masks for an image (one object per channel).
        """
        masks = []
        klasses = []
        for b in self.buildings:
            if b.color() == 5:
                continue
            img = np.zeros(S.MASKSHAPE[:2], dtype=np.uint8)
            coords = b.coords()
            if len(coords) > 0:
                cv2.fillPoly(img, np.array([coords]), 1)
                masks.append(img)
                klasses.append(b.color())
        if len(masks) == 0:
            return np.array(masks, copy=False), np.ones([0], dtype=np.int32)
        #masks.append(np.zeros(MASKSHAPE[:2], dtype=np.uint8))
        return np.dstack(masks).astype(bool), np.array(klasses)#np.ones([len(masks)], dtype=np.int32)

    def image(self):
        """Get this Target's input image (i.e. satellite chip) to feed to the model"""
        return skimage.io.imread(self.image_path())

    def image_path(self):
        return self.img_path

    @staticmethod
    def from_json(filename:str, transform:float=0.0, df:Dataflow=None):
        """Create a Target object from a path to a .JSON file"""
        with open(filename) as f:
            t = Target(f.read())
            t._df = df
            t.img_path = get_abs_path(t.img_name)
            t.transform = transform
            return t

    @staticmethod
    def from_png(filename:str, df:Dataflow=None):
        """
        Create a Target object from a path to a .PNG file.  For use with test images that have
        no metadata in a .json file.

        Note: from_json() will only store the base filename, but this
              function expects the string passed to be the full (absolute)
              path to the .png file.
        """
        target = Target()
        target._df = df
        target.img_name = filename
        target.img_path = filename
        target.metadata = dict()
        return target

    def chips(self, image=None, step=256, max_x=S.SAMPLESHAPE[0], max_y=S.SAMPLESHAPE[1]):
        """
        Turn an image into 1/16th-sized chips.
        """
        ret = []
        if image is None:
            image = self.image()
        for i in range(0, max_x, step):
            for j in range(0, max_y, step):
                ret.append(image[...,i:i+step,j:j+step,:])
        return ret

    @staticmethod
    def weave(chips):
        """
        Stitch 1/16th-sized chips back together into the full size image.
        """
        return np.vstack([np.hstack(chips[:4]), np.hstack(chips[4:8]), np.hstack(chips[8:12]), np.hstack(chips[12:])])


if __name__ == '__main__':
    # Testing and data inspection
    import time
    import train
    import score
    df = Dataflow()
    if len(sys.argv) > 1:
        df.samples = [df.prune_to(sys.argv[1])]
    model = train.build_model()
    train.load_weights(model)
    while True:
        idx = random.randint(0,len(df) - 1)
        xy= df.samples[idx]
        i = 1
        fig = plt.figure()
        for sample in xy:

            fig.add_subplot(2,4,i)
            plt.imshow(sample.image())
            plt.title(sample.img_name)
            i += 1

            fig.add_subplot(2,4,i)
            plt.imshow(sample.image())
            #background, no-damage, minor-damage, major-damage, destroyed, un-classified
            #    0           1           2             3            4            5
            colormap = {0: 'w', 1: 'b', 2: 'g', 3: 'y', 4: 'r', 5: 'k'}
            polys = []
            colors = set()
            for b in sample.buildings:
                coords = [(c[0], c[1]) for c in b.coords()]#[b.upvert(x,y,1024,1024) for (x,y) in zip(b.coords()[:,0], b.coords()[:,1])]
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
                elif S.CLASSES[b.color()] not in colors:
                    label = S.CLASSES[b.color()]
                else:
                    label = None
                colors.add(label)
                polys.append(plt.plot(xs, ys, colormap[b.color()], antialiased=True, lw=0.85, label=label))
            plt.title("building polygons")
            if len(sample.buildings) > 0:
                plt.legend()
            i += 1

            fig.add_subplot(2,4,i)
            mask = sample.multichannelmask()
            mask = convert_prediction(mask)
            plt.imshow(mask)
            plt.title("target mask")
            i += 1

#            fig.add_subplot(2,4,i)
#            pred = model.predict(np.array(sample.chips()))
#jj            plt.imshow(infer.weave_pred_no_argmax(pred))
 #           plt.title("Prediction")
            #i += 1
            #toggle += 1
            #fig.add_subplot(2,5,i)
            #plt.imshow(lossvalue.numpy().squeeze())
            #plt.title("Loss")
            #i += 1

        plt.show()
        time.sleep(1)
