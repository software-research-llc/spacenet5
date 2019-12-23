"""
Mask R-CNN
Train on the toy Ship dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 ship.py train --dataset=./datasets --weights=coco

    # Resume training a model that you had trained earlier
    python3 ship.py train --dataset=./datasets --weights=../../logs/ship20180815T0023/mask_rcnn_ship_0067.h5

    # Train a new model starting from ImageNet weights
    python3 ship.py train --dataset=./datasets --weights=imagenet

    # Apply color splash to an image
    python3 ship.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 ship.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import pandas as pd
from skimage.morphology import label

# Root directory of the project
ROOT_DIR = "./"
TRANSFORM = 0.30

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from settings import *

import flow
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class Xview2Config(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "xView2"

    #LEARNING_RATE = 0.002

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + ship

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 8251
    VALIDATION_STEPS = 50

    DETECTION_MIN_CONFIDENCE = 0.75

    #RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.0

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    #RPN_NMS_THRESHOLD = 0.8

    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (112, 112)

    MAX_GT_INSTANCES = 200
    DETECTION_MAX_INSTANCES = 300

    IMAGE_RESIZE_MODE = "none"

    #TRAIN_ROIS_PER_IMAGE = 512

    # # Length of square anchor side in pixels
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    #
    # # Ratios of anchors at each cell (width/height)
    # # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    # RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    #
    # # Image mean (RGB)
    # MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    MEAN_PIXEL = np.array([86.81837701005743, 90.15986695022038, 67.48294656034845])
    #MEAN_PIXEL = np.array([0.,0.,0.])

############################################################
#  Dataset
############################################################

class Xview2Dataset(utils.Dataset):

    def load_xview2(self, subset):
        """Load a subset of the Ship dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("building", 1, "building")

        # Load image ids (filenames) and run length encoded pixels
        if subset == "train":
            self.df = flow.Dataflow(files=flow.get_training_files(), buildings_only=True, transform=TRANSFORM, return_postmask=True, batch_size=1)
        elif subset == "val":
            self.df = flow.Dataflow(files=flow.get_validation_files(), buildings_only=True, return_postmask=True, batch_size=1)
        else:
            raise Exception("unrecognized subset %s" % subset)

        n = 0
        for (pre, post) in self.df.samples:
            path = pre.img_path
            if len(pre.buildings) == 0:
                continue
            n += 1
            self.add_image("building", image_id=path, width=1024, height=1024, target=pre, path=path)
        if subset == "train":
            old = Xview2Config.STEPS_PER_EPOCH
            Xview2Config.STEPS_PER_EPOCH = n // 2
            print("Changing STEPS_PER_EPOCH from {} to {}".format(old, Xview2Config.STEPS_PER_EPOCH))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        return self.image_info[image_id]['target'].rcnn_masks()
        masks = []
        for b in self.image_info[image_id]['target'].buildings:
            mask = np.zeros([MASKSHAPE[0], MASKSHAPE[1]])
            coords = b.coords()
            for x,y in coords:
                cv2.fillPoly(x,y)
        # If not a ship dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "ship":
            return super(self.__class__, self).load_mask(image_id)

        # Convert RLE Encoding to bitmap mask of shape [height, width, instance count]
        info = self.image_info[image_id]
        img_masks = info["img_masks"]
        shape = [info["height"], info["width"]]

        # Mask array placeholder
        mask_array = np.zeros([info["height"], info["width"], len(info["img_masks"])],dtype=np.uint8)

        # Build mask array
        for index, mask in enumerate(img_masks):
            mask_array[:,:,index] = self.rle_decode(mask, shape)

        return mask_array.astype(np.bool), np.ones([mask_array.shape[-1]], dtype=np.int32)

    def load_image(self, image_id):
        return self.image_info[image_id]['target'].rcnn_image()

    def image_reference(self, image_id):
        """Return the path of the image."""
        return self.image_info[image_id]['target'].image_path()
        info = self.image_info[image_id]
        if info["source"] == "ship":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def rle_encode(self,img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels = img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def rle_decode(self, mask_rle, shape=(768, 768)):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background

        '''
        if not isinstance(mask_rle, str):
            img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
            return img.reshape(shape).T

        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T

    def multi_rle_encode(self, mask_array):
        # Go back from Bitmask to RLE
        re_encoded_to_rle_list = []
        for i in np.arange(mask_array.shape[-1]):
            boolean_mask = mask_array[:,:,i]
            re_encoded_to_rle = self.rle_encode(boolean_mask)
            re_encoded_to_rle_list.append(re_encoded_to_rle)

        return re_encoded_to_rle_list

    def multi_rle_decode(self, rle_img_masks):
        # Build mask array
        mask_array = np.zeros([768, 768, len(rle_img_masks)],dtype=np.uint8)

        # Go from RLE to Bitmask
        for index, rle_mask in enumerate(rle_img_masks):
            mask_array[:,:,index] = self.rle_decode(rle_mask)

        return mask_array

    def test_endcode_decode(self):
        ROOT_DIR = os.path.abspath("../../")
        SHIP_DIR = os.path.join(ROOT_DIR, "./samples/ship/datasets")
        ship_segmentations_df = pd.read_csv(os.path.join(SHIP_DIR,"train_val","train_ship_segmentations.csv"))
        rle_img_masks = ship_segmentations_df.loc[ship_segmentations_df['ImageId'] == "0005d01c8.jpg", 'EncodedPixels']
        rle_img_masks_list = rle_img_masks.tolist()

        mask_array = self.multi_rle_decode(rle_img_masks)
        print("mask_array shape", mask_array.shape)
        # re_encoded_to_rle_list = self.multi_rle_encode(mask_array)
        re_encoded_to_rle_list = []
        for i in np.arange(mask_array.shape[-1]):
            boolean_mask = mask_array[:,:,i]
            re_encoded_to_rle = self.rle_encode(boolean_mask)
            re_encoded_to_rle_list.append(re_encoded_to_rle)

        print("Masks Match?", re_encoded_to_rle_list == rle_img_masks_list)
        print("Mask Count: ", len(rle_img_masks))
        print("rle_img_masks_list", rle_img_masks_list)
        print("re_encoded_to_rle_list", re_encoded_to_rle_list)

        # Check if re encoded rle masks are the same as the original ones
        return re_encoded_to_rle_list == rle_img_masks_list

    def masks_as_image(self,in_mask_list):
        # Take the individual ship masks and create a single mask array for all ships
        all_masks = np.zeros((768, 768), dtype = np.uint8)
        for mask in in_mask_list:
            if isinstance(mask, str):
                all_masks |= self.rle_decode(mask)
        return all_masks

    def multi_rle_encode(self,img):
        labels = label(img)
        if img.ndim > 2:
            return [self.rle_encode(np.sum(labels==k, axis=2)) for k in np.unique(labels[labels>0])]
        else:
            return [self.rle_encode(labels==k) for k in np.unique(labels[labels>0])]

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = Xview2Dataset()
    dataset_train.load_xview2("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = Xview2Dataset()
    dataset_val.load_xview2("val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=160, layers="all")


def output2prediction(output):
    output = output[0]
    rois = output['rois']
    class_ids = output['class_ids']
    scores = output['scores']
    masks = output['masks'].astype(np.uint8)

    return get_mask(masks)

def get_mask(masks):
    try:
        ret = np.zeros_like(masks[:,:,0])
    except:
        return np.zeros(MASKSHAPE)
    for i in range(masks.shape[-1]):
        ret = np.logical_or(masks[:,:,i], ret)
    return ret

def f1score(y_true, y_pred):
    if isinstance(y_true, list):
        tp = 0
        fp = 0
        fn = 0
        for (true, pred) in zip(y_true, y_pred):
            _, rtp, rfp, rfn = _f1(true, pred, return_counts=True)
            tp += rtp
            fp += rfp
            fn += rfn
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        return 2 * prec * rec / np.clip(prec + rec, 1e-10, 1e10)
    else:
        return _f1(y_true, y_pred)

def _f1(y_true, y_pred, return_counts=False):
    import keras.backend as K
    import tensorflow as tf
    y_true, y_pred = y_true.astype(np.int8), y_pred.astype(np.int8)
    #one = np.ones_like(y_pred)
    tp = np.sum(y_true * y_pred)
    fn = np.sum(np.clip(y_true - y_pred, 0, 1))
    fp = np.sum(np.clip(y_pred - y_true, 0, 1))
    #fn = K.sum(tf.logical_and(y_pred != y_true, y_pred == one))
    #fp = K.sum(tf.logical_and(y_pred == one, y_true != one))

    #tp = np.sum(np.logical_and(y_true, y_pred))
    #xor = np.logical_xor(y_true, y_pred)
    #fp = np.sum(np.logical_and(xor, y_pred))
    #fn = np.sum(np.logical_and(xor, y_true))

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    if return_counts:
        return 2 * prec * rec / np.clip(prec + rec, 1e-10, 1e10), tp, fp, fn
    else:
        return 2 * prec * rec / np.clip(prec + rec, 1e-10, 1e10)

def score_predictions(model):
    import tqdm
    dataset = Xview2Dataset()
    dataset.load_xview2("val")
    dataset.prepare()

    y_true, y_pred = [], []
    for image_id in tqdm.tqdm(dataset.image_ids, desc="Generating predictions"):
        image = dataset.load_image(image_id)
        gtmask = dataset.load_mask(image_id)
        mask = get_mask(gtmask[0])
        output = model.detect([image], verbose=0)
        pred = output2prediction(output)

        y_true.append(mask)
        y_pred.append(pred)

    print("Score: %f" % f1score(y_true, y_pred))

def show_predictions(model):
    import matplotlib.pyplot as plt
    import time
    import random
    dataset = Xview2Dataset()
    dataset.load_xview2("val")
    dataset.prepare()
    all_ids = dataset.image_ids.copy()
    random.shuffle(all_ids)

    for image_id in all_ids:
        image = dataset.load_image(image_id)
        gtmask = dataset.load_mask(image_id)
        mask = get_mask(gtmask[0])
        output = model.detect([image], verbose=1)
        pred = output2prediction(output)

        score = f1score(mask, pred)

        fig = plt.figure()
        fig.canvas.set_window_title(dataset.image_reference(image_id))

        fig.add_subplot(2,2,1)
        plt.imshow(image.squeeze())
        plt.title("Sample")

        fig.add_subplot(2,2,2)
        img = image.squeeze()
        try:
            img[pred] = [128,0,128]
        except Exception as exc:
            print("error: %s" % str(exc))
        plt.imshow(img)
        plt.title("Overlay")

        fig.add_subplot(2,2,3)
        plt.imshow(mask)
        plt.title("Ground truth")

        #import pdb; pdb.set_trace()
        fig.add_subplot(2,2,4)
        plt.imshow(pred.squeeze())
        plt.title("Prediction (F1-Score: %f)" % score)

        plt.show()
        time.sleep(1)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect ships.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = Xview2Config()
    else:
        class InferenceConfig(Xview2Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "score":
        score_predictions(model)
    else:
        show_predictions(model)
