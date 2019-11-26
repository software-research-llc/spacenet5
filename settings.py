import os
from inspect import getsourcefile
import segmentation_models as sm


BATCH_SIZE = 1
# Train/validation split proportion
SPLITFACTOR = 0.9

# The architecture and weights to use for feature extraction (encoding)
BACKBONE = 'resnext50'
# The architecture to use for decoding (upscaling)
TOPCLASS = sm.Unet

# Each pixel of the net outputs (the masks) is an index in to this list
#               0            1             2                 3             4              5
CLASSES = ["no-damage", "minor-damage", 'major-damage', 'destroyed', 'un-classified', None ]
N_CLASSES = len(CLASSES)

# Shape of the training input images; basically the same as MASKSHAPE
INPUTSHAPE = [1024,1024,3]
MASKSHAPE = [1024,1024,1]
# The shape of the input samples
SAMPLESHAPE = [1024,1024,3]
TARGETSHAPE = INPUTSHAPE

# The training and validation set files (pickled versions for faster loading)
PICKLED_TRAINSET = "trainingflow.pickle"
PICKLED_VALIDSET = "validationflow.pickle"

# Base directory of the Python files
HOMEDIR = os.path.abspath(os.path.dirname(getsourcefile(lambda:0)))
# Directories where the training images are located
IMAGEDIRS = ["/data/xview2/train/images", "/data/xview2/tier3/images"]
# Directories where the .json files describing the images are
LABELDIRS = ["/data/xview2/train/labels", "/data/xview2/tier3/labels"]
# Directories where the test images are kept
TESTDIRS = ["/data/xview2/test/images"]
# The following is not to be changed: used only to locate the .png file for reading
ALLIMAGEDIRS = IMAGEDIRS + TESTDIRS
