import os
from inspect import getsourcefile


BATCH_SIZE = 20
# Train/validation split proportion
SPLITFACTOR = 0.9

# The architecture and weights to use for feature extraction (encoding)
BACKBONE = 'resnet50'
MODELSTRING = "unet.hdf5"# % BACKBONE

# Each pixel of the net outputs (the masks) is an index in to this list
#           0        1             2                3              4              5
CLASSES = [None, "no-damage", "minor-damage", 'major-damage', 'destroyed', 'un-classified']
N_CLASSES = len(CLASSES)

# Shape of the training input images
INPUTSHAPE = [256,256,3]
MASKSHAPE = [256,256,2]
# The shape of the raw on-disk input images
SAMPLESHAPE = [1024,1024,3]
TARGETSHAPE = INPUTSHAPE

# The training and validation set files (pickled versions for faster loading)
PICKLED_TRAINSET = "trainingsamples.pickle"
PICKLED_VALIDSET = "validationsamples.pickle"

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
