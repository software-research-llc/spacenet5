import os
from inspect import getsourcefile


# The architecture to use
ARCHITECTURE = "motokimura"
DAMAGE_ARCHITECTURE = 'motokimura-mobilenetv2'

# Train/validation split proportion
SPLITFACTOR = 0.9
BATCH_SIZE = 2

#           0        1             2                3              4              5
CLASSES = [None, "no-damage", "minor-damage", 'major-damage', 'destroyed', 'un-classified']
N_CLASSES = len(CLASSES)

DAMAGE = True 

MODELSTRING = "%s-%d.hdf5" % (ARCHITECTURE, N_CLASSES)
DMG_MODELSTRING = "damage-%s.hdf5" % DAMAGE_ARCHITECTURE

MODELSTRING_BEST = MODELSTRING.replace(".hdf5", "-best.hdf5")
DMG_MODELSTRING_BEST = DMG_MODELSTRING.replace(".hdf5", "-best.hdf5")

# The maximum length of building patches for damage classification
DAMAGE_MAX_X = 224
DAMAGE_MAX_Y = 224

# Shape of the training input images
INPUTSHAPE = [1024,1024,6]
MASKSHAPE = [1024,1024,6]
# The shape of the raw on-disk input images
SAMPLESHAPE = [1024,1024,3]
TARGETSHAPE = INPUTSHAPE

# size of combination pre+post building images
DMG_SAMPLESHAPE = [128,128,3]
DMG_INPUTSHAPE = DMG_SAMPLESHAPE

# The training and validation set files (pickled versions for faster loading)
PICKLED_TRAINSET = "trainingsamples.pickle"
PICKLED_VALIDSET = "validationsamples.pickle"

# Base directory of the Python files
HOMEDIR = os.path.abspath(os.path.dirname(getsourcefile(lambda:0)))
# Directories where the training images are located
IMAGEDIRS = ["train/images", "tier3/images"]
# Directories where the .json files describing the images are
LABELDIRS = ["train/labels", "tier3/labels"]
# Directories where the test images are kept
TESTDIRS = ["/data/xview2/test/images"]
# The following is not to be changed: used only to locate the .png file for reading
ALLIMAGEDIRS = IMAGEDIRS + TESTDIRS
