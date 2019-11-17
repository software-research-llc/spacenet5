import os
from inspect import getsourcefile
import segmentation_models as sm


BATCH_SIZE = 8
# Train/validation split proportion
SPLITFACTOR = 0.9

BACKBONE = 'resnext50'
TOPCLASS = sm.FPN

CLASSES = ["background", "no-damage", "minor-damage", 'major-damage', 'destroyed', 'un-classified']
N_CLASSES = 1

TARGETSHAPE = [256,256,1]
SAMPLESHAPE = [256,256,3]
MASKSHAPE = [256,256,1]

PICKLED_TRAINSET = "trainingflow.pickle"
PICKLED_VALIDSET = "validationflow.pickle"

HOMEDIR = os.path.abspath(os.path.dirname(getsourcefile(lambda:0)))
IMAGEDIRS = ["/data/xview2/train/images", "/data/xview2/tier3/images"]
LABELDIRS = ["/data/xview2/train/labels", "/data/xview2/tier3/labels"]
TESTDIRS = ["/data/xview2/test/images"]
