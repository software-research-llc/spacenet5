import os
from inspect import getsourcefile

BACKBONE = 'resnext50'
CLASSES = ["no-damage", "minor-damage", 'major-damage', 'destroyed', 'un-classified']
N_CLASSES = len(CLASSES)

TARGETSHAPE = [1024,1024,1]
SAMPLESHAPE = [1024,1024,3]
MASKSHAPE = [1024,1024,N_CLASSES]

HOMEDIR = os.path.abspath(os.path.dirname(getsourcefile(lambda:0)))
IMAGEDIRS = ["/data/xview2/train/images", "/data/xview2/tier3/images"]
LABELDIRS = ["/data/xview2/train/labels", "/data/xview2/tier3/labels"]
TESTDIRS = ["/data/xview2/test/images"]
