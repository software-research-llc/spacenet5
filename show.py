#!/usr/bin/env python3
from skimage import io
import matplotlib.pyplot as plt
import sys
import glob
import os

DATADIR = "nfs/data/cosmiq/spacenet/competitions/SN5_roads/tiles_upload/train/AOI_7_Moscow/PAN/"
filename = sys.argv[1]
if not os.path.exists(filename):
    filename = os.path.join(DATADIR, sys.argv[1])
    if not os.path.exists(filename):
        filename = glob.glob("%s/%s" % (DATADIR, sys.argv[1]))
        if filename and issubclass(type(filename), list):
            filename = filename[0]
        if not filename:
            raise FileNotFoundError("File not found: %s" % sys.argv[1])
print("Displaying %s" % filename)
im = io.imread(filename)
plt.imshow(im)
plt.show()
