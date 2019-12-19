import skimage
import numpy as np
import os
import sys
import tqdm

for file in tqdm.tqdm(os.listdir("solution")):
    img = skimage.io.imread("solution/{}".format(file))
    if len(img[img > 4]) > 0:
        img[img > 4] = 1
        print("Corrected {}".format(file))
        skimage.io.imsave("solution/{}".format(file), img, check_contrast=False)
    elif len(img[img < 0]) > 0:
        img[img < 0] = 0
        print("Corrected {}".format(file))
        skimage.io.imsave("solution/{}".format(file), img, check_contrast=False)

