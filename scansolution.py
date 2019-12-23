import skimage
import numpy as np
import os
import sys
import tqdm
import random
from collections import Counter

counts = Counter()
totals = Counter()
areas = Counter()
filecount = 0

for file in tqdm.tqdm(os.listdir("solution")):
    img = skimage.io.imread("solution/{}".format(file))
    filecount += 1
    for i in range(6):
        totals = len(img[img>0])
        c = len(img[img==i])
        if totals > 0:
            c /= totals
        counts[i] += c
    """
    if len(img[img > 4]) > 0:
        img[img > 4] = random.randint(2,3)
        print("Corrected {}".format(file))
        skimage.io.imsave("solution/{}".format(file), img, check_contrast=False)
    elif len(img[img < 0]) > 0:
        img[img < 0] = 0
        print("Corrected {}".format(file))
        skimage.io.imsave("solution/{}".format(file), img, check_contrast=False)
    """
print('Counts:')
for i in range(6):
    print("{}: {}".format(i, counts[i]))
