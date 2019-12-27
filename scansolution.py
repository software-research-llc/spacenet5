"""
Scan solution .png files and return a proportional count of
pixel values.
"""
import skimage
import numpy as np
import os
import sys
import tqdm
import random
from collections import Counter
import settings as S

# fix invalid pixel values in a really dumb way
ALTER_SOLUTION = False

counts = Counter()
totals = Counter()
areas = Counter()
total = 0

for file in tqdm.tqdm(os.listdir("solution")):
    img = skimage.io.imread("solution/{}".format(file))
    for i in range(6):
        c = len(img[img==i])
        counts[i] += c
        total += c

    if not ALTER_SOLUTION:
        continue
    if len(img[img > 4]) > 0:
        img[img > 4] = random.randint(2,3)
        print("Corrected {}".format(file))
        skimage.io.imsave("solution/{}".format(file), img, check_contrast=False)
    elif len(img[img < 0]) > 0:
        img[img < 0] = 0
        print("Corrected {}".format(file))
        skimage.io.imsave("solution/{}".format(file), img, check_contrast=False)

print('Pixels by class:')
for i in range(6):
    print("  {:13s}: {:.5f}% ({})".format(str(S.CLASSES[i]), counts[i] / total, counts[i]))
