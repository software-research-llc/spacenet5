import random
import os
import sys
import snflow as flow
import pandas as pd
import numpy as np
import plac
import keras
import tqdm
import sngraph
import train
import infer
import create_submission
import random
from skimage.transform import resize
from skimage import io

USE_TRAINING_IMAGES = True

if not USE_TRAINING_IMAGES:
    flow.CITIES += ["AOI_9_San_Juan"]
    get_files = flow.get_test_filenames
else:
    get_files = flow.get_filenames

model = train.build_model()
train.load_weights(model)
graphs = []
masks = []
allfiles = get_files()
random.shuffle(allfiles)
for filename in tqdm.tqdm(allfiles):
    try:
        image = resize(io.imread(filename), flow.IMSHAPE, anti_aliasing=True)
        image = flow.normalize(image)
        mask, graph, preproc, skel = infer.infer(model, image, chipname=flow.Target.expand_imageid(filename))
        graphs.append(graph)
        masks.append(mask.squeeze())
    except KeyboardInterrupt:
        break

print("Creating submission")
create_submission.graphs_to_wkt(masks, graphs,
                                output_csv_path='solution/solution.csv')
