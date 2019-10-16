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

flow.CITIES += ["AOI_9_San_Juan"]
model = train.build_model()
train.load_weights(model)
graphs = []
masks = []
for filename in tqdm.tqdm(flow.get_test_filenames()):
    image = flow.get_image(filename)
    try:
        mask, graph, preproc, skel = infer.infer(model, image, chipname=flow.Target.expand_imageid(filename))
        graphs.append(graph)
        masks.append(image[0])
    except KeyboardInterrupt:
        break

print("Creating submission")
create_submission.graphs_to_wkt(masks, graphs,
                                output_csv_path='solution/solution.csv')
