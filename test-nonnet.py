import os
import sys
import snflow as flow
import pandas as pd
import numpy as np
import plac
import unet
import keras
import tqdm
import create_submission
import infer

#flow.CITIES += ["AOI_9_San_Juan"]
#flow.CITIES = sorted(flow.CITIES)[1:]
model = unet.load_model(flow.model_file)
graphs = []
masks = []
seq = flow.SpacenetSequence.all(transform=False, shuffle=False, batch_size=1)
flow.RUNNING_TESTS = True
for x, image, fn in tqdm.tqdm(seq):
    try:
        graph, _, skel = infer.infer_roads(image[0].astype(np.float32), chipname=fn[0])
        graphs.append(graph)
        masks.append(image[0])
        if len(graphs) > 100:
            break
    except KeyboardInterrupt:
        break

print("Creating submission")
create_submission.graphs_to_wkt(masks, graphs,
                                output_csv_path='solution/solution.csv')
