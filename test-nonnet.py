import os
import sys
import snflow as flow
import pandas as pd
import numpy as np
import plac
import keras
import tqdm
import create_submission
import infer

#flow.CITIES += ["AOI_9_San_Juan"]
#flow.CITIES = sorted(flow.CITIES)[1:]
#model = unet.load_model(flow.model_file)

def create_mask(img):
#    channels = [np.zeros((flow.IMSHAPE[0], flow.IMSHAPE[1])) for x in range(3)]
    channels = []
    for i in range(4):
        chan = np.zeros_like(img)
        chan[img==i+1] = 1.0
        channels.append(chan)
    return np.stack(channels, axis=2)

graphs = []
masks = []
seq = flow.SpacenetSequence.all(transform=False, shuffle=True, batch_size=1)
flow.RUNNING_TESTS = True
for x, image, fn in tqdm.tqdm(seq):
    try:
        mask = create_mask(image[0])
        graph, _, skel = infer.infer_roads(flow.normalize(image[0]), chipname=fn[0])
        graphs.append(graph)
        masks.append(mask)
        if len(graphs) > 1000:
            break
    except KeyboardInterrupt:
        break

print("Creating submission")
create_submission.graphs_to_wkt(masks, graphs,
                                output_csv_path='solution/solution.csv')
