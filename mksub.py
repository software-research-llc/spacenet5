import os
import sys
import snflow as flow
import pandas as pd
import numpy as np
import plac
import snmodel
import keras
import tqdm

def main(path: "Path to the test set image files",
         randomize: ("Choose images at random", "flag", "r"),
         verbose: ("Be more verbose", "flag", "v")):
    pass

import create_submission
import infer

model = keras.models.load_model("model.tf")
graphs = []
filenames = []
for city in flow.CITIES:
    for fn in os.listdir(os.path.join(flow.BASEDIR, city, flow.DATASET)):
        filenames.append(os.path.join(flow.BASEDIR, city, flow.DATASET, fn))
for fn in tqdm.tqdm(filenames[::-1]):
    try:
        image = flow.resize(flow.get_image(fn), flow.IMSHAPE).reshape(flow.IMSHAPE)
        _, graph, _ = infer.infer(model, image)
        graphs.append(graph)
    except Exception as exc:
        print(type(exc), exc)

#mask, graph, skel, filename = infer.do_all(loop=False)
#import pdb; pdb.set_trace()
create_submission.graphs_to_wkt(graphs,
                                filenames=filenames,
                                output_csv_path='solution/solution.csv',
                                verbose=False)
