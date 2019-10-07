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
filenames = []
graphs = []
for city in flow.CITIES:
    filenames += flow.get_filenames(datadir=flow.BASEDIR + city)
for fn in tqdm.tqdm(filenames):
    image = flow.resize(flow.get_image(fn), flow.IMSHAPE).reshape(flow.IMSHAPE)
    _, graph, _ = infer.infer(model, image)
    graphs.append(graph)

#mask, graph, skel, filename = infer.do_all(loop=False)
#import pdb; pdb.set_trace()
create_submission.graphs_to_wkt(graphs,
                                filenames=filenames,
                                output_csv_path='solution/solution.csv',
                                verbose=False)
