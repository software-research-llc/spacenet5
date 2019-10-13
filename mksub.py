import os
import sys
import snflow as flow
import pandas as pd
import numpy as np
import plac
import snmodel
import keras
import tqdm
import create_submission
import infer

#flow.CITIES += ["AOI_9_San_Juan"]
#flow.CITIES = sorted(flow.CITIES)[1:]
model = snmodel.load_model(flow.model_file)
graphs = []
masks = []
filenames = []
for city in flow.CITIES:
    for fn in os.listdir(os.path.join(flow.BASEDIR, city, flow.DATASET)):
        filenames.append(os.path.join(flow.BASEDIR, city, flow.DATASET, fn))
for fn in tqdm.tqdm(filenames):
    try:
#        image = flow.resize(flow.get_image(fn), flow.IMSHAPE)#.reshape(flow.IMSHAPE)
        image = flow.get_image(fn)
        mask, graph, preproc, skel = infer.infer(model, image, chipname=fn)
        graphs.append(graph)
        masks.append(mask)
    except Exception as exc:
        print(type(exc), exc)
    except KeyboardInterrupt:
        break
create_submission.graphs_to_wkt(masks, graphs,
                                output_csv_path='solution/solution.csv')
