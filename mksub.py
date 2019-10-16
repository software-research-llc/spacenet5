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

if __name__ == '__main__':
    model = train.build_model()
    train.load_weights(model)
    flow.RUNNING_TESTS = True
    seq = flow.Sequence(batch_size=1)
    for x,y,name in tqdm.tqdm(seq):
#        img = model.predict(x)
        graph = sngraph.SNGraph()
        graph.name = flow.Target.expand_imageid(name[0])
        graph.add_channel(y[0])
        linestrings = graph.toWKT()
        with open("solution/solution.csv", "a") as f:
            f.writelines(linestrings)
        import pdb; pdb.set_trace()
