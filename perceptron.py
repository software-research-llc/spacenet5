import segmentation_models as sm
import sys
import time
import numpy as np
import logging
import snflow as flow
import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout

logger = logging.getLogger(__name__)

def build_model():
    inp = Input(shape=(flow.IMSHAPE[0], flow.IMSHAPE[1]))
    x = Flatten()(inp)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='relu')(x)
    x = Dense(512 * 512, activation='sigmoid')(x)

    return keras.models.Model(inputs=[inp], outputs=[x])

def main():
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    seq = flow.Sequence(test=False, shuffle=True)
    for x,y in seq:
        inp = y.image()
        out = y.points()

