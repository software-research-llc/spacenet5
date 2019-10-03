import spacenetflow as flow
import keras
import keras.backend as K
import numpy as np
import snmodel
from keras.applications import xception


def train(model, seq, epochs=100):
    for x, y in seq:
        history = model.fit(x, y, batch_size=seq.batch_size, epochs=epochs, verbose=1)
    print(history)

def custom_accuracy(y_true, y_pred):
    return K.sum(K.round(y_true * y_pred)) / K.sum(y_true)
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)

def main():
    model = snmodel.build_model()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', custom_accuracy])
    model.summary()
    seq = flow.SpacenetSequence.all()
    train(model, seq)

if __name__ == '__main__':
    main()
