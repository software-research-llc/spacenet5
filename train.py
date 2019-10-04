import spacenetflow as flow
import keras
import keras.backend as K
import numpy as np
import snmodel
from keras.applications import xception
import time

model = None
model_file = "model.tf"

def train(model, seq, epochs=32):
    for x, y in seq:
        history = model.fit(x, y, batch_size=seq.batch_size, epochs=epochs, verbose=1)
    print(history)

def custom_accuracy(y_true, y_pred):
    return K.sum(K.round(y_true * y_pred)) / K.sum(y_true)
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., neg - pos + 1.)

def load(path=model_file):
    global model
    model = keras.models.load_model(model_file)
    return model

def main():
    global model
    if model is None:
        model = snmodel.build_model()
        print("WARNING: starting from a new model")
        time.sleep(5)
    else:
        print("Model was loaded successfully.")
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', custom_accuracy])
    seq = flow.SpacenetSequence.all()
    i = 0
    while True:
        train(model, seq)
        i += 1
        print("Loops through training data: %d" % i)

if __name__ == '__main__':
    try:
        load()
    except Exception as exc:
        print(exc)
    try:
        main()
    except Exception as exc:
        print("Saving and re-raising %s..." % str(exc))
        model.save(model_file)
        raise exc
    except KeyboardInterrupt:
        print("\nSaving to file in 5...")
        time.sleep(5)
        print("\nSaving...")
        model.save(model_file)
