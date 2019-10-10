import snflow as flow
import keras
import keras.backend as K
import numpy as np
import snmodel
from keras.applications import xception
import time
import tensorflow as tf
import loss

#tf.compat.v1.disable_eager_execution()

EPOCHS = 25
model = None
model_file = "model.tf"
i = 0

def train(model, seq, epochs=EPOCHS):
    iters = 0
    for x, y in seq:
        start = time.time()
        history = model.fit(x, y, batch_size=seq.batch_size, epochs=epochs, validation_split=0.1, verbose=1, use_multiprocessing=True)
        stop = time.time()
        steptime = stop - start
        totaltime = len(seq) * steptime / 60
        iters += 1
        print("{:15s} {:^20s} {:^20s} {:^15s} {:^20s}".format("Full epochs", "Secs per iteration", "Mins per full epoch", "Batch size", "Samples remaining"))
        print("{:15s} {:^20.2f} {:^20.2f} {:^15s} {:^20s}".format(str(i), steptime, totaltime, str(len(x)), str(len(seq) * len(x) - iters * len(x))))
    print(history.history['loss'])

def custom_accuracy(y_true, y_pred):
    """Return the percentage of pixels that were correctly predicted as belonging to a road"""
    tp = tf.math.count_nonzero(y_true * y_pred)
    total_pos = tf.math.count_nonzero(y_true)
    return tf.dtypes.cast(tp, dtype=tf.float64) / tf.maximum(tf.constant(1, dtype=tf.float64), tf.dtypes.cast(total_pos, dtype=tf.float64))

def custom_loss(y_true, y_pred):
    tp = tf.math.count_nonzero(y_true * y_pred)
    total_pos = tf.math.count_nonzero(y_true)
    loss = tf.constant(1, dtype=tf.float64) - tf.dtypes.cast(tp, dtype=tf.float64) / tf.maximum(tf.constant(1, dtype=tf.float64), tf.dtypes.cast(total_pos, dtype=tf.float64))
    return loss

def load(path=model_file):
    global model
    model = keras.models.load_model(model_file)
    return model

def main():
    global model
    global i
    if model is None:
        model = snmodel.build_model()
        print("WARNING: starting from a new model")
        time.sleep(5)
    else:
        print("Model was loaded successfully.")
        time.sleep(5)
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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
    except KeyboardInterrupt:
        print("Finished %d full epochs." % i)
        print("\nSaving to file in 5...")
        time.sleep(5)
        print("\nSaving...")
        model.save(model_file)
