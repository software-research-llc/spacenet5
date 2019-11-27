from settings import *
import time
import sys
import flow
import plac
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import segmentation_models as sm
import numpy as np
import logging
import ordinal_loss
#import tensorflow.keras.backend as K
import layers
import modeling

logger = logging.getLogger(__name__)

#tf.config.optimizer.set_jit(False)
#tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.hdf5', save_weights_only=True, save_best_only=True),
#    keras.callbacks.tensorboard_v2.TensorBoard(log_dir="logs", histogram_freq=0, batch_size=1, write_grads=False,
#                                            update_freq='epoch'),
]

#metrics = ['sparse_categorical_accuracy', sm.losses.CategoricalFocalLoss(), sm.metrics.IOUScore(), sm.metrics.FScore()]
#preprocess_input = sm.get_preprocessing(BACKBONE)


def f1score(y_true, y_pred):
    TP = tf.reduce_sum(y_true * y_pred)
    FN = tf.reduce_sum(y_true - y_pred)
    FP = tf.reduce_sum(y_pred - y_true)

#    FN = K.sum(tf.logical_and(pred != targ, targ == one))
#    FP = K.sum(tf.logical_and(pred == one, targ != one))

    prec = TP / (TP + FP + K.epsilon())
    rec = TP / (TP + FN + K.epsilon())
    f1 = 2 * (prec * rec) / (prec + rec + K.epsilon())
    return f1

def f1_loss(y_true, y_pred):
    return 1 - f1score(y_true, y_pred)

def save_model(model, save_path="model-%s.hdf5" % BACKBONE, pause=0):
    if pause > 0:
        sys.stderr.write("Saving in")
        for i in list(range(1,6))[::-1]:
            sys.stderr.write(" %d...\n" % i)
            time.sleep(pause)
    sys.stderr.write("Saving...\n")
    return model.save_weights(save_path)


def load_weights(model, save_path=MODELSTRING):
    try:
        model.load_weights(save_path)
        logger.info("Model file %s loaded successfully." % save_path)
    except OSError as exc:
        sys.stderr.write("!! ERROR LOADING %s:" % save_path)
        sys.stderr.write(str(exc) + "\n")
    return model


def build_model():
    return modeling.get_unet()


def main(save_path=MODELSTRING,
         optimizer=tf.keras.optimizers.Adam(lr=0.001),
         loss='categorical_crossentropy',
         metrics=['binary_accuracy', 'categorical_accuracy', 'mae'],
         restore=True,
         verbose=1,
         epochs=100):
    """
    Train the model.
    """
#    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    logger.info("Building model.")
    model = build_model()
    if restore:
        load_weights(model)

#    sm.utils.set_trainable(model, recompile=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info("Generating dataflows.")
    if os.path.exists("trainingsamples.pickle2"):
        train_seq = flow.Dataflow("trainingsamples.pickle")
    else:
        train_seq = flow.Dataflow(files=flow.get_training_files(), batch_size=BATCH_SIZE)#, transform=0.25)
    if os.path.exists("validationsamples.pickle2"):
        val_seq = flow.Dataflow("validationsamples.pickle")
    else:
        val_seq = flow.Dataflow(files=flow.get_validation_files(), batch_size=BATCH_SIZE)

    logger.info("Training.")
    train_step(model, train_seq, verbose, epochs, callbacks, save_path, val_seq)
    save_model(model, save_path)


def train_step(model, train_seq, verbose, epochs, callbacks, save_path, val_seq):
    try:
        model.fit(train_seq, validation_data=val_seq, epochs=epochs,
                            verbose=verbose, callbacks=callbacks)
    except KeyboardInterrupt:
            save_model(model, save_path, pause=1)
            sys.exit()
    except Exception as exc:
        raise(exc)


if __name__ == "__main__":
    plac.call(main)
