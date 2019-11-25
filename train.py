from settings import *
import time
import sys
import flow
import plac
import tensorflow as tf
import segmentation_models as sm
import tensorflow.keras as keras
import numpy as np
import logging
import ordinal_loss
import keras
import layers

logger = logging.getLogger(__name__)

tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.hdf5', save_weights_only=True, save_best_only=True),
]

#metrics = ['sparse_categorical_accuracy', sm.losses.CategoricalFocalLoss(), sm.metrics.IOUScore(), sm.metrics.FScore()]
#preprocess_input = sm.get_preprocessing(BACKBONE)

#@tf.custom_gradient
def f1score(y_true, y_pred):
    if y_true.shape[0] == 2:
        first = f1score(y_true[0], y_pred[0])
        second = f1score(y_true[1], y_pred[1])
        return (first + second) / 2.0
    pred = tf.cast(y_pred, dtype=tf.float32)
    targ = tf.cast(tf.argmax(y_true, axis=3), dtype=tf.float32)
    f1s = 0
    for c in [0,1,2,3,4,5]:
        c = tf.constant(c, dtype=tf.float32)
        TP = tf.reduce_sum(tf.cast(tf.logical_and(pred == c, targ == c), dtype=tf.float32))
        FN = tf.reduce_sum(tf.cast(tf.logical_and(pred != c, targ == c), dtype=tf.float32))
        FP = tf.reduce_sum(tf.cast(tf.logical_and(pred == c, targ != c), dtype=tf.float32))
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        f1 = tf.constant(2, dtype=tf.float32) * (prec * rec) / (prec + rec)
        f1s += f1
#    def grad(dy):
#        return tf.gradient(dy)
    return tf.constant(1, dtype=tf.float32) - (f1s / 6)#, grad

def save_model(model, save_path="model-%s.hdf5" % BACKBONE, pause=0):
    if pause > 0:
        sys.stderr.write("Saving in")
        for i in list(range(1,6))[::-1]:
            sys.stderr.write(" %d...\n" % i)
            time.sleep(pause)
    sys.stderr.write("Saving...\n")
    return model.save_weights(save_path)


def load_weights(model, save_path="model-%s.hdf5" % BACKBONE):
    try:
        model.load_weights(save_path)
        logger.info("Model file %s loaded successfully." % save_path)
    except OSError as exc:
        sys.stderr.write("!! ERROR LOADING %s:" % save_path)
        sys.stderr.write(str(exc) + "\n")
    return model


def build_model():
    base = TOPCLASS(BACKBONE,# classes=N_CLASSES,
                   input_shape=INPUTSHAPE, activation='sigmoid',
                   encoder_weights='imagenet')
    return base
    inp = keras.layers.Input(INPUTSHAPE)
    layer = layers.ChannelCompression()
    x = layer(inp)
    #out = keras.layers.Lambda(lambda x: tf.argmax(x, axis=3))(mid)
    return keras.models.Model(inputs=inp, outputs=x)


def main(save_path="model-%s.hdf5" % BACKBONE,
         optimizer=tf.keras.optimizers.Adam(),
         loss=sm.losses.bce_jaccard_loss,
         metrics=[sm.metrics.iou_score, 'accuracy'],
         restore=True,
         verbose=1,
         epochs=100):
    """
    Train the model.
    """
    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    logger.info("Building model.")
    model = build_model()
    if restore:
        load_weights(model)

#    sm.utils.set_trainable(model, recompile=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info("Generating dataflows.")
    if os.path.exists(PICKLED_TRAINSET):
        train_seq = flow.Dataflow.from_pickle(PICKLED_TRAINSET)
    else:
        train_seq = flow.Dataflow(batch_size=BATCH_SIZE)#, transform=0.30)
    if os.path.exists(PICKLED_VALIDSET):
        val_seq = flow.Dataflow.from_pickle(PICKLED_VALIDSET)
    else:
        val_seq = flow.Dataflow(batch_size=BATCH_SIZE, validation_set=True)

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
        save_model(model, save_path)
        raise(exc)


if __name__ == "__main__":
    plac.call(main)
