import os
import focal_loss
import settings as S
import time
import sys
import flow
import plac
import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import backend as K
#import keras
#import keras.backend as K
#import segmentation_models as sm
import tensorflow.keras as keras
import numpy as np
import logging
import ordinal_loss
#import tensorflow.keras.backend as K
import layers
import deeplabmodel
import infer
import score

logger = logging.getLogger(__name__)

#tf.config.optimizer.set_jit(False)
#tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

callbacks = [
    keras.callbacks.ModelCheckpoint('deeplab-xception-best.hdf5', save_weights_only=True, save_best_only=True)]
""""
    keras.callbacks.TensorBoard(log_dir="logs",
                                histogram_freq=1,
                                write_graph=True,
                                write_images=True,
                                embeddings_freq=0,
                                update_freq=100),
]
"""

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


def save_model(model, save_path=S.MODELSTRING, pause=0):
    if pause > 0:
        sys.stderr.write("Saving in")
        for i in list(range(1,6))[::-1]:
            sys.stderr.write(" %d...\n" % i)
            time.sleep(pause)
    sys.stderr.write("Saving...\n")
    return model.save_weights(save_path)


def load_weights(model, save_path=S.MODELSTRING):
    try:
        model.load_weights(save_path)
        logger.info("Model file %s loaded successfully." % save_path)
    except OSError as exc:
        sys.stderr.write("!! ERROR LOADING %s:" % save_path)
        sys.stderr.write(str(exc) + "\n")
    return model


def build_model(architecture=S.ARCHITECTURE, train=False):
    inp = tf.keras.layers.Input(S.INPUTSHAPE)
#    x = tf.image.resize(inp, size=(512,512), method=tf.image.ResizeMethod.BILINEAR)
    x = tf.keras.layers.GaussianNoise(0.0003)(inp)
    xception = deeplabmodel.Deeplabv3(input_tensor=x,#input_shape=INPUTSHAPE,
                                  weights='cityscapes',
                                  backbone=architecture,
                                  classes=S.N_CLASSES,
                                  OS=16 if train is True else 8,
#                                  activation='softmax',
#                                  alpha=1.0,
#                                  OS=8)
                                 )

    x = xception.get_layer('custom_logits_semantic').output
    x = tf.image.resize(x, size=S.INPUTSHAPE[:2],
                        #preserve_aspect_ratio=True,
                        method=tf.image.ResizeMethod.BILINEAR,
                        #name="resize_xception_logits",
                        #align_corners=True
                        )

    #x = xception.output
    x = tf.keras.layers.Reshape((-1,S.N_CLASSES))(x)
    x = tf.keras.layers.Activation('softmax')(x)
    return keras.models.Model(inputs=[inp], outputs=[x])


def main(restore: ("Restore from checkpoint", "flag", "r"),
         architecture: ("xception or mobilenetv2", "option", "a")=S.ARCHITECTURE,
         save_path: ("Save path", "option", "s")=S.MODELSTRING,
         optimizer=tf.keras.optimizers.Adam(),#(lr=0.0001, epsilon=0.9),
         loss='categorical_crossentropy',
         metrics=['binary_accuracy', 'categorical_accuracy', 'mae',
                  'binary_crossentropy', 'categorical_crossentropy'],
         verbose=1,
         epochs=100):
    """
    Train the model.
    """
#    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    S.MODELSTRING = "deeplab-%s.hdf5" % architecture
    logger.info("Building model.")
    model = build_model(architecture=architecture, train=True)
    if restore:
        load_weights(model, S.MODELSTRING)

#    sm.utils.set_trainable(model, recompile=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info("Generating dataflows.")
    if os.path.exists("trainingsamples.pickle"):
        train_seq = flow.Dataflow("trainingsamples.pickle")
    else:
        train_seq = flow.Dataflow(files=flow.get_training_files(), batch_size=S.BATCH_SIZE, transform=0.3, shuffle=True)
    if os.path.exists("validationsamples.pickle"):
        val_seq = flow.Dataflow("validationsamples.pickle")
    else:
        val_seq = flow.Dataflow(files=flow.get_validation_files(), batch_size=S.BATCH_SIZE)

    logger.info("Training.")
    train_step(model, train_seq, verbose, epochs, callbacks, save_path, val_seq)
    save_model(model, save_path)


def train_step(model, train_seq, verbose, epochs, callbacks, save_path, val_seq):
    try:
        model.fit(train_seq, validation_data=val_seq, epochs=epochs,
                            verbose=verbose, callbacks=callbacks,
                            validation_steps=100, shuffle=False,
                            use_multiprocessing=True,
                            max_queue_size=10, steps_per_epoch=16502 * 16)
    except KeyboardInterrupt:
            save_model(model, "tmp.hdf5", pause=0)
            save_model(model, save_path, pause=1)
            sys.exit()
    except Exception as exc:
        save_model(model, "tmp.hdf5", pause=0)
        raise(exc)


if __name__ == "__main__":
    plac.call(main)
