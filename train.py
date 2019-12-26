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
import ordinal_loss
import unet

logger = logging.getLogger(__name__)

#tf.config.optimizer.set_jit(False)
#tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})


def save_model(model, save_path=S.MODELSTRING, pause=0):
    if pause > 0:
        sys.stderr.write("Saving to {} in".format(save_path))
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


def to_categorical(tensor):
    return tf.keras.utils.to_categorical(tensor)#, S.N_CLASSES)


def build_sm_model(*args, **kwargs):
    tf.keras.backend.set_image_data_format('channels_last')
    import segmentation_models as sm
    sm.set_framework('tf.keras')

    inp_pre = tf.keras.layers.Input(S.INPUTSHAPE)
    inp = tf.keras.layers.Input(S.INPUTSHAPE)
    m = sm.Unet('efficientnetb0', weights=None, input_shape=S.INPUTSHAPE, activation='softmax', *args, **kwargs)
    #m.set_trainable('true')

    x = m(inp)
#    x = tf.one_hot(x, S.N_CLASSES, axis=-1)
    #tf.keras.utils.to_categorical(x)#, num_classes=S.N_CLASSES)
    #x = tf.keras.layers.Reshape((-1,S.N_CLASSES))(x)
    #x = tf.keras.layers.Activation('softmax')(x)

    return tf.keras.models.Model(inputs=[inp_pre, inp], outputs=[x])


def build_model(classes=2, damage=True, *args, **kwargs):
    L = tf.keras.layers
    R = tf.keras.regularizers

    decoder = unet.SegmentationModel(classes=6) if damage else unet.MotokimuraUnet(classes=classes)
    
    inp = L.Input(S.INPUTSHAPE)
    x = decoder(inp)
    x = L.Reshape((-1,classes))(x)
    x = L.Activation('softmax')(x)

    m = tf.keras.models.Model(inputs=[inp], outputs=[x])
    return m


def build_deeplab_model(architecture=S.ARCHITECTURE, train=False):
    deeplab = deeplabmodel.Deeplabv3(input_shape=S.INPUTSHAPE,
                                  weights='pascal_voc',
                                  backbone=architecture,
                                  classes=S.N_CLASSES,
                                  OS=16 if train is True else 8)

    decoder = keras.models.Model(inputs=deeplab.inputs, outputs=[deeplab.get_layer('custom_logits_semantic').output])

    inp_pre = tf.keras.layers.Input(S.INPUTSHAPE)
    inp_post = tf.keras.layers.Input(S.INPUTSHAPE)

#    x = decoder(inp_pre)
#    y = decoder(inp_post)

    x = tf.keras.layers.Add()([inp_pre, inp_post])
    x = decoder(x)
    x = tf.compat.v1.image.resize(x, size=S.INPUTSHAPE[:2], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    x = tf.keras.layers.Reshape((-1,S.N_CLASSES))(x)
    x = tf.keras.layers.Activation('softmax')(x)

    return tf.keras.models.Model(inputs=[inp_pre, inp_post], outputs=[x])


def main(restore: ("Restore from checkpoint", "flag", "r"),
         damage: ("Train a damage classifier (default is localization)", "flag", "d"),
         verbose: ("Keras verbosity level", "option", "v", int)=1,
         epochs: ("Number of epochs", "option", "e", int)=50,
         initial_epoch: ("Initial epoch to continue from", "option", "i", int)=1,
         optimizer=tf.keras.optimizers.RMSprop(),
         loss='categorical_crossentropy'):
    """
    Train a model.
    """
#    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    save_path = S.DMG_MODELSTRING if damage else S.MODELSTRING

    metrics=['accuracy',
             score.num_correct,
             score.recall,
             score.damage_f1_score if damage else None,
             score.tensor_f1_score]

    callbacks = [
        keras.callbacks.ModelCheckpoint(save_path.replace(".hdf5", "-best.hdf5"), save_weights_only=True, save_best_only=True),

        keras.callbacks.TensorBoard(log_dir="logs"),
    ]


    S.INPUTSHAPE[-1] = 6 if damage else 6
    S.DAMAGE = True if damage else False
    classes = 6 if damage else S.N_CLASSES
    logger.info("Building model.")
    model = build_model(classes=classes, damage=damage)

    S.N_CLASSES = 6 if damage else S.N_CLASSES
    S.MASKSHAPE[-1] = 6 if damage else S.MASKSHAPE[-1]
    if restore:
        load_weights(model, save_path)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    flowcall = flow.DamagedDataflow if damage else flow.Dataflow
    train_seq = flowcall(files=flow.get_training_files(), batch_size=S.BATCH_SIZE,
                         transform=0.3,
                         shuffle=True,
                         buildings_only=True,
                         return_postmask=True if damage else False,
                         return_stacked=True if damage else True,
                         return_post_only=False if damage else False,
                         return_average=False)
    val_seq = flow.Dataflow(files=flow.get_validation_files(), batch_size=S.BATCH_SIZE,
                       buildings_only=True,
                       shuffle=True,
                       return_postmask=True if damage else False,
                       return_stacked=True if damage else True,
                       return_post_only=False if damage else False,
                       return_average=False)

    logger.info("Training %s" % save_path)
    train_stepper(model, train_seq, verbose, epochs, callbacks, save_path, val_seq, initial_epoch)
    save_model(model, save_path)


def train_stepper(model, train_seq, verbose, epochs, callbacks, save_path, val_seq, initial_epoch):
    try:
        model.fit(train_seq, validation_data=val_seq, epochs=epochs,
                            verbose=verbose, callbacks=callbacks,
                            validation_steps=len(val_seq), shuffle=False,
                            #steps_per_epoch=500,
                            #use_multiprocessing=True,
                            max_queue_size=10)#, workers=8)
    except KeyboardInterrupt:
            save_model(model, "tmp.hdf5", pause=0)
            save_model(model, save_path, pause=1)
            sys.exit()
    except Exception as exc:
        save_model(model, "tmp.hdf5", pause=0)
        raise(exc)


if __name__ == "__main__":
    plac.call(main)
