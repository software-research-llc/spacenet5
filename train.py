import os
import settings as S
import time
import sys
import flow
import plac
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import logging
import deeplabmodel
import infer
import score
import unet

logger = logging.getLogger(__name__)

#tf.config.optimizer.set_jit(False)
#tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})


def save_model(model, save_path=S.MODELSTRING, pause=0):
    """
    Save a model, optionally pausing to give the user a chance to cancel the save
    (canceling training with a CTRL-C [KeyboardInterrupt] saves after 5 seconds).
    """
    if pause > 0:
        sys.stderr.write("Saving to {} in".format(save_path))
        for i in list(range(1,6))[::-1]:
            sys.stderr.write(" %d...\n" % i)
            time.sleep(pause)
    sys.stderr.write("Saving...\n")
    return model.save_weights(save_path)


def load_weights(model, save_path=S.MODELSTRING):
    try:
        model.load_weights(save_path)#, by_name=True)
        logger.info("Model file %s loaded successfully." % save_path)
    except OSError as exc:
        logger.error("unable to load %s: %s" % (save_path, str(exc)))
    return model


def build_model(classes=6, damage=False, *args, **kwargs):
    L = tf.keras.layers
    R = tf.keras.regularizers

    #decoder = unet.MotokimuraMobilenet(classes=classes) if damage else unet.MotokimuraUnet(classes=classes)
    #decoder = unet.Ensemble(classes=classes)
    decoder = unet.MotokimuraUnet(classes=classes)

    inp = L.Input(S.INPUTSHAPE)
    x = decoder(inp)

    # Take the model's output and reshape so we can use categorical_crossentropy loss. To undo this
    # and recover the predicted mask, see infer.convert_prediction().
    x = L.Reshape((-1,classes), name="logits")(x)
    x = L.Activation('softmax')(x)

    m = tf.keras.models.Model(inputs=[inp], outputs=[x])
    return m


def build_deeplab_model(classes=6, damage=False, train=False):
    deeplab = deeplabmodel.Deeplabv3(input_shape=S.INPUTSHAPE,
                                  weights=None,
                                  backbone='xception',
                                  classes=classes,
                                  OS=16 if train is True else 8)

    #decoder = keras.models.Model(inputs=deeplab.inputs, outputs=[deeplab.get_layer('custom_logits_semantic').output])

    #x = decoder(deeplab.input)
    x = deeplab.get_layer("custom_logits_semantic").output
    x = tf.compat.v1.image.resize(x, size=S.MASKSHAPE[:2], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    x = tf.keras.layers.Reshape((-1,classes))(x)
    x = tf.keras.layers.Activation('softmax')(x)

    return tf.keras.models.Model(inputs=deeplab.inputs, outputs=[x])


def main(restore: ("Restore from checkpoint", "flag", "r"),
         damage: ("Train a damage classifier (default is localization)", "flag", "d"),
         deeplab: ("Build and train a DeeplabV3+ model", "flag", "D"),
         motokimura: ("Build and train a Motokimura-designed Unet", "flag", "M"),
         verbose: ("Keras verbosity level", "option", "v", int)=1,
         epochs: ("Number of epochs", "option", "e", int)=50,
         initial_epoch: ("Initial epoch to continue from", "option", "i", int)=1,
         optimizer: ("Keras optimizer to use", "option", "o", str)='RMSprop',
         loss='categorical_crossentropy'):
    """
    Train a model.
    """
#    optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    if deeplab:
        logger.info("Building DeeplabV3+ model.")
        model = build_deeplab_model(classes=S.N_CLASSES, damage=damage, train=True)
        S.ARCHITECTURE = "deeplab-xception"
    elif motokimura:
        logger.info("Building MotokimuraUnet model.")
        model = build_model(classes=S.N_CLASSES, damage=damage, train=True)
        S.ARCHITECTURE = "motokimura"
    else:
        logger.error("Use -M (motokimura) or -D (deeplab) parameter.")
        sys.exit(-1)

    S.DAMAGE = True if damage else False
    save_path = S.MODELSTRING = f"{S.ARCHITECTURE}.hdf5"
    S.DMG_MODELSTRING = f"damage-{save_path}"
    if restore:
        load_weights(model, save_path)


    metrics=['accuracy',
             score.num_correct,
             score.recall,
             score.tensor_f1_score]

    callbacks = [
        keras.callbacks.ModelCheckpoint(save_path.replace(".hdf5", "-best.hdf5"), save_weights_only=True, save_best_only=True),
        #keras.callbacks.TensorBoard(log_dir="logs"),
    ]

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

    logger.info("Training and saving best weights after each epoch (CTRL+C to interrupt).")
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
            save_model(model, save_path, pause=1)
            sys.exit()
    except Exception as exc:
        save_model(model, "tmp.hdf5", pause=0)
        logger.error("Saved current weights to tmp.hdf5 file.")
        raise(exc)


if __name__ == "__main__":
    plac.call(main)
