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
    keras.callbacks.ModelCheckpoint(S.MODELSTRING.replace(".hdf5", "-best.hdf5"), save_weights_only=True, save_best_only=True)]
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


def _build_head(OS, classes, backbone, b4, skip1, x):
    from tensorflow.keras.layers import BatchNormalization, Activation, Lambda, Concatenate, Conv2D, Dropout
    from deeplabmodel import SepConv_BN

    b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # upsample. have to use compat because of the option align_corners
    #size_before = tf.keras.backend.int_shape(x)
    b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, (16,16),#size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = Concatenate(name="post_aspp")([b4, b0, b1, b2, b3])
    else:
        x = Concatenate(name="post_aspp")([b4, b0])

    x = Conv2D(256, (1, 1), padding='same',
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        size_before2 = tf.keras.backend.int_shape(x)
        x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                        skip1.shape[1:3],
                                                        method='bilinear', align_corners=True))(x)

        dec_skip1 = Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation('relu')(dec_skip1)
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

    last_layer_name = 'custom_logits_semantic'
    x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    return x

def build_model(architecture=S.ARCHITECTURE, train=False):
    OS=16 if train is True else 8
    inp_pre = tf.keras.layers.Input(S.INPUTSHAPE)
    inp_post = tf.keras.layers.Input(S.INPUTSHAPE)

    deeplab = deeplabmodel.Deeplabv3(input_tensor=inp_pre,#input_shape=S.INPUTSHAPE,
                                  weights='pascal_voc',
                                  backbone=architecture,
                                  classes=S.N_CLASSES,
                                  OS=OS)

    pool_features = tf.keras.models.Model(inputs=deeplab.inputs, outputs=[deeplab.get_layer('image_pooling').output])
    x = pool_features(inp_pre)
    y = pool_features(inp_post)

    x = tf.keras.layers.Add()([x,y])
    for layer in deeplab.layers:
        print(layer.name)

    skip = pool_features.get_layer("entry_flow_block2_separable_conv2_pointwise_BN").output
    x2 = pool_features.get_layer("activation_67").output#""exit_flow_block2_separable_conv3_pointwise").output
    #x2 = tf.keras.layers.BatchNormalization(epsilon=1e-3)(x2)
    #x2 = tf.keras.layers.Activation("relu")(x2)
    x = _build_head(OS=OS, x=x2, skip1=skip, b4=x, classes=S.N_CLASSES, backbone=architecture)

    x = tf.compat.v1.image.resize(x, size=S.INPUTSHAPE[:2], method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
    x = tf.keras.layers.Reshape((-1,S.N_CLASSES))(x)
    x = tf.keras.layers.Activation('softmax')(x)

    model = tf.keras.models.Model(inputs=[inp_pre, inp_post], outputs=[x])
    model.summary()
    return model

def main(restore: ("Restore from checkpoint", "flag", "r"),
         architecture: ("xception or mobilenetv2", "option", "a")=S.ARCHITECTURE,
         save_path: ("Save path", "option", "s")=S.MODELSTRING,
         optimizer=tf.keras.optimizers.RMSprop(),#tf.keras.optimizers.Adam(lr=0.0001),
         loss='categorical_crossentropy',
         metrics=['categorical_accuracy', 'mae',
                  score.iou_score, score.num_correct, score.pct_correct,
                  score.tensor_f1_score],
         verbose=1,
         epochs=15):
    """
    Train the model.
    """
    #optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
    S.MODELSTRING = "deeplab-%s.hdf5" % architecture
    logger.info("Building model.")
    model = build_model(architecture=architecture, train=True)
    if restore:
        load_weights(model, S.MODELSTRING)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    train_seq = flow.Dataflow(files=flow.get_training_files(), batch_size=S.BATCH_SIZE,
                              transform=0.3,
                              shuffle=True,
                              buildings_only=True)
    val_seq = flow.Dataflow(files=flow.get_validation_files(), batch_size=S.BATCH_SIZE)

    logger.info("Training.")
    train_stepper(model, train_seq, verbose, epochs, callbacks, save_path, val_seq)
    save_model(model, save_path)


def train_stepper(model, train_seq, verbose, epochs, callbacks, save_path, val_seq):
    try:
        model.fit(train_seq, validation_data=val_seq, epochs=epochs,
                            verbose=verbose, callbacks=callbacks,
                            validation_steps=len(val_seq), shuffle=False,
                            use_multiprocessing=False,
                            max_queue_size=10)
    except KeyboardInterrupt:
            save_model(model, "tmp.hdf5", pause=0)
            save_model(model, save_path, pause=1)
            sys.exit()
    except Exception as exc:
        save_model(model, "tmp.hdf5", pause=0)
        raise(exc)


if __name__ == "__main__":
    plac.call(main)
