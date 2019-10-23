import time
import sys
import snflow as flow
import plac
import tensorflow as tf
import segmentation_models as sm
import tensorflow.keras as keras
import numpy as np
import logging
logger = logging.getLogger(__name__)

callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.hdf5', save_weights_only=True, save_best_only=True, mode='min'),
]

BATCH_SIZE = 3
dice_loss = sm.losses.DiceLoss()#class_weights=np.array([1, 1, 1, 1]))
focal_loss = sm.losses.BinaryFocalLoss() if flow.N_CLASSES == 1 else sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + sm.losses.BinaryFocalLoss()#focal_loss#keras.losses.sparse_categorical_crossentropy + dice_loss
metrics = ['accuracy', sm.losses.CategoricalFocalLoss(), sm.metrics.IOUScore(), sm.metrics.FScore()]
optim = keras.optimizers.Adam()
#preprocess_input = sm.get_preprocessing(flow.BACKBONE)

def loss(y_true, y_pred):
    dice = dice_loss(y_true, y_pred)
    return dice# + keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

def save_model(model, save_path="model-%s.hdf5" % flow.BACKBONE, pause=0):
    if pause > 0:
        sys.stderr.write("Saving in")
        for i in list(range(1,6))[::-1]:
            sys.stderr.write(" %d...\n" % i)
            time.sleep(pause)
    sys.stderr.write("Saving...\n")
    return model.save_weights(save_path)

def load_weights(model, save_path="model-%s.hdf5" % flow.BACKBONE):
    try:
        model.load_weights(save_path)
        logger.warning("Model file %s loaded successfully." % save_path)
    except OSError as exc:
        sys.stderr.write("!! ERROR LOADING %s:" % save_path)
        sys.stderr.write(str(exc) + "\n")

def build_model():
    segm = sm.Unet(flow.BACKBONE, classes=flow.N_CLASSES,
                   input_shape=flow.IMSHAPE, activation='softmax',
                   encoder_weights='imagenet')
    
    x = keras.layers.Conv2D(4, (3,3), use_bias=True)(segm)
    x = keras.layers.Conv2D(2, (3,3), use_bias=True)(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Conv2D(2, (3,3), dilation_rate=2, use_bias=True)(x)
    x = keras.layers.Conv2D(2, (3,3), dilation_rate=2, use_bias=True)(x)
    x = keras.layers.MaxPooling2D((2,2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(flow.IMSHAPE[0], activation='softmax')(x)

    trans = keras.models.Model(inputs=segm.output, outputs=x)
    return keras.models.Model(inputs=segm.input, outputs=trans.output)

def main(save_path="model-%s.hdf5" % flow.BACKBONE,
         optimizer='adam',
         loss='sparse_categorical_crossentropy',#'sparse_categorical_crossentropy',#loss,#'sparse_categorical_crossentropy',
         restore=True,
         verbose=1,
         epochs=500,
         validation_split=0.1):
    model = build_model()
    if restore:
        load_weights(model)

    sm.utils.set_trainable(model, recompile=False)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    train_seq = flow.Sequence(batch_size=BATCH_SIZE, transform=0.30, test=False, shuffle=True)
    val_seq = flow.Sequence(batch_size=BATCH_SIZE, test=True, shuffle=True)

    train_step(model, train_seq, verbose, epochs, callbacks, save_path, val_seq)
    save_model(model, save_path)
    sys.exit()

    for epoch in range(epochs):
        vi = 0
        stepnum = 0
        for x,y in train_seq:
            stepnum += 1
            vx, vy = val_seq[vi]
            vi += 1
            if vi >= len(val_seq):
                vi = 0
            try:
                print("Epoch %d / %d, step %d / %d" % (epoch, epochs, stepnum, len(train_seq)))
                model.fit(x, y, validation_data=[vx, vy], epochs=1,
                            verbose=verbose, batch_size=BATCH_SIZE)
            except KeyboardInterrupt:
                save_model(model, save_path, pause=1)
                sys.exit()
            except Exception as exc:
                save_model(model, save_path)
                raise exc
    save_model(model, save_path)

def train_step(model, train_seq, verbose, epochs, callbacks, save_path, val_seq):
    try:
        model.fit(train_seq, validation_data=val_seq, epochs=epochs,
                            verbose=verbose, callbacks=callbacks)
                            #use_multiprocessing=True)
    except KeyboardInterrupt:
            save_model(model, save_path, pause=1)
            sys.exit()
    except Exception as exc:
        save_model(model, save_path)
        raise(exc)


def train_step_generator(model, train_seq, verbose, epochs, callbacks, save_path, val_seq=None):
    try:
        model.fit_generator(train_seq, validation_data=val_seq, epochs=epochs,
                            verbose=verbose, callbacks=callbacks)
                            #use_multiprocessing=True)
    except KeyboardInterrupt:
            save_model(model, save_path, pause=1)
            sys.exit()
    except Exception as exc:
        save_model(model, save_path)
        raise(exc)

if __name__ == "__main__":
    build_model()
    sys.exit()
    plac.call(main)
