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
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
]

dice_loss = sm.losses.DiceLoss()#class_weights=np.array([1, 1, 1, 1]))
focal_loss = sm.losses.BinaryFocalLoss() if flow.N_CLASSES == 1 else sm.losses.CategoricalFocalLoss()
total_loss = 0.25 * dice_loss + (1.0 * focal_loss)
metrics = ['accuracy', 'sparse_categorical_crossentropy', sm.metrics.IOUScore(), sm.metrics.FScore()]
optim = keras.optimizers.Adam()
preprocess_input = sm.get_preprocessing(flow.BACKBONE)

def save_model(model, save_path="model.hdf5", pause=0):
    if pause > 0:
        sys.stderr.write("Saving in")
        for i in list(range(1,6))[::-1]:
            sys.stderr.write(" %d...\n" % i)
            time.sleep(pause)
    sys.stderr.write("Saving...\n")
    return model.save_weights(save_path)

def load_weights(model, save_path="model.hdf5"):
    try:
        model.load_weights(save_path)
        logger.warning("Model loaded successfully.")
    except OSError as exc:
        sys.stderr.write("!! ERROR LOADING WEIGHTS:")
        sys.stderr.write(str(exc) + "\n")

def build_model():
    return sm.Unet(flow.BACKBONE, classes=flow.N_CLASSES, activation='softmax', encoder_weights='imagenet')

def main(save_path="model.hdf5",
         optimizer='adam',
         loss='sparse_categorical_crossentropy',
         restore=True,
         verbose=1,
         epochs=50,
         validation_split=0.1):
    model = build_model()
    if restore:
        load_weights(model)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    train_seq = flow.Sequence(batch_size=5, transform=0.00, test=False)
    val_seq = flow.Sequence(batch_size=1, test=True)

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
                            verbose=verbose, batch_size=3)
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
    plac.call(main)
