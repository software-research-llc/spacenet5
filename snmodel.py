import show
import keras
from keras.applications import xception
from scipy.ndimage.interpolation import zoom
import snflow as flow
import unet
import numpy as np
import cv2
import time


def build_model(train=True):
    return unet.get_unet(input_img=keras.layers.Input(flow.IMSHAPE))
    p2p = unet.build_gan()
    return p2p
    if train:
        return p2p
    else:
        return p2p.generator
    return unet.build_google_unet()
    return xception_model()

def simple_model():
    # simple MLP
    inp = keras.layers.Input(flow.IMSHAPE)
#    x = keras.layers.BatchNormalization()(inp)
    x = keras.layers.Flatten()(inp)
#    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(25 * 25 * 3, activation='linear')(x)
    return keras.models.Model(inputs=inp, outputs=x)

def conv_model():
    # Simple convolutional model
    inp = keras.layers.Input(flow.IMSHAPE)
    x = keras.layers.Conv2D(32, (2, 2), activation='relu', padding='same')(inp)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(32 * 2, (2, 2), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    x = keras.layers.Conv2D(32 * 3, (2, 2), activation='relu', padding='same')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10, activation='relu')(x)
    x = keras.layers.Dense(5, activation='relu')(x)
    x = keras.layers.Dense(75 * 75 * 3, activation='linear')(x)
    return keras.models.Model(inputs=inp, outputs=x)

def xception_model():
    # xception model
    xm = xception.Xception(include_top=False, input_shape=(299,299,3))
    x = xm.get_layer("block14_sepconv2_act").output
    # Add a decoder to the Xception network
    for i in range(3):
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2DTranspose(3, strides=(3,3), kernel_size=(3,3), padding='valid')(x)
        x = keras.layers.BatchNormalization()(x)
    return keras.models.Model(inputs=xm.input, outputs=x)

    x = keras.layers.Dense(3, activation='relu')(x)
    x = keras.layers.Dense(299, activation='relu')(x)
    x = keras.layers.Dense(299 * 299 * 3, activation='linear', name='predictions')(x)
    return keras.models.Model(inputs=xm.input, outputs=x)

def preprocess(image):
    return xception.preprocess_input(image)

def prep_for_skeletonize(img):
    img = np.array(np.round(img), dtype=np.float32)
    return img

def load_model(path="model.tf-2", train=True):
#    losses = { flow.LOSS.__qualname__: flow.LOSS }
#    return keras.models.load_model(path, custom_objects=losses)
    model = build_model(train)
    try:
        model.load_weights(path)
    except:
        stat = model.checkpoint.restore(path)
        stat.expect_partial()
    return model

def save_model(model, path=flow.model_file):
    try:
        model.save_weights(path)
    except:
        model.checkpoint.save(path)

def compile_model(model):
    print("Compiling model...")
    model.compile(optimizer=flow.OPTIMIZER, loss=flow.LOSS, metrics=['binary_accuracy', 'binary_crossentropy', flow.loss.ssim_metric])
