import show
import keras
from keras.applications import xception
from scipy.ndimage.interpolation import zoom
import spacenetflow as flow
import unet
import numpy as np
import train
import cv2


def build_model():
    return unet.get_unet(dropout=0.25, input_img=keras.layers.Input(flow.IMSHAPE))
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

if __name__ == '__main__':
    try:
        m = keras.models.load_model(train.model_file)
    except:
        try:
            m = build_model()
        except Exception as exc:
            print("ERROR BUILDING MODEL: %s" % exc)
    m.summary()
    import spacenetflow as flow
    import matplotlib.pyplot as plt
    import os
    while True:
        fpath = flow.get_file()
        tb = flow.TargetBundle()
        inp_im = flow.resize(flow.get_image(fpath), flow.IMSHAPE).reshape([1,] + flow.IMSHAPE)
        try:
            out_im = m.predict(inp_im)[0]
            out_im = cv2.cvtColor(out_im, cv2.COLOR_GRAY2RGB)
        except Exception as exc:
            print("ERROR: %s" % exc)
            out_im = np.zeros(flow.IMSHAPE)
            out_im = np.array(m.predict(inp_im)).reshape(flow.DECODER_OUTPUT_SHAPE)
        try:
            fig = plt.figure()
            fig.add_subplot(2, 2, 1)
            plt.imshow(inp_im[0])
            plt.title("Input")
            fig.add_subplot(2, 2, 2)
            plt.imshow(out_im)
            plt.title("Pred")
            fig.add_subplot(2, 2, 3)
            edges = cv2.Canny(np.cast['uint8'](out_im * 255), 1, 255)
            plt.imshow(edges)
            plt.title("Post-proc'd pred")
            fig.add_subplot(2, 2, 4)
            t_im = tb[os.path.basename(fpath)].image()
            t_im = cv2.cvtColor(t_im, cv2.COLOR_GRAY2RGB)
            plt.imshow(t_im)
            plt.title("Target")
        except Exception as exc:
            print("ERROR: %s" % exc)

        plt.show()
