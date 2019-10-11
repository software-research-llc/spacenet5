import show
import keras
from keras.applications import xception
from scipy.ndimage.interpolation import zoom
import snflow as flow
import unet
import numpy as np
import train
import cv2
import time
import loss


def build_model():
    return unet.Generator()
#    return unet.get_unet(input_img=keras.layers.Input(flow.IMSHAPE))
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

def load_model(path="model.tf"):
#    losses = { flow.LOSS.__qualname__: flow.LOSS }
#    return keras.models.load_model(path, custom_objects=losses)
    model = build_model()
    model.load_weights(path)
    compile_model(model)
    return model

def save_model(model, path="model.tf"):
    model.save_weights(path)

def compile_model(model):
    model.compile(optimizer=flow.OPTIMIZER, loss=flow.LOSS, metrics=['accuracy', 'binary_accuracy', 'mae'])

if __name__ == '__main__':
    try:
        m = keras.models.load_model(train.model_file)
    except:
        try:
            m = build_model()
        except Exception as exc:
            print("ERROR BUILDING MODEL: %s" % exc)
    m.summary()
    import snflow as flow
    import matplotlib.pyplot as plt
    import os
    import sknw
    from skimage.morphology import skeletonize
    tb = flow.TargetBundle()
    while True:
        fpath = flow.get_file()#"/home/mmiron/src/spacenet5/data/train/AOI_7_Moscow/PS-RGB/SN5_roads_train_AOI_7_Moscow_PS-RGB_chip0.tif"
        inp_im = flow.resize(flow.get_image(fpath), flow.IMSHAPE).reshape([1,] + flow.IMSHAPE)
        try:
            out_im = m.predict(inp_im)[0]
            #out_im = cv2.cvtColor(out_im, cv2.COLOR_GRAY2RGB)
        except Exception as exc:
            print("ERROR in first try: %s" % exc)
            out_im = np.zeros(flow.IMSHAPE)
            out_im = np.array(m.predict(inp_im))
        try:
            fig = plt.figure()
            fig.add_subplot(2, 2, 1)
            plt.imshow(inp_im[0])
            plt.title("1. Satellite image (input)")

            fig.add_subplot(2, 2, 2)
            out_im = cv2.cvtColor(out_im, cv2.COLOR_GRAY2RGB)
            plt.imshow(out_im)#[:,:,0])
            plt.title("2. Road predictions (output)")

            fig.add_subplot(2, 2, 3)
            buf = cv2.cvtColor(out_im, cv2.COLOR_RGB2GRAY)
            buf = prep_for_skeletonize(buf)
            skel = skeletonize(buf)#[:,:,0])
            plt.imshow(skel)
            # build graph from skeleton
            graph = sknw.build_sknw(skel, True)
            print(graph.nodes(), graph.edges())
            try:
                # draw edges by pts
                for (s,e) in graph.edges():
                    ps = graph[s][e][0]['pts']
                    plt.plot(ps[:,1], ps[:,0], 'green')
                    print(ps[:,1], ps[:,0])

                # draw node by o
                node, nodes = graph.node, graph.nodes()
                ps = np.array([node[i]['o'] for i in nodes])
                plt.plot(ps[:,1], ps[:,0], 'r.')
            except Exception as exc:
                print("ERROR drawing graph: %s" % exc)
            plt.title("3. Graph built from predictions")

            fig.add_subplot(2, 2, 4)
            t_im = tb[os.path.basename(fpath)].image()
            #t_im = cv2.cvtColor(t_im, cv2.COLOR_GRAY2RGB)
            plt.imshow(t_im[:,:,0])
            plt.title("Ground truth (drawn by a person)")
        except Exception as exc:
            print("ERROR in second try: %s" % exc)
            raise exc
        time.sleep(1)
        plt.show()
