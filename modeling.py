import keras
import keras_applications
keras_applications.resnet_common.backend = keras.backend
from keras_applications import resnext
from keras.models import Model
from keras.layers import Flatten, Reshape, MaxPooling2D, Dropout, BatchNormalization, Conv2D, Activation, Conv2DTranspose, Concatenate, concatenate, Conv3D, Cropping2D, Permute, Activation
from settings import *

keras.backend.set_image_data_format("channels_last")


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),  kernel_initializer="he_normal",
#               kernel_regularizer=keras.regularizers.l1_l2(0.01),
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
#               kernel_regularizer=keras.regularizers.l1_l2(0.01),
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def resnet_unet(filters=128, dropout=0.5):
    backbone = keras.applications.ResNet50(input_shape=INPUTSHAPE, include_top=False, pooling=None)
    inp = backbone.input
    x = backbone.output

    gl = lambda x: backbone.get_layer(x).output
    skips = [gl("activation_10"), gl("activation_22"), gl("activation_40"), gl("activation_46")]

    x = concatenate([x, skips.pop()])
    x = Conv2DTranspose(filters * 8, kernel_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Dropout(dropout)(x)
    x = conv2d_block(x, filters * 8)

    x = concatenate([x, skips.pop()])
    x = Conv2DTranspose(filters * 4, kernel_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Dropout(dropout)(x)
    x = conv2d_block(x, filters * 4)

    x = concatenate([x, skips.pop()])
    x = Conv2DTranspose(filters * 2, kernel_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Dropout(dropout)(x)
    x = conv2d_block(x, filters * 2)
    
    x = concatenate([x, skips.pop()])
    x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Dropout(dropout)(x)
    x = conv2d_block(x, filters)

    x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Conv2D(2, (1, 1))(x)

    x = Reshape((2, INPUTSHAPE[0] * INPUTSHAPE[1]))(x)
    x = Permute((2, 1))(x)
    
    out = Activation("softmax")(x)
    return Model(inputs=[inp], outputs=[out])


def resnetv2_unet(filters=128, dropout=0.5):
    backbone = keras.applications.ResNet50V2(input_shape=INPUTSHAPE, include_top=False, pooling=None)
    inp = backbone.input
    x = backbone.output

    skips = []
    for i in range(2,6):
        skips.append(backbone.get_layer("conv%d_block1_out" % i).output)
#    skips = skips[::-1]

    x = concatenate([x, skips.pop()])
    x = Conv2DTranspose(filters * 8, kernel_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Dropout(dropout)(x)
    x = conv2d_block(x, filters * 8)

    x = concatenate([x, skips.pop()])
    x = Conv2DTranspose(filters * 4, kernel_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Dropout(dropout)(x)
    x = conv2d_block(x, filters * 4)

    x = concatenate([x, skips.pop()])
    x = Conv2DTranspose(filters * 2, kernel_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Dropout(dropout)(x)
    x = conv2d_block(x, filters * 2)
    
    x = concatenate([x, skips.pop()])
    x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Dropout(dropout)(x)
    x = conv2d_block(x, filters)

    x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2), padding='valid')(x)
    x = Conv2D(2, (1, 1))(x)

    x = Reshape((2, INPUTSHAPE[0] * INPUTSHAPE[1]))(x)
    x = Permute((2, 1))(x)
    
    out = Activation("softmax")(x)
    return Model(inputs=[inp], outputs=[out])

def build_model():
    return resnet_unet()

def get_unet(n_filters=16, dropout=0.5, batchnorm=True):
    inp = keras.layers.Input(INPUTSHAPE)
    c1 = conv2d_block(inp, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout / 2)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
   
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    #outputs = Conv2D(flow.N_CLASSES, (1, 1), activation=flow.LAST_LAYER_ACTIVATION) (c9)
    #x = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    x = Conv2D(2, (1, 1))(c9)

    x = Reshape((2, INPUTSHAPE[0] * INPUTSHAPE[1]))(x)
    x = Permute((2, 1))(x)
    
    out = Activation("softmax")(x)
    model = Model(inputs=[inp], outputs=[out])
    return model


if __name__ == '__main__':
    m = resnet_unet()
    m.summary()
