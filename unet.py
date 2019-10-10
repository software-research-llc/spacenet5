from __future__ import absolute_import, division, print_function, unicode_literals
from keras.layers import MaxPooling2D, Dropout, BatchNormalization, Conv2D, Activation, Conv2DTranspose, Concatenate, concatenate, Conv3D, Cropping2D
from keras.models import Model
import keras
import tensorflow as tf
import snflow as flow
from tensorflow_examples.models.pix2pix import pix2pix

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt


DOWN = -1
UP = 1
OUT = 0

def conv3x3_relu_block(inp, n_filters, kernel_size=3, direction=DOWN):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), activation='relu')(inp)
    y = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), activation='relu')(x)

    if direction is DOWN:
        x = MaxPooling2D((1, 1), strides=(2, 2))(y)
    elif direction is UP:
        x = Conv2DTranspose(n_filters, kernel_size=(kernel_size, kernel_size),
                            strides=(2, 2), activation='relu')(y)
#        x = Conv2D(n_filters, kernel_size=(kernel_size, kernel_size), activation='relu')(x)
    else:
        x = Conv2D(n_filters, kernel_size=(1,1))(y)

    return x, y

def build(inp):
    base_filters = 64
    n_classes = 4
    x1,y1 = conv3x3_relu_block(inp, n_filters=base_filters)
    x2,y2 = conv3x3_relu_block(x1, n_filters=base_filters * 2)
    x3,y3 = conv3x3_relu_block(x2, n_filters=base_filters * 2 ** 2)
    x4,y4 = conv3x3_relu_block(x3, n_filters=base_filters * 2 ** 3)

    x5,y5 = conv3x3_relu_block(x4, kernel_size=3, n_filters=base_filters * 2 ** 4, direction=UP)
    x5 = Cropping2D((15,15))(x5)

    x = concatenate([x5, y5])
    x, _ = conv3x3_relu_block(x, kernel_size=2, n_filters=base_filters * 2 ** 3, direction=UP)
    y3 = concatenate([x3, y3])
    y2 = conv3x3_relu_block(y3, kernel_size=2, n_filters=base_filters * 2 ** 2, direction=UP)
    y2 = concatenate([x2,y2])
    y1 = conv3x3_relu_block(y2, kernel_size=2, n_filters=base_filters * 2, direction=UP)
    y1 = concatenate([x1,y1])

    x = conv3x3_relu_block(y1, n_filters=n_classes, direction=OUT)
    return keras.models.Model(inputs=[inp], outputs=[x])

def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),  kernel_initializer="he_normal",
               #kernel_regularizer=keras.regularizers.l2(),
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               #kernel_regularizer=keras.regularizers.l2(),
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

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
    
    outputs = Conv2D(flow.N_CLASSES, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def build_google_unet():
    output_channels = 1
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
    base_model.trainable = False

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'block_16_project',      # 4x4
    ]
    layers = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

#    down_stack.trainable = False

    up_stack = [
        pix2pix.upsample(512, 3),  # 4x4 -> 8x8
        pix2pix.upsample(256, 3),  # 8x8 -> 16x16
        pix2pix.upsample(128, 3),  # 16x16 -> 32x32
        pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    ]

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same', activation='sigmoid')  #64x64 -> 128x128

    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # Downsampling through the model
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    
    # Upsampling and establishing the skip connections
    i = 0
    for up, skip in zip(up_stack, skips):
        try:
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
    
        except Exception as exc:
            print("on iteration %d" % i)
            raise exc
        i += 1
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

