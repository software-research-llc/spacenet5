from __future__ import absolute_import, division, print_function, unicode_literals
from keras.layers import Input, Flatten, Reshape, MaxPooling2D, Dropout, BatchNormalization, Conv2D, Activation, Conv2DTranspose, Concatenate, concatenate, Conv3D, Cropping2D
from keras.models import Model
import keras
import tensorflow as tf
import snflow as flow
from tensorflow_examples.models.pix2pix import pix2pix

import numpy as np

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

from IPython.display import clear_output
import matplotlib.pyplot as plt


DOWN = -1
UP = 1
OUT = 0
OUTPUT_CHANNELS = flow.N_CLASSES
generator = None
discriminator = None

def build_model(train=True):
#    return build_google_unet()
    return get_unet(input_img=keras.layers.Input(flow.IMSHAPE))
    p2p = build_gan()
    return p2p
    if train:
        return p2p
    else:
        return p2p.generator
    return build_google_unet()
    return xception_model()

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
    model.compile(optimizer=flow.OPTIMIZER, loss=flow.LOSS, metrics=['binary_accuracy', flow.loss.jaccard_distance, flow.loss.binary_focal_loss_fixed, flow.loss.ssim_metric])

def load_initial_gan_inp():
  image = tf.Variable(np.array(flow.IMSHAPE), name='gan_input')#tf.io.read_file(image_file)
  #image = tf.image.decode_jpeg(image)

  #w = tf.shape(image)[1]

  #w = w // 2
  #real_image = image[:, :w, :]
  #input_image = image[:, w:, :]

  #input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(image, tf.float32)

  return real_image, real_image

def build_gan():
    return Generator()
    return pix2pix.Pix2pix(enable_function=False, epochs=flow.EPOCHS)
    global generator
    global discriminator
    tf.compat.v1.disable_eager_execution()
    generator = Generator()
    inp = tf.compat.v1.placeholder(tf.float32, shape=[1] + flow.IMSHAPE)#tf.Variable(np.zeros([flow.BATCH_SIZE] + flow.IMSHAPE), shape=(flow.BATCH_SIZE, 256, 256, 3), dtype=tf.float32)#tf.keras.layers.Input(shape=[None, None, 3])
#flow.get_image().reshape([1] + flow.IMSHAPE).astype(np.float32)

    gen_out = keras.models.Model(inputs=inp, outputs=generator(inp))

    discriminator = Discriminator()
    disc_out = discriminator([inp, gen_out])

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[None, None, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def Generator():
  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(128, 4), # (bs, 64, 64, 128)
    downsample(256, 4), # (bs, 32, 32, 256)
    downsample(512, 4), # (bs, 16, 16, 512)
    downsample(512, 4), # (bs, 8, 8, 512)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation=flow.LAST_LAYER_ACTIVATION) # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[None,None,3])
  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)
#  x = Conv2D(flow.N_CLASSES, (1, 1), activation=flow.LAST_LAYER_ACTIVATION) (x)
  return tf.keras.Model(inputs=inputs, outputs=x)

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

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

def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
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

    outputs = Conv2D(flow.N_CLASSES, (1, 1), activation=flow.LAST_LAYER_ACTIVATION) (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model

def build_google_unet():
    output_channels = 3
    base_model = tf.keras.applications.MobileNetV2(input_shape=flow.IMSHAPE, include_top=False)
#    base_model.trainable = True

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

