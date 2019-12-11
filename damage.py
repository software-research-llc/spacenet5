import numpy as np
import logging
import infer
from flow import Dataflow
import train
import deeplabmodel
import settings as S
import tensorflow as tf

logger = logging.getLogger(__name__)


def build_model(backbone=S.ARCHITECTURE,
                train=False,
                classes=4):
    height = 128
    width = 128
    deeplab = deeplabmodel.Deeplabv3(input_shape=(height*2,width*2),
                                     backbone=backbone,
                                     OS=16 if train is True else 8,
                                     weights='pascal_voc',
                                     classes=classes)

    logits = tf.keras.models.Model(inputs=deeplab.inputs, outputs=[deeplab.get_layer("image_pooling_BN").output])

    inp_pre = tf.keras.layers.Input(input_shape=(height,width,3))
    inp_post = tf.keras.layers.Input(input_shape=(height,width,3))

    x = tf.image.pad_to_bounding_box(inp_pre, 0, 0, height, width)
    y = tf.image.pad_to_bounding_box(inp_post, 0, 0, height, width)

    x = tf.keras.layers.Concatenate(name='concatenated_input')([x,y])
    x = logits(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)

    return tf.keras.models.Model(inputs=[inp_pre, inp_post], outputs=[x])


def extract_patches(pre, post, mask):
    preboxes = []
    postboxes = []
    klasses = []
    rectangles = infer.bounding_rectangles(mask)
    for rect in rectangles:
        x,y = rect
        if x.start+x.stop <= 1 or y.start+y.stop <= 1:
            continue
        prebox = pre[x.start:x.stop,y.start:y.stop]
        postbox = post[x.start:x.stop,y.start:y.stop]

        klass = np.mean(mask[x.start:x.stop,y.start:y.stop])
        klass = int(round(klass))

        preboxes.append(prebox)#.astype(np.uint8))
        postboxes.append(postbox)#.astype(np.uint8))
        klasses.append(klass)

    if len(preboxes) < 1:
        return (None,None), None

    return (np.array(preboxes), np.array(postboxes)), np.array(klasses)


class DamageDataflow(Dataflow):
    def __getitem__(self, idx):
        (x,y), mask = Dataflow.__getitem__(self, idx, preprocess=False)
        mask = infer.weave_pred(mask)
        x = infer.weave(x)
        y = infer.weave(y)

        return extract_patches(x, y, mask)


if __name__ == '__main__':
    from show import display_images
    df = DamageDataflow()
    for (x,y), klass in df:
        if x is None:
            continue
        display_images(x)
        display_images(y)
