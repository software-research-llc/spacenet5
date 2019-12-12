import numpy as np
import logging
import infer
from flow import Dataflow
import train
import deeplabmodel
import settings as S
import tensorflow as tf
import plac

logger = logging.getLogger(__name__)


def mode(ary):
    ary = ary.ravel()
    counts = {}
    for val in ary:
        if counts.get(val, None) is None:
            counts[val] = 1
        else:
            counts[val] += 1
    
    return sorted(counts, key=lambda x: counts[x])[-1]


def build_model(backbone=S.ARCHITECTURE,
                train=False,
                classes=4):
    height = S.DAMAGE_MAX_X
    width = S.DAMAGE_MAX_Y
    deeplab = deeplabmodel.Deeplabv3(input_shape=(height*2,width*2),
                                     backbone=backbone,
                                     OS=16 if train is True else 8,
                                     weights='pascal_voc',
                                     classes=classes)

    logits = tf.keras.models.Model(inputs=deeplab.inputs, outputs=[deeplab.get_layer("image_pooling_BN").output])

    inp_pre = tf.keras.layers.Input(input_shape=(height,width,3))
    inp_post = tf.keras.layers.Input(input_shape=(height,width,3))

    # FIXME: can't do this, need a predetermined input shape
    shape_b4 = tf.int_shape(inp_pre)
    h = height - shape_b4[1]
    w = width - shape_b4[2]
    x = tf.image.pad_to_bounding_box(inp_pre, 0, w, height, 0)
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


def extract_patches(pre, post, mask, return_masks=False, max_x=S.DAMAGE_MAX_X, max_y=S.DAMAGE_MAX_Y):
    preboxes = []
    postboxes = []
    klasses = []
    masks = []
    rectangles = infer.bounding_rectangles(mask)
    for rect in rectangles:
        x,y = rect
        if x.stop-x.start <= 5 or y.stop-y.start <= 5:
            continue
        if x.stop-x.start >= max_x:
            x = slice(x.start, x.start + max_x - 1)
        if y.stop-y.start >= max_y:
            y = slice(y.start, y.start + max_y - 1)

        prebox = pre[x.start:x.stop,y.start:y.stop]
        postbox = post[x.start:x.stop,y.start:y.stop]
        retmask = mask[x.start:x.stop,y.start:y.stop]

        klass = mask[x.start:x.stop,y.start:y.stop]
        klass = mode(klass[np.nonzero(klass)])

        preboxes.append(prebox)#.astype(np.uint8))
        postboxes.append(postbox)#.astype(np.uint8))
        masks.append(retmask)
        klasses.append(klass)


    if return_masks is True:
        if len(preboxes) < 1:
            return (None,None),None,None
        return (np.array(preboxes), np.array(postboxes)), np.array(klasses), np.array(masks)
    else:
        if len(preboxes) < 1:
            return (None,None),None
        return (np.array(preboxes), np.array(postboxes)), np.array(klasses)


class DamageDataflow(Dataflow):
    def __init__(self, return_masks=False, *args, **kwargs):
        super(DamageDataflow, self).__init__(*args, **kwargs)
        self.return_masks = return_masks

    def __getitem__(self, idx):
        (x,y), mask = Dataflow.__getitem__(self, idx, preprocess=False, return_postmask=True)
        mask = infer.weave_pred(mask)
        x = infer.weave(x)
        y = infer.weave(y)

        return extract_patches(x, y, mask, return_masks=self.return_masks)


def epoch(model, train_seq, val_seq, step=16):
    for (pre,post), mask in train_seq:
        for i in range(0, len(pre), step):
            if i+step > len(pre):
                step = i + (i - len(pre))
            history = model.fit([pre[i:i+step], post[i:i+step]], mask[i:i+step],
                                verbose=1, shuffle=False)


def main(epochs=25):
    from flow import get_training_files, get_validation_files
    model = build_model()
    train_seq = DamageDataflow(files=get_training_files(), shuffle=True, transform=0.3)
    valid_seq = DamageDataflow(files=get_validation_files(), shuffle=True, transform=False)

    for i in range(epochs):
        logger.info("Epoch %d of %d" % (i, epochs))
        epoch(model, train_seq, val_seq)


def display():
    from show import display_images
    df = DamageDataflow(True)
    for (xs,ys), klasses, masks in df:
        if xs is None:
            continue
        images = []
        names = []
        for i in range(len(xs)):
            images.append(xs[i])
            images.append(ys[i])
            images.append(masks[i])
            names.append(klasses[i])
            names.append(klasses[i])
            names.append(klasses[i])
        display_images(images, names)


if __name__ == "__main__":
    plac.call(main)
