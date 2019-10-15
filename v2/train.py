import snflow as flow
import plac
import keras_segmentation


def main():
    model = keras_segmentation.models.resnet50_unet(n_classes=flow.N_CLASSES,
                                                    input_height=flow.IMSHAPE[0],
                                                    input_width=flow.IMSHAPE[1])
    model.checkpoint = tf.train.Checkpoint()

    seq = flow.Sequence()
    
