import tensorflow as tf
import tensorflow.keras.layers as L
import settings as S
import logging

logger = logging.getLogger(__name__)


class MotokimuraUnet():
    """
    Slightly downsized Unet design by Motokimura (winner of SpaceNet challenge).

    Model for single or multiclass semantic segmentation.

    With a GeForce GTX 1080 Ti, trains in about 24 hours with model weight files of 82MB.

    Xview2 scores:
        Multiclass - 0.778 localization, 0.310 damage classification (0.450 overall).
        Single class - 0.818 localization, 0.0 damage classification (0.304 overall).
    """
    def __init__(self, *args, **kwargs):
        if 'classes' not in kwargs:
            raise KeyError("pass number of classes as classes=N")
        input_shape = S.INPUTSHAPE
        damage = S.DAMAGE
        s = self
        factor = 5
        s.c0 = L.Conv2D(2 ** (factor), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c1 = L.Conv2D(2 ** (factor+1), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c2 = L.Conv2D(2 ** (factor+1), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c3 = L.Conv2D(2 ** (factor+2), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c4 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c5 = L.Conv2D(2 ** (factor+3), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c6 = L.Conv2D(2 ** (factor+3), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c7 = L.Conv2D(2 ** (factor+4), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c8 = L.Conv2D(2 ** (factor+4), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))

        s.dc8 = L.Conv2DTranspose(2 ** (factor+4), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc7 = L.Conv2D(2 ** (factor+4), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.dc6 = L.Conv2DTranspose(2 ** (factor+3), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc5 = L.Conv2D(2 ** (factor+3), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.dc4 = L.Conv2DTranspose(2 ** (factor+2), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc3 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.dc2 = L.Conv2DTranspose(2 ** (factor), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc1 = L.Conv2D(2 ** (factor), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.dc0 = L.Conv2D(kwargs['classes'], kernel_size=(3,3), strides=1, padding='same', name='decoder_out')

        s.bnc0 = L.BatchNormalization()
        s.bnc1 = L.BatchNormalization()
        s.bnc2 = L.BatchNormalization()
        s.bnc3 = L.BatchNormalization()
        s.bnc4 = L.BatchNormalization()
        s.bnc5 = L.BatchNormalization()
        s.bnc6 = L.BatchNormalization()
        s.bnc7 = L.BatchNormalization()
        s.bnc8 = L.BatchNormalization()
        
        s.bnd8 = L.BatchNormalization()
        s.bnd7 = L.BatchNormalization()
        s.bnd6 = L.BatchNormalization()
        s.bnd5 = L.BatchNormalization()
        s.bnd4 = L.BatchNormalization()
        s.bnd3 = L.BatchNormalization()
        s.bnd2 = L.BatchNormalization()
        s.bnd1 = L.BatchNormalization()

        inp = L.Input(input_shape)
        e0 = L.Activation('relu')(s.bnc0(s.c0(inp)))
        e1 = L.Activation('relu')(s.bnc1(s.c1(e0)))
        e2 = L.Activation('relu')(s.bnc2(s.c2(e1)))
        e3 = L.Activation('relu')(s.bnc3(s.c3(e2)))
        e4 = L.Activation('relu')(s.bnc4(s.c4(e3)))
        e5 = L.Activation('relu')(s.bnc5(s.c5(e4)))
        e6 = L.Activation('relu')(s.bnc6(s.c6(e5)))
        e7 = L.Activation('relu')(s.bnc7(s.c7(e6)))
        e8 = L.Activation('relu', name='encoder_out')(s.bnc8(s.c8(e7)))

        d8 = L.Activation('relu')(s.bnd8(s.dc8(L.Concatenate()([e7,e8]))))
        d7 = L.Activation('relu')(s.bnd7(s.dc7(d8)))
        d6 = L.Activation('relu')(s.bnd6(s.dc6(L.Concatenate()([e6,d7]))))
        d5 = L.Activation('relu')(s.bnd5(s.dc5(d6)))
        d4 = L.Activation('relu')(s.bnd4(s.dc4(L.Concatenate()([e4,d5]))))
        d3 = L.Activation('relu')(s.bnd3(s.dc3(d4)))
        d2 = L.Activation('relu')(s.bnd2(s.dc2(L.Concatenate()([e2,d3]))))
        d1 = L.Activation('relu')(s.bnd1(s.dc1(d2)))
        d0 = s.dc0(L.Concatenate()([e0,d1]))

        self.model = tf.keras.models.Model(inputs=[inp], outputs=[d0])

    def compile(self, *args, **kwargs):
        return self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.model.load_weights(*args, **kwargs)


class GenerativeUnet(MotokimuraUnet):
    def __init__(self, *args, **kwargs):
        if 'classes' not in kwargs:
            raise KeyError("pass number of classes as classes=N")
        input_shape = S.INPUTSHAPE
        damage = S.DAMAGE
        s = self
        factor = 5
        s.c0 = L.Conv2D(2 ** (factor), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c1 = L.Conv2D(2 ** (factor+1), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c2 = L.Conv2D(2 ** (factor+1), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c3 = L.Conv2D(2 ** (factor+2), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c4 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c5 = L.Conv2D(2 ** (factor+3), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c6 = L.Conv2D(2 ** (factor+3), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c7 = L.Conv2D(2 ** (factor+4), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c8 = L.Conv2D(2 ** (factor+4), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))

        s.dc8 = L.Conv2DTranspose(2 ** (factor+4), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc7 = L.Conv2D(2 ** (factor+4), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.dc6 = L.Conv2DTranspose(2 ** (factor+3), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc5 = L.Conv2D(2 ** (factor+3), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.dc4 = L.Conv2DTranspose(2 ** (factor+2), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc3 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.dc2 = L.Conv2DTranspose(2 ** (factor), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc1 = L.Conv2D(2 ** (factor), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.dc0 = L.Conv2D(kwargs['classes'], kernel_size=(3,3), strides=1, padding='same', name='decoder_out')

        s.bnc0 = L.BatchNormalization()
        s.bnc1 = L.BatchNormalization()
        s.bnc2 = L.BatchNormalization()
        s.bnc3 = L.BatchNormalization()
        s.bnc4 = L.BatchNormalization()
        s.bnc5 = L.BatchNormalization()
        s.bnc6 = L.BatchNormalization()
        s.bnc7 = L.BatchNormalization()
        s.bnc8 = L.BatchNormalization()
        
        s.bnd8 = L.BatchNormalization()
        s.bnd7 = L.BatchNormalization()
        s.bnd6 = L.BatchNormalization()
        s.bnd5 = L.BatchNormalization()
        s.bnd4 = L.BatchNormalization()
        s.bnd3 = L.BatchNormalization()
        s.bnd2 = L.BatchNormalization()
        s.bnd1 = L.BatchNormalization()

        inp = L.Input(input_shape)
        e0 = L.Activation('relu')(s.bnc0(s.c0(inp)))
        e1 = L.Activation('relu')(s.bnc1(s.c1(e0)))
        e2 = L.Activation('relu')(s.bnc2(s.c2(e1)))
        e3 = L.Activation('relu')(s.bnc3(s.c3(e2)))
        e4 = L.Activation('relu')(s.bnc4(s.c4(e3)))
        e5 = L.Activation('relu')(s.bnc5(s.c5(e4)))
        e6 = L.Activation('relu')(s.bnc6(s.c6(e5)))
        e7 = L.Activation('relu')(s.bnc7(s.c7(e6)))
        e8 = L.Activation('relu', name='encoder_out')(s.bnc8(s.c8(e7)))

        d8 = L.Activation('relu')(s.bnd8(s.dc8(L.Concatenate()([e7,e8]))))
        d7 = L.Activation('relu')(s.bnd7(s.dc7(d8)))
        d6 = L.Activation('relu')(s.bnd6(s.dc6(L.Concatenate()([e6,d7]))))
        d5 = L.Activation('relu')(s.bnd5(s.dc5(d6)))
        d4 = L.Activation('relu')(s.bnd4(s.dc4(L.Concatenate()([e4,d5]))))
        d3 = L.Activation('relu')(s.bnd3(s.dc3(d4)))
        d2 = L.Activation('relu')(s.bnd2(s.dc2(L.Concatenate()([e2,d3]))))
        d1 = L.Activation('relu')(s.bnd1(s.dc1(d2)))
        d0 = s.dc0(L.Concatenate()([e0,d1]))

        self.model = tf.keras.models.Model(inputs=[inp], outputs=[d0])

class DoubleMotokimuraUnet(MotokimuraUnet):
    def __init__(self, *args, **kwargs):
        if 'classes' not in kwargs:
            raise KeyError("pass number of classes as classes=N")
        input_shape = S.INPUTSHAPE
        damage = S.DAMAGE
        s = self
        factor = 5
        s.c0 = L.Conv2D(2 ** (factor), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.c1 = L.Conv2D(2 ** (factor), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.c2 = L.Conv2D(2 ** (factor), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.c3 = L.Conv2D(2 ** (factor), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.c4 = L.Conv2D(2 ** (factor+1), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.c5 = L.Conv2D(2 ** (factor+1), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.c6 = L.Conv2D(2 ** (factor+1), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.c7 = L.Conv2D(2 ** (factor+1), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.c8 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)

        s.c9 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)

        s.c10 = L.Conv2D(2 ** (factor+2), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.c11 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.c12 = L.Conv2D(2 ** (factor+3), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.c13 = L.Conv2D(2 ** (factor+3), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.c14 = L.Conv2D(2 ** (factor+3), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.c15 = L.Conv2D(2 ** (factor+3), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.c16 = L.Conv2D(2 ** (factor+4), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.c17 = L.Conv2D(2 ** (factor+4), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)

        s.dc17 = L.Conv2DTranspose(2 ** (factor+4), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc16 = L.Conv2D(2 ** (factor+4), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.dc15 = L.Conv2DTranspose(2 ** (factor+3), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc14 = L.Conv2D(2 ** (factor+3), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.dc13 = L.Conv2DTranspose(2 ** (factor+3), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc12 = L.Conv2D(2 ** (factor+3), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.dc11 = L.Conv2DTranspose(2 ** (factor+2), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc10 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)

        s.dc9 = L.Conv2DTranspose(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)

        s.dc8 = L.Conv2DTranspose(2 ** (factor+2), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc7 = L.Conv2D(2 ** (factor+1), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.dc6 = L.Conv2DTranspose(2 ** (factor+1), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc5 = L.Conv2D(2 ** (factor+1), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.dc4 = L.Conv2DTranspose(2 ** (factor+1), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc3 = L.Conv2D(2 ** (factor), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.dc2 = L.Conv2DTranspose(2 ** (factor), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc1 = L.Conv2D(2 ** (factor), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.dc0 = L.Conv2D(kwargs['classes'], kernel_size=(3,3), strides=1, padding='same', name='decoder_out')

        s.bnc0 = L.BatchNormalization()
        s.bnc1 = L.BatchNormalization()
        s.bnc2 = L.BatchNormalization()
        s.bnc3 = L.BatchNormalization()
        s.bnc4 = L.BatchNormalization()
        s.bnc5 = L.BatchNormalization()
        s.bnc6 = L.BatchNormalization()
        s.bnc7 = L.BatchNormalization()
        s.bnc8 = L.BatchNormalization()

        s.bnc9 = L.BatchNormalization()

        s.bnc10 = L.BatchNormalization()
        s.bnc11 = L.BatchNormalization()
        s.bnc12 = L.BatchNormalization()
        s.bnc13 = L.BatchNormalization()
        s.bnc14 = L.BatchNormalization()
        s.bnc15 = L.BatchNormalization()
        s.bnc16 = L.BatchNormalization()
        s.bnc17 = L.BatchNormalization()
        
        s.bnd17 = L.BatchNormalization()
        s.bnd16 = L.BatchNormalization()
        s.bnd15 = L.BatchNormalization()
        s.bnd14 = L.BatchNormalization()
        s.bnd13 = L.BatchNormalization()
        s.bnd12 = L.BatchNormalization()
        s.bnd11 = L.BatchNormalization()
        s.bnd10 = L.BatchNormalization()

        s.bnd9 = L.BatchNormalization()

        s.bnd8 = L.BatchNormalization()
        s.bnd7 = L.BatchNormalization()
        s.bnd6 = L.BatchNormalization()
        s.bnd5 = L.BatchNormalization()
        s.bnd4 = L.BatchNormalization()
        s.bnd3 = L.BatchNormalization()
        s.bnd2 = L.BatchNormalization()
        s.bnd1 = L.BatchNormalization()

        inp = L.Input(input_shape)
        e0 = L.Activation('relu')(s.bnc0(s.c0(inp)))
        e1 = L.Activation('relu')(s.bnc1(s.c1(e0)))
        e2 = L.Activation('relu')(s.bnc2(s.c2(e1)))
        e3 = L.Activation('relu')(s.bnc3(s.c3(e2)))
        e4 = L.Activation('relu')(s.bnc4(s.c4(e3)))
        e5 = L.Activation('relu')(s.bnc5(s.c5(e4)))
        e6 = L.Activation('relu')(s.bnc6(s.c6(e5)))
        e7 = L.Activation('relu')(s.bnc7(s.c7(e6)))
        e8 = L.Activation('relu')(s.bnc8(s.c8(e7)))

        e9 = L.Activation('relu')(s.bnc9(s.c9(e8)))

        e10 = L.Activation('relu')(s.bnc10(s.c10(e9)))
        e11 = L.Activation('relu')(s.bnc11(s.c11(e10)))
        e12 = L.Activation('relu')(s.bnc12(s.c12(e11)))
        e13 = L.Activation('relu')(s.bnc13(s.c13(e12)))
        e14 = L.Activation('relu')(s.bnc14(s.c14(e13)))
        e15 = L.Activation('relu')(s.bnc15(s.c15(e14)))
        e16 = L.Activation('relu')(s.bnc16(s.c16(e15)))
        e17 = L.Activation('relu')(s.bnc17(s.c17(e16)))

        d17 = L.Activation('relu')(s.bnd17(s.dc17(L.Concatenate()([e16,e17]))))
        d16 = L.Activation('relu')(s.bnd16(s.dc16(d17)))
        d15 = L.Activation('relu')(s.bnd15(s.dc15(L.Concatenate()([e14,d16]))))
        d14 = L.Activation('relu')(s.bnd14(s.dc14(d15)))
        d13 = L.Activation('relu')(s.bnd13(s.dc13(L.Concatenate()([e12,d14]))))
        d12 = L.Activation('relu')(s.bnd12(s.dc12(d13)))
        d11 = L.Activation('relu')(s.bnd11(s.dc11(L.Concatenate()([e10,d12]))))
        d10 = L.Activation('relu')(s.bnd10(s.dc10(d11)))

        d9 = L.Activation('relu')(s.bnd9(s.dc9(L.Concatenate()([e8,d10]))))

        d8 = L.Activation('relu')(s.bnd8(s.dc8(L.Concatenate()([e8,d9]))))
        d7 = L.Activation('relu')(s.bnd7(s.dc7(d8)))
        d6 = L.Activation('relu')(s.bnd6(s.dc6(L.Concatenate()([e6,d7]))))
        d5 = L.Activation('relu')(s.bnd5(s.dc5(d6)))
        d4 = L.Activation('relu')(s.bnd4(s.dc4(L.Concatenate()([e4,d5]))))
        d3 = L.Activation('relu')(s.bnd3(s.dc3(d4)))
        d2 = L.Activation('relu')(s.bnd2(s.dc2(L.Concatenate()([e2,d3]))))
        d1 = L.Activation('relu')(s.bnd1(s.dc1(d2)))
        d0 = s.dc0(L.Concatenate()([e0,d1]))

        self.model = tf.keras.models.Model(inputs=[inp], outputs=[d0])


class MotokimuraMobilenet(MotokimuraUnet):
    """
    MotokimuraUnet modified to use a MobileNetV2 encoder and an extra deconv layer.

    Model for single class or multiclass semantic segmentation.

    With a GeForce GTX 1080 Ti, trains in about 24 hours with model weight files of 71MB.
    """
    def __init__(self, *args, **kwargs):
        # The encoder (feature extractor)
        self.mobilenetv2 = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=S.INPUTSHAPE,
                                                                          weights=None,
                                                                          include_top=False)
        if 'classes' not in kwargs:
            raise KeyError("pass number of classes as classes=N")
        input_shape = S.INPUTSHAPE
        damage = S.DAMAGE
        s = self
        factor = 5

        # The decoder layers
        s.dc10 = L.Conv2DTranspose(2 ** (factor+4), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        #s.dc9 = L.Conv2D(2 ** (factor+4), kernel_size=(3,3), strides=1, padding='same',
        #        use_bias=True if damage else False)
        s.dc8 = L.Conv2DTranspose(2 ** (factor+3), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc7 = L.Conv2D(2 ** (factor+3), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.dc6 = L.Conv2DTranspose(2 ** (factor+2), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc5 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.dc4 = L.Conv2DTranspose(2 ** (factor+2), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc3 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.dc2 = L.Conv2DTranspose(2 ** (factor+1), kernel_size=(4,4), strides=2, padding='same',
                use_bias=True if damage else False)
        s.dc1 = L.Conv2D(2 ** (factor+1), kernel_size=(3,3), strides=1, padding='same',
                use_bias=True if damage else False)
        s.dc0 = L.Conv2D(kwargs['classes'], kernel_size=(3,3), strides=1, padding='same', name='decoder_out')

        # BatchNormalization for every layer
        s.bnd9 = L.BatchNormalization()
        s.bnd8 = L.BatchNormalization()
        s.bnd7 = L.BatchNormalization()
        s.bnd6 = L.BatchNormalization()
        s.bnd5 = L.BatchNormalization()
        s.bnd4 = L.BatchNormalization()
        s.bnd3 = L.BatchNormalization()
        s.bnd2 = L.BatchNormalization()
        s.bnd1 = L.BatchNormalization()

        # Final MobileNetV2 output
        mobilenet_out = self.mobilenetv2.get_layer("out_relu").output # 32x32
        
        # Skip connections
        e10 = self.mobilenetv2.get_layer("block_15_project_BN").output # 32x32
        e8 = self.mobilenetv2.get_layer("block_12_project_BN").output # 64x64
        e6 = self.mobilenetv2.get_layer("block_5_project_BN").output # 128x128
        e4 = self.mobilenetv2.get_layer("block_2_project_BN").output # 256x256
        e2 = self.mobilenetv2.get_layer("expanded_conv_project_BN").output # 512x512
        e0 = self.mobilenetv2.get_layer("Conv1_relu").output # 512x512

        # Put it all together
        d10 = L.Activation('relu')(s.bnd9(s.dc10(L.Concatenate()([mobilenet_out,e10]))))
        #d9 = L.Activation('relu')(s.bnd9(s.dc9(d10)))
        d8 = L.Activation('relu')(s.bnd8(s.dc8(L.Concatenate()([e8,d10]))))
        d7 = L.Activation('relu')(s.bnd7(s.dc7(d8)))
        d6 = L.Activation('relu')(s.bnd6(s.dc6(L.Concatenate()([e6,d7]))))
        d5 = L.Activation('relu')(s.bnd5(s.dc5(d6)))
        d4 = L.Activation('relu')(s.bnd4(s.dc4(L.Concatenate()([e4,d5]))))
        d3 = L.Activation('relu')(s.bnd3(s.dc3(d4)))
        d2 = L.Activation('relu')(s.bnd2(s.dc2(L.Concatenate()([e2,d3]))))
        d1 = L.Activation('relu')(s.bnd1(s.dc1(d2)))
        d0 = s.dc0(d1)#L.Concatenate()([e0,d1]))

        self.model = tf.keras.models.Model(inputs=self.mobilenetv2.inputs, outputs=[d0])


class Ensemble(MotokimuraMobilenet):
    def __init__(self, *args, **kwargs):
        if 'classes' not in kwargs:
            raise KeyError("pass number of classes as classes=N")
        self.motokimura = MotokimuraUnet(*args, **kwargs)
        self.mobilenet = MotokimuraMobilenet(*args, **kwargs)

        inp = L.Input(S.INPUTSHAPE)

        one = self.motokimura(inp)
        one = L.Reshape((-1,kwargs['classes']))(one)

        two = self.mobilenet(inp)
        two = L.Reshape((-1,kwargs['classes']))(two)

        out = L.Add()([one, two])
        out = L.Activation('softmax')(out)
        self.model = tf.keras.models.Model(inputs=[inp], outputs=[out])

    def load_individual_weights(self, onefile="damage-motokimura-best.hdf5", twofile="damage-motokimura-mobilenetv2-best.hdf5"):
        self.motokimura.load_weights(onefile, by_name=True)
        self.mobilenet.load_weights(twofile, by_name=True)
