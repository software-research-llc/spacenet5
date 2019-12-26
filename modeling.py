import tensorflow.keras.layers as L
import tensorflow as tf
import settings as S

class MotokimuraUnet():
    def __init__(self, input_shape=S.INPUTSHAPE, *args, **kwargs):
        if 'classes' not in kwargs:
            raise KeyError("pass number of classes as classes=N")
        s = self
        factor = 5
        s.c0 = L.Conv2D(2 ** (factor), kernel_size=(3,3), strides=1, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c1 = L.Conv2D(2 ** (factor+1), kernel_size=(4,4), strides=2, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c2 = L.Conv2D(2 ** (factor+1), kernel_size=(3,3), strides=1, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c3 = L.Conv2D(2 ** (factor+2), kernel_size=(4,4), strides=2, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c4 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c5 = L.Conv2D(2 ** (factor+3), kernel_size=(4,4), strides=2, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c6 = L.Conv2D(2 ** (factor+3), kernel_size=(3,3), strides=1, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c7 = L.Conv2D(2 ** (factor+4), kernel_size=(4,4), strides=2, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.c8 = L.Conv2D(2 ** (factor+4), kernel_size=(3,3), strides=1, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))

        s.dc8 = L.Conv2DTranspose(2 ** (factor+4), kernel_size=(4,4), strides=2, padding='same',
                use_bias=False)
        s.dc7 = L.Conv2D(2 ** (factor+4), kernel_size=(3,3), strides=1, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.dc6 = L.Conv2DTranspose(2 ** (factor+3), kernel_size=(4,4), strides=2, padding='same',
                use_bias=False)
        s.dc5 = L.Conv2D(2 ** (factor+3), kernel_size=(3,3), strides=1, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.dc4 = L.Conv2DTranspose(2 ** (factor+2), kernel_size=(4,4), strides=2, padding='same',
                use_bias=False)
        s.dc3 = L.Conv2D(2 ** (factor+2), kernel_size=(3,3), strides=1, padding='same',
                use_bias=False)
                        #kernel_regularizer=tf.keras.regularizers.l2(0.000000001))
        s.dc2 = L.Conv2DTranspose(2 ** (factor), kernel_size=(4,4), strides=2, padding='same',
                use_bias=False)
        s.dc1 = L.Conv2D(2 ** (factor), kernel_size=(3,3), strides=1, padding='same',
                use_bias=False)
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


