import show
import keras
from keras.applications import xception
from scipy.ndimage.interpolation import zoom

def build_model():
    xm = xception.Xception()
    x = xm.get_layer("avg_pool").output
    x = keras.layers.Dense(3, activation='relu')(x)
    x = keras.layers.Dense(299, activation='relu')(x)
    x = keras.layers.Dense(299 * 299 * 3, activation='linear', name='predictions')(x)
    return keras.models.Model(inputs=xm.input, outputs=x)
    """
    i = keras.layers.Input((299,299,3))
    x = keras.layers.BatchNormalization()(i)
    x = keras.layers.Conv2D(7, kernel_size=(3,3), padding='same')(x)
    x = keras.layers.Conv2D(5, kernel_size=(3,3), padding='same')(x)
    x = keras.layers.Conv2D(3, kernel_size=(3,3), padding='same')(x)
    x = keras.layers.Dense(1, activation='relu')(x)
    x = keras.layers.Dense(3, activation='sigmoid')(x)
    return keras.models.Model(i, x)
    """

def preprocess(image):
    return xception.preprocess(image)

if __name__ == '__main__':
    m = build_model()
    m.summary()
