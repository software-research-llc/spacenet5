import show
import keras
from keras.applications import xception
from scipy.ndimage.interpolation import zoom

def build_model():
    m = xception.Xception()
    o = m.layers[-1]
    d = keras.layers.Dense(units=50)
    d = d(o.input)
    return keras.models.Model(input=m.input, output=d)

def preprocess(image):
    return xception.preprocess(image)
if __name__ == '__main__':
    m = build_model()
    m.summary()
