import keras
import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec
from keras.models import Model
import keras.backend as K

# Johnson et al. suggests reflection padding
# Keras impl. from https://stackoverflow.com/a/53349976 
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        # If you are using "channels_last" configuration
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

# Residual block as in
# https://web.eecs.umich.edu/~justincj/papers/eccv16/JohnsonECCV16Supplementary.pdf
class ResidualBlock(Layer):
    def __init__(self, filters=1, **kwargs):
        self.filters = filters
        super(ResidualBlock, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1]-4, s[1]-4, self.filters)

    def call(self, x):
        y = keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3))(x)
        y = keras.layers.BatchNormalization()(y)
        y = keras.activations.relu(y)
        y = keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3))(y)
        y = keras.layers.BatchNormalization()(y)
        # Residual: add input to output
        res = keras.layers.Cropping2D(((2,2),(2,2)))(x)
        return K.sum([y, res], axis=0)

def make_network():
    x = keras.layers.Input((256,256,3)) # (None,None,3)
    y = ReflectionPadding2D((40,40))(x)
    y = keras.layers.Conv2D(filters=32, kernel_size=(9,9), strides=1, padding="same")(y)
    y = keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=2, padding="same")(y)
    y = keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=2, padding="same")(y)
    y = ResidualBlock(128)(y)
    y = ResidualBlock(128)(y)
    y = ResidualBlock(128)(y)
    y = ResidualBlock(128)(y)
    y = ResidualBlock(128)(y)
    y = keras.layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=2, padding="same")(y)
    y = keras.layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=2, padding="same")(y)
    y = keras.layers.Conv2DTranspose(filters=3, kernel_size=(9,9), strides=1, padding="same")(y)

    model = Model(inputs=x, outputs=y)
    model.summary()
    return model
