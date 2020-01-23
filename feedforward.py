import tensorflow as tf
import tensorflow_addons as tfa

# Residual block as in
# https://web.eecs.umich.edu/~justincj/papers/eccv16/JohnsonECCV16Supplementary.pdf
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.filters = filters

        # Use instance normalization instead as suggested in 
        # https://arxiv.org/abs/1607.08022
        self.layers = [
            tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3)),
            tfa.layers.InstanceNormalization(),
            tf.keras.activations.relu,
            tf.keras.layers.Conv2D(filters=self.filters, kernel_size=(3,3)),
            tfa.layers.InstanceNormalization()
        ]
        for (i, layer) in enumerate(self.layers):
             self.__setattr__("layer_%d"%i, layer)

    def call(self, x):
        # Residual: add input to output
        res = tf.keras.layers.Cropping2D(((2,2),(2,2)))(x)
        for layer in self.layers:
            x = layer(x)
        return tf.keras.backend.sum([x, res], axis=0)

def make_network(scale=32):
    x = tf.keras.layers.Input((256,256,3))
    # Johnson et al. suggests reflection padding
    y = tf.pad(x, [[0,0], [40,40], [40,40], [0,0] ], 'REFLECT')
    y = tf.keras.layers.Conv2D(filters=scale, kernel_size=(9,9), strides=1, padding="same")(y)
    y = tfa.layers.InstanceNormalization()(y)
    y = tf.keras.activations.relu(y)
    y = tf.keras.layers.Conv2D(filters=scale*2, kernel_size=(3,3), strides=2, padding="same")(y)
    y = tfa.layers.InstanceNormalization()(y)
    y = tf.keras.activations.relu(y)
    y = tf.keras.layers.Conv2D(filters=scale*4, kernel_size=(3,3), strides=2, padding="same")(y)
    y = tfa.layers.InstanceNormalization()(y)
    y = tf.keras.activations.relu(y)
    y = ResidualBlock(scale*4)(y)
    y = ResidualBlock(scale*4)(y)
    y = ResidualBlock(scale*4)(y)
    y = ResidualBlock(scale*4)(y)
    y = ResidualBlock(scale*4)(y)
    # equivalent to "fractionally strided convolutions"
    y = tf.keras.layers.Conv2DTranspose(filters=scale*2, kernel_size=(3,3), strides=2, padding="same")(y)
    y = tfa.layers.InstanceNormalization()(y)
    y = tf.keras.activations.relu(y)
    y = tf.keras.layers.Conv2DTranspose(filters=scale, kernel_size=(3,3), strides=2, padding="same")(y)
    y = tfa.layers.InstanceNormalization()(y)
    y = tf.keras.activations.relu(y)
    y = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(9,9), strides=1, padding="same")(y)
    y = tfa.layers.InstanceNormalization()(y)
    y = tf.keras.activations.tanh(y)
    y = y * 127.5
    
    model = tf.keras.Model(inputs=x, outputs=y)
    model.summary()
    return model
