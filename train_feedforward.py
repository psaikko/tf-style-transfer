import numpy as np
import argparse
from glob import glob
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import feedforward

# Workaround for tf 2.0 issue
# https://stackoverflow.com/a/58684421
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser(description='Train a style transfer network with tf.keras')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=100.0, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# Using 256x256 images as in Johnson et al.
# The trained network is fully convolutional, so it generalizes to higher resolution inputs
img_nrows = 256
img_ncols = 256

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_nrows, img_ncols))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.vgg19.preprocess_input(img)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x = x.numpy().reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')

# https://www.tensorflow.org/tutorials/generative/style_transfer
def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  return tf.keras.Model([vgg.input], outputs)

# https://www.tensorflow.org/tutorials/generative/style_transfer
@tf.function
def gram_matrix(x):
  #""" Computes the gram matrix of an image tensor (feature-wise outer product)."""
  return tf.linalg.einsum('bijc,bijd->bcd', x, x)

# "style loss" to maintain the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image and from the generated image
@tf.function
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    div = tf.constant(4.0 * (3 ** 2) * ((img_nrows * img_ncols) ** 2))
    return tf.reduce_sum(tf.square(S - C)) / div

# auxiliary loss function to maintain the "content" of the base image
@tf.function
def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

# total variation loss to keep the generated image locally coherent
@tf.function
def total_variation_loss(x):
    return tf.image.total_variation(x)

# Named layers of VGG model
# Using deeper layers (=higher level features) for content loss, 
# lower level features (=textures etc) for style loss
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
layer_model = vgg_layers(style_layers + content_layers)

# Wrap the loss function, including VGG model in a keras model
class StyleLossModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers, style_reference):
        super(StyleLossModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_reference = style_reference
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False
    
    def call(self, base_image, combined_image):
        base_image_features = self.vgg(base_image)
        style_reference_features = self.vgg(self.style_reference)
        combination_features = self.vgg(combined_image)
        
        loss = content_weight * content_loss(base_image_features[-1], combination_features[-1])
        loss = loss + total_variation_weight * total_variation_loss(combined_image)
        for i in range(self.num_style_layers):
            sl = style_loss(style_reference_features[i], combination_features[i])
            loss = loss + (style_weight / self.num_style_layers) * sl
        return loss

# Create the loss evaluator and wrapper function
loss_model = StyleLossModel(style_layers, content_layers, tf.constant(preprocess_image(style_reference_image_path)))
def call_loss_model(y_true, y_pred, sample_weight=None):
    return loss_model(y_true, y_pred)

ff = feedforward.make_network()
ff.compile(loss=call_loss_model, optimizer="adam")

image = tf.constant(preprocess_image(base_image_path))
plt.imshow(deprocess_image(ff(image)))
plt.pause(0.1)

while True:
    # Note: not training an identity function!
    # Custom loss function for the output is wrt. the input image
    ff.fit([image], [image], epochs=10)
    plt.imshow(deprocess_image(ff(image)))
    plt.pause(0.1)