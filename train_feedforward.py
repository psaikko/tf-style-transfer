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
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('--content_weight', type=float, default=1e0, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=2e-5, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=2e-4, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
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
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_nrows, img_ncols), interpolation="bicubic")
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    #return img
    return tf.keras.applications.vgg19.preprocess_input(img)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x = x.numpy().reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # # 'BGR'->'RGB'
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
  result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
  input_shape = tf.shape(x)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result / num_locations

# "style loss" to maintain the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image and from the generated image
@tf.function
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    return style_weight * tf.reduce_mean(tf.square(S - C))

# auxiliary loss function to maintain the "content" of the base image
@tf.function
def content_loss(base, combination):
    return content_weight * tf.reduce_mean(tf.square(combination - base))

# total variation loss to keep the generated image locally coherent
@tf.function
def total_variation_loss(x):
    return total_variation_weight * tf.image.total_variation(x)

# Named layers of VGG model
# Using deeper layers (=higher level features) for content loss, 
# lower level features (=textures, colors etc) for style loss
content_layers = ['block3_conv2']
style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv2']
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
        
        loss = content_loss(base_image_features[-1], combination_features[-1])
        loss += total_variation_loss(combined_image)
        for i in range(self.num_style_layers):
            sl = style_loss(style_reference_features[i], combination_features[i])
            loss += (sl / self.num_style_layers)
        return loss

# Create the loss evaluator and wrapper function
loss_model = StyleLossModel(style_layers, content_layers, tf.constant(preprocess_image(style_reference_image_path)))
def call_loss_model(y_true, y_pred, sample_weight=None):
    return loss_model(y_true, y_pred)

ff = feedforward.make_network(scale=16)
ff.compile(loss=call_loss_model, optimizer="adam")

#
# Training on the MS-COCO dataset
# as in https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf
# We use only the images data, any diverse set of images could be used. 
# 
coco_train, info = tfds.load(name="coco", split="train", with_info=True)

def resize(x): return tf.image.resize(x, (img_nrows, img_ncols))
def get_image(x): return x["image"]

#
# Set up tf.data pipeline
#
batch_size = 4
n_samples = info.splits["train"].num_examples
feed = coco_train.repeat()
feed = feed.map(get_image).map(resize)
feed = feed.map(tf.keras.applications.vgg16.preprocess_input)
feed = feed.shuffle(42).batch(batch_size)

i = 0
fig, ax = plt.subplots(3,2)
plt.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
for batch in feed: 
    # Note: not training an identity function!
    # Custom loss function for the output is wrt. the input image
    ff.fit(batch, batch)
    fig.suptitle("Iteration %d" % i)
    if i % 10 == 0:
        for j in range(3):
            plt.subplot(321+2*j)
            plt.cla()
            plt.imshow(deprocess_image(batch[j]))
            plt.axis('off')
            plt.subplot(322+2*j)
            plt.cla()
            plt.imshow(deprocess_image(ff(tf.expand_dims(batch[j], axis=0))))
            plt.axis('off')
        plt.pause(0.01)

    if i % 1000 == 0:
        ff.save_weights("checkpoints/%d" % i)
        plt.savefig("checkpoints/%d.png" % i)
    # train for about 2 epochs
    if i * batch_size > 2 * n_samples:
        break
    i += 1
    