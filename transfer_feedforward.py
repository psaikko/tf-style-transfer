import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import feedforward
import math
import time

# Workaround for tf 2.0 issue
# https://stackoverflow.com/a/58684421
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser(description='Apply style to image with pre-trained network')
parser.add_argument('base_image_path', metavar='img', type=str,
                    help='Path to image to transform.')
parser.add_argument('weights_path', metavar='wts', type=str,
                    help='Trained weights of the feedforward network.')

args = parser.parse_args()
base_image_path = args.base_image_path
weights_path = args.weights_path

width, height = tf.keras.preprocessing.image.load_img(base_image_path).size
target_h = 256
target_w = int(math.ceil(width / height) * target_h)

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(target_h, target_w), interpolation="bicubic")
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return tf.keras.applications.vgg19.preprocess_input(img)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x = x.numpy().reshape((target_h, target_w, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')

ff = feedforward.make_network(scale=16)
ff.load_weights(weights_path)
img = preprocess_image(base_image_path)

ff(img) # trigger tf to load dynamic libraries for more accurate timing

start = time.time()
res = deprocess_image(ff(img))
end = time.time()
print("Applied style in %d ms" % (int((end-start)*1000)))

plt.subplots(2,1)
plt.subplot(211)
plt.title("Original")
plt.imshow(tf.keras.preprocessing.image.load_img(base_image_path, target_size=(target_h, target_w)))
plt.axis("off")

plt.subplot(212)
plt.title("Style applied")
plt.imshow(res)
plt.axis("off")

plt.show()