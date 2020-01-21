import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import feedforward

# Workaround for tf 2.0 issue
# https://stackoverflow.com/a/58684421
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser(description='Train a style transfer network with tf.keras')
parser.add_argument('base_image_path', metavar='img', type=str,
                    help='Path to image to transform.')

args = parser.parse_args()
base_image_path = args.base_image_path

width, height = tf.keras.preprocessing.image.load_img(base_image_path).size
target_h = 500
target_w = int((width / height) * target_h)

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(target_h, target_w))
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

ff = feedforward.make_network()
ff.load_weights("checkpoints/3000")
img = preprocess_image(base_image_path)
res = deprocess_image(ff(img))

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