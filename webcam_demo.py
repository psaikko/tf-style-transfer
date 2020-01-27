import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import feedforward
import math
import time
import cv2

# Workaround for tf 2.0 issue
# https://stackoverflow.com/a/58684421
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

parser = argparse.ArgumentParser(description='Apply style to webcam stream with pre-trained network')
parser.add_argument('weights_path', metavar='wts', type=str,
                    help='Trained weights of the feedforward network.')

args = parser.parse_args()
weights_path = args.weights_path

target_h = 300
target_w = 400

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(np_image):
    img = cv2.resize(np_image, dsize=(target_w, target_h), interpolation=cv2.INTER_CUBIC)
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
    vmax = np.max(x)
    vmin = np.min(x)
    x -= vmin
    x /= (vmax - vmin)
    x *= 255
    return x.astype('uint8')

ff = feedforward.make_network(scale=16)
ff.load_weights(weights_path)

plt.subplots(2,1)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break

    img = preprocess_image(frame)

    res = deprocess_image(ff(img))

    plt.subplot(211)
    plt.cla()
    plt.title("Original")
    plt.imshow(frame)
    plt.axis("off")

    plt.subplot(212)
    plt.cla()
    plt.title("Style applied")
    plt.imshow(res)
    plt.axis("off")

    plt.pause(0.001)
    # plt.show()
    # break