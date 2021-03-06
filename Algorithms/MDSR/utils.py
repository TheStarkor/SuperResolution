import math
import multiprocessing
import random

import cv2 as cv
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, add, Lambda
from tensorflow.python.client import device_lib

from config import img_size, max_scale


def custom_loss(y_true, y_pred):
    diff = y_pred - y_true
    return K.mean(K.sqrt(K.square(diff) + K.epsilon))


# getting the number of GPUs
def get_available_gpus():
    local_devices_protos = device_lib.list_local_devices()
    return [x.name for x in local_devices_protos if x.device_type == 'GPU']


# getting the number of CPUs
def get_available_cpus():
    return multiprocessing.cpu_count()


# convolutional residual block
def res_block(input_tensor, features=64, kernel=3):
    x = Conv2D(features, kernel, activation='relu', padding='same')(input_tensor)
    x = Conv2D(features, kernel, padding='same')(x)
    return add([input_tensor, x])


# upscale layer
def upsample(x, scale=2, features=64, kernel=3):
    assert scale in [2, 3, 4]
    
    def upsample_1(x, factor, **kwargs):
        x = Conv2D(features * (factor ** 2), kernel, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2)
    elif scale == 3:
        x = upsample_1(x, 3)
    elif scale == 4:
        x = upsample_1(x, 2)
        x = upsample_1(x, 2)
    
    return x


# shuffle layer
def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)


def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def draw_str(dst, target, s):
    x, y = target
    cv.putText(dst, s, (x + 1, y + 1), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv.LINE_AA)
    cv.putText(dst, s, (x, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv.LINE_AA)


def random_crop(image_bgr, scale):
    full_height, full_width = image_bgr.shape[:2]
    gt_size = img_size * scale
    if full_height < gt_size or full_width < gt_size:
        gt = np.zeros((gt_size, gt_size, 3))
        u = min(full_width, gt_size)
        v = min(full_height, gt_size)
        gt[0:v, 0:u, :] = image_bgr[0:v, 0:u, :]
    else:
        u = random.randint(0, full_width - gt_size)
        v = random.randint(0, full_height - gt_size)
        gt = image_bgr[v:v + gt_size, u:u + gt_size]

    return gt


def preprocess_input(x):
    # subtract the mean RGB value of the ImageNet dataset.
    b_mean = 104.00698793
    g_mean = 116.66876762
    r_mean = 122.67891434
    x = x.astype(np.float32)
    x[:, :, 0] -= b_mean
    x[:, :, 1] -= g_mean
    x[:, :, 2] -= r_mean
    return x


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def get_example_numbers():
    with open('train_names.txt', 'r') as f:
        names = f.read().splitlines()
        num_train_samples = len(names)
    with open('valid_names.txt', 'r') as f:
        names = f.read().splitlines()
        num_valid_samples = len(names)
    return num_train_samples, num_valid_samples