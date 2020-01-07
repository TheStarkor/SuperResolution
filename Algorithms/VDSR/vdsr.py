import tensorflow as tf
import numpy as np

import os, time
from tqdm import tqdm

import h5py

# from utils import *
# from dataset import Dataset

class VDSR():
    def __init__(self, config):
        # Network setting
        self.layer_depth = config.layer_depth

        # Learning schedule
        self.epoch = config.epoch
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size

        # Log interval
        self.PRINT_INTERVAL = config.print_interval
        self.EVAL_INTERVAL = config.eval_interval

        # Upscale factor, only availale in test stage
        self.scale = config.scale

        # Others
        self.CHECKPOINT_PATH = config.checkpoint_path
        self.MODEL_PATH = os.path.join(config.model_path, 'model')
        self.TRAIN_DATASET_PATH = config.train_dataset_path
        self.TRAIN_DATASET = config.train_dataset
        self.VALID_DATASET = config.valid_dataset
        self.TEST_DATASET_PATH = config.test_dataset_path
        self.TEST_DATASET = config.test_dataset
        self.RESULT_DIR = config.result_dir

        # Build network computational graph
        self.build_network()

    # ???????????
    # Build network architecture
    def build_network(self):
        initializer_w = tf.initializers.he_normal()
        initializer_b = tf.constant_initializer(0)

        with tf.variable_scope('VDSR'):
            self.weights_t = {
                'w1': tf.get_variable('w1', [3, 3, 1, 64], initializer=initializer_w, dtype=tf.float32),
                'w{:d}'.format(self.layer_depth): tf.get_variable
            }