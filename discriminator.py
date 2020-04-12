import tensorflow as tf
from utils import *
from hyperparams import Hyperparams as hp
from modules import *


class Discriminator:
    def __init__(self, name, is_training):
        self.name = name
        self.is_training = is_training

    def __call__(self, x):
        """
        Args:
          input: batch_size x image_size x image_size x 3
        Returns:
          output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                  filled with 0.9 if real, 0.0 if fake
        """

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            with tf.variable_scope('dense_1'):
                h = tf.layers.dense(x, hp.hidden_units)
            h = tf.reshape(h, [-1, hp.maxlen, hp.hidden_units, 1])
            with tf.variable_scope('conv2d_2_1'):
                hc1 = tf.layers.conv2d(
                    h,
                    filters=16,
                    kernel_size=[2, hp.hidden_units],
                    strides=[1, 1],
                    padding='SAME',
                    kernel_initializer=tf.contrib.layers.xavier_initializer()
                )
                hc1 = tf.nn.relu(hc1)

            with tf.variable_scope('conv2d_2_2'):
                hc2 = tf.layers.conv2d(
                    h,
                    filters=16,
                    kernel_size=[3, hp.hidden_units],
                    strides=[1, 1],
                    padding='SAME',
                    kernel_initializer=tf.contrib.layers.xavier_initializer()
                )
                hc2 = tf.nn.relu(hc2)

            with tf.variable_scope('conv2d_2_3'):
                hc3 = tf.layers.conv2d(
                    h,
                    filters=16,
                    kernel_size=[4, hp.hidden_units],
                    strides=[1, 1],
                    padding='SAME',
                    kernel_initializer=tf.contrib.layers.xavier_initializer()
                )
                hc3 = tf.nn.relu(hc3)
            with tf.variable_scope('conv2d_2_4'):
                hc4 = tf.layers.conv2d(
                    h,
                    filters=16,
                    kernel_size=[5, hp.hidden_units],
                    strides=[1, 1],
                    padding='SAME',
                    kernel_initializer=tf.contrib.layers.xavier_initializer()
                )
                hc4 = tf.nn.relu(hc4)

            h = tf.concat([hc1, hc2, hc3, hc4], 2)

            with tf.variable_scope('max_pooling_3'):
                h = tf.layers.max_pooling2d(
                    h,
                    pool_size=[hp.maxlen, 1],
                    strides=1
                )

            with tf.variable_scope('dense_4'):
                h = tf.layers.dense(h, 32)
                h = tf.nn.sigmoid(h)
            with tf.variable_scope('dense_5'):
                h = tf.layers.dense(h, 1)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return h

