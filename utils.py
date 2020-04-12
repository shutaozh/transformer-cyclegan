import tensorflow as tf
import random
import numpy as np
import os
import pickle as pkl
import logging
from tensorflow.contrib import learn
from datetime import datetime
from tensorflow.python import pywrap_tensorflow
from hyperparams import Hyperparams as hp


def token2text(token, checkpoints_dir):
    vocab_path = os.path.join(checkpoints_dir, 'vocab')
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    vocab_dict = vocab_processor.vocabulary_._mapping
    text = []
    for i in token:
        text.append(list(vocab_dict.keys())[list(vocab_dict.values()).index(i)])

    return ''.join(text).split("e")[0].strip()


def text2token(text, checkpoints_dir):
    vocab_path = os.path.join(checkpoints_dir, 'vocab')
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    token = np.array(list(vocab_processor.fit_transform(text)))
    return token


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1],
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def gradient_penalty(D, fake_y, y):
    alpha = tf.random_uniform(
        shape=[hp.batch_size, 1, 1],
        minval=0.,
        maxval=1.
    )
    differences = y - fake_y
    interpolates = y + alpha * differences
    gradients = tf.gradients(D(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    return gradient_penalty

