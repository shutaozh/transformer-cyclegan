import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import os
from modules import *
from hyperparams import Hyperparams as hp


class Generator:
    def __init__(self, name, is_training, vocab_size):
        self.name = name
        self.reuse = False
        self.vocab_size = vocab_size
        self.is_training = is_training

    def __call__(self, x, y):
        """
        Args:
          input: batch_size x width x height x 3
        Returns:
          output: same size as input
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # define decoder inputs
            decoder_inputs = tf.concat((tf.ones_like(y[:, :1]) * 1, y[:, :-1]), -1)  # 2:<S>

            # Encoder
            with tf.variable_scope("encoder"):
                # Embedding
                enc = embedding(x,
                                vocab_size=self.vocab_size,
                                num_units=hp.hidden_units,
                                scale=True,
                                scope="enc_embed")

                # Positional Encoding
                enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(x)[1]), 0), [tf.shape(x)[0], 1]),
                                 vocab_size=hp.maxlen,
                                 num_units=hp.hidden_units,
                                 zero_pad=False,
                                 scale=False,
                                 scope="enc_pe")

                # Dropout
                enc = tf.layers.dropout(enc,
                                        rate=hp.dropout_rate,
                                        training=tf.convert_to_tensor(self.is_training))

                # Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        # Multihead Attention
                        enc = multihead_attention(queries=enc,
                                                  keys=enc,
                                                  num_units=hp.hidden_units,
                                                  num_heads=hp.num_heads,
                                                  dropout_rate=hp.dropout_rate,
                                                  is_training=self.is_training,
                                                  causality=False)

                        # Feed Forward
                        enc = feedforward(enc, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Decoder
            with tf.variable_scope("decoder"):
                # Embedding
                dec = embedding(decoder_inputs,
                                vocab_size=self.vocab_size,
                                num_units=hp.hidden_units,
                                scale=True,
                                scope="dec_embed")

                # Positional Encoding
                dec += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(decoder_inputs)[1]), 0), [tf.shape(decoder_inputs)[0], 1]),
                    vocab_size=hp.maxlen,
                    num_units=hp.hidden_units,
                    zero_pad=False,
                    scale=False,
                    scope="dec_pe")

                # Dropout
                dec = tf.layers.dropout(dec,
                                        rate=hp.dropout_rate,
                                        training=tf.convert_to_tensor(self.is_training))

                # Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        # Multihead Attention (self-attention)
                        dec = multihead_attention(queries=dec,
                                                  keys=dec,
                                                  num_units=hp.hidden_units,
                                                  num_heads=hp.num_heads,
                                                  dropout_rate=hp.dropout_rate,
                                                  is_training=self.is_training,
                                                  causality=True,
                                                  scope="self_attention")

                        # Multihead Attention (vanilla attention)
                        dec = multihead_attention(queries=dec,
                                                  keys=enc,
                                                  num_units=hp.hidden_units,
                                                  num_heads=hp.num_heads,
                                                  dropout_rate=hp.dropout_rate,
                                                  is_training=self.is_training,
                                                  causality=False,
                                                  scope="vanilla_attention")

                        # Feed Forward
                        dec = feedforward(dec, num_units=[4 * hp.hidden_units, hp.hidden_units])
            logits = tf.layers.dense(dec, self.vocab_size)
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return logits

