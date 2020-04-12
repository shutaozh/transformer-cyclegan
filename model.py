import tensorflow as tf
from utils import *
from discriminator import Discriminator
from generator import Generator
from hyperparams import Hyperparams as hp


class CycleGAN:
    def __init__(self, vocab_size, LAMBDA, is_training):
        self.vocab_size = vocab_size
        self.LAMBDA = LAMBDA
        self.is_training = is_training

        self.G = Generator('G', self.is_training, self.vocab_size)
        self.D_Y = Discriminator('D_Y', self.is_training)
        self.F = Generator('F', self.is_training, self.vocab_size)
        self.D_X = Discriminator('D_X', self.is_training)

        self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
        self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))

        self._creat_model()

    def _creat_model(self):
        self.x_ = tf.one_hot(indices=self.x, depth=self.vocab_size)
        self.y_ = tf.one_hot(indices=self.y, depth=self.vocab_size)
        # x -> y
        self.fake_y = self.G(self.x, self.y)

        self.D_Y_real = self.D_Y(self.y_)
        self.D_Y_fake = self.D_Y(self.fake_y)

        self.preds_y = tf.to_int32(tf.arg_max(self.fake_y, dimension=-1))

        # y -> x
        self.fake_x = self.F(self.y, self.x)

        self.D_X_real = self.D_X(self.x_)
        self.D_X_fake = self.D_X(self.fake_x)

        self.preds_x = tf.to_int32(tf.arg_max(self.fake_x, dimension=-1))

        if self.is_training:
            # Loss
            G_loss = self.generator_loss(self.D_Y_fake)
            fake_y = self.G(self.preds_x, self.y)
            G_cycle_loss = self.cycle_loss(self.fake_y, self.y)

            D_Y_loss = self.discriminator_loss(self.D_Y_fake, self.D_Y_real)

            penalty = gradient_penalty(self.D_Y, self.fake_y, self.y_)

            self.D_Y_loss = D_Y_loss + self.LAMBDA * penalty
            self.G_loss = G_cycle_loss
            self.D_Y_opt = tf.train.AdamOptimizer(
                learning_rate=hp.D_learning_rate,
                beta1=0.5,
                beta2=0.9
            ).minimize(self.D_Y_loss, var_list=self.D_Y.variables)
            self.G_opt = tf.train.AdamOptimizer(
                learning_rate=hp.G_learning_rate,
                beta1=0.8,
                beta2=0.98,
                epsilon=1e-8
            ).minimize(self.G_loss, var_list=self.G.variables)

            F_loss = self.generator_loss(self.D_X_fake)
            fake_x = self.F(self.preds_y, self.x)
            F_cycle_loss = self.cycle_loss(self.fake_x, self.x)

            D_X_loss = self.discriminator_loss(self.D_X_fake, self.D_X_real)

            penalty = gradient_penalty(self.D_X, self.fake_x, self.x_)

            self.D_X_loss = D_X_loss + self.LAMBDA * penalty
            self.F_loss = F_cycle_loss
            self.D_X_opt = tf.train.AdamOptimizer(
                learning_rate=hp.D_learning_rate,
                beta1=0.5,
                beta2=0.9
            ).minimize(self.D_X_loss, var_list=self.D_X.variables)
            self.F_opt = tf.train.AdamOptimizer(
                learning_rate=hp.G_learning_rate,
                beta1=0.8,
                beta2=0.98,
                epsilon=1e-8
            ).minimize(self.F_loss, var_list=self.F.variables)

            self.merged = tf.summary.merge_all()

    def generator_loss(self, D_fake):
        gen_loss = -tf.reduce_mean(D_fake)
        # gen_loss = tf.reduce_mean(tf.squared_difference(D_fake, 1.))
        return gen_loss

    def discriminator_loss(self, D_fake, D_real):
        disc_loss = -tf.reduce_mean(D_real) + tf.reduce_mean(D_fake)
        # error_real = tf.reduce_mean(tf.squared_difference(D_real, 1.))
        # error_fake = tf.reduce_mean(tf.square(D_fake))
        # disc_loss = (error_real + error_fake) / 2
        return disc_loss

    def cycle_loss(self, fake, real):
        istarget = tf.to_float(tf.not_equal(real, 0))
        y_smoothed = label_smoothing(tf.one_hot(real, depth=self.vocab_size))
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=fake, labels=y_smoothed)
        cycle_loss = tf.reduce_sum(loss * istarget) / (tf.reduce_sum(istarget))
        return self.LAMBDA * cycle_loss


