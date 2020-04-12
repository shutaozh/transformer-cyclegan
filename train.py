import tensorflow as tf
from model import CycleGAN
from datetime import datetime
import os
import logging
from tensorflow.contrib import learn
import numpy as np
import data_helpers
import utils
from hyperparams import Hyperparams as hp


def train():
    if not os.path.exists(hp.checkpoint_dir):
        os.mkdir(hp.checkpoint_dir)

    x, y = data_helpers.load_data(hp.source_train)
    # 建立词汇表
    max_document_length = hp.maxlen
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    vocab_processor.fit_transform(['<s>', '<e>'] + x + y)

    x_train = np.array(list(vocab_processor.fit_transform(x)))
    y_train = np.array(list(vocab_processor.fit_transform(y)))

    vocab_size = len(vocab_processor.vocabulary_)
    vocab_processor.save(os.path.join(hp.checkpoint_dir, 'vocab'))

    # 生成batches
    batches = data_helpers.batch_iter(
        list(zip(x_train, y_train)), hp.batch_size, hp.num_epochs)

    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(
            vocab_size=vocab_size,
            LAMBDA=10,
            is_training=True)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            gs = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)

                # train
                for i in range(5):
                    _, D_Y_loss = (sess.run([cycle_gan.D_Y_opt, cycle_gan.D_Y_loss],
                                            feed_dict={cycle_gan.x: x_batch,
                                                       cycle_gan.y: y_batch}
                                            )
                                   )
                _, G_loss, fake_y_val = (sess.run([cycle_gan.G_opt, cycle_gan.G_loss, cycle_gan.preds_y],
                                                  feed_dict={cycle_gan.x: x_batch,
                                                             cycle_gan.y: y_batch}
                                                  )
                                         )

                for i in range(5):
                    _, D_X_loss = (sess.run([cycle_gan.D_X_opt, cycle_gan.D_X_loss],
                                            feed_dict={cycle_gan.x: x_batch,
                                                       cycle_gan.y: y_batch}
                                            )
                                   )
                _, F_loss, fake_x_val = (sess.run([cycle_gan.F_opt, cycle_gan.F_loss, cycle_gan.preds_x],
                                                  feed_dict={cycle_gan.x: x_batch,
                                                             cycle_gan.y: y_batch}
                                                  )
                                         )

                if gs % 100 == 0:
                    print('********step: {}, DY_loss = {:.8f}, G_loss = {:.8f}, '
                          'DX_loss = {:.8f}, F_loss = {:.8f}********'.format(gs, D_Y_loss, G_loss, D_X_loss, F_loss))

                if gs % 100 == 0:
                    saver.save(sess, hp.checkpoint_dir + "/model.ckpt", global_step=gs)

                gs += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, hp.checkpoint_dir + "/model.ckpt", global_step=gs)
            logging.info("Model saved in file: %s" % save_path)
            coord.request_stop()
            coord.join(threads)


def main(unused_argv):
    train()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
