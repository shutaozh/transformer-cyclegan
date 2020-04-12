# -*- coding: utf-8 -*-
# @Time    : 2019/10/28 15:17
# @Author  : shutaozhang
import data_helpers
from hyperparams import Hyperparams as hp
import utils
import tensorflow as tf
from model import CycleGAN
import numpy as np
import codecs
import os
from nltk.translate.bleu_score import corpus_bleu


def eval():
    x, y, label = data_helpers.load_data_label(hp.source_test)
    x_test = utils.text2token(x, hp.checkpoint_dir)
    y_test = utils.text2token(y, hp.checkpoint_dir)

    # 生成batches
    batches = data_helpers.batch_iter(
        list(zip(x_test, y_test, label)), hp.batch_size, num_epochs=1, shuffle=False)

    graph = tf.Graph()
    with graph.as_default():
        cycle_gan = CycleGAN(
            vocab_size=3961,
            LAMBDA=10,
            is_training=True)
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(hp.checkpoint_dir))

        if not os.path.exists('results'):
            os.mkdir('results')
        with codecs.open("results/" + 'fake.txt', "w", "utf-8") as fout:
            list_of_refs, hypotheses = [], []
            i = 1
            for batch in batches:
                x_batch, y_batch, label = zip(*batch)

                preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                for j in range(hp.maxlen):
                    _preds = sess.run(cycle_gan.preds_y, feed_dict={cycle_gan.x: x_batch, cycle_gan.y: preds})
                    preds[:, j] = _preds[:, j]

                for labels, targets, pred in zip(label, y_batch, preds):  # sentence-wise
                    got = utils.token2text(pred, hp.checkpoint_dir)
                    target = utils.token2text(targets, hp.checkpoint_dir)
                    print(str(i))

                    fout.write(got + "," + target + ',' + str(labels) + "\n")
                    fout.flush()
                    i += 1

                    # bleu score
                    ref = target.split()
                    hypothesis = got.split()
                    if len(ref) > 3 and len(hypothesis) > 3:
                        list_of_refs.append([ref])
                        hypotheses.append(hypothesis)

            # Calculate bleu score
            score = corpus_bleu(list_of_refs, hypotheses)
            fout.write("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
    eval()



