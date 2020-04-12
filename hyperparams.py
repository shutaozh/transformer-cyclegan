# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''


class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'data/train.txt'
    target_train = 'data/train_y'
    source_test = 'data/test.txt'
    target_test = 'corpora/cleaned_ham_5000.utf8'

    # training
    batch_size = 32  # alias = N
    D_learning_rate = 0.0005
    G_learning_rate = 0.0001
    checkpoint_dir = 'checkpoints'  # log directory

    # model
    maxlen = 15  # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    hidden_units = 512
    num_blocks = 6  # number of encoder/decoder blocks
    num_epochs = 30
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.




