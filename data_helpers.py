import numpy as np
import re
import os

import pandas as pd


def load_data(data_file):
    """
    1. 加载所有数据和标签
    2. 可以进行多分类，每个类别的数据单独放在一个文件中
    3. 保存处理后的数据
    """
    x_text = []
    y_text = []
    lines = open(data_file, 'r', encoding='utf-8').readlines()
    for line in lines:
        text = line.strip('\n').split('\t')
        if int(text[2]) == 1:
            x_text.append(clean_str(seperate_line(text[0])) + ' e')
            y_text.append(clean_str(seperate_line(text[1])) + ' e')

    return x_text, y_text


def load_data_label(data_file):
    """
    1. 加载所有数据和标签
    2. 可以进行多分类，每个类别的数据单独放在一个文件中
    3. 保存处理后的数据
    """
    x_text = []
    y_text = []
    label = []
    lines = open(data_file, 'r', encoding='utf-8').readlines()
    for line in lines:
        text = line.strip('\n').split('\t')
        x_text.append(clean_str(seperate_line(text[0])) + ' e')
        y_text.append(clean_str(seperate_line(text[1])) + ' e')
        label.append(text[2])

    return x_text, y_text, label


def clean_str(string):
    """
    1. 将除汉字外的字符转为一个空格
    2. 将连续的多个空格转为一个空格
    3. 除去句子前后的空格字符
    """
    string = re.sub(r'[^\u4e00-\u9fff]', ' ', string)
    string = re.sub(r'\s{2,}', ' ', string)
    return string.strip()


def seperate_line(line):
    """
    将句子中的每个字用空格分隔开
    """
    return ''.join([word + ' ' for word in line])


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    生成一个batch迭代器
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]

