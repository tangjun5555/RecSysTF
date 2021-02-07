# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/29 5:24 下午
# desc:

import math
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def get_embedding_variable(scope, name, vocab_size, embed_size):
    """
    获取embedding矩阵变量
    :param scope:
    :param name:
    :param vocab_size:
    :param embed_size:
    :return:
    """
    return get_normal_variable(scope, name, (vocab_size, embed_size))


def get_normal_variable(scope, name, shape):
    """
    获取矩阵变量，便于共享
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        variable = tf.get_variable(
            name=name,
            initializer=tf.truncated_normal(shape=shape, stddev=1.0 / math.sqrt(shape[-1])),
            dtype=tf.float32,
        )
    return variable
