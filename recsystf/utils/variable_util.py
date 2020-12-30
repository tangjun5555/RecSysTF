# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/29 5:24 下午
# desc:

import math
import tensorflow as tf


def get_embedding_variable(scope, name, vocab_size, embed_size):
    """
    获取embedding矩阵变量，便于共享
    :param scope:
    :param name:
    :param vocab_size:
    :param embed_size:
    :return:
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        embedding_variable = tf.get_variable(
            name=name,
            initializer=tf.truncated_normal([vocab_size, embed_size], stddev=1.0 / math.sqrt(embed_size)),
            dtype=tf.float32,
        )
    return embedding_variable
