# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/1/7 3:48 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def l2_norm(input_net):
    """
    L2标准化
    :param input_net:
    :return:
    """
    norm = tf.norm(input_net, axis=-1, keepdims=True)
    return tf.python.math_ops.div(input_net, tf.maximum(norm, 1e-12))
