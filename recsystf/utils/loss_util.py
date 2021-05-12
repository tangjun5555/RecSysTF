# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/4/26 6:14 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def compute_focal_loss(labels, predictions, alpha=0.25, gamma=2.0):
    zeros = tf.zeros_like(predictions, dtype=predictions.dtype)
    pos_corr = tf.where(labels > zeros, 1.0 - predictions, zeros)
    neg_corr = tf.where(labels > zeros, zeros, predictions)
    loss = -alpha * (pos_corr ** gamma) * tf.log(predictions) \
           - (1 - alpha) * (neg_corr ** gamma) * tf.log(1.0 - predictions)
    return tf.reduce_mean(loss)
