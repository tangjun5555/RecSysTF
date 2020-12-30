# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/21 22:42
# desc:

import numpy as np
import tensorflow as tf


def test_04():
    t1 = np.reshape(np.array(list(range(20))), (2, 2, 5))
    t2 = np.reshape(np.array([1] * 20), (2, 5, 2))
    t3 = np.matmul(t1, t2)
    print(t1)
    print(t2)
    print(t3)
    print(t3.shape)


def test_03():
    tf.enable_eager_execution()
    query_embed = tf.constant(
        value=[
            [1.0, 2.0, 3.0],
            [11.0, 12.0, 13.0]
        ],
        dtype=tf.float32,
    )
    query_embed = tf.tile(query_embed, [1, 5])
    print(query_embed)


def test_02():
    tf.enable_eager_execution()
    t1 = tf.constant(
        value=[
            [1, 2, -1, -1],
            [1, 4, 5, 10]
        ],
        dtype=tf.int64,
    )
    t2 = tf.expand_dims(t1, axis=1)
    t3 = tf.reshape(t2, [-1, t2.get_shape().as_list()[-1]])
    print(t2)
    print(t3)


def test_01():
    tf.enable_eager_execution()
    t1 = tf.constant(
        value=[
            [1, 2, -1, -1],
            [1, 4, 5, 10]
        ],
        dtype=tf.int64,
    )
    t2 = tf.zeros_like(t1)
    t3 = tf.ones_like(t1)
    r1 = tf.where(tf.equal(t1, -1), t2, t3)
    r2 = tf.reduce_sum(r1, axis=1, keep_dims=False)
    print(r1)
    print(r2)
