# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/3/23 4:44 下午
# desc:

import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class DINAttentionLayer(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, input_value):
        """
        input_value shape
            - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

        output_value shape
            - 2D tensor with shape: ``(batch_size, embedding_size)``.
                """
        pass


# class
