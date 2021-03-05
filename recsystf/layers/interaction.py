# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/2/23 12:00 下午
# desc:

import tensorflow as tf
from recsystf.utils.variable_util import get_normal_variable

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class FMLayer(object):
    """
    Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, input_value):
        """
        input_value shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.

        output_value shape
        - 2D tensor with shape: ``(batch_size, 1)``.
        """
        input_value_shape = input_value.get_shape().as_list()
        if len(input_value_shape) != 3:
            raise ValueError("Unexpected input_value dimensions %d, expect to be 3 dimensions" % (str(input_value)))

        square_of_sum = tf.square(tf.reduce_sum(input_value, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(input_value * input_value, axis=1, keepdims=True)
        output_value = square_of_sum - sum_of_square
        output_value = 0.5 * tf.reduce_sum(output_value, axis=2, keepdims=False)
        return output_value


class CrossLayer(object):
    """
    The Cross Network part of Deep&Cross Network model, which leans both low and high degree cross feature.
    """

    def __init__(self, name, cross_network_layer_size):
        self.name = name
        self.cross_network_layer_size = cross_network_layer_size

    def __call__(self, input_value):
        """
        input_value shape
          - 2D tensor with shape: ``(batch_size, dim)``
        output_value shape
          - 2D tensor with shape: ``(batch_size, dim)``
        """
        input_dim = input_value.get_shape().as_list()[1]
        kernels = [
            get_normal_variable("CrossLayer", "kernel_" + str(i), [input_dim])
            for i in range(self.cross_network_layer_size)
        ]
        bias = [
            get_normal_variable("CrossLayer", "bias_" + str(i), [input_dim])
            for i in range(self.cross_network_layer_size)
        ]
        x_0 = input_value
        x_l = x_0
        for i in range(self.cross_network_layer_size):
            x_b = tf.tensordot(a=tf.reshape(x_l, [-1, 1, input_dim]), b=kernels[i], axes=1)
            x_l = x_0 * x_b + bias[i] + x_l
        return x_l


class AFMLayer(object):
    """
    Attentonal Factorization Machine models pairwise (order-2) feature interactions without linear term and bias.
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, input_value):
        """
        input_value shape
          - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

        output_value shape
          - 2D tensor with shape: ``(batch_size, 1)``.
        """
        input_value_shape = input_value.get_shape().as_list()


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
