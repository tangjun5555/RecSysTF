# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/2/23 12:00 下午
# desc:

import tensorflow as tf

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


class CrossLayer(object):
    """

    """

    def __init__(self, name):
        self.name = name

    def __call__(self, input_value):
        pass


class InnerProductLayer(object):
    """

    """
    def __init__(self, name):
        self.name = name

    def __call__(self, input_value):
        pass
