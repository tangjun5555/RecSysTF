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


class DCNCrossLayer(object):
    """
    The Cross Network part of Deep&Cross Network model,
    which leans both low and high degree cross feature.
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
            get_normal_variable("%s_CrossLayer" % self.name, "kernel_" + str(i), [input_dim])
            for i in range(self.cross_network_layer_size)
        ]
        bias = [
            get_normal_variable("%s_CrossLayer" % self.name, "bias_" + str(i), [input_dim])
            for i in range(self.cross_network_layer_size)
        ]
        x_0 = input_value
        x_l = x_0
        for i in range(self.cross_network_layer_size):
            x_b = tf.tensordot(a=tf.reshape(x_l, [-1, 1, input_dim]), b=kernels[i], axes=1)
            x_l = x_0 * x_b + bias[i] + x_l
        return x_l


class PNNInnerProductLayer(object):
    """
    InnerProduct Layer used in PNN,
    which compute the element-wise product or inner product between feature vectors.

    References
        [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.](https://arxiv.org/pdf/1611.00144.pdf)
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, input_value):
        """
        input_value shape
        - 3D tensor with shape: (batch_size, field_size, embedding_size)

        output_value shape
        - 2D tensor with shape: (batch_size, num_pairs)
        """
        field_size = input_value.get_shape().as_list()[1]
        embedding_size = input_value.get_shape().as_list()[-1]
        num_pairs = int(field_size * (field_size - 1) / 2)

        row = []
        col = []
        for i in range(field_size - 1):
            for j in range(i + 1, field_size):
                row.append(i)
                col.append(j)

        # batch_size * pair * embedding_size
        p = tf.transpose(
            # pair * batch_size * embedding_size
            tf.gather(
                # field_size * batch_size * embedding_size
                tf.transpose(input_value, [1, 0, 2]),
                row
            ),
            [1, 0, 2]
        )

        # batch_size * pair * embedding_size
        q = tf.transpose(
            tf.gather(
                # field_size * batch_size * embedding_size
                tf.transpose(input_value, [1, 0, 2]),
                col
            ),
            [1, 0, 2]
        )

        p = tf.reshape(p, [-1, num_pairs, embedding_size])
        q = tf.reshape(q, [-1, num_pairs, embedding_size])

        return tf.reshape(tf.reduce_sum(p * q, [-1]), [-1, num_pairs])


class PNNOuterProductLayer(object):
    """
    OuterProduct Layer used in PNN.

    References
        [Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//Data Mining (ICDM), 2016 IEEE 16th International Conference on. IEEE, 2016: 1149-1154.](https://arxiv.org/pdf/1611.00144.pdf)
    """

    def __init__(self, name, kernel_type="mat"):
        self.name = name
        self.kernel_type = kernel_type

    def __call__(self, input_value):
        """
        input_value shape
            - 3D tensor with shape: (batch_size, field_size, embedding_size)

        output_value shape
            - 2D tensor with shape: (batch_size, num_pairs)
        """
        field_size = input_value.get_shape().as_list()[1]
        embedding_size = input_value.get_shape().as_list()[-1]
        num_pairs = int(field_size * (field_size - 1) / 2)

        if self.kernel_type == "mat":
            kernel = get_normal_variable("%s_OuterProductLayer" % self.name, "kernel",
                                         (embedding_size, num_pairs, embedding_size))
        elif self.kernel_type == "vec":
            kernel = get_normal_variable("%s_OuterProductLayer" % self.name, "kernel", (num_pairs, embedding_size))
        elif self.kernel_type == "num":
            kernel = get_normal_variable("%s_OuterProductLayer" % self.name, "kernel", (num_pairs, 1))
        else:
            raise Exception("OuterProductLayer don't support %s" % self.kernel_type)
        row = []
        col = []
        for i in range(field_size - 1):
            for j in range(i + 1, field_size):
                row.append(i)
                col.append(j)

        # batch_size * pair * embedding_size
        p = tf.transpose(
            # pair * batch_size * embedding_size
            tf.gather(
                # field_size * batch_size * embedding_size
                tf.transpose(input_value, [1, 0, 2]),
                row
            ),
            [1, 0, 2]
        )

        # batch_size * pair * embedding_size
        q = tf.transpose(
            tf.gather(
                # field_size * batch_size * embedding_size
                tf.transpose(input_value, [1, 0, 2]),
                col
            ),
            [1, 0, 2]
        )

        # batch_size * pair * embedding_size
        p = tf.reshape(p, [-1, num_pairs, embedding_size])
        # batch_size * pair * embedding_size
        q = tf.reshape(q, [-1, num_pairs, embedding_size])

        if self.kernel_type == "mat":
            # batch_size * 1 * pair * embedding_size
            p = tf.expand_dims(p, 1)
            # batch_size * pair
            kp = tf.reduce_sum(
                # batch_size * pair * embedding_size
                tf.multiply(
                    # batch_size * pair * embedding_size
                    tf.transpose(
                        # batch_size * embedding_size * pair
                        tf.reduce_sum(
                            # batch_size * embedding_size * pair * embedding_size
                            tf.multiply(p, kernel),
                            -1
                        ),
                        [0, 2, 1]
                    ),
                    q),
                -1
            )
        else:
            # 1 * pair * (k or 1)
            kernel = tf.expand_dims(kernel, 0)
            # batch_size * pair
            kp = tf.reduce_sum(p * q * kernel, -1)

        return kp
