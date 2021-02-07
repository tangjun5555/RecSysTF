# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/25 5:59 下午
# desc:

import tensorflow as tf
from collections import namedtuple
from recsystf.utils.variable_util import get_normal_variable

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

DNNConfig = namedtuple(
    typename="DNNConfig",
    field_names=["name", "hidden_units", "use_bias", "activation", "dropout_ratio", "use_bn", "is_training"],
)


class DNN(object):
    """
    Multi-Layer Fully Connected Layer
    """

    def __init__(self, name, hidden_units,
                 use_bias=True,
                 activation=tf.nn.relu,
                 dropout_ratio=None,
                 use_bn=False, is_training=False,
                 ):
        self.name = name
        self.hidden_units = hidden_units
        self.use_bias = use_bias
        self.activation = activation
        self.dropout_ratio = dropout_ratio
        self.use_bn = use_bn
        self.is_training = is_training

    def __call__(self, deep_fea):
        hidden_units = [deep_fea.get_shape().as_list()[-1]] + self.hidden_units
        self.kernels = [get_normal_variable("dnn_kernels", self.name, (hidden_units[i], hidden_units[i + 1]))
                        for i in range(len(self.hidden_units))]
        self.biases = [get_normal_variable("dnn_biases", self.name, (v,)) for v in self.hidden_units]
        for i, unit in enumerate(self.hidden_units):
            # deep_fea = tf.layers.dense(
            #     inputs=deep_fea,
            #     units=unit,
            #     use_bias=self.use_bias,
            #     name="%s/dnn_%d" % (self.name, i),
            # )
            deep_fea = tf.nn.bias_add(
                value=tf.tensordot(deep_fea, self.kernels[i], axes=(-1, 0)),
                bias=self.biases[i],
            )

            if self.use_bn:
                deep_fea = tf.layers.batch_normalization(
                    deep_fea,
                    training=self.is_training,
                    trainable=True,
                    name="%s/dnn_%d/bn" % (self.name, i),
                )

            if self.activation:
                deep_fea = self.activation(deep_fea, name="%s/dnn_%d/activation" % (self.name, i))

            if self.is_training and self.dropout_ratio and isinstance(self.dropout_ratio, float):
                assert 0.0 < self.dropout_ratio < 1.0, "invalid dropout_ratio: %.3f" % self.dropout_ratio
                deep_fea = tf.nn.dropout(
                    deep_fea,
                    keep_prob=1 - self.dropout_ratio,
                    name="%s/%d/dropout" % (self.name, i),
                )
            if self.is_training and self.dropout_ratio and isinstance(self.dropout_ratio, list):
                assert self.dropout_ratio[i] < 1, "invalid dropout_ratio: %.3f" % self.dropout_ratio[i]
                deep_fea = tf.nn.dropout(
                    deep_fea,
                    keep_prob=1 - self.dropout_ratio[i],
                    name="%s/%d/dropout" % (self.name, i),
                )

        return deep_fea
