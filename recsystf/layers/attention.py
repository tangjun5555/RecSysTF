# -*- coding: utf-8 -*-
# author: 唐俊 tangjun0612@shihuo.cn
# time: 2021/1/13 10:58 上午
# desc:

import math
import tensorflow as tf
from recsystf.utils.normalization_ops_util import l2_norm
from recsystf.layers.dnn import DNN

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def dot_product_attention_function(query_embed, seq_embed):
    """
    点积
    :param query_embed [B, H], 候选物品embedding向量
    :param seq_embed [B, T, H], 用户行为序列embedding矩阵
    """
    seq_max_length = seq_embed.get_shape().as_list()[1]
    embed_size = seq_embed.get_shape().as_list()[-1]

    query_embed = tf.tile(query_embed, [1, seq_max_length])
    query_embed = tf.reshape(query_embed, [-1, seq_max_length, embed_size])

    attention_score = tf.multiply(seq_embed, query_embed)
    attention_score = tf.reduce_sum(attention_score, axis=2, keepdims=False)

    return tf.reshape(attention_score, [-1, 1, seq_max_length])


def scaled_dot_product_attention_function(query_embed, seq_embed):
    """
    缩放点积
    :param query_embed [B, H], 候选物品embedding向量
    :param seq_embed [B, T, H], 用户行为序列embedding矩阵
    """
    seq_max_length = seq_embed.get_shape().as_list()[1]
    embed_size = seq_embed.get_shape().as_list()[-1]

    query_embed = tf.tile(query_embed, [1, seq_max_length])
    query_embed = tf.reshape(query_embed, [-1, seq_max_length, embed_size])

    attention_score = tf.multiply(seq_embed, query_embed)
    attention_score = tf.reduce_sum(attention_score, axis=2, keepdims=False)
    attention_score = attention_score / math.sqrt(embed_size)

    return tf.reshape(attention_score, [-1, 1, seq_max_length])


def consine_attention_function(query_embed, seq_embed):
    """
    缩放点积
    :param query_embed [B, H], 候选物品embedding向量
    :param seq_embed [B, T, H], 用户行为序列embedding矩阵
    """
    seq_max_length = seq_embed.get_shape().as_list()[1]
    embed_size = seq_embed.get_shape().as_list()[-1]

    query_embed = tf.tile(query_embed, [1, seq_max_length])
    query_embed = tf.reshape(query_embed, [-1, seq_max_length, embed_size])

    query_embed = l2_norm(query_embed)
    seq_embed = l2_norm(seq_embed)

    attention_score = tf.multiply(seq_embed, query_embed)
    attention_score = tf.reduce_sum(attention_score, axis=2, keepdims=False)

    return tf.reshape(attention_score, [-1, 1, seq_max_length])


def mlp_attention_function(query_embed, seq_embed, dnn_name, dnn_units=(32, 16)):
    """
    缩放点积
    :param query_embed [B, H], 候选物品embedding向量
    :param seq_embed [B, T, H], 用户行为序列embedding矩阵
    :param dnn_name
    :param dnn_units
    """
    seq_max_length = seq_embed.get_shape().as_list()[1]
    embed_size = seq_embed.get_shape().as_list()[-1]

    query_embed = tf.tile(query_embed, [1, seq_max_length])
    query_embed = tf.reshape(query_embed, [-1, seq_max_length, embed_size])

    attention_input = tf.concat([query_embed, seq_embed], axis=-1)
    attention_dnn = DNN(name=dnn_name, hidden_units=dnn_units, activation=tf.nn.relu, use_bias=True)
    attention_score = attention_dnn(attention_input)
    attention_score = tf.layers.dense(attention_score, 1, name=dnn_name + "_latest")

    return tf.reshape(attention_score, [-1, 1, seq_max_length])


class NormalAttention(object):
    def __init__(self, name, attention_function="mlp"):
        self.name = name
        self.attention_function = attention_function

    def __call__(self, query_embed, seq_embed, seq_real_length):
        seq_max_length = seq_embed.get_shape().as_list()[1]

        # Score
        if "mlp" == self.attention_function:
            attention_score = mlp_attention_function(query_embed, seq_embed, self.name + "_mlp")
        elif "cosine" == self.attention_function:
            attention_score = consine_attention_function(query_embed, seq_embed)
        elif "scaled_dot_product" == self.attention_function:
            attention_score = scaled_dot_product_attention_function(query_embed, seq_embed)
        else:
            attention_score = dot_product_attention_function(query_embed, seq_embed)

        # Mask
        key_masks = tf.sequence_mask(seq_real_length, seq_max_length)
        key_masks = tf.expand_dims(key_masks, 1)
        paddings = tf.ones_like(attention_score) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, attention_score, paddings)

        # Softmax
        outputs = tf.nn.softmax(outputs)

        # Weighted sum
        outputs = tf.matmul(outputs, seq_embed)
        outputs = tf.reshape(outputs, [-1, outputs.get_shape().as_list()[-1]])

        return outputs


class SelfAttention(object):
    def __init__(self, name):
        self.name = name
        self.attention_function = "scaled_dot_product"

    def __call__(self, seq_embed, seq_real_length):
        """
        TODO 使用矩阵乘法高效计算
        :param seq_embed [B, T, H], 用户行为序列embedding矩阵
        """
        seq_max_length = seq_embed.get_shape().as_list()[1]
        attention_outputs = []
        for i in range(seq_max_length):
            query_embed = tf.slice(seq_embed, [0, i, 0], [-1, 1, -1])
            normal_attention = NormalAttention(
                name="%s_attention_%d" % (self.name, i),
                attention_function=self.attention_function
            )
            attention_outputs.append(normal_attention(query_embed, seq_embed, seq_real_length))
        attention_outputs = tf.stack(attention_outputs, axis=1)
        return attention_outputs


class MultiHeadAttention(object):
    """
    多个 Self-Attention 的组合
    """

    def __init__(self, name, head_num=4):
        self.name = name
        self.head_num = head_num

    def __call__(self, seq_embed, seq_real_length):
        attention_outputs = []
        for i in range(self.head_num):
            self_attention = SelfAttention(name="%s_self_att_%d" % (self.name, i))
            attention_outputs.append(
                self_attention(seq_embed, seq_real_length)
            )
        attention_outputs = tf.concat(attention_outputs, axis=2)
        return attention_outputs
