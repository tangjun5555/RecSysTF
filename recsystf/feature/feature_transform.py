# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/2/8 12:31 下午
# desc:

import logging
import tensorflow as tf
from recsystf.utils.variable_util import get_embedding_variable
from recsystf.feature.feature_type import EmbeddingFeature

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def embedding_feature_to_vector(scope, feature_values, feature_config: EmbeddingFeature):
    embedding_matrix = get_embedding_variable(scope, feature_config.embedding_name, feature_config.vocab_size + 1,
                                              feature_config.embedding_size)

    seq = feature_values + 1
    keys_length = tf.where(tf.equal(seq, 0), tf.zeros_like(seq), tf.ones_like(seq))
    keys_length = tf.reduce_sum(keys_length, axis=1, keep_dims=False)

    key_masks = tf.sequence_mask(keys_length, feature_config.length)
    key_masks = tf.expand_dims(key_masks, -1)
    key_masks = tf.concat([key_masks] * feature_config.embedding_size, axis=-1)

    embedding_value = tf.nn.embedding_lookup(embedding_matrix, seq)
    paddings = tf.zeros_like(embedding_value)
    embedding_value = tf.where(key_masks, embedding_value, paddings)

    logging.info(
        "embedding_feature_to_vector, %s embedding_matrix.shape:%s" %
        (feature_config.feature_name, embedding_matrix.shape)
    )
    logging.info(
        "embedding_feature_to_vector, %s embedding_value.shape:%s" %
        (feature_config.feature_name, embedding_value.shape)
    )
    if feature_config.pooling == "mean":
        real_keys_length = tf.where(tf.equal(keys_length, 0), tf.ones_like(keys_length), keys_length)
        return tf.div(x=tf.reduce_sum(embedding_value, axis=1, keepdims=False),
                      y=tf.cast(tf.expand_dims(real_keys_length, axis=-1), tf.dtypes.float32),
                      )
    elif feature_config.pooling == "sum":
        return tf.reduce_sum(embedding_value, axis=1, keepdims=False)
    elif feature_config.pooling == "max":
        return tf.reduce_max(embedding_value, axis=1, keepdims=False)
    else:
        raise Exception("embedding_feature_to_vector, feature:%s pooling isn't supported" % feature_config.feature_name)
