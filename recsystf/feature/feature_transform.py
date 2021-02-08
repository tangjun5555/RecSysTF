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
    embedding_matrix = get_embedding_variable(scope, feature_config.name, feature_config.vocab_size,
                                              feature_config.embedding_size)
    embedding_value = tf.nn.embedding_lookup(embedding_matrix, feature_values)
    logging.info(
        "embedding_feature_to_vector, %s embedding_matrix.shape:%s" %
        (feature_config.name, embedding_matrix.shape)
    )
    logging.info(
        "embedding_feature_to_vector, %s embedding_value.shape:%s" %
        (feature_config.name, embedding_value.shape)
    )
    if feature_config.pooling == "mean":
        return tf.reduce_mean(embedding_value, axis=1, keepdims=False)
    elif feature_config.pooling == "sum":
        return tf.reduce_sum(embedding_value, axis=1, keepdims=False)
    elif feature_config.pooling == "max":
        return tf.reduce_max(embedding_value, axis=1, keepdims=False)
    else:
        raise Exception("embedding_feature_to_vector, feature:%s pooling isn't supported" % feature_config.name)
