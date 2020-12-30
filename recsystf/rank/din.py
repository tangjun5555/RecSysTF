# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:47 下午
# desc:

import time
from collections import namedtuple
from typing import List
import tensorflow as tf
from recsystf.utils.variable_util import get_embedding_variable
from tensorflow.python.estimator.canned import optimizers

AttentionSequenceFeature = namedtuple(
    typename="AttentionSequenceFeature",
    field_names=["group_name", "query_name", "seq_name", "vocab_size", "embedding_size"],
)


def compute_dnn_out(input_value, hidden_units):
    output_value = input_value
    for units in hidden_units:
        output_value = tf.layers.dense(output_value, units, activation=tf.nn.sigmoid, use_bias=True)
    return output_value


def attention(query_embed, seq_embed, seq_real_length, attention_dnn_units=(16, 8), need_softmax=False):
    """
      query_embed: [B, H], 候选物品embedding向量
      seq_embed:   [B, T, H], 用户行为序列embedding矩阵
      seq_real_length: [B], 用户历史行为序列的真实有效长度
      attention_dnn_units:
      need_softmax: attention权重归一化
    """
    seq_max_length = seq_embed.get_shape().as_list()[1]
    embed_size = seq_embed.get_shape().as_list()[-1]

    query_embed = tf.tile(query_embed, [1, seq_max_length])
    query_embed = tf.reshape(query_embed, [-1, seq_max_length, embed_size])

    attention_input = tf.concat([query_embed, seq_embed, query_embed - seq_embed, query_embed * seq_embed], axis=-1)
    attention_output = compute_dnn_out(attention_input, attention_dnn_units)
    attention_output = tf.layers.dense(attention_output, 1, use_bias=False)
    attention_output = tf.reshape(attention_output, [-1, 1, seq_max_length])

    # Mask
    key_masks = tf.sequence_mask(seq_real_length, seq_max_length)
    key_masks = tf.expand_dims(key_masks, 1)
    if need_softmax:
        paddings = tf.ones_like(attention_output) * (-2 ** 32 + 1)
    else:
        paddings = tf.zeros_like(attention_output)
    outputs = tf.where(key_masks, attention_output, paddings)

    if need_softmax:
        outputs = tf.nn.softmax(outputs)

    # Weighted sum
    outputs = tf.matmul(outputs, seq_embed)
    outputs = tf.reshape(outputs, [-1, outputs.get_shape().as_list()[-1]])

    return outputs


class DeepInterestNetworkEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,

                 weight_column=None,
                 att_feature_columns: List[AttentionSequenceFeature] = None,
                 dnn_feature_columns=None,

                 dnn_hidden_units=None,

                 optimizer_name="Adam",
                 learning_rate=0.01,
                 ):
        def custom_model_fn(features, labels, mode, params=None, config=None):
            net = tf.feature_column.input_layer(features, feature_columns=dnn_feature_columns)
            tf.logging.info(
                "%s DeepInterestNetworkEstimator custom_model_fn, net.shape:%s" %
                (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    net.shape
                )
            )

            if att_feature_columns:
                att_net = [net]
                for column in att_feature_columns:
                    embedding_variable = get_embedding_variable("attention", column.group_name, column.vocab_size,
                                                                column.embedding_size)
                    query_embed = tf.nn.embedding_lookup(embedding_variable, features[column.query_name])
                    query_embed = tf.reshape(query_embed, [-1, query_embed.get_shape().as_list()[-1]])

                    seq = features[column.seq_name]
                    seq_embed = tf.nn.embedding_lookup(embedding_variable, features[seq])

                    seq_zeros = tf.zeros_like(seq)
                    seq_ones = tf.ones_like(seq)
                    keys_length = tf.where(tf.equal(seq, -1), seq_zeros, seq_ones)
                    keys_length = tf.reduce_sum(keys_length, axis=1, keep_dims=False)

                    att_net.append(attention(query_embed, seq_embed, keys_length))
                net = tf.concat(att_net, axis=1)
                tf.logging.info(
                    "%s DeepInterestNetworkEstimator custom_model_fn, net.shape:%s" %
                    (
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        net.shape
                    )
                )

            logits = compute_dnn_out(net, dnn_hidden_units)
            logits = tf.layers.dense(logits, 1, use_bias=False)
            logits = tf.reshape(logits, (-1,))
            tf.logging.info(
                "%s DeepInterestNetworkEstimator custom_model_fn, logits.shape:%s" %
                (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    logits.shape
                )
            )

            predictions = tf.sigmoid(logits)
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={"predictions": predictions},
                )

            if weight_column:
                loss = tf.losses.log_loss(
                    labels=tf.cast(labels, tf.float32),
                    predictions=predictions,
                    weights=features[weight_column],
                )
            else:
                loss = tf.losses.log_loss(
                    labels=tf.cast(labels, tf.float32),
                    predictions=predictions,
                )

            eval_metric_ops = {
                "auc": tf.metrics.auc(labels, predictions),
            }
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metric_ops=eval_metric_ops,
                )

            assert mode == tf.estimator.ModeKeys.TRAIN
            optimizer_instance = optimizers.get_optimizer_instance(optimizer_name, learning_rate=learning_rate, )
            train_op = optimizer_instance.minimize(
                loss,
                global_step=tf.train.get_global_step(),
            )
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                eval_metric_ops=eval_metric_ops,
            )

        super().__init__(
            model_fn=custom_model_fn,
            model_dir=model_dir,
            config=config,
            params=None,
            warm_start_from=warm_start_from,
        )
        tf.logging.info(
            "[%s] DeepInterestNetworkEstimator init" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
