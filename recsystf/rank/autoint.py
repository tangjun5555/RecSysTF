# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/1/13 12:35 下午
# desc:
# Reference: AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks

import logging
import tensorflow as tf
from recsystf.utils.variable_util import get_embedding_variable
from tensorflow.python.estimator.canned import optimizers

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def multihead_attention(queries,
                        keys,
                        values,
                        num_units=None,
                        num_heads=1,
                        has_residual=True,
                        ):
    if num_units is None:
        num_units = queries.get_shape().as_list[-1]

    # Linear projections
    Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
    K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
    V = tf.layers.dense(values, num_units, activation=tf.nn.relu)
    if has_residual:
        V_res = tf.layers.dense(values, num_units, activation=tf.nn.relu)

    # Split and concat
    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

    # Multiplication
    weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

    # Scale
    weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)

    # Activation
    weights = tf.nn.softmax(weights)

    # # Dropouts
    # weights = tf.layers.dropout(weights, rate=1 - dropout_keep_prob,
    #                             training=tf.convert_to_tensor(is_training))

    # Weighted sum
    outputs = tf.matmul(weights, V_)

    # Restore shape
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

    # Residual connection
    if has_residual:
        outputs += V_res

    outputs = tf.nn.relu(outputs)

    # # Normalize
    # outputs = normalize(outputs)

    return outputs


class AutoIntEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,

                 weight_column=None,
                 category_feature_columns=None,
                 numeric_feature_columns=None,

                 embedding_size=32,
                 att_layer_num=3,
                 att_head_num=2,

                 optimizer_name="SGD",
                 learning_rate=0.01,
                 ):
        def custom_model_fn(features, labels, mode, params=None, config=None):
            if category_feature_columns:
                category_fea_value = tf.feature_column.input_layer(
                    features=features,
                    feature_columns=category_feature_columns,
                )
                category_fea_field_num = len(category_feature_columns)
                category_fea_value = tf.reshape(
                    tensor=category_fea_value,
                    shape=(-1, category_fea_field_num, embedding_size),
                )
                logging.info(
                    "AutoIntEstimator custom_model_fn, category_fea_value.shape:%s" %
                    (
                        category_fea_value.shape
                    )
                )
            else:
                category_fea_field_num = 0
                category_fea_value = None

            if numeric_feature_columns:
                numeric_fea_value = tf.feature_column.input_layer(
                    features=features,
                    feature_columns=numeric_feature_columns,
                )
                numeric_fea_field_num = numeric_fea_value.get_shape().as_list()[1]
                numeric_fea_value = tf.reshape(
                    tensor=numeric_fea_value,
                    shape=(-1, numeric_fea_field_num, 1),
                )
                logging.info(
                    "AutoIntEstimator custom_model_fn, numeric_fea_value.shape:%s" %
                    (
                        numeric_fea_value.shape
                    )
                )
                numeric_fea_embedding = get_embedding_variable(
                    scope="feature",
                    name="numeric_fea_embedding",
                    vocab_size=numeric_fea_field_num,
                    embed_size=embedding_size,
                )
                numeric_fea_value = tf.multiply(numeric_fea_value, numeric_fea_embedding)
                logging.info(
                    "AutoIntEstimator custom_model_fn, numeric_fea_value.shape:%s" %
                    (
                        numeric_fea_value.shape
                    )
                )
            else:
                numeric_fea_field_num = 0
                numeric_fea_value = None

            if category_fea_value is not None and numeric_fea_value is not None:
                attention_input = tf.concat([category_fea_value, numeric_fea_value], axis=1)
            elif category_fea_value is not None:
                attention_input = category_fea_value
            elif numeric_fea_value is not None:
                attention_input = numeric_fea_value
            else:
                raise Exception("attention_input is empty.")
            logging.info(
                "AutoIntEstimator custom_model_fn, attention_input.shape:%s" %
                (
                    attention_input.shape
                )
            )

            for i in range(att_layer_num):
                attention_input = multihead_attention(
                    attention_input, attention_input, attention_input,
                    num_heads=att_head_num, has_residual=True,
                )
                logging.info(
                    "AutoIntEstimator custom_model_fn, layer:%d, attention_input.shape:%s" %
                    (
                        i, attention_input.shape
                    )
                )
            attention_output = tf.reshape(
                tensor=attention_input,
                shape=(-1, (category_fea_field_num + numeric_fea_field_num) * embedding_size),
            )
            logging.info(
                "AutoIntEstimator custom_model_fn, attention_output.shape:%s" %
                (
                    attention_output.shape
                )
            )

            logits = tf.layers.dense(attention_output, 1, activation=None, use_bias=True)
            logits = tf.reshape(logits, (-1,))
            logging.info(
                "AutoIntEstimator custom_model_fn, logits.shape:%s" %
                (
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
            optimizer_instance = optimizers.get_optimizer_instance(optimizer_name, learning_rate=learning_rate)
            train_op = optimizer_instance.minimize(loss, global_step=tf.train.get_global_step())
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
        logging.info("AutoIntEstimator init")
