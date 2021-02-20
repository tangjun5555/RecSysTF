# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:47 下午
# desc:
# Reference: Deep & Cross Network for Ad Click Predictions

import logging
import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers
from recsystf.layers.dnn import DNN

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


def cross_interaction(input_value, cross_network_layer_size):
    input_dim = input_value.get_shape().as_list()[1]
    kernels = [
        tf.Variable(
            initial_value=tf.truncated_normal([input_dim]),
            trainable=True,
            name="kernel" + str(i),
        )
        for i in range(cross_network_layer_size)
    ]
    bias = [
        tf.Variable(
            initial_value=tf.zeros([input_dim]),
            trainable=True,
            name="bias" + str(i),
        )
        for i in range(cross_network_layer_size)
    ]
    x_0 = input_value
    x_l = x_0
    for i in range(cross_network_layer_size):
        x_b = tf.tensordot(a=tf.reshape(x_l, [-1, 1, input_dim]), b=kernels[i], axes=1)
        x_l = x_0 * x_b + bias[i] + x_l
    return x_l


class DeepAndCrossNetworkEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,

                 weight_column=None,
                 feature_columns=None,

                 cross_network_layer_size=3,

                 dense_network_hidden_units=(128, 64, 32),
                 dense_network_activation_fn=tf.nn.relu,
                 dense_network_dropout_rate=None,
                 dense_network_use_bn=False,

                 optimizer_name="SGD",
                 learning_rate=0.01,
                 ):

        def custom_model_fn(features, labels, mode, params=None, config=None):
            net = tf.feature_column.input_layer(features, feature_columns=feature_columns)
            logging.info(
                "DeepAndCrossNetworkEstimator custom_model_fn, net.shape:%s" %
                (
                    net.shape
                )
            )

            def compute_dense_out(input_value):
                dense_dnn = DNN(
                    name="dense",
                    hidden_units=dense_network_hidden_units,
                    use_bias=True,
                    activation=dense_network_activation_fn,
                    dropout_ratio=dense_network_dropout_rate,
                    use_bn=dense_network_use_bn,
                    is_training=mode == tf.estimator.ModeKeys.TRAIN,
                )
                return dense_dnn(input_value)

            # Deep & Cross
            if dense_network_hidden_units and cross_network_layer_size:
                deep_out = compute_dense_out(net)
                logging.info(
                    "DeepAndCrossNetworkEstimator custom_model_fn, deep_out.shape:%s" %
                    (
                        str(deep_out.shape)
                    )
                )
                cross_out = cross_interaction(net, cross_network_layer_size)
                logging.info(
                    "DeepAndCrossNetworkEstimator custom_model_fn, cross_out.shape:%s" %
                    (
                        str(cross_out.shape)
                    )
                )
                concat_out = tf.concat([deep_out, cross_out], axis=1)
                logging.info(
                    "DeepAndCrossNetworkEstimator custom_model_fn, concat_out.shape:%s" %
                    (
                        str(concat_out.shape)
                    )
                )
                logits = tf.layers.dense(concat_out, 1, activation=None, use_bias=True)
            # Only Deep
            elif dense_network_hidden_units:
                deep_out = compute_dense_out(net)
                logits = tf.layers.dense(deep_out, 1, activation=None, use_bias=True)
            # Only Cross
            elif cross_network_layer_size:
                cross_out = cross_interaction(net, cross_network_layer_size)
                logits = tf.layers.dense(cross_out, 1, activation=None, use_bias=True)
            else:
                raise NotImplementedError

            logits = tf.reshape(logits, (-1,))
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
        logging.info("DeepAndCrossNetworkEstimator init")
