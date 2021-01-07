# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:47 下午
# desc:

import time
import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers

if tf.__version__ >= '2.0':
    tf = tf.compat.v1


class DeepAndCrossNetworkEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,

                 feature_columns=None,

                 cross_layer_size=3,

                 dnn_hidden_units=(300, 200, 100),
                 dnn_activation_fn=tf.nn.relu,
                 dnn_dropout_rate=None,
                 dnn_use_bn=False,

                 sample_weight_column=None,
                 optimizer_name="Adam",
                 learning_rate=0.01,
                 ):

        def custom_model_fn(features, labels, mode, params=None, config=None):
            net = tf.feature_column.input_layer(features, feature_columns=feature_columns)
            tf.logging.info(
                "%s DeepAndCrossNetworkEstimator custom_model_fn, net.shape:%s" %
                (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    net.shape
                )
            )
            net_dim = int(net.shape[1])

            def compute_deep_out(input_value):
                output_value = input_value
                for units in dnn_hidden_units:
                    if dnn_dropout_rate is not None and dnn_dropout_rate > 0.0:
                        output_value = tf.layers.dense(output_value, units=units, activation=dnn_activation_fn,
                                                       use_bias=True)
                        output_value = tf.layers.dropout(output_value, dnn_dropout_rate,
                                                         training=(mode == tf.estimator.ModeKeys.TRAIN))
                    elif dnn_use_bn:
                        output_value = tf.layers.dense(output_value, units=units, activation=None, use_bias=False)
                        output_value = tf.nn.relu(
                            tf.layers.batch_normalization(output_value, training=(mode == tf.estimator.ModeKeys.TRAIN)))
                    else:
                        output_value = tf.layers.dense(output_value, units=units, activation=dnn_activation_fn,
                                                       use_bias=True)
                return output_value

            def compute_cross_out(input_value):
                kernels = [
                    tf.Variable(
                        initial_value=tf.truncated_normal([net_dim, 1]),
                        trainable=True,
                        name="kernel" + str(i),
                    )
                    for i in range(cross_layer_size)
                ]
                bias = [
                    tf.Variable(
                        initial_value=tf.zeros([net_dim, 1]),
                        trainable=True,
                        name="bias" + str(i),
                    )
                    for i in range(cross_layer_size)
                ]
                x_0 = tf.expand_dims(input_value, axis=2)
                x_l = x_0
                for i in range(cross_layer_size):
                    xl_w = tf.tensordot(x_l, kernels[i], axes=(1, 0))
                    dot_ = tf.matmul(x_0, xl_w)
                    x_l = dot_ + bias[i] + x_l
                x_l = tf.squeeze(x_l, axis=2)
                return x_l

            if len(dnn_hidden_units) > 0 and cross_layer_size > 0:  # Deep & Cross
                deep_out = compute_deep_out(net)
                cross_out = compute_cross_out(net)
                stack_out = tf.concat([deep_out, cross_out], axis=1)
                logits = tf.layers.dense(stack_out, 1, activation=None, use_bias=True)
            elif len(dnn_hidden_units) > 0:  # Only Deep
                deep_out = compute_deep_out(net)
                logits = tf.layers.dense(deep_out, 1, activation=None, use_bias=True)
            elif cross_layer_size > 0:  # Only Cross
                cross_out = compute_cross_out(net)
                logits = tf.layers.dense(cross_out, 1, activation=None, use_bias=True)
            else:
                raise NotImplementedError
            logits = tf.reshape(logits, (-1,))
            tf.logging.info(
                "%s DeepAndCrossNetworkEstimator custom_model_fn, logits.shape:%s" %
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

            if sample_weight_column:
                loss = tf.losses.log_loss(
                    labels=tf.cast(labels, tf.float32),
                    predictions=predictions,
                    weights=features[sample_weight_column],
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
            "[%s] DeepAndCrossNetworkEstimator init" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
