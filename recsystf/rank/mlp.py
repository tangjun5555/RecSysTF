# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:47 下午
# desc:

import logging
import tensorflow as tf
from recsystf.layers.dnn import DNN
from tensorflow.python.estimator.canned import optimizers

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class MLPEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 params=None,
                 warm_start_from=None,

                 weight_column=None,
                 feature_columns=None,

                 hidden_units=(300, 200, 100),
                 activation_fn=tf.nn.relu,
                 dropout=None,
                 batch_norm=False,

                 optimizer_name="SGD",
                 learning_rate=0.01,
                 ):
        def custom_model_fn(features, labels, mode, params, config=None):
            net = tf.feature_column.input_layer(features, feature_columns=feature_columns)
            logging.info(
                "MLPEstimator custom_model_fn, net.shape:%s" %
                (
                    net.shape
                )
            )

            dnn = DNN(
                name="dense",
                hidden_units=hidden_units,
                use_bias=True,
                activation=activation_fn,
                dropout_ratio=dropout,
                use_bn=batch_norm,
                is_training=mode == tf.estimator.ModeKeys.TRAIN,
            )
            dnn_out = dnn(net)

            logits = tf.layers.dense(dnn_out, 1, activation=None, use_bias=True)
            logits = tf.reshape(logits, (-1,))
            predictions = tf.sigmoid(logits)

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
            params=params,
            warm_start_from=warm_start_from,
        )
        logging.info("MLPEstimator init")


# class MLPEstimator(tf.estimator.DNNLinearCombinedClassifier):
#     def __init__(self,
#                  model_dir=None,
#                  config=None,
#                  warm_start_from=None,
#
#                  weight_column=None,
#
#                  dnn_feature_columns=None,
#                  dnn_optimizer="SGD",
#                  dnn_hidden_units=None,
#                  dnn_activation_fn=tf.nn.relu,
#                  dnn_dropout=None,
#
#                  batch_norm=False,
#                  ):
#         super().__init__(
#                  model_dir=model_dir,
#                  dnn_feature_columns=dnn_feature_columns,
#                  dnn_optimizer=dnn_optimizer,
#                  dnn_hidden_units=dnn_hidden_units,
#                  dnn_activation_fn=dnn_activation_fn,
#                  dnn_dropout=dnn_dropout,
#                  n_classes=2,
#                  weight_column=weight_column,
#                  config=config,
#                  warm_start_from=warm_start_from,
#                  batch_norm=batch_norm,
#         )
#         logging.info(
#             "[%s] MLPEstimator init" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#         )
