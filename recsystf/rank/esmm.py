# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:47 下午
# desc:

import time
import tensorflow as tf
from recsystf.layers.dnn import DNN
from tensorflow.python.estimator.canned import optimizers


class ESSMEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 params=None,
                 warm_start_from=None,

                 ctr_column_name="ctr_label",
                 ctcvr_column_name="ctcvr_label",
                 dnn_feature_columns=None,

                 dnn_hidden_units=(300, 200, 100),
                 dnn_activation_fn=tf.nn.relu,
                 dnn_dropout=None,
                 dnn_use_bn=None,

                 optimizer_name="Adam",
                 learning_rate=0.01,
                 ):
        def custom_model_fn(features, labels, mode, params, config=None):
            net = tf.feature_column.input_layer(features, feature_columns=dnn_feature_columns)
            tf.logging.info(
                "%s ESSMEstimator custom_model_fn, net.shape:%s" %
                (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    net.shape
                )
            )

            ctr_model = DNN("CTR", hidden_units=dnn_hidden_units, activation=dnn_activation_fn,
                            dropout_ratio=dnn_dropout, use_bn=dnn_use_bn,
                            is_training=mode == tf.estimator.ModeKeys.TRAIN,
                            )
            ctcvr_model = DNN("CTCVR", hidden_units=dnn_hidden_units, activation=dnn_activation_fn,
                              dropout_ratio=dnn_dropout, use_bn=dnn_use_bn,
                              is_training=mode == tf.estimator.ModeKeys.TRAIN,
                              )

            ctr_logits = ctr_model(net)
            if dnn_hidden_units[-1] != 1:
                ctr_logits = tf.layers.dense(ctr_logits, 1)
            ctr_logits = tf.reshape(ctr_logits, (-1,))
            ctr = tf.sigmoid(ctr_logits, name="CTR")

            ctcvr_logits = ctcvr_model(net)
            if dnn_hidden_units[-1] != 1:
                ctcvr_logits = tf.layers.dense(ctcvr_logits, 1)
            ctcvr_logits = tf.reshape(ctcvr_logits, (-1,))
            ctcvr = tf.sigmoid(ctcvr_logits, name="CTCVR")

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    "ctr": ctr,
                    "ctcvr": ctcvr,
                }
                export_outputs = {
                    "prediction": tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            ctr_label = labels[ctr_column_name]
            ctcvr_label = labels[ctcvr_column_name]
            ctr_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(ctr_label, tf.float32), logits=ctr_logits),
                name="ctr_logloss",
            )
            ctcvr_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(ctcvr_label, tf.float32), logits=ctr_logits),
                name="ctcvr_logloss",
            )
            all_loss = tf.add(ctr_loss, ctcvr_loss, name="all_loss")

            eval_metric_ops = {
                "ctr_auc": tf.metrics.auc(ctr_label, ctr),
                "ctcvr_auc": tf.metrics.auc(ctcvr_label, ctcvr),
            }
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=all_loss,
                    eval_metric_ops=eval_metric_ops,
                )

            assert mode == tf.estimator.ModeKeys.TRAIN
            optimizer_instance = optimizers.get_optimizer_instance(optimizer_name, learning_rate=learning_rate, )
            train_op = optimizer_instance.minimize(
                all_loss,
                global_step=tf.train.get_global_step(),
            )
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=all_loss,
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
        tf.logging.info(
            "[%s] ESSMEstimator:%s init" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                            str(self.__class__.__name__)),
        )
