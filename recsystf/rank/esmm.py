# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:47 下午
# desc:
# Reference: Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate

import logging
import tensorflow as tf
from recsystf.layers.dnn import DNN
from tensorflow.python.estimator.canned import optimizers

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


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

                 optimizer_name="SGD",
                 learning_rate=0.01,
                 ):
        def custom_model_fn(features, labels, mode, params, config=None):
            net = tf.feature_column.input_layer(features, feature_columns=dnn_feature_columns)
            logging.info("ESSMEstimator custom_model_fn, net.shape:%s" % (str(net.shape)))

            ctr_model = DNN("CTR", hidden_units=dnn_hidden_units, activation=dnn_activation_fn,
                            dropout_ratio=dnn_dropout, use_bn=dnn_use_bn,
                            is_training=mode == tf.estimator.ModeKeys.TRAIN,
                            )
            cvr_model = DNN("CVR", hidden_units=dnn_hidden_units, activation=dnn_activation_fn,
                            dropout_ratio=dnn_dropout, use_bn=dnn_use_bn,
                            is_training=mode == tf.estimator.ModeKeys.TRAIN,
                            )

            ctr_logits = ctr_model(net)
            ctr_logits = tf.layers.dense(ctr_logits, 1, activation=None, use_bias=True)
            ctr_logits = tf.reshape(ctr_logits, (-1,))
            ctr = tf.sigmoid(ctr_logits, name="CTR")

            cvr_logits = cvr_model(net)
            cvr_logits = tf.layers.dense(cvr_logits, 1, activation=None, use_bias=True)
            cvr_logits = tf.reshape(cvr_logits, (-1,))
            cvr = tf.sigmoid(cvr_logits, name="CVR")

            ctcvr = tf.multiply(ctr, cvr, name="CTCVR")

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    "ctr": ctr,
                    "cvr": cvr,
                    "ctcvr": ctcvr,
                }
                export_outputs = {
                    "prediction": tf.estimator.export.PredictOutput(predictions)
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

            ctr_label = labels[ctr_column_name]
            ctcvr_label = labels[ctcvr_column_name]

            ctr_loss = tf.losses.log_loss(
                labels=tf.cast(ctr_label, tf.float32),
                predictions=ctr,
            )
            ctcvr_loss = tf.losses.log_loss(
                labels=tf.cast(ctcvr_label, tf.float32),
                predictions=ctcvr,
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
        logging.info("ESSMEstimator init")
