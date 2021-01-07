# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:47 下午
# desc:

import time
import tensorflow as tf
from recsystf.utils.normalization_ops_util import l2_norm
from tensorflow.python.estimator.canned import optimizers


class DSSMEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,

                 predict_type="logits",

                 user_id_column_name="user_id",
                 user_feature_columns=None,
                 item_id_column_name="item_id",
                 item_feature_columns=None,

                 user_transform_fn=None,
                 item_transform_fn=None,

                 need_l2_norm=False,

                 optimizer_name="Adam",
                 learning_rate=0.01,
                 ):
        """

        :param model_dir:
        :param config:
        :param warm_start_from:
        :param user_feature_columns:
        :param item_feature_columns:
        :param user_transform_fn: user transform function. Follows the signature:
            * Args:
                * `user_net`:
                * `features`:
        :param item_transform_fn: item transform function. Follows the signature:
            * Args:
                * `net`:
                * `features`:
        :param optimizer_name:
        :param learning_rate:
        """

        assert predict_type in ["logits", "user", "item"]

        def custom_model_fn(features, labels, mode, params=None, config=None):
            if mode == tf.estimator.ModeKeys.PREDICT and predict_type == "user":
                user_id = features[user_id_column_name]
                user_net = tf.feature_column.input_layer(features, feature_columns=user_feature_columns)
                user_vector = user_transform_fn(net=user_net, features=features)
                if need_l2_norm:
                    user_vector = l2_norm(user_vector)
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={
                        "user_id": user_id,
                        "user_vector": tf.reduce_join(tf.as_string(user_vector), axis=-1, separator=','),
                    },
                )

            if mode == tf.estimator.ModeKeys.PREDICT and predict_type == "item":
                item_id = features[item_id_column_name]
                item_net = tf.feature_column.input_layer(features, feature_columns=item_feature_columns)
                item_vector = item_transform_fn(net=item_net, features=features)
                if need_l2_norm:
                    item_vector = l2_norm(item_vector)
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={
                        "item_id": item_id,
                        "item_vector": tf.reduce_join(tf.as_string(item_vector), axis=-1, separator=','),
                    },
                )

            user_id = features[user_id_column_name]
            user_net = tf.feature_column.input_layer(features, feature_columns=user_feature_columns)
            user_vector = user_transform_fn(net=user_net, features=features)

            item_id = features[item_id_column_name]
            item_net = tf.feature_column.input_layer(features, feature_columns=item_feature_columns)
            item_vector = item_transform_fn(net=item_net, features=features)

            assert user_vector.get_shape().as_list() == item_vector.get_shape().as_list()

            if need_l2_norm:
                user_vector = l2_norm(user_vector)
                item_vector = l2_norm(item_vector)

            user_item_sim = tf.reduce_sum(
                tf.multiply(user_vector, item_vector),
                axis=1,
                keep_dims=True,
            )
            tf.logging.info(
                "%s DSSMEstimator custom_model_fn, user_item_sim.shape:%s" %
                (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    user_item_sim.shape
                )
            )
            sim_w = tf.get_variable("sim_w", dtype=tf.float32, shape=(1, 1), initializer=tf.ones_initializer())
            sim_b = tf.get_variable("sim_b", dtype=tf.float32, shape=(1,), initializer=tf.zeros_initializer())
            logits = tf.matmul(user_item_sim, tf.abs(sim_w)) + sim_b
            logits = tf.reshape(logits, (-1,))
            tf.logging.info(
                "%s DSSMEstimator custom_model_fn, logits.shape:%s" %
                (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    logits.shape
                )
            )

            predictions = tf.nn.sigmoid(logits)
            if mode == tf.estimator.ModeKeys.PREDICT and predict_type == "logits":
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={
                        "user_id": user_id,
                        "user_vector": tf.reduce_join(tf.as_string(user_vector), axis=-1, separator=','),
                        "item_id": item_id,
                        "item_vector": tf.reduce_join(tf.as_string(item_vector), axis=-1, separator=','),
                        "user_item_sim": user_item_sim,
                        "predictions": predictions
                    },
                )

            loss = tf.losses.log_loss(
                labels=tf.cast(labels, tf.float32),
                predictions=predictions,
            )
            eval_metric_ops = dict()
            eval_metric_ops["auc"] = tf.metrics.auc(labels, predictions)
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
            "[%s] DSSMEstimator init" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
