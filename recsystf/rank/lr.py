# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/4/19 6:55 下午
# desc:

import logging
import tensorflow as tf
from recsystf.rank.rank_model import RankModelEstimator

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class LREstimator(RankModelEstimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 params=None,
                 warm_start_from=None,

                 weight_column=None,
                 feature_column=None,

                 optimizer_name="SGD",
                 learning_rate=0.01,
                 ):
        def custom_model_fn(features, labels, mode, params, config=None):
            input_net = features[feature_column]

            predictions = tf.layers.dense(
                inputs=input_net,
                units=1,
                use_bias=True,
                activation=tf.nn.sigmoid,
            )
            self.add_to_prediction_dict("predictions", predictions)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=self.get_prediction_dict(),
                )

            loss = self.build_pointwise_loss(labels, predictions, 1.0 if not weight_column else features[weight_column])
            eval_metric_ops = self.build_pointwise_metric(labels, predictions)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metric_ops=eval_metric_ops,
                )

            assert mode == tf.estimator.ModeKeys.TRAIN

            optimizer_instance = self.get_optimizer_instance(optimizer_name, learning_rate)
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
        logging.info("LREstimator init")
