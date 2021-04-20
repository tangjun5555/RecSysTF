# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/4/20 4:04 下午
# desc:

import logging
import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class RankModelEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_fn=None,
                 model_dir=None,
                 config=None,
                 params=None,
                 warm_start_from=None,
                 ):
        self._prediction_dict = {}
        super().__init__(
            model_fn=model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from,
        )
        logging.info("RankModelEstimator init")

    def get_prediction_dict(self):
        return self._prediction_dict

    def add_to_prediction_dict(self, name, predictions):
        self._prediction_dict[name] = predictions

    def build_binary_loss(self, labels, predictions, weights):
        return tf.losses.log_loss(
            labels=tf.cast(labels, tf.float32),
            predictions=predictions,
            weights=weights,
        )

    def build_binary_metric(self, labels, predictions):
        return {
            "auc": tf.metrics.auc(labels=labels,
                                  predictions=predictions,
                                  ),
        }

    def get_optimizer_instance(self, optimizer_name, learning_rate):
        return optimizers.get_optimizer_instance(opt=optimizer_name,
                                                 learning_rate=learning_rate)
