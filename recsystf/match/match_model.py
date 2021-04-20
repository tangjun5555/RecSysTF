# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/4/20 5:11 下午
# desc:

import logging
import tensorflow as tf
from tensorflow.python.estimator.canned import optimizers

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class MatchModelEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_fn=None,
                 model_dir=None,
                 config=None,
                 params=None,
                 warm_start_from=None,
                 ):
        super().__init__(
            model_fn=model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from,
        )
        logging.info("MatchModelEstimator init")

    def build_pointwise_loss(self, labels, predictions, weights):
        pass

    def build_pairwise_loss(self, ):
        pass

    def build_listwise_loss(self, ):
        pass

    def get_optimizer_instance(self, optimizer_name, learning_rate):
        return optimizers.get_optimizer_instance(
            opt=optimizer_name,
            learning_rate=learning_rate,
        )
