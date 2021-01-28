# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/1/13 3:37 下午
# desc:
# Reference: SDM: Sequential Deep Matching Model for Online Large-scale Recommender System

import time
import logging
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class SDMEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,
                 ):
        def custom_model_fn(features, labels, mode, params=None, config=None):
            pass

        super().__init__(
            model_fn=custom_model_fn,
            model_dir=model_dir,
            config=config,
            params=None,
            warm_start_from=warm_start_from,
        )
        logging.info(
            "[%s] SDMEstimator init" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
