# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/1/26 7:12 下午
# desc:
# Reference: Entire Space Multi-Task Modeling via Post-Click Behavior Decomposition for Conversion Rate Prediction

import logging
import tensorflow as tf
from recsystf.layers.dnn import DNN
from tensorflow.python.estimator.canned import optimizers

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class ESM2Estimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 params=None,
                 warm_start_from=None,
                 ):
        def custom_model_fn(features, labels, mode, params, config=None):
            pass

        super().__init__(
            model_fn=custom_model_fn,
            model_dir=model_dir,
            config=config,
            params=params,
            warm_start_from=warm_start_from,
        )
        logging.info("ESM2Estimator init")
