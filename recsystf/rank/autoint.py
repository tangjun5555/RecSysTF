# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/1/13 12:35 下午
# desc:
# Reference: AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks

import logging
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class AutoIntEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,

                 weight_column=None,
                 category_embed_feature_columns=None,
                 numeric_feature_columns=None,

                 embedding_size=32,

                 optimizer_name="Adam",
                 learning_rate=0.01,
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
        logging.info("AutoIntEstimator init")
