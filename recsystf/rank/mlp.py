# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:47 下午
# desc:

import time
import logging
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class MLPEstimator(tf.estimator.DNNLinearCombinedClassifier):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,

                 weight_column=None,

                 dnn_feature_columns=None,
                 dnn_optimizer="SGD",
                 dnn_hidden_units=None,
                 dnn_activation_fn=tf.nn.relu,
                 dnn_dropout=None,

                 batch_norm=False,
                 ):
        super().__init__(
                 model_dir=model_dir,
                 dnn_feature_columns=dnn_feature_columns,
                 dnn_optimizer=dnn_optimizer,
                 dnn_hidden_units=dnn_hidden_units,
                 dnn_activation_fn=dnn_activation_fn,
                 dnn_dropout=dnn_dropout,
                 n_classes=2,
                 weight_column=weight_column,
                 config=config,
                 warm_start_from=warm_start_from,
                 batch_norm=batch_norm,
        )
        logging.info(
            "[%s] MLPEstimator init" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        )
