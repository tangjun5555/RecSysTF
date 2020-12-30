# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:47 下午
# desc:

import time
import tensorflow as tf


class DSSMEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 warm_start_from=None,

                 weight_column=None,
                 user_feature_columns=None,
                 item_feature_columns=None,

                 user_transform_fn=None,
                 item_transform_fn=None,
                 loss_type='pointwise',
                 ):
        assert loss_type in ['pointwise', 'pairwise']
        
        def custom_model_fn(features, labels, mode, params=None, config=None):
            user_vector = user_transform_fn(user_feature_columns)
            item_vector = item_transform_fn(item_feature_columns)
        
        super().__init__(
            model_fn=custom_model_fn,
            model_dir=model_dir,
            config=config,
            params=None,
            warm_start_from=warm_start_from,
        )
        tf.logging.info(
            '[%s] DeepAndCrossNetworkEstimator init' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        )
