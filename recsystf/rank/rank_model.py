# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/4/20 4:04 下午
# desc:

import os
import json
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
        if config is None:
            config = tf.estimator.RunConfig()
        # 是否使用ps分布式训练
        if params["use_ps_distribute"]:
            # 配置样例
            # os.environ["TF_CONFIG"] = json.dumps({
            #     "cluster": {
            #         "chief": ["127.0.0.1:5000"],  # 调度节点
            #         "worker": ["127.0.0.1:5001"],  # 计算节点
            #         "ps": ["127.0.0.1:5002"]  # 参数服务器节点，可不必使用GPU
            #     },
            #     "task": {"type": "chief", "index": 0}  # 定义本进程为worker节点，即["127.0.0.1:5002"]为ps节点
            # })
            os.environ["TF_CONFIG"] = json.dumps(params.get("TF_CONFIG"))
            strategy = tf.distribute.experimental.ParameterServerStrategy()
            config.train_distribute = strategy
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

    def build_pointwise_loss(self, labels, predictions, weights):
        return tf.losses.log_loss(
            labels=tf.cast(labels, tf.float32),
            predictions=predictions,
            weights=weights,
        )

    def build_pointwise_metric(self, labels, predictions):
        return {
            "auc": tf.metrics.auc(labels=labels,
                                  predictions=predictions,
                                  ),
        }

    def get_optimizer_instance(self, optimizer_name, learning_rate):
        return optimizers.get_optimizer_instance(
            opt=optimizer_name,
            learning_rate=learning_rate,
        )

    def saved_model(self, feature_placeholder):
        export_model_dir = self.export_saved_model(
            export_dir_base=self.model_dir,
            serving_input_receiver_fn=tf.estimator.export.build_raw_serving_input_receiver_fn(feature_placeholder),
            as_text=True,
        )
        logging.warning("保存模型到" + export_model_dir.decode())
