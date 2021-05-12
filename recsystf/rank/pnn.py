# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/4/26 4:03 下午
# desc:
# Reference: Product-based Neural Networks for User Response Prediction

import logging
from typing import List
import tensorflow as tf
from recsystf.feature.feature_type import EmbeddingFeature
from recsystf.feature.feature_transform import embedding_feature_to_vector
from recsystf.rank.rank_model import RankModelEstimator
from recsystf.layers.interaction import PNNInnerProductLayer, PNNOuterProductLayer
from recsystf.layers.dnn import DNN

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class PNNEstimator(RankModelEstimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 params=None,
                 warm_start_from=None,

                 weight_column=None,
                 embedding_columns: List[EmbeddingFeature] = None,

                 embedding_size=32,

                 use_inner=True,
                 use_outer=True,
                 outer_kernel_type="mat",

                 dense_network_hidden_units=(128, 64, 32),
                 dense_network_activation_fn=tf.nn.relu,
                 dense_network_dropout_rate=None,
                 dense_network_use_bn=False,

                 optimizer_name="SGD",
                 learning_rate=0.01,
                 ):
        def custom_model_fn(features, labels, mode, params, config=None):
            embedding_fea_value = []
            for column_config in embedding_columns:
                embedding_fea_value.append(
                    embedding_feature_to_vector("feature", features[column_config.feature_name], column_config)
                )
            embedding_fea_value = tf.stack(values=embedding_fea_value, axis=1)
            logging.info(
                "PNNEstimator custom_model_fn, embedding_fea_value.shape:%s" %
                (
                    embedding_fea_value.shape
                )
            )

            linear_signal = tf.reshape(embedding_fea_value, [-1, len(embedding_columns) * embedding_size])

            inner_interaction = PNNInnerProductLayer("inner_interaction")
            outer_interaction = PNNOuterProductLayer("inner_interaction", outer_kernel_type)

            if use_inner and use_outer:
                inner_out = inner_interaction(embedding_fea_value)
                outer_out = outer_interaction(embedding_fea_value)
                deep_input = tf.concat([linear_signal, inner_out, outer_out], axis=1)
            elif use_inner:
                inner_out = inner_interaction(embedding_fea_value)
                deep_input = tf.concat([linear_signal, inner_out], axis=1)
            elif use_outer:
                outer_out = outer_interaction(embedding_fea_value)
                deep_input = tf.concat([linear_signal, outer_out], axis=1)
            else:
                deep_input = linear_signal

            dnn = DNN(
                name="dense",
                hidden_units=dense_network_hidden_units,
                use_bias=True,
                activation=dense_network_activation_fn,
                dropout_ratio=dense_network_dropout_rate,
                use_bn=dense_network_use_bn,
                is_training=mode == tf.estimator.ModeKeys.TRAIN,
            )
            dnn_out = dnn(deep_input)

            logits = tf.layers.dense(dnn_out, 1, activation=None, use_bias=True)
            logits = tf.reshape(logits, (-1,))
            predictions = tf.sigmoid(logits)
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
        logging.info("PNNEstimator init")
