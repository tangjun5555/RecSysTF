# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:47 下午
# desc:
# Reference: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
# Core Network Structure: Embedding + Full Connection + Mixture-of-Experts + Multi-Gate

import logging
from typing import List
import tensorflow as tf
from recsystf.layers.dnn import DNN, DNNConfig
from recsystf.feature.feature_type import EmbeddingFeature
from recsystf.feature.feature_transform import embedding_feature_to_vector
from tensorflow.python.estimator.canned import optimizers

if tf.__version__ >= "2.0":
    tf = tf.compat.v1


class MMoEEstimator(tf.estimator.Estimator):
    def __init__(self,
                 model_dir=None,
                 config=None,
                 params=None,
                 warm_start_from=None,

                 feature_columns=None,
                 embedding_columns: List[EmbeddingFeature] = None,

                 task_names=None,
                 export_dnn_configs: List[DNNConfig] = None,

                 optimizer_name="SGD",
                 learning_rate=0.01,
                 ):
        assert 0.0 < learning_rate < 1.0
        assert (feature_columns or embedding_columns) and export_dnn_configs and task_names
        expert_num = len(export_dnn_configs)
        task_num = len(task_names)

        def custom_model_fn(features, labels, mode, params, config=None):
            net = []
            if feature_columns:
                net.append(tf.feature_column.input_layer(features, feature_columns=feature_columns))
            if embedding_columns:
                for column_config in embedding_columns:
                    net.append(
                        embedding_feature_to_vector("feature", features[column_config.feature_name], column_config))
            net = tf.concat(net, axis=1)
            logging.info(
                "MMoEEstimator custom_model_fn, net.shape:%s" %
                (
                    net.shape
                )
            )

            expert_output_list = []
            for expert_id in range(expert_num):
                expert_dnn = DNN(
                    name="expert_%d/dnn" % expert_id,
                    hidden_units=export_dnn_configs[expert_id].hidden_units,
                    activation=export_dnn_configs[expert_id].activation,
                    dropout_ratio=export_dnn_configs[expert_id].dropout_ratio,
                    use_bn=export_dnn_configs[expert_id].use_bn,
                    is_training=mode == tf.estimator.ModeKeys.TRAIN,
                )
                # shape: (batch_size, hidden_units)
                export_output = expert_dnn(net)
                expert_output_list.append(export_output)
            # shape: (batch_size, expert_num, hidden_units)
            expert_output_list = tf.stack(expert_output_list, axis=1)

            gate_output_list = []
            for task_id in range(task_num):
                gate_output = tf.layers.dense(
                    inputs=net,
                    units=expert_num,
                    name="gate_%d/dnn" % task_id,
                    reuse=tf.AUTO_REUSE,
                )
                gate_output = tf.nn.softmax(gate_output, axis=1)
                # shape: (batch_size, expert_num, 1)
                gate_output = tf.expand_dims(gate_output, -1)

                # shape: (batch_size, expert_num, hidden_units)
                gate_output = tf.multiply(expert_output_list, gate_output)
                # shape: (batch_size, hidden_units)
                gate_output = tf.reduce_sum(gate_output, axis=1)
                gate_output_list.append(gate_output)

            task_logits = dict()
            task_predictions = dict()
            for task_id in range(task_num):
                task_name = task_names[task_id]
                tower_output = tf.layers.dense(
                    inputs=gate_output_list[task_id],
                    units=1,
                    name="tower_%s/dnn" % task_name,
                    reuse=tf.AUTO_REUSE,
                )
                task_logits[task_name] = tower_output
                task_predictions[task_name + "_predictions"] = tf.nn.sigmoid(tower_output)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=task_predictions,
                )

            all_loss = []
            all_auc = dict()
            for task_name in task_names:
                task_loss = tf.losses.log_loss(
                    labels=tf.cast(labels[task_name], tf.float32),
                    predictions=task_predictions[task_name + "_predictions"],
                )
                all_loss.append(task_loss)
                all_auc[task_name] = tf.metrics.auc(
                    labels=tf.cast(labels[task_name], tf.float32),
                    predictions=task_predictions[task_name + "_predictions"],
                )
            all_loss = tf.add_n(all_loss)
            eval_metric_ops = dict()
            eval_metric_ops.update(all_auc)
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=all_loss,
                    eval_metric_ops=eval_metric_ops,
                )

            assert mode == tf.estimator.ModeKeys.TRAIN
            optimizer_instance = optimizers.get_optimizer_instance(optimizer_name, learning_rate=learning_rate, )
            train_op = optimizer_instance.minimize(
                all_loss,
                global_step=tf.train.get_global_step(),
            )
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=all_loss,
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
        logging.info("MMoEEstimator:%s init" % (str(self.__class__.__name__)))
