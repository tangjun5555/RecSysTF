# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/2/24 3:05 下午
# desc:

import os
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1

from tensorflow.python.framework import graph_util


def get_all_variable_shape_map_from_ckpt(checkpoint_path):
    """
    获取模型所有的变量和相应形状
    :param checkpoint_path:
    :return:
    """
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    return reader.get_variable_to_shape_map()


def get_tensor_values_from_ckpt(checkpoint_path, tensor_name):
    """
    获取模型具体变量的值
    :param checkpoint_path:
    :param tensor_name:
    :return:
    """
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    return list(reader.get_tensor(tensor_name))


def ckpt_to_pb(checkpoint_path, pb_path, output_node_names):
    """
    tensorflow模型保存为pb
    :param checkpoint_path:
    :param pb_path:
    :param output_node_names
    :return:
    """

    meta_files = [x for x in os.listdir(checkpoint_path) if x.endswith(".meta")]
    sorted(meta_files)
    tf.logging.error("used meta file:" + meta_files[-1])
    saver = tf.train.import_meta_graph(checkpoint_path + "/" + meta_files[-1], clear_devices=True)
    # 获得默认的图
    graph = tf.get_default_graph()
    # 返回一个序列化的图代表当前的图
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        # 恢复图并得到数据
        saver.restore(sess, checkpoint_path + "/" + meta_files[-1][:-5])
        # 模型持久化，将变量值固定
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(",")
        )
        with tf.gfile.GFile(pb_path, mode="wb") as f:
            f.write(output_graph_def.SerializeToString())

        for op in graph.get_operations():
            print(op.name, op.values())


def keras_h5_to_pb(keras_model_path, pb_path):
    with tf.device("/cpu:0"):
        model = tf.keras.models.load_model(keras_model_path)
        tf.saved_model.simple_save(
            session=tf.keras.backend.get_session(),
            export_dir=pb_path,
            inputs={"inputs": model.input},
            outputs={"outputs": model.output}
        )
