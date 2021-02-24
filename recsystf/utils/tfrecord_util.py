# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/2/24 3:05 下午
# desc:

import logging
import tensorflow as tf

if tf.__version__ >= "2.0":
    tf = tf.compat.v1
    
    
def decode_libsvm(line, feature_len):
    columns = tf.string_split([line], delimiter=" ")
    labels = tf.reshape(tf.string_to_number(columns.values[0], out_type=tf.int32), [-1])
    splits = tf.string_split(columns.values[1:], delimiter=":")
    id_vals = tf.reshape(splits.values, splits.dense_shape)
    feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
    feat_ids = tf.reshape(tf.string_to_number(feat_ids, out_type=tf.int64), [-1])
    feat_vals = tf.reshape(tf.string_to_number(feat_vals, out_type=tf.float32), [-1])
    indices = tf.cast(tf.reshape(tf.range(0, limit=tf.shape(feat_ids)[0], dtype=tf.int32), [-1, 1]), tf.int64)
    feat_ids = tf.SparseTensor(indices, feat_ids, [feature_len])
    feat_vals = tf.SparseTensor(indices, feat_vals, [feature_len])
    return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels


def libsvm_file_to_tfrecord_file(libsvm_filenames, tfrecord_filenames):
    logging.info("libsvm_filenames:%s" % str(libsvm_filenames))
    logging.info("tfrecord_filenames:%s" % str(tfrecord_filenames))
    assert type(libsvm_filenames) == type(tfrecord_filenames)
    if isinstance(libsvm_filenames, str):
        libsvm_filenames = [libsvm_filenames]
        tfrecord_filenames = [tfrecord_filenames]

    for i in range(len(libsvm_filenames)):
        libsvm_filename = libsvm_filenames[i]
        tfrecord_filename = tfrecord_filenames[i]
        tf.logging.info("Begin to process %s" % libsvm_filename)
        writer = tf.python_io.TFRecordWriter(tfrecord_filename)
        line_num = 0
        with open(libsvm_filename, mode="r", encoding="utf-8") as fread:
            for line in fread:
                line_num += 1

                if line_num % 1000 == 0:
                    tf.logging.info("Processing the {0} line sample".format(line_num))

                feature_ids = []
                vals = []
                line_components = line.strip().split(" ")

                try:
                    label = float(line_components[0])
                    features = line_components[1:]
                except IndexError:
                    tf.logging.info("Index Error, line: {0}".format(line))
                    continue
                for feature in features:
                    feature_components = feature.split(":")
                    try:
                        feature_id = int(feature_components[0])
                        val = float(feature_components[1])
                    except IndexError:
                        tf.logging.info("Index Error: , feature_components: {0}", format(feature))
                        continue
                    except ValueError:
                        tf.logging.info("Value Error: feature_components[0]: {0}".format(feature_components[0]))
                    feature_ids.append(feature_id)
                    vals.append(val)
                tfrecord_feature = {
                    "label": tf.train.Feature(float_list=tf.train.FloatList(value=[label])),
                    "feature_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=feature_ids)),
                    "feature_vals": tf.train.Feature(float_list=tf.train.FloatList(value=vals))
                }
                example = tf.train.Example(features=tf.train.Features(feature=tfrecord_feature))
                writer.write(example.SerializeToString())
            writer.close()
        logging.info("{0} transform to tfrecord: {1} successfully".format(libsvm_filename, tfrecord_filename))
