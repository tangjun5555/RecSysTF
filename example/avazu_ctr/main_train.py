# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/5/16 6:23 下午
# desc:

import time
import logging
import argparse
import tensorflow as tf
from recsystf.feature.feature_type import EmbeddingFeature
from recsystf.rank.pnn import PNNEstimator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %a",
)

logging.info("当前tf版本:%s" % (str(tf.__version__)))
if tf.__version__ >= "2.0":
    tf = tf.compat.v1
    tf.disable_eager_execution()

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, required=False, default="pnn")
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--export_model_dir", type=str, required=True)
parser.add_argument("--train_filename", type=str, required=True)
parser.add_argument("--valid_filename", type=str, required=True)
parser.add_argument("--save_checkpoints_steps", type=int, required=False, default=1000)
parser.add_argument("--epoch", type=int, required=False, default=2)
parser.add_argument("--batch_size", type=int, required=False, default=128)

args = parser.parse_args()
logging.info("程序运行参数:" + str(args))

vocab_size = 1000
embedding_size = 16
names = "id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id," \
        "device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21".split(",")

embedding_feature_configs = [
    EmbeddingFeature(feature_name=names[i], embedding_name=names[i],
                     vocab_size=vocab_size, embedding_size=embedding_size,
                     pooling="mean", length=1,
                     )
    for i in range(3, len(names))
]

feature_placeholder = dict()
for i in range(3, len(names)):
    feature_placeholder[names[i]] = tf.placeholder(
        shape=(None, 1),
        dtype=tf.dtypes.int32,
        name=names[i]
    )


def parse_example(line):
    columns = tf.string_split([line], ",")
    labels = tf.reshape(tf.string_to_number(columns.values[1], out_type=tf.float32), [-1])
    features = {}
    for i in range(3, len(names)):
        features[names[i]] = tf.reshape(tf.string_to_hash_bucket_fast(columns.values[i], vocab_size), [-1])
    return features, labels


def train_input_fn(filename, batch_size, epoch):
    dataset = tf.data.TextLineDataset([filename])
    dataset = dataset.map(parse_example, num_parallel_calls=4)
    dataset = dataset.shuffle(batch_size * 2)
    dataset = dataset.prefetch(batch_size * 2)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(epoch)
    return dataset


def valid_input_fn(filename, batch_size):
    dataset = tf.data.TextLineDataset([filename])
    dataset = dataset.map(parse_example, num_parallel_calls=4)
    dataset = dataset.batch(batch_size)
    return dataset


if __name__ == '__main__':
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    session_config = tf.ConfigProto(log_device_placement=True)
    if args.model_name == "pnn":
        model = PNNEstimator(
            model_dir=args.model_dir,
            config=tf.estimator.RunConfig(
                save_checkpoints_steps=args.save_checkpoints_steps,
                save_summary_steps=args.save_checkpoints_steps / 10,
                log_step_count_steps=int(args.save_checkpoints_steps / 10),
                tf_random_seed=555,
                keep_checkpoint_max=2,
                session_config=session_config,
            ),
            embedding_columns=embedding_feature_configs,
            embedding_size=embedding_size,
        )
    else:
        raise Exception("暂不支持")

    # 训练模型
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(args.train_filename, args.batch_size, args.epoch),
    )
    valid_spec = tf.estimator.EvalSpec(
        input_fn=lambda: valid_input_fn(args.valid_filename, args.batch_size),
    )
    tf.estimator.train_and_evaluate(model, train_spec, valid_spec)

    # 保存模型
    model.saved_model(feature_placeholder)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    logging.info("开始时间:%s, 结束时间:%s" % (start_time, end_time))
