# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2021/2/8 3:59 下午
# desc:

from collections import namedtuple

RawFeatureConfig = namedtuple(
    typename="RawFeatureConfig",
    field_names=["feature_name", "feature_type", "feature_shape", "feature_vocab_size"],
)

EmbeddingFeature = namedtuple(
    typename="EmbeddingFeature",
    field_names=["feature_name", "embedding_name",
                 "vocab_size", "embedding_size",
                 "pooling",
                 "length"
                 ],
)
