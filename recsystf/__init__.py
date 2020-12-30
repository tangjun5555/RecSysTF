# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:49 下午
# desc:

import os
import sys

from recsystf.version import __version__

curr_dir, _ = os.path.split(__file__)
parent_dir = os.path.dirname(curr_dir)
sys.path.insert(0, parent_dir)
