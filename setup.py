# -*- coding: utf-8 -*-
# author: tangj 1844250138@qq.com
# time: 2020/12/10 3:36 下午
# desc:

from setuptools import setup, find_packages

version_file = 'recsystf/version.py'


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name='recsystf',
    version=get_version(),
    description='An Framework Based On Tensorflow For Recommender System',
    packages=find_packages(),
    author='tangj',
    author_email='1844250138@qq.com',
)
