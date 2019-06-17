#
# Copyright (C) 2019 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Commonly used utils in Koalas.
"""

import functools
from typing import Callable, Dict, Union

from pyspark import sql as spark
import pandas as pd


def default_session(conf=None):
    if conf is None:
        conf = dict()
    builder = spark.SparkSession.builder.appName("Koalas")
    for key, value in conf.items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def validate_arguments_and_invoke_function(pobj: Union[pd.DataFrame, pd.Series],
                                           koalas_func: Callable, pandas_func: Callable,
                                           input_args: Dict):
    """
    Invokes a pandas function.

    This is created because different versions of pandas support different parameters, and as a
    result when we code against the latest version, our users might get a confusing
    "got an unexpected keyword argument" error if they are using an older version of pandas.

    This function validates all the arguments, removes the ones that are not supported if they
    are simply the default value (i.e. most likely the user didn't explicitly specify it). It
    throws a TypeError if the user explicitly specify an argument that is not supported by the
    pandas version available.

    For example usage, look at DataFrame.to_html().

    :param pobj: the pandas DataFrame or Series to operate on
    :param koalas_func: koalas function, used to get default parameter values
    :param pandas_func: pandas function, used to check whether pandas supports all the arguments
    :param input_args: arguments to pass to the pandas function, often created by using locals().
                       Make sure locals() call is at the top of the function so it captures only
                       input parameters, rather than local variables.
    :return: whatever pandas_func returns
    """
    import inspect

    # Makes a copy since whatever passed in is likely created by locals(), and we can't delete
    # 'self' key from that.
    args = input_args.copy()
    del args['self']

    if 'kwargs' in args:
        # explode kwargs
        kwargs = args['kwargs']
        del args['kwargs']
        args = {**args, **kwargs}

    koalas_params = inspect.signature(koalas_func).parameters
    pandas_params = inspect.signature(pandas_func).parameters

    for param in koalas_params.values():
        if param.name not in pandas_params:
            if args[param.name] == param.default:
                del args[param.name]
            else:
                raise TypeError(
                    ("The pandas version [%s] available does not support parameter '%s' " +
                     "for function '%s'.") % (pd.__version__, param.name, pandas_func.__name__))

    args['self'] = pobj
    return pandas_func(**args)


def lazy_property(fn):
    """
    Decorator that makes a property lazy-evaluated.

    Copied from https://stevenloria.com/lazy-properties/
    """
    attr_name = '_lazy_' + fn.__name__

    @property
    @functools.wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property
