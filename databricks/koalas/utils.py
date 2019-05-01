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

from pyspark import sql as spark


def default_session():
    return spark.SparkSession.builder.getOrCreate()


def validate_arguments_and_invoke_function(pdf, koalas_func, pandas_func, input_args):
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

    :param pdf: the pandas DataFrame to operate on
    :param koalas_func: koalas function, used to get default parameter values
    :param pandas_func: pandas function, used to check whether pandas supports all the arguments
    :param input_args: arguments to pass to the pandas function
    :return: whatever pandas_func returns
    """
    import inspect

    # Makes a copy since whatever passed in is likely created by locals(), and we can't delete
    # 'self' key from that.
    args = input_args.copy()
    del args['self']

    koalas_params = inspect.signature(koalas_func).parameters
    pandas_params = inspect.signature(pandas_func).parameters

    for param in koalas_params.values():
        if param.name not in pandas_params:
            if args[param.name] == param.default:
                del args[param.name]
            else:
                raise TypeError(
                    ("The version of pandas available does not support parameter '%s' " +
                        "for function '%s'.") % (param.name, pandas_func.__name__))

    args['self'] = pdf
    return pandas_func(**args)
