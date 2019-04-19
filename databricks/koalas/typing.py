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
Utilities to deal with types. This is mostly focused on python3.
"""
from decorator import decorate
from decorator import getfullargspec
import typing

import numpy as np
from pyspark.sql import Column
from pyspark.sql.functions import pandas_udf
import pyspark.sql.types as types


T = typing.TypeVar("T")


class Col(typing.Generic[T]):
    def is_col(self):
        return self


class _Column(object):
    def __init__(self, inner):
        self.inner = inner

    def __repr__(self):
        return "_ColumnType[{}]".format(self.inner)


class _DataFrame(object):
    def __repr__(self):
        return "_DataFrameType"


class _Regular(object):
    def __init__(self, tpe):
        self.type = tpe

    def __repr__(self):
        return "_RegularType[{}]".format(self.type)


class _Unknown(object):
    def __init__(self, tpe):
        self.type = tpe

    def __repr__(self):
        return "_UnknownType"


X = typing.Union[_Column, _DataFrame, _Regular, _Unknown]


def _is_col(tpe):
    return hasattr(tpe, "is_col")


def _get_col_inner(tpe):
    return tpe.__args__[0]


def _to_stype(tpe) -> X:
    if _is_col(tpe):
        inner = as_spark_type(_get_col_inner(tpe))
        return _Column(inner)
    inner = as_spark_type(tpe)
    if inner is None:
        return _Unknown(tpe)
    else:
        return _Regular(inner)


# First element of the list is the python base type
_base = {
    types.StringType(): [str, 'str', 'string'],
    types.ByteType(): [np.int8, 'int8', 'byte'],
    types.ShortType(): [np.int16, 'int16', 'short'],
    types.IntegerType(): [int, 'int', np.int],
    types.LongType(): [np.int64, 'int64', 'long'],
    types.FloatType(): [float, 'float', np.float],
    types.DoubleType(): [np.float64, 'float64', 'double'],
    types.TimestampType(): [np.datetime64],
    types.BooleanType(): [bool, 'boolean', 'bool', np.bool],
}


def _build_type_dict():
    return dict([(other_type, spark_type) for (spark_type, l) in _base.items() for other_type in l])


def _build_py_type_dict():
    return dict([(spark_type, l[0]) for (spark_type, l) in _base.items()])


_known_types = _build_type_dict()


_py_conversions = _build_py_type_dict()


def as_spark_type(tpe):
    """
    Given a python type, returns the equivalent spark type.
    Accepts:
    - the built-in types in python
    - the built-in types in numpy
    - list of pairs of (field_name, type)
    - dictionaries of field_name -> type
    - python3's typing system
    :param tpe:
    :return:
    """
    return _known_types.get(tpe, None)


def as_python_type(spark_tpe):
    return _py_conversions.get(spark_tpe, None)


def _check_compatible(arg, sig_arg: X):
    if isinstance(sig_arg, _Unknown):
        return arg
    if isinstance(sig_arg, _Regular):
        t = as_spark_type(type(arg))
        if t != sig_arg.type:
            raise ValueError("Passing an argument {} of type {}, but the function only accepts "
                             "type {} for this argument".format(arg, t, sig_arg))
    if isinstance(sig_arg, _Column):
        if not isinstance(arg, Column):
            raise ValueError(
                "Expected a column argument, but got argument of type {} instead".format(type(arg)))
        s = arg.schema
        if s != sig_arg.inner:
            raise ValueError("Passing an argument {} of type {}, but the function only accepts "
                             "columns of type {} for this argument".format(arg, s, sig_arg))
    assert False, (arg, sig_arg)


def make_fun(f, *args, **kwargs):
    """
    This function calls the function f while taking into account some of the
    limitations of the pandas UDF support:
    - support for keyword arguments
    - support for scalar values (as long as they are picklable)
    - support for type hints and input checks.
    :param f:
    :param args:
    :param kwargs:
    :return:
    """
    sig_args = f.sig_args  # type: typing.List[X]
    final_args = []
    col_indexes = []
    frozen_args = []  # None for columns or the value for non-columns
    for (idx, (arg, sig_arg)) in enumerate(zip(args, sig_args)):
        arg2 = _check_compatible(arg, sig_arg)
        if isinstance(arg2, (Column,)):
            col_indexes.append(idx)
            frozen_args.append(None)
        else:
            frozen_args.append(arg2)
        final_args.append(arg2)
    sig_kwargs = f.sig_kwargs  # type: typing.Dict[str, X]
    final_kwargs = {}
    col_keys = []
    frozen_kwargs = {}  # Value is none for kwargs that are columns, and the value otherwise
    for (key, arg) in kwargs:
        sig_arg = sig_kwargs[key]
        arg2 = _check_compatible(arg, sig_arg)
        final_kwargs[key] = arg2
        if isinstance(arg2, (Column,)):
            col_keys.append(key)
            frozen_kwargs[key] = None
        else:
            frozen_kwargs[key] = arg2
    if not col_keys and not col_indexes:
        # No argument is related to spark
        # The function is just called through without other considerations.
        return f(*args, **kwargs)
    # We detected some columns. They need to be wrapped in a UDF to spark.

    # Only handling the case of columns for now.
    ret_type = f.sig_return
    assert isinstance(ret_type, _Column), ret_type
    spark_ret_type = ret_type.inner
    # Spark UDFs do not handle extra data that is not a column.
    # We build a new UDF that only takes arguments from columns, the rest is
    # sent inside the closure into the function.
    all_indexes = col_indexes + col_keys  # type: typing.Union[str, int]

    def clean_fun(*args2):
        assert len(args2) == len(all_indexes),\
            "Missing some inputs:{}!={}".format(all_indexes, [str(c) for c in args2])
        full_args = list(frozen_args)
        full_kwargs = dict(frozen_kwargs)
        for (arg, idx) in zip(args2, all_indexes):
            if isinstance(idx, int):
                full_args[idx] = arg
            else:
                assert isinstance(idx, str), str(idx)
                full_kwargs[idx] = arg
        return f(*full_args, **full_kwargs)
    udf = pandas_udf(clean_fun, returnType=spark_ret_type)
    wrapped_udf = udf  # udf #_wrap_callable(udf)
    col_args = []
    for idx in col_indexes:
        col_args.append(final_args[idx])
    for key in col_keys:
        col_args.append(final_kwargs[key])
    col = wrapped_udf(*col_args)
    # TODO: make more robust
    col._spark_ref_dataframe = col_args[0]._spark_ref_dataframe
    return col


def _wrap_callable(obj):
    f0 = obj.__call__

    def f(*args, **kwargs):
        return f0(*args, **kwargs)
    obj.__call__ = f
    return obj


def pandas_wrap(f, return_col=None):
    # Extract the signature arguments from this function.
    spec = getfullargspec(f)
    rtype = None
    return_sig = spec.annotations.get("return", None)
    if not (return_col or return_sig):
        raise ValueError(
            "Missing type information. It should either be provided as an argument to pandas_wrap,"
            "or as a python typing hint")
    if return_col is not None:
        rtype = _to_stype(return_col)
    if return_sig is not None:
        rtype = _to_stype(return_sig)

    # Extract the input signatures, if any:
    sig_args = []
    sig_kwargs = {}
    for key in spec.args:
        t = spec.annotations.get(key, None)
        if t is not None:
            dt = _to_stype(t)
        else:
            dt = _Unknown(None)
        sig_kwargs[key] = dt
        sig_args.append(dt)
    f.sig_return = rtype
    f.sig_args = sig_args
    f.sig_kwargs = sig_kwargs
    return decorate(f, make_fun)
