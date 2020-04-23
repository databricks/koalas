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
import typing
import datetime
from inspect import getfullargspec

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_datetime64tz_dtype
import pyarrow as pa
import pyspark.sql.types as types
from pyspark.sql.types import UserDefinedType

try:
    from pyspark.sql.types import to_arrow_type, from_arrow_type
except ImportError:
    from pyspark.sql.pandas.types import to_arrow_type, from_arrow_type

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.typedef.string_typehints import resolve_string_type_hint


# A column of data, with the data type.
class SeriesType(object):
    def __init__(self, tpe):
        self.tpe = tpe  # type: types.DataType

    def __repr__(self):
        return "SeriesType[{}]".format(self.tpe)


class DataFrameType(object):
    def __init__(self, tpe):
        # Seems we cannot specify field names. I currently gave some default names
        # `c0, c1, ... cn`.
        self.tpe = types.StructType(
            [types.StructField("c%s" % i, tpe[i]) for i in range(len(tpe))]
        )  # type: types.StructType

    def __repr__(self):
        return "DataFrameType[{}]".format(self.tpe)


# The type is a scalar type that is furthermore understood by Spark.
class ScalarType(object):
    def __init__(self, tpe):
        self.tpe = tpe  # type: types.DataType

    def __repr__(self):
        return "ScalarType[{}]".format(self.tpe)


# The type is left unspecified or we do not know about this type.
class UnknownType(object):
    def __init__(self, tpe):
        self.tpe = tpe

    def __repr__(self):
        return "UnknownType[{}]".format(self.tpe)


def _to_stype(tpe) -> typing.Union[SeriesType, DataFrameType, ScalarType, UnknownType]:
    if isinstance(tpe, str):
        # This type hint can happen when given hints are string to avoid forward reference.
        tpe = resolve_string_type_hint(tpe)
    if hasattr(tpe, "__origin__") and tpe.__origin__ == ks.Series:
        inner = as_spark_type(tpe.__args__[0])
        return SeriesType(inner)
    if hasattr(tpe, "__origin__") and tpe.__origin__ == ks.DataFrame:
        tuple_type = tpe.__args__[0]
        if hasattr(tuple_type, "__tuple_params__"):
            # Python 3.5.0 to 3.5.2 has '__tuple_params__' instead.
            # See https://github.com/python/cpython/blob/v3.5.2/Lib/typing.py
            parameters = getattr(tuple_type, "__tuple_params__")
        else:
            parameters = getattr(tuple_type, "__args__")
        return DataFrameType([as_spark_type(t) for t in parameters])
    inner = as_spark_type(tpe)
    if inner is None:
        return UnknownType(tpe)
    else:
        return ScalarType(inner)


def as_spark_type(tpe) -> types.DataType:
    """
    Given a python type, returns the equivalent spark type.
    Accepts:
    - the built-in types in python
    - the built-in types in numpy
    - list of pairs of (field_name, type)
    - dictionaries of field_name -> type
    - python3's typing system
    """
    if tpe in (str, "str", "string"):
        return types.StringType()
    elif tpe in (bytes,):
        return types.BinaryType()
    elif tpe in (np.int8, "int8", "byte"):
        return types.ByteType()
    elif tpe in (np.int16, "int16", "short"):
        return types.ShortType()
    elif tpe in (int, "int", np.int, np.int32):
        return types.IntegerType()
    elif tpe in (np.int64, "int64", "long", "bigint"):
        return types.LongType()
    elif tpe in (float, "float", np.float):
        return types.FloatType()
    elif tpe in (np.float64, "float64", "double"):
        return types.DoubleType()
    elif tpe in (datetime.datetime, np.datetime64):
        return types.TimestampType()
    elif tpe in (datetime.date,):
        return types.DateType()
    elif tpe in (bool, "boolean", "bool", np.bool):
        return types.BooleanType()
    elif tpe in ():
        return types.ArrayType(types.StringType())


def spark_type_to_pandas_dtype(spark_type):
    """ Return the given Spark DataType to pandas dtype. """
    if isinstance(spark_type, UserDefinedType):
        return np.dtype("object")
    elif isinstance(spark_type, types.TimestampType):
        return np.dtype("datetime64[ns]")
    else:
        return np.dtype(to_arrow_type(spark_type).to_pandas_dtype())


def infer_pd_series_spark_type(s: pd.Series) -> types.DataType:
    """Infer Spark DataType from pandas Series dtype.

    :param s: :class:`pandas.Series` to be inferred
    :return: the inferred Spark data type
    """
    dt = s.dtype
    if dt == np.dtype("object"):
        if len(s) == 0 or s.isnull().all():
            raise ValueError("can not infer schema from empty or null dataset")
        elif hasattr(s[0], "__UDT__"):
            return s[0].__UDT__
        else:
            return from_arrow_type(pa.Array.from_pandas(s).type)
    elif is_datetime64_dtype(dt) or is_datetime64tz_dtype(dt):
        return types.TimestampType()
    else:
        return from_arrow_type(pa.from_numpy_dtype(dt))


def infer_return_type(
    f, return_col=None, return_scalar=None
) -> typing.Union[SeriesType, DataFrameType, ScalarType, UnknownType]:
    """
    >>> def func() -> int:
    ...    pass
    >>> infer_return_type(func).tpe
    IntegerType

    >>> def func() -> ks.Series[int]:
    ...    pass
    >>> infer_return_type(func).tpe
    IntegerType

    >>> def func() -> ks.DataFrame[np.float, str]:
    ...    pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(c0,FloatType,true),StructField(c1,StringType,true)))

    >>> def func() -> ks.DataFrame[np.float]:
    ...    pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(c0,FloatType,true)))

    >>> def func() -> 'int':
    ...    pass
    >>> infer_return_type(func).tpe
    IntegerType

    >>> def func() -> 'ks.Series[int]':
    ...    pass
    >>> infer_return_type(func).tpe
    IntegerType

    >>> def func() -> 'ks.DataFrame[np.float, str]':
    ...    pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(c0,FloatType,true),StructField(c1,StringType,true)))

    >>> def func() -> 'ks.DataFrame[np.float]':
    ...    pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(c0,FloatType,true)))
    """
    spec = getfullargspec(f)
    return_sig = spec.annotations.get("return", None)

    if not (return_col or return_sig or return_scalar):
        raise ValueError(
            "Missing type information. It should either be provided as an argument to "
            "pandas_wraps, or as a python typing hint"
        )
    if return_col is not None:
        if isinstance(return_col, ks.Series):
            return _to_stype(return_col)
        inner = as_spark_type(return_col)
        return SeriesType(inner)
    if return_scalar is not None:
        if isinstance(return_scalar, ks.Series):
            raise ValueError(
                "Column return type {}, you should use 'return_col' to specify"
                " it.".format(return_scalar)
            )
        inner = as_spark_type(return_scalar)
        return ScalarType(inner)
    if return_sig is not None:
        return _to_stype(return_sig)
    assert False
