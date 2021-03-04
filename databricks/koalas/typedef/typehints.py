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
import decimal
from inspect import getfullargspec, isclass

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas.api.extensions import ExtensionDtype

try:
    from pandas import Int8Dtype, Int16Dtype, Int32Dtype, Int64Dtype

    extension_dtypes_available = True
    extension_dtypes = (Int8Dtype, Int16Dtype, Int32Dtype, Int64Dtype)  # type: typing.Tuple

    try:
        from pandas import BooleanDtype, StringDtype

        extension_object_dtypes_available = True
        extension_dtypes += (BooleanDtype, StringDtype)
    except ImportError:
        extension_object_dtypes_available = False

    try:
        from pandas import Float32Dtype, Float64Dtype

        extension_float_dtypes_available = True
        extension_dtypes += (Float32Dtype, Float64Dtype)
    except ImportError:
        extension_float_dtypes_available = False

except ImportError:
    extension_dtypes_available = False
    extension_object_dtypes_available = False
    extension_float_dtypes_available = False
    extension_dtypes = ()

import pyarrow as pa
import pyspark.sql.types as types

try:
    from pyspark.sql.types import to_arrow_type, from_arrow_type
except ImportError:
    from pyspark.sql.pandas.types import to_arrow_type, from_arrow_type

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.typedef.string_typehints import resolve_string_type_hint

T = typing.TypeVar("T")

Scalar = typing.Union[
    int, float, bool, str, bytes, decimal.Decimal, datetime.date, datetime.datetime, None
]

Dtype = typing.Union[np.dtype, ExtensionDtype]


# A column of data, with the data type.
class SeriesType(typing.Generic[T]):
    def __init__(self, tpe):
        self.tpe = tpe  # type: types.DataType

    def __repr__(self):
        return "SeriesType[{}]".format(self.tpe)


class DataFrameType(object):
    def __init__(self, tpe, names=None):
        if names is None:
            # Default names `c0, c1, ... cn`.
            self.tpe = types.StructType(
                [types.StructField("c%s" % i, tpe[i]) for i in range(len(tpe))]
            )  # type: types.StructType
        else:
            self.tpe = types.StructType(
                [types.StructField(n, t) for n, t in zip(names, tpe)]
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


class NameTypeHolder(object):
    name = None
    tpe = None


def as_spark_type(
    tpe: typing.Union[str, type, Dtype], *, raise_error: bool = True
) -> types.DataType:
    """
    Given a Python type, returns the equivalent spark type.
    Accepts:
    - the built-in types in Python
    - the built-in types in numpy
    - list of pairs of (field_name, type)
    - dictionaries of field_name -> type
    - Python3's typing system
    """
    # TODO: Add "boolean" and "string" types.
    # ArrayType
    if tpe in (np.ndarray,):
        return types.ArrayType(types.StringType())
    elif hasattr(tpe, "__origin__") and issubclass(tpe.__origin__, list):  # type: ignore
        element_type = as_spark_type(tpe.__args__[0], raise_error=raise_error)  # type: ignore
        if element_type is None:
            return None
        return types.ArrayType(element_type)
    # BinaryType
    elif tpe in (bytes, np.character, np.bytes_, np.string_):
        return types.BinaryType()
    # BooleanType
    elif tpe in (bool, np.bool, "bool", "?"):
        return types.BooleanType()
    # DateType
    elif tpe in (datetime.date,):
        return types.DateType()
    # NumericType
    elif tpe in (np.int8, np.byte, "int8", "byte", "b"):
        return types.ByteType()
    elif tpe in (decimal.Decimal,):
        # TODO: considering about the precision & scale for decimal type.
        return types.DecimalType(38, 18)
    elif tpe in (float, np.float, np.float64, "float", "float64", "double"):
        return types.DoubleType()
    elif tpe in (np.float32, "float32", "f"):
        return types.FloatType()
    elif tpe in (np.int32, "int32", "i"):
        return types.IntegerType()
    elif tpe in (int, np.int, np.int64, "int", "int64", "long"):
        return types.LongType()
    elif tpe in (np.int16, "int16", "short"):
        return types.ShortType()
    # StringType
    elif tpe in (str, np.unicode_, "str", "U"):
        return types.StringType()
    # TimestampType
    elif tpe in (datetime.datetime, np.datetime64, "datetime64[ns]", "M"):
        return types.TimestampType()

    # categorical types
    elif isinstance(tpe, CategoricalDtype) or (isinstance(tpe, str) and type == "category"):
        return types.LongType()

    # extension types
    elif extension_dtypes_available:
        # IntegralType
        if isinstance(tpe, Int8Dtype) or (isinstance(tpe, str) and tpe == "Int8"):
            return types.ByteType()
        elif isinstance(tpe, Int16Dtype) or (isinstance(tpe, str) and tpe == "Int16"):
            return types.ShortType()
        elif isinstance(tpe, Int32Dtype) or (isinstance(tpe, str) and tpe == "Int32"):
            return types.IntegerType()
        elif isinstance(tpe, Int64Dtype) or (isinstance(tpe, str) and tpe == "Int64"):
            return types.LongType()

        if extension_object_dtypes_available:
            # BooleanType
            if isinstance(tpe, BooleanDtype) or (isinstance(tpe, str) and tpe == "boolean"):
                return types.BooleanType()
            # StringType
            elif isinstance(tpe, StringDtype) or (isinstance(tpe, str) and tpe == "string"):
                return types.StringType()

        if extension_float_dtypes_available:
            # FractionalType
            if isinstance(tpe, Float32Dtype) or (isinstance(tpe, str) and tpe == "Float32"):
                return types.FloatType()
            elif isinstance(tpe, Float64Dtype) or (isinstance(tpe, str) and tpe == "Float64"):
                return types.DoubleType()

    if raise_error:
        raise TypeError("Type %s was not understood." % tpe)
    else:
        return None


def spark_type_to_pandas_dtype(
    spark_type: types.DataType, *, use_extension_dtypes: bool = False
) -> Dtype:
    """ Return the given Spark DataType to pandas dtype. """

    if use_extension_dtypes and extension_dtypes_available:
        # IntegralType
        if isinstance(spark_type, types.ByteType):
            return Int8Dtype()
        elif isinstance(spark_type, types.ShortType):
            return Int16Dtype()
        elif isinstance(spark_type, types.IntegerType):
            return Int32Dtype()
        elif isinstance(spark_type, types.LongType):
            return Int64Dtype()

        if extension_object_dtypes_available:
            # BooleanType
            if isinstance(spark_type, types.BooleanType):
                return BooleanDtype()
            # StringType
            elif isinstance(spark_type, types.StringType):
                return StringDtype()

        # FractionalType
        if extension_float_dtypes_available:
            if isinstance(spark_type, types.FloatType):
                return Float32Dtype()
            elif isinstance(spark_type, types.DoubleType):
                return Float64Dtype()

    if isinstance(
        spark_type, (types.DateType, types.NullType, types.StructType, types.UserDefinedType)
    ):
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
            return types.NullType()
        elif hasattr(s.iloc[0], "__UDT__"):
            return s.iloc[0].__UDT__
        else:
            return from_arrow_type(pa.Array.from_pandas(s).type)
    elif isinstance(dt, CategoricalDtype):
        return as_spark_type(s.cat.codes.dtype)
    else:
        return as_spark_type(dt)


def infer_return_type(f) -> typing.Union[SeriesType, DataFrameType, ScalarType, UnknownType]:
    """
    >>> def func() -> int:
    ...    pass
    >>> infer_return_type(func).tpe
    LongType

    >>> def func() -> ks.Series[int]:
    ...    pass
    >>> infer_return_type(func).tpe
    LongType

    >>> def func() -> ks.DataFrame[np.float, str]:
    ...    pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(c0,DoubleType,true),StructField(c1,StringType,true)))

    >>> def func() -> ks.DataFrame[np.float]:
    ...    pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(c0,DoubleType,true)))

    >>> def func() -> 'int':
    ...    pass
    >>> infer_return_type(func).tpe
    LongType

    >>> def func() -> 'ks.Series[int]':
    ...    pass
    >>> infer_return_type(func).tpe
    LongType

    >>> def func() -> 'ks.DataFrame[np.float, str]':
    ...    pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(c0,DoubleType,true),StructField(c1,StringType,true)))

    >>> def func() -> 'ks.DataFrame[np.float]':
    ...    pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(c0,DoubleType,true)))

    >>> def func() -> ks.DataFrame['a': np.float, 'b': int]:
    ...     pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(a,DoubleType,true),StructField(b,LongType,true)))

    >>> def func() -> "ks.DataFrame['a': np.float, 'b': int]":
    ...     pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(a,DoubleType,true),StructField(b,LongType,true)))

    >>> pdf = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    >>> def func() -> ks.DataFrame[pdf.dtypes]:
    ...     pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(c0,LongType,true),StructField(c1,LongType,true)))

    >>> pdf = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})
    >>> def func() -> ks.DataFrame[zip(pdf.columns, pdf.dtypes)]:
    ...     pass
    >>> infer_return_type(func).tpe
    StructType(List(StructField(a,LongType,true),StructField(b,LongType,true)))
    """
    # We should re-import to make sure the class 'SeriesType' is not treated as a class
    # within this module locally. See Series.__class_getitem__ which imports this class
    # canonically.
    from databricks.koalas.typedef import SeriesType, NameTypeHolder

    spec = getfullargspec(f)
    tpe = spec.annotations.get("return", None)
    if isinstance(tpe, str):
        # This type hint can happen when given hints are string to avoid forward reference.
        tpe = resolve_string_type_hint(tpe)
    if hasattr(tpe, "__origin__") and (
        issubclass(tpe.__origin__, SeriesType) or tpe.__origin__ == ks.Series
    ):
        # TODO: remove "tpe.__origin__ == ks.Series" when we drop Python 3.5 and 3.6.
        inner = as_spark_type(tpe.__args__[0])
        return SeriesType(inner)

    if hasattr(tpe, "__origin__") and tpe.__origin__ == ks.DataFrame:
        # When Python version is lower then 3.7. Unwrap it to a Tuple type
        # hints.
        tpe = tpe.__args__[0]

    # Note that, DataFrame type hints will create a Tuple.
    # Python 3.6 has `__name__`. Python 3.7 and 3.8 have `_name`.
    # Check if the name is Tuple.
    name = getattr(tpe, "_name", getattr(tpe, "__name__", None))
    if name == "Tuple":
        tuple_type = tpe
        if hasattr(tuple_type, "__tuple_params__"):
            # Python 3.5.0 to 3.5.2 has '__tuple_params__' instead.
            # See https://github.com/python/cpython/blob/v3.5.2/Lib/typing.py
            parameters = getattr(tuple_type, "__tuple_params__")
        else:
            parameters = getattr(tuple_type, "__args__")
        if len(parameters) > 0 and all(
            isclass(p) and issubclass(p, NameTypeHolder) for p in parameters
        ):
            names = [p.name for p in parameters if issubclass(p, NameTypeHolder)]
            types = [p.tpe for p in parameters if issubclass(p, NameTypeHolder)]
            return DataFrameType([as_spark_type(t) for t in types], names)
        return DataFrameType([as_spark_type(t) for t in parameters])
    inner = as_spark_type(tpe)
    if inner is None:
        return UnknownType(tpe)
    else:
        return ScalarType(inner)
