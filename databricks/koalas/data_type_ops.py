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
Classes for binary operations between Koalas objects, classified by object data types.
"""
from abc import ABCMeta, abstractmethod
import datetime
import decimal
import numpy as np

from pandas.api.types import CategoricalDtype
import pyspark.sql.types as types
from pyspark.sql import Column, functions as F

from databricks.koalas.base import column_op, IndexOpsMixin
from databricks.koalas.typedef import Dtype


class DataTypeOps(object, metaclass=ABCMeta):
    """The base class for binary operations of Koalas objects (of different data types)."""

    def __new__(cls, dtype: Dtype, spark_type: types.DataType):
        if dtype in (np.float32,) or (dtype in (float, np.float, np.float64)):
            # FloatType, DoubleType
            return object.__new__(FractionalOps)
        elif dtype in (decimal.Decimal,):
            return object.__new__(DecimalOps)
        elif (
            dtype in (int, np.int, np.int64)
            or (dtype in (np.int32,))
            or (dtype in (np.int8, np.byte))
            or (dtype in (np.int16,))
        ):
            # LongType, IntegerType, ByteType, ShortType
            return object.__new__(IntegralOps)
        elif isinstance(dtype, CategoricalDtype):
            return object.__new__(CategoricalOps)
        elif dtype in (np.unicode_,):
            return object.__new__(StringOps)
        elif dtype in (bool, np.bool):
            return object.__new__(BooleanOps)
        elif dtype in (datetime.datetime, np.datetime64):
            return object.__new__(DatetimeOps)
        elif dtype == np.dtype("object"):
            if isinstance(spark_type, types.DecimalType):
                return object.__new__(DecimalOps)
            elif isinstance(spark_type, types.DateType):
                return object.__new__(DatetimeOps)
            else:
                raise TypeError("Type %s cannot be inferred." % dtype)
        else:
            raise TypeError("Type %s was not understood." % dtype)

    def __init__(self, dtype: Dtype, spark_type: types.DataType):
        self.dtype = dtype
        self.spark_type = spark_type

    @abstractmethod
    def __add__(self, left, right):
        raise NotImplementedError()

    # @abstractmethod
    def __sub__(self, left, right):
        raise NotImplementedError()

    # @abstractmethod
    def __mul__(self, left, right):
        raise NotImplementedError()

    # @abstractmethod
    def __truediv__(self, left, right):
        raise NotImplementedError()

    # @abstractmethod
    def __floordiv__(self, left, right):
        raise NotImplementedError()

    # @abstractmethod
    def __mod__(self, left, right):
        raise NotImplementedError()

    # @abstractmethod
    def __pow__(self, left, right):
        raise NotImplementedError()


class NumericOps(DataTypeOps):
    """
    The class for binary operations of numeric Koalas objects.
    """

    pass


class IntegralOps(NumericOps):
    """
    The class for binary operations of Koalas objects with spark types: LongType, IntegerType,
    ByteType, and ShortType.
    """

    def __add__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, types.StringType)
        ) or isinstance(right, str):
            raise TypeError("string addition can only be applied to string series or literals.")
        return column_op(Column.__add__)(left, right)

    def __sub__(self, left, right):
        return column_op(Column.__sub__)(left, right)

    def __mul__(self, left, right):
        return column_op(Column.__mul__)(left, right)

    def __truediv__(self, left, right):
        return column_op(Column.__truediv__)(left, right)

    def __floordiv__(self, left, right):
        return column_op(Column.__floordiv__)(left, right)

    def __mod__(self, left, right):
        return column_op(Column.__mod__)(left, right)

    def __pow__(self, left, right):
        raise column_op(Column.__pow__)(left, right)


class FractionalOps(NumericOps):
    """
    The class for binary operations of Koalas objects with spark types: FloatType and DoubleType.
    """

    def __add__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, types.StringType)
        ) or isinstance(right, str):
            raise TypeError("string addition can only be applied to string series or literals.")
        return column_op(Column.__add__)(left, right)

    def __sub__(self, left, right):
        return column_op(Column.__sub__)(left, right)

    def __mul__(self, left, right):
        return column_op(Column.__mul__)(left, right)

    def __truediv__(self, left, right):
        return column_op(Column.__truediv__)(left, right)

    def __floordiv__(self, left, right):
        return column_op(Column.__floordiv__)(left, right)

    def __mod__(self, left, right):
        return column_op(Column.__mod__)(left, right)

    def __pow__(self, left, right):
        raise column_op(Column.__pow__)(left, right)


class DecimalOps(FractionalOps):
    """
    The class for binary operations of Koalas objects with spark type: DecimalType.
    """

    def __add__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, types.StringType)
        ) or isinstance(right, str):
            raise TypeError("string addition can only be applied to string series or literals.")

        return column_op(Column.__add__)(left, right)

    def __sub__(self, left, right):
        return column_op(Column.__sub__)(left, right)

    def __mul__(self, left, right):
        return column_op(Column.__mul__)(left, right)

    def __truediv__(self, left, right):
        return column_op(Column.__truediv__)(left, right)

    def __floordiv__(self, left, right):
        return column_op(Column.__floordiv__)(left, right)

    def __mod__(self, left, right):
        return column_op(Column.__mod__)(left, right)

    def __pow__(self, left, right):
        raise column_op(Column.__pow__)(left, right)


class StringOps(DataTypeOps):
    """
    The class for binary operations of Koalas objects with spark type: StringType.
    """

    def __add__(self, left, right):
        if isinstance(right.spark.data_type, types.StringType):
            return column_op(F.concat)(left, right)
        elif isinstance(right, str):
            return column_op(F.concat)(left, F.lit(right))
        else:
            raise TypeError("string addition can only be applied to string series or literals.")

    def __sub__(self, left, right):
        return column_op(Column.__sub__)(left, right)

    def __mul__(self, left, right):
        return column_op(Column.__mul__)(left, right)

    def __truediv__(self, left, right):
        return column_op(Column.__truediv__)(left, right)

    def __floordiv__(self, left, right):
        return column_op(Column.__floordiv__)(left, right)

    def __mod__(self, left, right):
        return column_op(Column.__mod__)(left, right)

    def __pow__(self, left, right):
        raise column_op(Column.__pow__)(left, right)


class CategoricalOps(DataTypeOps):
    """
    The class for binary operations of Koalas objects with categorical types.
    """

    def __add__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, types.StringType)
        ) or isinstance(right, str):
            raise TypeError("string addition can only be applied to string series or literals.")
        return column_op(Column.__add__)(left, right)

    def __sub__(self, left, right):
        return column_op(Column.__sub__)(left, right)

    def __mul__(self, left, right):
        return column_op(Column.__mul__)(left, right)

    def __truediv__(self, left, right):
        return column_op(Column.__truediv__)(left, right)

    def __floordiv__(self, left, right):
        return column_op(Column.__floordiv__)(left, right)

    def __mod__(self, left, right):
        return column_op(Column.__mod__)(left, right)

    def __pow__(self, left, right):
        raise column_op(Column.__pow__)(left, right)


class BooleanOps(DataTypeOps):
    """
    The class for binary operations of Koalas objects with spark type: BooleanType.
    """

    def __add__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, types.StringType)
        ) or isinstance(right, str):
            raise TypeError("string addition can only be applied to string series or literals.")
        return column_op(Column.__add__)(left, right)

    def __sub__(self, left, right):
        return column_op(Column.__sub__)(left, right)

    def __mul__(self, left, right):
        return column_op(Column.__mul__)(left, right)

    def __truediv__(self, left, right):
        return column_op(Column.__truediv__)(left, right)

    def __floordiv__(self, left, right):
        return column_op(Column.__floordiv__)(left, right)

    def __mod__(self, left, right):
        return column_op(Column.__mod__)(left, right)

    def __pow__(self, left, right):
        raise column_op(Column.__pow__)(left, right)


class DatetimeOps(DataTypeOps):
    """
    The class for binary operations of Koalas objects with spark type: TimestampType.
    """

    def __add__(self, left, right):
        raise TypeError("addition can not be applied to date times.")

    def __sub__(self, left, right):
        return column_op(Column.__sub__)(left, right)

    def __mul__(self, left, right):
        return column_op(Column.__mul__)(left, right)

    def __truediv__(self, left, right):
        return column_op(Column.__truediv__)(left, right)

    def __floordiv__(self, left, right):
        return column_op(Column.__floordiv__)(left, right)

    def __mod__(self, left, right):
        return column_op(Column.__mod__)(left, right)

    def __pow__(self, left, right):
        raise column_op(Column.__pow__)(left, right)
