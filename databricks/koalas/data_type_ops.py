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
import warnings

from pandas.api.types import CategoricalDtype
import pyspark.sql.types as types
from pyspark.sql import Column, functions as F

from databricks.koalas.base import column_op, IndexOpsMixin
from databricks.koalas.spark import functions as SF
from databricks.koalas.typedef import Dtype, as_spark_type


class DataTypeOps(object, metaclass=ABCMeta):
    """The base class for binary operations of Koalas objects (of different data types)."""

    def __new__(cls, dtype: Dtype, spark_type: types.DataType):
        if dtype == np.dtype("object"):
            if isinstance(spark_type, types.DecimalType):
                return object.__new__(DecimalOps)
            elif isinstance(spark_type, types.DateType):
                return object.__new__(DatetimeOps)
            elif isinstance(spark_type, types.StringType):
                return object.__new__(StringOps)
            else:
                raise TypeError("Type %s cannot be inferred." % dtype)
        elif dtype in (np.float32,) or (dtype in (float, np.float, np.float64)):
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
        elif dtype in (datetime.date,):
            return object.__new__(DateOps)
        else:
            raise TypeError("Type %s was not understood." % dtype)

    def __init__(self, dtype: Dtype, spark_type: types.DataType):
        self.dtype = dtype
        self.spark_type = spark_type

    @abstractmethod
    def __add__(self, left, right):
        raise NotImplementedError()

    @abstractmethod
    def __sub__(self, left, right):
        raise NotImplementedError()

    @abstractmethod
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

    def __add__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, types.StringType)
        ) or isinstance(right, str):
            raise TypeError("string addition can only be applied to string series or literals.")
        return column_op(Column.__add__)(left, right)

    def __sub__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, types.StringType)
        ) or isinstance(right, str):
            raise TypeError("substraction can not be applied to string series or literals.")
        return column_op(Column.__sub__)(left, right)

    def __mul__(self, left, right):
        if isinstance(right, str):
            raise TypeError("multiplication can not be applied to a string literal.")

        if isinstance(right.spark.data_type, types.TimestampType):
            raise TypeError("multiplication can not be applied to date times.")


class IntegralOps(NumericOps):
    """
    The class for binary operations of Koalas objects with spark types: LongType, IntegerType,
    ByteType, and ShortType.
    """

    def __mul__(self, left, right):
        if isinstance(right, str):
            raise TypeError("multiplication can not be applied to a string literal.")

        if isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, types.StringType):
            return column_op(SF.repeat)(right, left)
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
        if isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, types.StringType):
            return column_op(F.concat)(left, right)
        elif isinstance(right, str):
            return column_op(F.concat)(left, F.lit(right))
        else:
            raise TypeError("string addition can only be applied to string series or literals.")

    def __sub__(self, left, right):
        raise TypeError("substraction can not be applied to string series or literals.")

    def __mul__(self, left, right):
        if isinstance(right, str):
            raise TypeError("multiplication can not be applied to a string literal.")

        if (
            isinstance(right, IndexOpsMixin)
            and isinstance(right.spark.data_type, types.IntegralType)
        ) or isinstance(right, int):
            return column_op(SF.repeat)(left, right)
        else:
            raise TypeError("a string series can only be multiplied to an int series or literal")

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
        raise TypeError("Object with dtype category cannot perform the numpy op add")

    def __sub__(self, left, right):
        raise TypeError("Object with dtype category cannot perform the numpy op subtract")

    def __mul__(self, left, right):
        raise TypeError("Object with dtype category cannot perform the numpy op multiply")

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
        raise TypeError("numpy boolean subtract, the `-` operator, is not supported")

    def __mul__(self, left, right):
        if isinstance(right, str):
            raise TypeError("multiplication can not be applied to a string literal.")

        if isinstance(right.spark.data_type, types.TimestampType):
            raise TypeError("multiplication can not be applied to date times.")

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
        # Note that timestamp subtraction casts arguments to integer. This is to mimic pandas's
        # behaviors. pandas returns 'timedelta64[ns]' from 'datetime64[ns]'s subtraction.
        msg = (
            "Note that there is a behavior difference of timestamp subtraction. "
            "The timestamp subtraction returns an integer in seconds, "
            "whereas pandas returns 'timedelta64[ns]'."
        )
        if isinstance(right, IndexOpsMixin) and isinstance(
            right.spark.data_type, types.TimestampType
        ):
            warnings.warn(msg, UserWarning)
            return left.astype("long") - right.astype("long")
        elif isinstance(right, datetime.datetime):
            warnings.warn(msg, UserWarning)
            return left.astype("long") - F.lit(right).cast(as_spark_type("long"))
        else:
            raise TypeError("datetime subtraction can only be applied to datetime series.")

    def __mul__(self, left, right):
        raise TypeError("cannot perform __mul__ with this index type: DatetimeArray")

    def __truediv__(self, left, right):
        return column_op(Column.__truediv__)(left, right)

    def __floordiv__(self, left, right):
        return column_op(Column.__floordiv__)(left, right)

    def __mod__(self, left, right):
        return column_op(Column.__mod__)(left, right)

    def __pow__(self, left, right):
        raise column_op(Column.__pow__)(left, right)


class DateOps(DataTypeOps):
    """
    The class for binary operations of Koalas objects with spark type: DateType.
    """

    def __add__(self, left, right):
        raise TypeError("addition can not be applied to date.")

    def __sub__(self, left, right):
        # Note that date subtraction casts arguments to integer. This is to mimic pandas's
        # behaviors. pandas returns 'timedelta64[ns]' in days from date's subtraction.
        msg = (
            "Note that there is a behavior difference of date subtraction. "
            "The date subtraction returns an integer in days, "
            "whereas pandas returns 'timedelta64[ns]'."
        )
        if isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, types.DateType):
            warnings.warn(msg, UserWarning)
            return column_op(F.datediff)(left, right).astype("long")
        elif isinstance(right, datetime.date) and not isinstance(right, datetime.datetime):
            warnings.warn(msg, UserWarning)
            return column_op(F.datediff)(self, F.lit(right)).astype("long")
        else:
            raise TypeError("date subtraction can only be applied to date series.")

    def __mul__(self, left, right):
        raise TypeError("multiplication can not be applied to date.")
