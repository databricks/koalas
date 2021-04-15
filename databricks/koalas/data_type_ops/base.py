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

from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegralType,
    ShortType,
    StringType,
    TimestampType,
)

from databricks.koalas.base import column_op, IndexOpsMixin
from databricks.koalas.typedef import Dtype, extension_dtypes


class DataTypeOps(object, metaclass=ABCMeta):
    """The base class for binary operations of Koalas objects (of different data types)."""

    def __new__(cls, dtype: Dtype, spark_type: DataType):
        from databricks.koalas.data_type_ops.boolean_ops import BooleanOps
        from databricks.koalas.data_type_ops.categorical_ops import CategoricalOps
        from databricks.koalas.data_type_ops.date_ops import DateOps
        from databricks.koalas.data_type_ops.datetime_ops import DatetimeOps
        from databricks.koalas.data_type_ops.num_ops import (
            IntegralOps,
            FractionalOps,
            DecimalOps,
        )
        from databricks.koalas.data_type_ops.string_ops import StringOps

        if isinstance(dtype, CategoricalDtype):
            return object.__new__(CategoricalOps)
        if dtype == np.dtype("object"):
            if isinstance(spark_type, DecimalType):
                return object.__new__(DecimalOps)
            elif isinstance(spark_type, DateType):
                return object.__new__(DateOps)
            elif isinstance(spark_type, StringType):
                return object.__new__(StringOps)
            else:
                raise TypeError("Type %s cannot be inferred." % dtype)
        elif isinstance(spark_type, FloatType) or isinstance(spark_type, DoubleType):
            return object.__new__(FractionalOps)
        elif isinstance(spark_type, DecimalType):
            return object.__new__(DecimalOps)
        elif (
            isinstance(spark_type, IntegralType)
            or isinstance(spark_type, ByteType)
            or isinstance(spark_type, ShortType)
        ):
            return object.__new__(IntegralOps)
        elif isinstance(spark_type, StringType):
            return object.__new__(StringOps)
        elif isinstance(spark_type, BooleanType):
            return object.__new__(BooleanOps)
        elif isinstance(spark_type, TimestampType):
            return object.__new__(DatetimeOps)
        elif isinstance(spark_type, DateType):
            return object.__new__(DateOps)
        else:
            raise TypeError("Type %s was not understood." % dtype)

    def __init__(self, dtype: Dtype, spark_type: DataType):
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

    @abstractmethod
    def __truediv__(self, left, right):
        raise NotImplementedError()

    @abstractmethod
    def __floordiv__(self, left, right):
        raise NotImplementedError()

    @abstractmethod
    def __mod__(self, left, right):
        raise NotImplementedError()

    @abstractmethod
    def __pow__(self, left, right):
        raise NotImplementedError()

    @abstractmethod
    def __radd__(self, left, right=None):
        raise NotImplementedError()

    @abstractmethod
    def __rsub__(self, left, right=None):
        raise NotImplementedError()

    @abstractmethod
    def __rmul__(self, left, right=None):
        raise NotImplementedError()

    @abstractmethod
    def __rtruediv__(self, left, right=None):
        raise NotImplementedError()

    @abstractmethod
    def __rfloordiv__(self, left, right=None):
        raise NotImplementedError()

    @abstractmethod
    def __rpow__(self, left, right=None):
        raise NotImplementedError()

    @abstractmethod
    def __rmod__(self, left, right=None):
        raise NotImplementedError()

    def __and__(self, left, right):
        if isinstance(left.dtype, extension_dtypes) or (
            isinstance(right, IndexOpsMixin) and isinstance(right.dtype, extension_dtypes)
        ):

            def and_func(left, right):
                if not isinstance(right, spark.Column):
                    if pd.isna(right):
                        right = F.lit(None)
                    else:
                        right = F.lit(right)
                return left & right

        else:

            def and_func(left, right):
                if not isinstance(right, spark.Column):
                    if pd.isna(right):
                        right = F.lit(None)
                    else:
                        right = F.lit(right)
                scol = left & right
                return F.when(scol.isNull(), False).otherwise(scol)

        return column_op(and_func)(left, right)

    def __rand__(self, left, right=None):
        return self.__and__(left, right)

    def __or__(self, left, right):
        if isinstance(left.dtype, extension_dtypes) or (
            isinstance(right, IndexOpsMixin) and isinstance(right.dtype, extension_dtypes)
        ):

            def or_func(left, right):
                if not isinstance(right, spark.Column):
                    if pd.isna(right):
                        right = F.lit(None)
                    else:
                        right = F.lit(right)
                return left | right

        else:

            def or_func(left, right):
                if not isinstance(right, spark.Column) and pd.isna(right):
                    return F.lit(False)
                else:
                    scol = left | F.lit(right)
                    return F.when(left.isNull() | scol.isNull(), False).otherwise(scol)

        return column_op(or_func)(left, right)

    def __ror__(self, left, right=None):
        return self.__or__(left, right)
