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
from pandas.api.types import CategoricalDtype

from pyspark.sql.types import (
    BooleanType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegralType,
    StringType,
    TimestampType,
)

from databricks.koalas.typedef import Dtype


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
        )
        from databricks.koalas.data_type_ops.string_ops import StringOps

        if isinstance(dtype, CategoricalDtype):
            return object.__new__(CategoricalOps)
        elif (
            isinstance(spark_type, FloatType)
            or isinstance(spark_type, DoubleType)
            or isinstance(spark_type, DecimalType)
        ):
            return object.__new__(FractionalOps)
        elif isinstance(spark_type, IntegralType):
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

    def restore(self, col):
        """Restore column when to_pandas."""
        return col

    def prepare(self, col):
        """Prepare column when from_pandas"""
        return col.replace({np.nan: None})
