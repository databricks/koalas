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
import numbers

import numpy as np
from pandas.api.types import CategoricalDtype

from pyspark.sql import Column, functions as F
from pyspark.sql.types import (
    NumericType,
    StringType,
    TimestampType,
)

from databricks.koalas.base import column_op, IndexOpsMixin, numpy_column_op
from databricks.koalas.data_type_ops.base import DataTypeOps


class BooleanOps(DataTypeOps):
    """
    The class for binary operations of Koalas objects with spark type: BooleanType.
    """

    def __add__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, StringType)
        ) or isinstance(right, str):
            raise TypeError("string addition can only be applied to string series or literals.")

        if (
            isinstance(right, IndexOpsMixin)
            and (
                isinstance(right.dtype, CategoricalDtype)
                or (not isinstance(right.spark.data_type, NumericType))
            )
        ) and not isinstance(right, numbers.Number):
            raise TypeError("addition can not be applied to given types.")

        return column_op(Column.__add__)(left, right)

    def __sub__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin)
            and (
                isinstance(right.dtype, CategoricalDtype)
                or (not isinstance(right.spark.data_type, NumericType))
            )
        ) and not isinstance(right, numbers.Number):
            raise TypeError("substraction can not be applied to given types.")

        return column_op(Column.__sub__)(left, right)

    def __mul__(self, left, right):
        if isinstance(right, str):
            raise TypeError("multiplication can not be applied to a string literal.")

        if isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, TimestampType):
            raise TypeError("multiplication can not be applied to date times.")

        if (
            isinstance(right, IndexOpsMixin)
            and (
                isinstance(right.dtype, CategoricalDtype)
                or (not isinstance(right.spark.data_type, NumericType))
            )
        ) and not isinstance(right, numbers.Number):
            raise TypeError("multiplication can not be applied to given types.")

        return column_op(Column.__mul__)(left, right)

    def __truediv__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, StringType)
        ) or isinstance(right, str):
            raise TypeError("division can not be applied on string series or literals.")

        if (
            isinstance(right, IndexOpsMixin)
            and (
                isinstance(right.dtype, CategoricalDtype)
                or (not isinstance(right.spark.data_type, NumericType))
            )
        ) and not isinstance(right, numbers.Number):
            raise TypeError("division can not be applied to given types.")

        def truediv(left, right):
            return F.when(F.lit(right != 0) | F.lit(right).isNull(), left.__div__(right)).otherwise(
                F.when(F.lit(left == np.inf) | F.lit(left == -np.inf), left).otherwise(
                    F.lit(np.inf).__div__(left)
                )
            )

        return numpy_column_op(truediv)(left, right)

    def __floordiv__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, StringType)
        ) or isinstance(right, str):
            raise TypeError("division can not be applied on string series or literals.")

        if (
            isinstance(right, IndexOpsMixin)
            and (
                isinstance(right.dtype, CategoricalDtype)
                or (not isinstance(right.spark.data_type, NumericType))
            )
        ) and not isinstance(right, numbers.Number):
            raise TypeError("division can not be applied to given types.")

        def floordiv(left, right):
            return F.when(F.lit(right is np.nan), np.nan).otherwise(
                F.when(
                    F.lit(right != 0) | F.lit(right).isNull(), F.floor(left.__div__(right))
                ).otherwise(
                    F.when(F.lit(left == np.inf) | F.lit(left == -np.inf), left).otherwise(
                        F.lit(np.inf).__div__(left)
                    )
                )
            )

        return numpy_column_op(floordiv)(left, right)

    def __mod__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, StringType)
        ) or isinstance(right, str):
            raise TypeError("modulo can not be applied on string series or literals.")

        if (
            isinstance(right, IndexOpsMixin)
            and (
                isinstance(right.dtype, CategoricalDtype)
                or (not isinstance(right.spark.data_type, NumericType))
            )
        ) and not isinstance(right, numbers.Number):
            raise TypeError("division can not be applied to given types.")

        def mod(left, right):
            return ((left % right) + right) % right

        return column_op(mod)(left, right)

    def __pow__(self, left, right):
        if (
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, StringType)
        ) or isinstance(right, str):
            raise TypeError("exponentiation can not be applied on string series or literals.")

        if (
            isinstance(right, IndexOpsMixin)
            and (
                isinstance(right.dtype, CategoricalDtype)
                or (not isinstance(right.spark.data_type, NumericType))
            )
        ) and not isinstance(right, numbers.Number):
            raise TypeError("division can not be applied to given types.")

        def pow_func(left, right):
            return F.when(left == 1, left).otherwise(Column.__pow__(left, right))

        return column_op(pow_func)(left, right)

    def __radd__(self, left, right=None):
        if isinstance(right, str):
            raise TypeError("string addition can only be applied to string series or literals.")
        if not isinstance(right, numbers.Number):
            raise TypeError("addition can not be applied to given types.")
        return column_op(Column.__radd__)(left, right)

    def __rsub__(self, left, right=None):
        if isinstance(right, str):
            raise TypeError("substraction can not be applied to string series or literals.")
        if not isinstance(right, numbers.Number):
            raise TypeError("substraction can not be applied to given types.")
        return column_op(Column.__rsub__)(left, right)

    def __rmul__(self, left, right=None):
        if isinstance(right, str):
            raise TypeError("multiplication can not be applied to a string literal.")
        if not isinstance(right, numbers.Number):
            raise TypeError("multiplication can not be applied to given types.")
        return column_op(Column.__rmul__)(left, right)

    def __rtruediv__(self, left, right=None):
        if isinstance(right, str):
            raise TypeError("division can not be applied on string series or literals.")
        if not isinstance(right, numbers.Number):
            raise TypeError("division can not be applied to given types.")

        def rtruediv(left, right):
            return F.when(left == 0, F.lit(np.inf).__div__(right)).otherwise(
                F.lit(right).__truediv__(left)
            )

        return numpy_column_op(rtruediv)(left, right)

    def __rfloordiv__(self, left, right=None):
        if isinstance(right, str):
            raise TypeError("division can not be applied on string series or literals.")
        if not isinstance(right, numbers.Number):
            raise TypeError("division can not be applied to given types.")

        def rfloordiv(left, right):
            return F.when(F.lit(left == 0), F.lit(np.inf).__div__(right)).otherwise(
                F.when(F.lit(left) == np.nan, np.nan).otherwise(F.floor(F.lit(right).__div__(left)))
            )

        return numpy_column_op(rfloordiv)(left, right)

    def __rpow__(self, left, right=None):
        if isinstance(right, str):
            raise TypeError("exponentiation can not be applied on string series or literals.")
        if not isinstance(right, numbers.Number):
            raise TypeError("exponentiation can not be applied to given types.")

        def rpow_func(left, right):
            return F.when(F.lit(right == 1), right).otherwise(Column.__rpow__(left, right))

        return column_op(rpow_func)(left, right)

    def __rmod__(self, left, right=None):
        if isinstance(right, str):
            raise TypeError("modulo can not be applied on string series or literals.")
        if not isinstance(right, numbers.Number):
            raise TypeError("modulo can not be applied to given types.")

        def rmod(left, right):
            return ((right % left) + left) % left

        return column_op(rmod)(left, right)
