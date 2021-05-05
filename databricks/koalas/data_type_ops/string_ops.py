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

from pyspark.sql import functions as F
from pyspark.sql.types import IntegralType, StringType

from databricks.koalas.base import column_op, IndexOpsMixin
from databricks.koalas.data_type_ops.base import DataTypeOps
from databricks.koalas.spark import functions as SF


class StringOps(DataTypeOps):
    """
    The class for binary operations of Koalas objects with spark type: StringType.
    """

    def __add__(self, left, right):
        if isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, StringType):
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
            isinstance(right, IndexOpsMixin) and isinstance(right.spark.data_type, IntegralType)
        ) or isinstance(right, int):
            return column_op(SF.repeat)(left, right)
        else:
            raise TypeError("a string series can only be multiplied to an int series or literal")

    def __truediv__(self, left, right):
        raise TypeError("division can not be applied on string series or literals.")

    def __floordiv__(self, left, right):
        raise TypeError("division can not be applied on string series or literals.")

    def __mod__(self, left, right):
        raise TypeError("modulo can not be applied on string series or literals.")

    def __pow__(self, left, right):
        raise TypeError("exponentiation can not be applied on string series or literals.")

    def __radd__(self, left, right=None):
        if isinstance(right, str):
            return left._with_new_scol(F.concat(F.lit(right), left.spark.column))  # TODO: dtype?
        else:
            raise TypeError("string addition can only be applied to string series or literals.")

    def __rsub__(self, left, right=None):
        raise TypeError("substraction can not be applied to string series or literals.")

    def __rmul__(self, left, right=None):
        if isinstance(right, int):
            return column_op(SF.repeat)(left, right)
        else:
            raise TypeError("a string series can only be multiplied to an int series or literal")

    def __rtruediv__(self, left, right=None):
        raise TypeError("division can not be applied on string series or literals.")

    def __rfloordiv__(self, left, right=None):
        raise TypeError("division can not be applied on string series or literals.")

    def __rpow__(self, left, right=None):
        raise TypeError("exponentiation can not be applied on string series or literals.")

    def __rmod__(self, left, right=None):
        raise TypeError("modulo can not be applied on string series or literals.")
