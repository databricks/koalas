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
Base and utility classes for Koalas objects.
"""
from abc import ABCMeta, abstractmethod
import datetime
from functools import wraps, partial
from typing import Any, Callable, Tuple, Union
import warnings

import numpy as np
import pandas as pd  # noqa: F401
from pandas.api.types import is_list_like
from pyspark import sql as spark
from pyspark.sql import functions as F, Window, Column
from pyspark.sql.types import (
    DateType,
    DoubleType,
    FloatType,
    IntegralType,
    LongType,
    StringType,
    TimestampType,
)

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas import numpy_compat
from databricks.koalas.internal import (
    InternalFrame,
    NATURAL_ORDER_COLUMN_NAME,
    SPARK_DEFAULT_INDEX_NAME,
)
from databricks.koalas.spark import functions as SF
from databricks.koalas.spark.accessors import SparkIndexOpsMethods
from databricks.koalas.typedef import as_spark_type, spark_type_to_pandas_dtype
from databricks.koalas.utils import align_diff_series, same_anchor, scol_for, validate_axis
from databricks.koalas.frame import DataFrame


def booleanize_null(left_scol, scol, f):
    """
    Booleanize Null in Spark Column
    """
    comp_ops = [
        getattr(Column, "__{}__".format(comp_op))
        for comp_op in ["eq", "ne", "lt", "le", "ge", "gt"]
    ]

    if f in comp_ops:
        # if `f` is "!=", fill null with True otherwise False
        filler = f == Column.__ne__
        scol = F.when(scol.isNull(), filler).otherwise(scol)

    elif f == Column.__or__:
        scol = F.when(left_scol.isNull() | scol.isNull(), False).otherwise(scol)

    elif f == Column.__and__:
        scol = F.when(scol.isNull(), False).otherwise(scol)

    return scol


def column_op(f):
    """
    A decorator that wraps APIs taking/returning Spark Column so that Koalas Series can be
    supported too. If this decorator is used for the `f` function that takes Spark Column and
    returns Spark Column, decorated `f` takes Koalas Series as well and returns Koalas
    Series.

    :param f: a function that takes Spark Column and returns Spark Column.
    :param self: Koalas Series
    :param args: arguments that the function `f` takes.
    """

    @wraps(f)
    def wrapper(self, *args):
        # It is possible for the function `f` takes other arguments than Spark Column.
        # To cover this case, explicitly check if the argument is Koalas Series and
        # extract Spark Column. For other arguments, they are used as are.
        cols = [arg for arg in args if isinstance(arg, IndexOpsMixin)]

        if all(same_anchor(self, col) for col in cols):
            # Same DataFrame anchors
            args = [arg.spark.column if isinstance(arg, IndexOpsMixin) else arg for arg in args]
            scol = f(self.spark.column, *args)
            scol = booleanize_null(self.spark.column, scol, f)

            kser = self._with_new_scol(scol)
        else:
            # Different DataFrame anchors
            def apply_func(this_column, *that_columns):
                scol = f(this_column, *that_columns)
                return booleanize_null(this_column, scol, f)

            kser = align_diff_series(apply_func, self, *args, how="full")

        if not all(self.name == col.name for col in cols):
            kser = kser.rename()

        return kser

    return wrapper


def numpy_column_op(f):
    @wraps(f)
    def wrapper(self, *args):
        # PySpark does not support NumPy type out of the box. For now, we convert NumPy types
        # into some primitive types understandable in PySpark.
        new_args = []
        for arg in args:
            # TODO: This is a quick hack to support NumPy type. We should revisit this.
            if isinstance(self.spark.data_type, LongType) and isinstance(arg, np.timedelta64):
                new_args.append(float(arg / np.timedelta64(1, "s")))
            else:
                new_args.append(arg)
        return column_op(f)(self, *new_args)

    return wrapper


class IndexOpsMixin(object, metaclass=ABCMeta):
    """common ops mixin to support a unified interface / docs for Series / Index

    Assuming there are following attributes or properties and function.
    """

    @property
    @abstractmethod
    def _internal(self) -> InternalFrame:
        pass

    @property
    @abstractmethod
    def _kdf(self) -> DataFrame:
        pass

    @abstractmethod
    def _with_new_scol(self, scol: spark.Column):
        pass

    @property
    @abstractmethod
    def _column_label(self) -> Tuple:
        pass

    @property
    @abstractmethod
    def spark(self) -> SparkIndexOpsMethods:
        pass

    @property
    def spark_column(self) -> Column:
        warnings.warn(
            "Series.spark_column is deprecated as of Series.spark.column. "
            "Please use the API instead.",
            FutureWarning,
        )
        return self.spark.column

    spark_column.__doc__ = SparkIndexOpsMethods.column.__doc__

    # arithmetic operators
    __neg__ = column_op(Column.__neg__)

    def __add__(self, other):
        if not isinstance(self.spark.data_type, StringType) and (
            (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError("string addition can only be applied to string series or literals.")
        if isinstance(self.spark.data_type, StringType):
            # Concatenate string columns
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType):
                return column_op(F.concat)(self, other)
            # Handle df['col'] + 'literal'
            elif isinstance(other, str):
                return column_op(F.concat)(self, F.lit(other))
            else:
                raise TypeError("string addition can only be applied to string series or literals.")
        else:
            return column_op(Column.__add__)(self, other)

    def __sub__(self, other):
        if (
            isinstance(self.spark.data_type, StringType)
            or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError("substraction can not be applied to string series or literals.")

        if isinstance(self.spark.data_type, TimestampType):
            # Note that timestamp subtraction casts arguments to integer. This is to mimic pandas's
            # behaviors. pandas returns 'timedelta64[ns]' from 'datetime64[ns]'s subtraction.
            msg = (
                "Note that there is a behavior difference of timestamp subtraction. "
                "The timestamp subtraction returns an integer in seconds, "
                "whereas pandas returns 'timedelta64[ns]'."
            )
            if isinstance(other, IndexOpsMixin) and isinstance(
                other.spark.data_type, TimestampType
            ):
                warnings.warn(msg, UserWarning)
                return self.astype("bigint") - other.astype("bigint")
            elif isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return self.astype("bigint") - F.lit(other).cast(as_spark_type("bigint"))
            else:
                raise TypeError("datetime subtraction can only be applied to datetime series.")
        elif isinstance(self.spark.data_type, DateType):
            # Note that date subtraction casts arguments to integer. This is to mimic pandas's
            # behaviors. pandas returns 'timedelta64[ns]' in days from date's subtraction.
            msg = (
                "Note that there is a behavior difference of date subtraction. "
                "The date subtraction returns an integer in days, "
                "whereas pandas returns 'timedelta64[ns]'."
            )
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, DateType):
                warnings.warn(msg, UserWarning)
                return column_op(F.datediff)(self, other).astype("bigint")
            elif isinstance(other, datetime.date) and not isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return column_op(F.datediff)(self, F.lit(other)).astype("bigint")
            else:
                raise TypeError("date subtraction can only be applied to date series.")
        return column_op(Column.__sub__)(self, other)

    def __mul__(self, other):
        if isinstance(other, str):
            raise TypeError("multiplication can not be applied to a string literal.")

        if (
            isinstance(self.spark.data_type, IntegralType)
            and isinstance(other, IndexOpsMixin)
            and isinstance(other.spark.data_type, StringType)
        ):
            return column_op(SF.repeat)(other, self)

        if isinstance(self.spark.data_type, StringType):
            if (
                isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, IntegralType)
            ) or isinstance(other, int):
                return column_op(SF.repeat)(self, other)
            else:
                raise TypeError(
                    "a string series can only be multiplied to an int series or literal"
                )

        return column_op(Column.__mul__)(self, other)

    def __truediv__(self, other):
        """
        __truediv__ has different behaviour between pandas and PySpark for several cases.
        1. When divide np.inf by zero, PySpark returns null whereas pandas returns np.inf
        2. When divide positive number by zero, PySpark returns null whereas pandas returns np.inf
        3. When divide -np.inf by zero, PySpark returns null whereas pandas returns -np.inf
        4. When divide negative number by zero, PySpark returns null whereas pandas returns -np.inf

        +-------------------------------------------+
        | dividend (divisor: 0) | PySpark |  pandas |
        |-----------------------|---------|---------|
        |         np.inf        |   null  |  np.inf |
        |        -np.inf        |   null  | -np.inf |
        |           10          |   null  |  np.inf |
        |          -10          |   null  | -np.inf |
        +-----------------------|---------|---------+
        """

        if (
            isinstance(self.spark.data_type, StringType)
            or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError("division can not be applied on string series or literals.")

        def truediv(left, right):
            return F.when(F.lit(right != 0) | F.lit(right).isNull(), left.__div__(right)).otherwise(
                F.when(F.lit(left == np.inf) | F.lit(left == -np.inf), left).otherwise(
                    F.lit(np.inf).__div__(left)
                )
            )

        return numpy_column_op(truediv)(self, other)

    def __mod__(self, other):
        if (
            isinstance(self.spark.data_type, StringType)
            or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError("modulo can not be applied on string series or literals.")

        def mod(left, right):
            return ((left % right) + right) % right

        return column_op(mod)(self, other)

    def __radd__(self, other):
        # Handle 'literal' + df['col']
        if not isinstance(self.spark.data_type, StringType) and isinstance(other, str):
            raise TypeError("string addition can only be applied to string series or literals.")

        if isinstance(self.spark.data_type, StringType):
            if isinstance(other, str):
                return self._with_new_scol(F.concat(F.lit(other), self.spark.column))
            else:
                raise TypeError("string addition can only be applied to string series or literals.")
        else:
            return column_op(Column.__radd__)(self, other)

    def __rsub__(self, other):
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError("substraction can not be applied to string series or literals.")

        if isinstance(self.spark.data_type, TimestampType):
            # Note that timestamp subtraction casts arguments to integer. This is to mimic pandas's
            # behaviors. pandas returns 'timedelta64[ns]' from 'datetime64[ns]'s subtraction.
            msg = (
                "Note that there is a behavior difference of timestamp subtraction. "
                "The timestamp subtraction returns an integer in seconds, "
                "whereas pandas returns 'timedelta64[ns]'."
            )
            if isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return -(self.astype("bigint") - F.lit(other).cast(as_spark_type("bigint")))
            else:
                raise TypeError("datetime subtraction can only be applied to datetime series.")
        elif isinstance(self.spark.data_type, DateType):
            # Note that date subtraction casts arguments to integer. This is to mimic pandas's
            # behaviors. pandas returns 'timedelta64[ns]' in days from date's subtraction.
            msg = (
                "Note that there is a behavior difference of date subtraction. "
                "The date subtraction returns an integer in days, "
                "whereas pandas returns 'timedelta64[ns]'."
            )
            if isinstance(other, datetime.date) and not isinstance(other, datetime.datetime):
                warnings.warn(msg, UserWarning)
                return -column_op(F.datediff)(self, F.lit(other)).astype("bigint")
            else:
                raise TypeError("date subtraction can only be applied to date series.")
        return column_op(Column.__rsub__)(self, other)

    def __rmul__(self, other):
        if isinstance(other, str):
            raise TypeError("multiplication can not be applied to a string literal.")

        if isinstance(self.spark.data_type, StringType):
            if isinstance(other, int):
                return column_op(SF.repeat)(self, other)
            else:
                raise TypeError(
                    "a string series can only be multiplied to an int series or literal"
                )

        return column_op(Column.__rmul__)(self, other)

    def __rtruediv__(self, other):
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError("division can not be applied on string series or literals.")

        def rtruediv(left, right):
            return F.when(left == 0, F.lit(np.inf).__div__(right)).otherwise(
                F.lit(right).__truediv__(left)
            )

        return numpy_column_op(rtruediv)(self, other)

    def __floordiv__(self, other):
        """
        __floordiv__ has different behaviour between pandas and PySpark for several cases.
        1. When divide np.inf by zero, PySpark returns null whereas pandas returns np.inf
        2. When divide positive number by zero, PySpark returns null whereas pandas returns np.inf
        3. When divide -np.inf by zero, PySpark returns null whereas pandas returns -np.inf
        4. When divide negative number by zero, PySpark returns null whereas pandas returns -np.inf

        +-------------------------------------------+
        | dividend (divisor: 0) | PySpark |  pandas |
        |-----------------------|---------|---------|
        |         np.inf        |   null  |  np.inf |
        |        -np.inf        |   null  | -np.inf |
        |           10          |   null  |  np.inf |
        |          -10          |   null  | -np.inf |
        +-----------------------|---------|---------+
        """
        if (
            isinstance(self.spark.data_type, StringType)
            or (isinstance(other, IndexOpsMixin) and isinstance(other.spark.data_type, StringType))
            or isinstance(other, str)
        ):
            raise TypeError("division can not be applied on string series or literals.")

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

        return numpy_column_op(floordiv)(self, other)

    def __rfloordiv__(self, other):
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError("division can not be applied on string series or literals.")

        def rfloordiv(left, right):
            return F.when(F.lit(left == 0), F.lit(np.inf).__div__(right)).otherwise(
                F.when(F.lit(left) == np.nan, np.nan).otherwise(F.floor(F.lit(right).__div__(left)))
            )

        return numpy_column_op(rfloordiv)(self, other)

    def __rmod__(self, other):
        if isinstance(self.spark.data_type, StringType) or isinstance(other, str):
            raise TypeError("modulo can not be applied on string series or literals.")

        def rmod(left, right):
            return ((right % left) + left) % left

        return column_op(rmod)(self, other)

    __pow__ = column_op(Column.__pow__)
    __rpow__ = column_op(Column.__rpow__)
    __abs__ = column_op(F.abs)

    # comparison operators
    __eq__ = column_op(Column.__eq__)
    __ne__ = column_op(Column.__ne__)
    __lt__ = column_op(Column.__lt__)
    __le__ = column_op(Column.__le__)
    __ge__ = column_op(Column.__ge__)
    __gt__ = column_op(Column.__gt__)

    # `and`, `or`, `not` cannot be overloaded in Python,
    # so use bitwise operators as boolean operators
    __and__ = column_op(Column.__and__)
    __or__ = column_op(Column.__or__)
    __invert__ = column_op(Column.__invert__)
    __rand__ = column_op(Column.__rand__)
    __ror__ = column_op(Column.__ror__)

    # NDArray Compat
    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any):
        # Try dunder methods first.
        result = numpy_compat.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )

        # After that, we try with PySpark APIs.
        if result is NotImplemented:
            result = numpy_compat.maybe_dispatch_ufunc_to_spark_func(
                self, ufunc, method, *inputs, **kwargs
            )

        if result is not NotImplemented:
            return result
        else:
            # TODO: support more APIs?
            raise NotImplementedError("Koalas objects currently do not support %s." % ufunc)

    @property
    def dtype(self):
        """Return the dtype object of the underlying data.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3])
        >>> s.dtype
        dtype('int64')

        >>> s = ks.Series(list('abc'))
        >>> s.dtype
        dtype('O')

        >>> s = ks.Series(pd.date_range('20130101', periods=3))
        >>> s.dtype
        dtype('<M8[ns]')

        >>> s.rename("a").to_frame().set_index("a").index.dtype
        dtype('<M8[ns]')
        """
        return spark_type_to_pandas_dtype(self.spark.data_type)

    @property
    def empty(self):
        """
        Returns true if the current object is empty. Otherwise, returns false.

        >>> ks.range(10).id.empty
        False

        >>> ks.range(0).id.empty
        True

        >>> ks.DataFrame({}, index=list('abc')).index.empty
        False
        """
        return self._internal.resolved_copy.spark_frame.rdd.isEmpty()

    @property
    def hasnans(self):
        """
        Return True if it has any missing values. Otherwise, it returns False.

        >>> ks.DataFrame({}, index=list('abc')).index.hasnans
        False

        >>> ks.Series(['a', None]).hasnans
        True

        >>> ks.Series([1.0, 2.0, np.nan]).hasnans
        True

        >>> ks.Series([1, 2, 3]).hasnans
        False

        >>> (ks.Series([1.0, 2.0, np.nan]) + 1).hasnans
        True

        >>> ks.Series([1, 2, 3]).rename("a").to_frame().set_index("a").index.hasnans
        False
        """
        sdf = self._internal.spark_frame
        scol = self.spark.column

        if isinstance(self.spark.data_type, (DoubleType, FloatType)):
            return sdf.select(F.max(scol.isNull() | F.isnan(scol))).collect()[0][0]
        else:
            return sdf.select(F.max(scol.isNull())).collect()[0][0]

    @property
    def is_monotonic(self):
        """
        Return boolean if values in the object are monotonically increasing.

        .. note:: the current implementation of is_monotonic requires to shuffle
            and aggregate multiple times to check the order locally and globally,
            which is potentially expensive. In case of multi-index, all data are
            transferred to single node which can easily cause out-of-memory error currently.

        Returns
        -------
        is_monotonic : boolean

        Examples
        --------
        >>> ser = ks.Series(['1/1/2018', '3/1/2018', '4/1/2018'])
        >>> ser.is_monotonic
        True

        >>> df = ks.DataFrame({'dates': [None, '1/1/2018', '2/1/2018', '3/1/2018']})
        >>> df.dates.is_monotonic
        False

        >>> df.index.is_monotonic
        True

        >>> ser = ks.Series([1])
        >>> ser.is_monotonic
        True

        >>> ser = ks.Series([])
        >>> ser.is_monotonic
        True

        >>> ser.rename("a").to_frame().set_index("a").index.is_monotonic
        True

        >>> ser = ks.Series([5, 4, 3, 2, 1], index=[1, 2, 3, 4, 5])
        >>> ser.is_monotonic
        False

        >>> ser.index.is_monotonic
        True

        Support for MultiIndex

        >>> midx = ks.MultiIndex.from_tuples(
        ... [('x', 'a'), ('x', 'b'), ('y', 'c'), ('y', 'd'), ('z', 'e')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'b'),
                    ('y', 'c'),
                    ('y', 'd'),
                    ('z', 'e')],
                   )
        >>> midx.is_monotonic
        True

        >>> midx = ks.MultiIndex.from_tuples(
        ... [('z', 'a'), ('z', 'b'), ('y', 'c'), ('y', 'd'), ('x', 'e')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('z', 'a'),
                    ('z', 'b'),
                    ('y', 'c'),
                    ('y', 'd'),
                    ('x', 'e')],
                   )
        >>> midx.is_monotonic
        False
        """
        return self._is_monotonic("increasing")

    is_monotonic_increasing = is_monotonic

    @property
    def is_monotonic_decreasing(self):
        """
        Return boolean if values in the object are monotonically decreasing.

        .. note:: the current implementation of is_monotonic_decreasing requires to shuffle
            and aggregate multiple times to check the order locally and globally,
            which is potentially expensive. In case of multi-index, all data are transferred
            to single node which can easily cause out-of-memory error currently.

        Returns
        -------
        is_monotonic : boolean

        Examples
        --------
        >>> ser = ks.Series(['4/1/2018', '3/1/2018', '1/1/2018'])
        >>> ser.is_monotonic_decreasing
        True

        >>> df = ks.DataFrame({'dates': [None, '3/1/2018', '2/1/2018', '1/1/2018']})
        >>> df.dates.is_monotonic_decreasing
        False

        >>> df.index.is_monotonic_decreasing
        False

        >>> ser = ks.Series([1])
        >>> ser.is_monotonic_decreasing
        True

        >>> ser = ks.Series([])
        >>> ser.is_monotonic_decreasing
        True

        >>> ser.rename("a").to_frame().set_index("a").index.is_monotonic_decreasing
        True

        >>> ser = ks.Series([5, 4, 3, 2, 1], index=[1, 2, 3, 4, 5])
        >>> ser.is_monotonic_decreasing
        True

        >>> ser.index.is_monotonic_decreasing
        False

        Support for MultiIndex

        >>> midx = ks.MultiIndex.from_tuples(
        ... [('x', 'a'), ('x', 'b'), ('y', 'c'), ('y', 'd'), ('z', 'e')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'b'),
                    ('y', 'c'),
                    ('y', 'd'),
                    ('z', 'e')],
                   )
        >>> midx.is_monotonic_decreasing
        False

        >>> midx = ks.MultiIndex.from_tuples(
        ... [('z', 'e'), ('z', 'd'), ('y', 'c'), ('y', 'b'), ('x', 'a')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('z', 'a'),
                    ('z', 'b'),
                    ('y', 'c'),
                    ('y', 'd'),
                    ('x', 'e')],
                   )
        >>> midx.is_monotonic_decreasing
        True
        """
        return self._is_monotonic("decreasing")

    def _is_locally_monotonic_spark_column(self, order):
        window = (
            Window.partitionBy(F.col("__partition_id"))
            .orderBy(NATURAL_ORDER_COLUMN_NAME)
            .rowsBetween(-1, -1)
        )

        if order == "increasing":
            return (F.col("__origin") >= F.lag(F.col("__origin"), 1).over(window)) & F.col(
                "__origin"
            ).isNotNull()
        else:
            return (F.col("__origin") <= F.lag(F.col("__origin"), 1).over(window)) & F.col(
                "__origin"
            ).isNotNull()

    def _is_monotonic(self, order):
        assert order in ("increasing", "decreasing")

        sdf = self._internal.spark_frame

        sdf = (
            sdf.select(
                F.spark_partition_id().alias(
                    "__partition_id"
                ),  # Make sure we use the same partition id in the whole job.
                F.col(NATURAL_ORDER_COLUMN_NAME),
                self.spark.column.alias("__origin"),
            )
            .select(
                F.col("__partition_id"),
                F.col("__origin"),
                self._is_locally_monotonic_spark_column(order).alias(
                    "__comparison_within_partition"
                ),
            )
            .groupby(F.col("__partition_id"))
            .agg(
                F.min(F.col("__origin")).alias("__partition_min"),
                F.max(F.col("__origin")).alias("__partition_max"),
                F.min(F.coalesce(F.col("__comparison_within_partition"), F.lit(True))).alias(
                    "__comparison_within_partition"
                ),
            )
        )

        # Now we're windowing the aggregation results without partition specification.
        # The number of rows here will be as the same of partitions, which is expected
        # to be small.
        window = Window.orderBy(F.col("__partition_id")).rowsBetween(-1, -1)
        if order == "increasing":
            comparison_col = F.col("__partition_min") >= F.lag(F.col("__partition_max"), 1).over(
                window
            )
        else:
            comparison_col = F.col("__partition_min") <= F.lag(F.col("__partition_max"), 1).over(
                window
            )

        sdf = sdf.select(
            comparison_col.alias("__comparison_between_partitions"),
            F.col("__comparison_within_partition"),
        )

        ret = sdf.select(
            F.min(F.coalesce(F.col("__comparison_between_partitions"), F.lit(True)))
            & F.min(F.coalesce(F.col("__comparison_within_partition"), F.lit(True)))
        ).collect()[0][0]
        if ret is None:
            return True
        else:
            return ret

    @property
    def ndim(self):
        """
        Return an int representing the number of array dimensions.

        Return 1 for Series / Index / MultiIndex.

        Examples
        --------

        For Series

        >>> s = ks.Series([None, 1, 2, 3, 4], index=[4, 5, 2, 1, 8])
        >>> s.ndim
        1

        For Index

        >>> s.index.ndim
        1

        For MultiIndex

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [1, 1, 1, 1, 1, 2, 1, 2, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        >>> s.index.ndim
        1
        """
        return 1

    def astype(self, dtype):
        """
        Cast a Koalas object to a specified dtype ``dtype``.

        Parameters
        ----------
        dtype : data type
            Use a numpy.dtype or Python type to cast entire pandas object to
            the same type.

        Returns
        -------
        casted : same type as caller

        See Also
        --------
        to_datetime : Convert argument to datetime.

        Examples
        --------
        >>> ser = ks.Series([1, 2], dtype='int32')
        >>> ser
        0    1
        1    2
        dtype: int32

        >>> ser.astype('int64')
        0    1
        1    2
        dtype: int64

        >>> ser.rename("a").to_frame().set_index("a").index.astype('int64')
        Int64Index([1, 2], dtype='int64', name='a')
        """
        spark_type = as_spark_type(dtype)
        if not spark_type:
            raise ValueError("Type {} not understood".format(dtype))
        return self._with_new_scol(self.spark.column.cast(spark_type))

    def isin(self, values):
        """
        Check whether `values` are contained in Series.

        Return a boolean Series showing whether each element in the Series
        matches an element in the passed sequence of `values` exactly.

        Parameters
        ----------
        values : list or set
            The sequence of values to test.

        Returns
        -------
        isin : Series (bool dtype)

        Examples
        --------
        >>> s = ks.Series(['lama', 'cow', 'lama', 'beetle', 'lama',
        ...                'hippo'], name='animal')
        >>> s.isin(['cow', 'lama'])
        0     True
        1     True
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        Passing a single string as ``s.isin('lama')`` will raise an error. Use
        a list of one element instead:

        >>> s.isin(['lama'])
        0     True
        1    False
        2     True
        3    False
        4     True
        5    False
        Name: animal, dtype: bool

        >>> s.rename("a").to_frame().set_index("a").index.isin(['lama'])
        Index([True, False, True, False, True, False], dtype='object', name='a')
        """
        if not is_list_like(values):
            raise TypeError(
                "only list-like objects are allowed to be passed"
                " to isin(), you passed a [{values_type}]".format(values_type=type(values).__name__)
            )

        return self._with_new_scol(self.spark.column.isin(list(values)))

    def isnull(self):
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as None or numpy.NaN, gets mapped to True values.
        Everything else gets mapped to False values. Characters such as empty strings '' or
        numpy.inf are not considered NA values
        (unless you set pandas.options.mode.use_inf_as_na = True).

        Returns
        -------
        Series : Mask of bool values for each element in Series
            that indicates whether an element is not an NA value.

        Examples
        --------
        >>> ser = ks.Series([5, 6, np.NaN])
        >>> ser.isna()  # doctest: +NORMALIZE_WHITESPACE
        0    False
        1    False
        2     True
        dtype: bool

        >>> ser.rename("a").to_frame().set_index("a").index.isna()
        Index([False, False, True], dtype='object', name='a')
        """
        from databricks.koalas.indexes import MultiIndex

        if isinstance(self, MultiIndex):
            raise NotImplementedError("isna is not defined for MultiIndex")
        if isinstance(self.spark.data_type, (FloatType, DoubleType)):
            return self._with_new_scol(self.spark.column.isNull() | F.isnan(self.spark.column))
        else:
            return self._with_new_scol(self.spark.column.isNull())

    isna = isnull

    def notnull(self):
        """
        Detect existing (non-missing) values.
        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to True.
        Characters such as empty strings '' or numpy.inf are not considered NA values
        (unless you set pandas.options.mode.use_inf_as_na = True).
        NA values, such as None or numpy.NaN, get mapped to False values.

        Returns
        -------
        Series : Mask of bool values for each element in Series
            that indicates whether an element is not an NA value.

        Examples
        --------
        Show which entries in a Series are not NA.

        >>> ser = ks.Series([5, 6, np.NaN])
        >>> ser
        0    5.0
        1    6.0
        2    NaN
        dtype: float64

        >>> ser.notna()
        0     True
        1     True
        2    False
        dtype: bool

        >>> ser.rename("a").to_frame().set_index("a").index.notna()
        Index([True, True, False], dtype='object', name='a')
        """
        from databricks.koalas.indexes import MultiIndex

        if isinstance(self, MultiIndex):
            raise NotImplementedError("notna is not defined for MultiIndex")
        return (~self.isnull()).rename(self.name)

    notna = notnull

    # TODO: axis, skipna, and many arguments should be implemented.
    def all(self, axis: Union[int, str] = 0) -> bool:
        """
        Return whether all elements are True.

        Returns True unless there at least one element within a series that is
        False or equivalent (e.g. zero or empty)

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Indicate which axis or axes should be reduced.

            * 0 / 'index' : reduce the index, return a Series whose index is the
              original column labels.

        Examples
        --------
        >>> ks.Series([True, True]).all()
        True

        >>> ks.Series([True, False]).all()
        False

        >>> ks.Series([0, 1]).all()
        False

        >>> ks.Series([1, 2, 3]).all()
        True

        >>> ks.Series([True, True, None]).all()
        True

        >>> ks.Series([True, False, None]).all()
        False

        >>> ks.Series([]).all()
        True

        >>> ks.Series([np.nan]).all()
        True

        >>> df = ks.Series([True, False, None]).rename("a").to_frame()
        >>> df.set_index("a").index.all()
        False
        """
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')

        sdf = self._internal.spark_frame.select(self.spark.column)
        col = scol_for(sdf, sdf.columns[0])

        # Note that we're ignoring `None`s here for now.
        # any and every was added as of Spark 3.0
        # ret = sdf.select(F.expr("every(CAST(`%s` AS BOOLEAN))" % sdf.columns[0])).collect()[0][0]
        # Here we use min as its alternative:
        ret = sdf.select(F.min(F.coalesce(col.cast("boolean"), F.lit(True)))).collect()[0][0]
        if ret is None:
            return True
        else:
            return ret

    # TODO: axis, skipna, and many arguments should be implemented.
    def any(self, axis: Union[int, str] = 0) -> bool:
        """
        Return whether any element is True.

        Returns False unless there at least one element within a series that is
        True or equivalent (e.g. non-zero or non-empty).

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Indicate which axis or axes should be reduced.

            * 0 / 'index' : reduce the index, return a Series whose index is the
              original column labels.

        Examples
        --------
        >>> ks.Series([False, False]).any()
        False

        >>> ks.Series([True, False]).any()
        True

        >>> ks.Series([0, 0]).any()
        False

        >>> ks.Series([0, 1, 2]).any()
        True

        >>> ks.Series([False, False, None]).any()
        False

        >>> ks.Series([True, False, None]).any()
        True

        >>> ks.Series([]).any()
        False

        >>> ks.Series([np.nan]).any()
        False

        >>> df = ks.Series([True, False, None]).rename("a").to_frame()
        >>> df.set_index("a").index.any()
        True
        """
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')

        sdf = self._internal.spark_frame.select(self.spark.column)
        col = scol_for(sdf, sdf.columns[0])

        # Note that we're ignoring `None`s here for now.
        # any and every was added as of Spark 3.0
        # ret = sdf.select(F.expr("any(CAST(`%s` AS BOOLEAN))" % sdf.columns[0])).collect()[0][0]
        # Here we use max as its alternative:
        ret = sdf.select(F.max(F.coalesce(col.cast("boolean"), F.lit(False)))).collect()[0][0]
        if ret is None:
            return False
        else:
            return ret

    # TODO: add frep and axis parameter
    def shift(self, periods=1, fill_value=None):
        """
        Shift Series/Index by desired number of periods.

        .. note:: the current implementation of shift uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        periods : int
            Number of periods to shift. Can be positive or negative.
        fill_value : object, optional
            The scalar value to use for newly introduced missing values.
            The default depends on the dtype of self. For numeric data, np.nan is used.

        Returns
        -------
        Copy of input Series/Index, shifted.

        Examples
        --------
        >>> df = ks.DataFrame({'Col1': [10, 20, 15, 30, 45],
        ...                    'Col2': [13, 23, 18, 33, 48],
        ...                    'Col3': [17, 27, 22, 37, 52]},
        ...                   columns=['Col1', 'Col2', 'Col3'])

        >>> df.Col1.shift(periods=3)
        0     NaN
        1     NaN
        2     NaN
        3    10.0
        4    20.0
        Name: Col1, dtype: float64

        >>> df.Col2.shift(periods=3, fill_value=0)
        0     0
        1     0
        2     0
        3    13
        4    23
        Name: Col2, dtype: int64

        >>> df.index.shift(periods=3, fill_value=0)
        Int64Index([0, 0, 0, 0, 1], dtype='int64')
        """
        return self._shift(periods, fill_value)

    def _shift(self, periods, fill_value, part_cols=()):
        if not isinstance(periods, int):
            raise ValueError("periods should be an int; however, got [%s]" % type(periods).__name__)

        col = self.spark.column
        window = (
            Window.partitionBy(*part_cols)
            .orderBy(NATURAL_ORDER_COLUMN_NAME)
            .rowsBetween(-periods, -periods)
        )
        lag_col = F.lag(col, periods).over(window)
        col = F.when(lag_col.isNull() | F.isnan(lag_col), fill_value).otherwise(lag_col)
        return self._with_new_scol(col)

    # TODO: Update Documentation for Bins Parameter when its supported
    def value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
        """
        Return a Series containing counts of unique values.
        The resulting object will be in descending order so that the
        first element is the most frequently-occurring element.
        Excludes NA values by default.

        Parameters
        ----------
        normalize : boolean, default False
            If True then the object returned will contain the relative
            frequencies of the unique values.
        sort : boolean, default True
            Sort by values.
        ascending : boolean, default False
            Sort in ascending order.
        bins : Not Yet Supported
        dropna : boolean, default True
            Don't include counts of NaN.

        Returns
        -------
        counts : Series

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.

        Examples
        --------
        For Series

        >>> df = ks.DataFrame({'x':[0, 0, 1, 1, 1, np.nan]})
        >>> df.x.value_counts()  # doctest: +NORMALIZE_WHITESPACE
        1.0    3
        0.0    2
        Name: x, dtype: int64

        With `normalize` set to `True`, returns the relative frequency by
        dividing all values by the sum of values.

        >>> df.x.value_counts(normalize=True)  # doctest: +NORMALIZE_WHITESPACE
        1.0    0.6
        0.0    0.4
        Name: x, dtype: float64

        **dropna**
        With `dropna` set to `False` we can also see NaN index values.

        >>> df.x.value_counts(dropna=False)  # doctest: +NORMALIZE_WHITESPACE
        1.0    3
        0.0    2
        NaN    1
        Name: x, dtype: int64

        For Index

        >>> idx = ks.Index([3, 1, 2, 3, 4, np.nan])
        >>> idx
        Float64Index([3.0, 1.0, 2.0, 3.0, 4.0, nan], dtype='float64')

        >>> idx.value_counts().sort_index()
        1.0    1
        2.0    1
        3.0    2
        4.0    1
        dtype: int64

        **sort**

        With `sort` set to `False`, the result wouldn't be sorted by number of count.

        >>> idx.value_counts(sort=True).sort_index()
        1.0    1
        2.0    1
        3.0    2
        4.0    1
        dtype: int64

        **normalize**

        With `normalize` set to `True`, returns the relative frequency by
        dividing all values by the sum of values.

        >>> idx.value_counts(normalize=True).sort_index()
        1.0    0.2
        2.0    0.2
        3.0    0.4
        4.0    0.2
        dtype: float64

        **dropna**

        With `dropna` set to `False` we can also see NaN index values.

        >>> idx.value_counts(dropna=False).sort_index()  # doctest: +SKIP
        1.0    1
        2.0    1
        3.0    2
        4.0    1
        NaN    1
        dtype: int64

        For MultiIndex.

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [1, 1, 1, 1, 1, 2, 1, 2, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        >>> s.index  # doctest: +SKIP
        MultiIndex([(  'lama', 'weight'),
                    (  'lama', 'weight'),
                    (  'lama', 'weight'),
                    (   'cow', 'weight'),
                    (   'cow', 'weight'),
                    (   'cow', 'length'),
                    ('falcon', 'weight'),
                    ('falcon', 'length'),
                    ('falcon', 'length')],
                   )

        >>> s.index.value_counts().sort_index()
        (cow, length)       1
        (cow, weight)       2
        (falcon, length)    2
        (falcon, weight)    1
        (lama, weight)      3
        dtype: int64

        >>> s.index.value_counts(normalize=True).sort_index()
        (cow, length)       0.111111
        (cow, weight)       0.222222
        (falcon, length)    0.222222
        (falcon, weight)    0.111111
        (lama, weight)      0.333333
        dtype: float64

        If Index has name, keep the name up.

        >>> idx = ks.Index([0, 0, 0, 1, 1, 2, 3], name='koalas')
        >>> idx.value_counts().sort_index()
        0    3
        1    2
        2    1
        3    1
        Name: koalas, dtype: int64
        """
        from databricks.koalas.series import first_series

        if bins is not None:
            raise NotImplementedError("value_counts currently does not support bins")

        if dropna:
            sdf_dropna = self._internal.spark_frame.select(self.spark.column).dropna()
        else:
            sdf_dropna = self._internal.spark_frame.select(self.spark.column)
        index_name = SPARK_DEFAULT_INDEX_NAME
        column_name = self._internal.data_spark_column_names[0]
        sdf = sdf_dropna.groupby(scol_for(sdf_dropna, column_name).alias(index_name)).count()
        if sort:
            if ascending:
                sdf = sdf.orderBy(F.col("count"))
            else:
                sdf = sdf.orderBy(F.col("count").desc())

        if normalize:
            sum = sdf_dropna.count()
            sdf = sdf.withColumn("count", F.col("count") / F.lit(sum))

        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_column_names=[index_name],
            column_labels=self._internal.column_labels,
            data_spark_columns=[scol_for(sdf, "count")],
            column_label_names=self._internal.column_label_names,
        )

        return first_series(DataFrame(internal))

    def nunique(self, dropna: bool = True, approx: bool = False, rsd: float = 0.05) -> int:
        """
        Return number of unique elements in the object.
        Excludes NA values by default.

        Parameters
        ----------
        dropna : bool, default True
            Don’t include NaN in the count.
        approx: bool, default False
            If False, will use the exact algorithm and return the exact number of unique.
            If True, it uses the HyperLogLog approximate algorithm, which is significantly faster
            for large amount of data.
            Note: This parameter is specific to Koalas and is not found in pandas.
        rsd: float, default 0.05
            Maximum estimation error allowed in the HyperLogLog algorithm.
            Note: Just like ``approx`` this parameter is specific to Koalas.

        Returns
        -------
        int

        See Also
        --------
        DataFrame.nunique: Method nunique for DataFrame.
        Series.count: Count non-NA/null observations in the Series.

        Examples
        --------
        >>> ks.Series([1, 2, 3, np.nan]).nunique()
        3

        >>> ks.Series([1, 2, 3, np.nan]).nunique(dropna=False)
        4

        On big data, we recommend using the approximate algorithm to speed up this function.
        The result will be very close to the exact unique count.

        >>> ks.Series([1, 2, 3, np.nan]).nunique(approx=True)
        3

        >>> idx = ks.Index([1, 1, 2, None])
        >>> idx
        Float64Index([1.0, 1.0, 2.0, nan], dtype='float64')

        >>> idx.nunique()
        2

        >>> idx.nunique(dropna=False)
        3
        """
        res = self._internal.spark_frame.select([self._nunique(dropna, approx, rsd)])
        return res.collect()[0][0]

    def _nunique(self, dropna=True, approx=False, rsd=0.05):
        colname = self._internal.data_spark_column_names[0]
        count_fn = partial(F.approx_count_distinct, rsd=rsd) if approx else F.countDistinct
        if dropna:
            return count_fn(self.spark.column).alias(colname)
        else:
            return (
                count_fn(self.spark.column)
                + F.when(
                    F.count(F.when(self.spark.column.isNull(), 1).otherwise(None)) >= 1, 1
                ).otherwise(0)
            ).alias(colname)

    def take(self, indices):
        """
        Return the elements in the given *positional* indices along an axis.

        This means that we are not indexing according to actual values in
        the index attribute of the object. We are indexing according to the
        actual position of the element in the object.

        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take.

        Returns
        -------
        taken : same type as caller
            An array-like containing the elements taken from the object.

        See Also
        --------
        DataFrame.loc : Select a subset of a DataFrame by labels.
        DataFrame.iloc : Select a subset of a DataFrame by positions.
        numpy.take : Take elements from an array along an axis.

        Examples
        --------

        Series

        >>> kser = ks.Series([100, 200, 300, 400, 500])
        >>> kser
        0    100
        1    200
        2    300
        3    400
        4    500
        dtype: int64

        >>> kser.take([0, 2, 4]).sort_index()
        0    100
        2    300
        4    500
        dtype: int64

        Index

        >>> kidx = ks.Index([100, 200, 300, 400, 500])
        >>> kidx
        Int64Index([100, 200, 300, 400, 500], dtype='int64')

        >>> kidx.take([0, 2, 4]).sort_values()
        Int64Index([100, 300, 500], dtype='int64')

        MultiIndex

        >>> kmidx = ks.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("x", "c")])
        >>> kmidx  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'b'),
                    ('x', 'c')],
                   )

        >>> kmidx.take([0, 2])  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'c')],
                   )
        """
        if not is_list_like(indices) or isinstance(indices, (dict, set)):
            raise ValueError("`indices` must be a list-like except dict or set")
        if isinstance(self, ks.Series):
            result = self.iloc[indices]
        elif isinstance(self, ks.Index):
            result = self._kdf.iloc[indices].index
        return result
