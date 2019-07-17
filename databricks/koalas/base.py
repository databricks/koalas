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

from functools import wraps
from typing import Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
from pyspark import sql as spark
from pyspark.sql import functions as F, Window
from pyspark.sql.types import DoubleType, FloatType, LongType, StringType, TimestampType, \
    to_arrow_type

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.internal import _InternalFrame
from databricks.koalas.typedef import pandas_wraps


def _column_op(f):
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
        assert all((not isinstance(arg, IndexOpsMixin))
                   or (arg._kdf is self._kdf) for arg in args), \
            "Cannot combine column argument because it comes from a different dataframe"

        # It is possible for the function `f` takes other arguments than Spark Column.
        # To cover this case, explicitly check if the argument is Koalas Series and
        # extract Spark Column. For other arguments, they are used as are.
        args = [arg._scol if isinstance(arg, IndexOpsMixin) else arg for arg in args]
        scol = f(self._scol, *args)
        return self._with_new_scol(scol)
    return wrapper


def _numpy_column_op(f):
    @wraps(f)
    def wrapper(self, *args):
        # PySpark does not support NumPy type out of the box. For now, we convert NumPy types
        # into some primitive types understandable in PySpark.
        new_args = []
        for arg in args:
            # TODO: This is a quick hack to support NumPy type. We should revisit this.
            if isinstance(self.spark_type, LongType) and isinstance(arg, np.timedelta64):
                new_args.append(float(arg / np.timedelta64(1, 's')))
            else:
                new_args.append(arg)
        return _column_op(f)(self, *new_args)
    return wrapper


def _wrap_accessor_spark(accessor, fn, return_type=None):
    """
    Wrap an accessor property or method, e.g., Series.dt.date with a spark function.
    """
    if return_type:
        return _column_op(
            lambda col: fn(col).cast(return_type)
        )(accessor._data)
    else:
        return _column_op(fn)(accessor._data)


def _wrap_accessor_pandas(accessor, fn, return_type):
    """
    Wrap an accessor property or method, e.g, Series.dt.date with a pandas function.
    """
    return pandas_wraps(fn, return_col=return_type)(accessor._data)


class IndexOpsMixin(object):
    """common ops mixin to support a unified interface / docs for Series / Index

    Assuming there are following attributes or properties and function.

    :ivar _scol: Spark Column instance
    :type _scol: pyspark.Column
    :ivar _kdf: Parent's Koalas DataFrame
    :type _kdf: ks.DataFrame

    :ivar spark_type: Spark data type
    :type spark_type: spark.types.DataType

    def _with_new_scol(self, scol: spark.Column) -> IndexOpsMixin
        Creates new object with the new column
    """
    def __init__(self, internal: _InternalFrame, kdf):
        self._internal = internal  # type: _InternalFrame
        self._kdf = kdf

    @property
    def _scol(self):
        return self._internal.scol

    # arithmetic operators
    __neg__ = _column_op(spark.Column.__neg__)

    def __add__(self, other):
        if isinstance(self.spark_type, StringType):
            # Concatenate string columns
            if isinstance(other, IndexOpsMixin) and isinstance(other.spark_type, StringType):
                return _column_op(F.concat)(self, other)
            # Handle df['col'] + 'literal'
            elif isinstance(other, str):
                return _column_op(F.concat)(self, F.lit(other))
            else:
                raise TypeError('string addition can only be applied to string series or literals.')
        else:
            return _column_op(spark.Column.__add__)(self, other)

    def __sub__(self, other):
        # Note that timestamp subtraction casts arguments to integer. This is to mimic Pandas's
        # behaviors. Pandas returns 'timedelta64[ns]' from 'datetime64[ns]'s subtraction.
        if isinstance(other, IndexOpsMixin) and isinstance(self.spark_type, TimestampType):
            if not isinstance(other.spark_type, TimestampType):
                raise TypeError('datetime subtraction can only be applied to datetime series.')
            return self.astype('bigint') - other.astype('bigint')
        else:
            return _column_op(spark.Column.__sub__)(self, other)

    __mul__ = _column_op(spark.Column.__mul__)
    __div__ = _numpy_column_op(spark.Column.__div__)
    __truediv__ = _numpy_column_op(spark.Column.__truediv__)
    __mod__ = _column_op(spark.Column.__mod__)

    def __radd__(self, other):
        # Handle 'literal' + df['col']
        if isinstance(self.spark_type, StringType) and isinstance(other, str):
            return self._with_new_scol(F.concat(F.lit(other), self._scol))
        else:
            return _column_op(spark.Column.__radd__)(self, other)

    __rsub__ = _column_op(spark.Column.__rsub__)
    __rmul__ = _column_op(spark.Column.__rmul__)
    __rdiv__ = _numpy_column_op(spark.Column.__rdiv__)
    __rtruediv__ = _numpy_column_op(spark.Column.__rtruediv__)

    def __floordiv__(self, other):
        return self._with_new_scol(
            F.floor(_numpy_column_op(spark.Column.__div__)(self, other)._scol))

    def __rfloordiv__(self, other):
        return self._with_new_scol(
            F.floor(_numpy_column_op(spark.Column.__rdiv__)(self, other)._scol))

    __rmod__ = _column_op(spark.Column.__rmod__)
    __pow__ = _column_op(spark.Column.__pow__)
    __rpow__ = _column_op(spark.Column.__rpow__)

    # logistic operators
    __eq__ = _column_op(spark.Column.__eq__)
    __ne__ = _column_op(spark.Column.__ne__)
    __lt__ = _column_op(spark.Column.__lt__)
    __le__ = _column_op(spark.Column.__le__)
    __ge__ = _column_op(spark.Column.__ge__)
    __gt__ = _column_op(spark.Column.__gt__)

    # `and`, `or`, `not` cannot be overloaded in Python,
    # so use bitwise operators as boolean operators
    __and__ = _column_op(spark.Column.__and__)
    __or__ = _column_op(spark.Column.__or__)
    __invert__ = _column_op(spark.Column.__invert__)
    __rand__ = _column_op(spark.Column.__rand__)
    __ror__ = _column_op(spark.Column.__ror__)

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
        """
        if type(self.spark_type) == TimestampType:
            return np.dtype('datetime64[ns]')
        else:
            return np.dtype(to_arrow_type(self.spark_type).to_pandas_dtype())

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
        return self._kdf._sdf.rdd.isEmpty()

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
        """
        sdf = self._kdf._sdf.select(self._scol)
        col = self._scol

        ret = sdf.select(F.max(col.isNull() | F.isnan(col))).collect()[0][0]
        return ret

    @property
    def is_monotonic(self):
        """
        Return boolean if values in the object are monotonically increasing.

        .. note:: the current implementation of is_monotonic_increasing uses Spark's
            Window without specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

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
        """
        if len(self._kdf._internal.index_columns) == 0:
            raise ValueError("Index must be set.")

        col = self._scol
        index_columns = self._kdf._internal.index_columns
        window = Window.orderBy(index_columns).rowsBetween(-1, -1)
        sdf = self._kdf._sdf.withColumn(
            "__monotonic_col", (col >= F.lag(col, 1).over(window)) & col.isNotNull())
        kdf = ks.DataFrame(
            self._kdf._internal.copy(
                sdf=sdf, data_columns=self._kdf._internal.data_columns + ["__monotonic_col"]))
        return kdf["__monotonic_col"].all()

    is_monotonic_increasing = is_monotonic

    @property
    def is_monotonic_decreasing(self):
        """
        Return boolean if values in the object are monotonically decreasing.

        .. note:: the current implementation of is_monotonic_decreasing uses Spark's
            Window without specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

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
        """
        if len(self._kdf._internal.index_columns) == 0:
            raise ValueError("Index must be set.")

        col = self._scol
        index_columns = self._kdf._internal.index_columns
        window = Window.orderBy(index_columns).rowsBetween(-1, -1)
        sdf = self._kdf._sdf.withColumn(
            "__monotonic_col", (col <= F.lag(col, 1).over(window)) & col.isNotNull())
        kdf = ks.DataFrame(
            self._kdf._internal.copy(
                sdf=sdf, data_columns=self._kdf._internal.data_columns + ["__monotonic_col"]))
        return kdf["__monotonic_col"].all()

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
        Name: 0, dtype: int32

        >>> ser.astype('int64')
        0    1
        1    2
        Name: 0, dtype: int64
        """
        from databricks.koalas.typedef import as_spark_type
        spark_type = as_spark_type(dtype)
        if not spark_type:
            raise ValueError("Type {} not understood".format(dtype))
        return self._with_new_scol(self._scol.cast(spark_type))

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
        """
        if not is_list_like(values):
            raise TypeError("only list-like objects are allowed to be passed"
                            " to isin(), you passed a [{values_type}]"
                            .format(values_type=type(values).__name__))

        return self._with_new_scol(self._scol.isin(list(values)).alias(self.name))

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
        Name: 0, dtype: bool
        """
        if isinstance(self.spark_type, (FloatType, DoubleType)):
            return self._with_new_scol(self._scol.isNull() | F.isnan(self._scol)).alias(self.name)
        else:
            return self._with_new_scol(self._scol.isNull()).alias(self.name)

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
        Name: 0, dtype: float64

        >>> ser.notna()
        0     True
        1     True
        2    False
        Name: 0, dtype: bool
        """
        return (~self.isnull()).alias(self.name)

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
        """

        if axis not in [0, 'index']:
            raise ValueError('axis should be either 0 or "index" currently.')

        sdf = self._kdf._sdf.select(self._scol)
        col = self._scol

        # Note that we're ignoring `None`s here for now.
        # any and every was added as of Spark 3.0
        # ret = sdf.select(F.expr("every(CAST(`%s` AS BOOLEAN))" % sdf.columns[0])).collect()[0][0]
        # Here we use min as its alternative:
        ret = sdf.select(F.min(F.coalesce(col.cast('boolean'), F.lit(True)))).collect()[0][0]
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
        """

        if axis not in [0, 'index']:
            raise ValueError('axis should be either 0 or "index" currently.')

        sdf = self._kdf._sdf.select(self._scol)
        col = self._scol

        # Note that we're ignoring `None`s here for now.
        # any and every was added as of Spark 3.0
        # ret = sdf.select(F.expr("any(CAST(`%s` AS BOOLEAN))" % sdf.columns[0])).collect()[0][0]
        # Here we use max as its alternative:
        ret = sdf.select(F.max(F.coalesce(col.cast('boolean'), F.lit(False)))).collect()[0][0]
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

        """
        if len(self._internal.index_columns) == 0:
            raise ValueError("Index must be set.")

        if not isinstance(periods, int):
            raise ValueError('periods should be an int; however, got [%s]' % type(periods))

        col = self._scol
        index_columns = self._kdf._internal.index_columns
        window = Window.orderBy(index_columns).rowsBetween(-periods, -periods)
        shifted_col = F.lag(col, periods).over(window)
        col = F.when(
            shifted_col.isNull() | F.isnan(shifted_col), fill_value
        ).otherwise(shifted_col)

        return self._with_new_scol(col).alias(self.name)
