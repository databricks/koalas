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
A wrapper class for Spark Column to behave similar to pandas Series.
"""
import re
import inspect
from collections import Iterable
from functools import partial, wraps
from typing import Any, Optional, List, Union, Generic, TypeVar

import numpy as np
import pandas as pd
from pandas.core.accessor import CachedAccessor

from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StructType
from pyspark.sql.window import Window

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import DataFrame
from databricks.koalas.generic import _Frame, max_display_count
from databricks.koalas.internal import IndexMap, _InternalFrame
from databricks.koalas.missing.series import _MissingPandasLikeSeries
from databricks.koalas.plot import KoalasSeriesPlotMethods
from databricks.koalas.utils import validate_arguments_and_invoke_function
from databricks.koalas.datetimes import DatetimeMethods
from databricks.koalas.strings import StringMethods


# This regular expression pattern is complied and defined here to avoid to compile the same
# pattern every time it is used in _repr_ in Series.
# This pattern basically seeks the footer string from Pandas'
REPR_PATTERN = re.compile(r"Length: (?P<length>[0-9]+)")

_flex_doc_SERIES = """
Return {desc} of series and other, element-wise (binary operator `{op_name}`).

Equivalent to ``{equiv}``

Parameters
----------
other : Series or scalar value

Returns
-------
Series
    The result of the operation.

See Also
--------
Series.{reverse}

{series_examples}
"""

_add_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [1, 1, 1, np.nan],
...                    'b': [1, np.nan, 1, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  1.0  1.0
b  1.0  NaN
c  1.0  1.0
d  NaN  NaN

>>> df.a.add(df.b)
a    2.0
b    NaN
c    2.0
d    NaN
Name: a, dtype: float64
"""

_sub_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [1, 1, 1, np.nan],
...                    'b': [1, np.nan, 1, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  1.0  1.0
b  1.0  NaN
c  1.0  1.0
d  NaN  NaN

>>> df.a.subtract(df.b)
a    0.0
b    NaN
c    0.0
d    NaN
Name: a, dtype: float64
"""

_mul_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.multiply(df.b)
a    4.0
b    NaN
c    8.0
d    NaN
Name: a, dtype: float64
"""

_div_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.divide(df.b)
a    1.0
b    NaN
c    2.0
d    NaN
Name: a, dtype: float64
"""

_pow_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.pow(df.b)
a     4.0
b     NaN
c    16.0
d     NaN
Name: a, dtype: float64
"""

_mod_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.mod(df.b)
a    0.0
b    NaN
c    0.0
d    NaN
Name: a, dtype: float64
"""

_floordiv_example_SERIES = """
Examples
--------
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.floordiv(df.b)
a    1.0
b    NaN
c    2.0
d    NaN
Name: a, dtype: float64
"""

T = TypeVar("T")

# Needed to disambiguate Series.str and str type
str_type = str


class Series(_Frame, IndexOpsMixin, Generic[T]):
    """
    Koala Series that corresponds to Pandas Series logically. This holds Spark Column
    internally.

    :ivar _internal: an internal immutable Frame to manage metadata.
    :type _internal: _InternalFrame
    :ivar _kdf: Parent's Koalas DataFrame
    :type _kdf: ks.DataFrame

    Parameters
    ----------
    data : array-like, dict, or scalar value, Pandas Series
        Contains data stored in Series
        If data is a dict, argument order is maintained for Python 3.6
        and later.
        Note that if `data` is a Pandas Series, other arguments should not be used.
    index : array-like or Index (1d)
        Values must be hashable and have the same length as `data`.
        Non-unique index values are allowed. Will default to
        RangeIndex (0, 1, 2, ..., n) if not provided. If both a dict and index
        sequence are used, the index will override the keys found in the
        dict.
    dtype : numpy.dtype or None
        If None, dtype will be inferred
    copy : boolean, default False
        Copy input data
    """

    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False,
                 anchor=None):
        if isinstance(data, _InternalFrame):
            assert dtype is None
            assert name is None
            assert not copy
            assert not fastpath
            IndexOpsMixin.__init__(self, data, anchor)
        else:
            if isinstance(data, pd.Series):
                assert index is None
                assert dtype is None
                assert name is None
                assert not copy
                assert anchor is None
                assert not fastpath
                s = data
            else:
                s = pd.Series(
                    data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath)
            kdf = DataFrame(s)
            IndexOpsMixin.__init__(self, kdf._internal.copy(
                scol=kdf._internal._sdf[kdf._internal.data_columns[0]]), kdf)

    @property
    def _index_map(self) -> List[IndexMap]:
        return self._internal.index_map

    def _with_new_scol(self, scol: spark.Column) -> 'Series':
        """
        Copy Koalas Series with the new Spark Column.

        :param scol: the new Spark Column
        :return: the copied Series
        """
        return Series(self._kdf._internal.copy(scol=scol), anchor=self._kdf)

    @property
    def dtypes(self):
        """Return the dtype object of the underlying data.

        >>> s = ks.Series(list('abc'))
        >>> s.dtype == s.dtypes
        True
        """
        return self.dtype

    @property
    def spark_type(self):
        """ Returns the data type as defined by Spark, as a Spark DataType object."""
        return self.schema.fields[-1].dataType

    plot = CachedAccessor("plot", KoalasSeriesPlotMethods)

    # Arithmetic Operators
    def add(self, other):
        return (self + other).rename(self.name)

    add.__doc__ = _flex_doc_SERIES.format(
        desc='Addition',
        op_name="+",
        equiv="series + other",
        reverse='radd',
        series_examples=_add_example_SERIES)

    def radd(self, other):
        return (other + self).rename(self.name)

    radd.__doc__ = _flex_doc_SERIES.format(
        desc='Addition',
        op_name="+",
        equiv="other + series",
        reverse='add',
        series_examples=_add_example_SERIES)

    def div(self, other):
        return (self / other).rename(self.name)

    div.__doc__ = _flex_doc_SERIES.format(
        desc='Floating division',
        op_name="/",
        equiv="series / other",
        reverse='rdiv',
        series_examples=_div_example_SERIES)

    divide = div

    def rdiv(self, other):
        return (other / self).rename(self.name)

    rdiv.__doc__ = _flex_doc_SERIES.format(
        desc='Floating division',
        op_name="/",
        equiv="other / series",
        reverse='div',
        series_examples=_div_example_SERIES)

    def truediv(self, other):
        return (self / other).rename(self.name)

    truediv.__doc__ = _flex_doc_SERIES.format(
        desc='Floating division',
        op_name="/",
        equiv="series / other",
        reverse='rtruediv',
        series_examples=_div_example_SERIES)

    def rtruediv(self, other):
        return (other / self).rename(self.name)

    rtruediv.__doc__ = _flex_doc_SERIES.format(
        desc='Floating division',
        op_name="/",
        equiv="other / series",
        reverse='truediv',
        series_examples=_div_example_SERIES)

    def mul(self, other):
        return (self * other).rename(self.name)

    mul.__doc__ = _flex_doc_SERIES.format(
        desc='Multiplication',
        op_name="*",
        equiv="series * other",
        reverse='rmul',
        series_examples=_mul_example_SERIES)

    multiply = mul

    def rmul(self, other):
        return (other * self).rename(self.name)

    rmul.__doc__ = _flex_doc_SERIES.format(
        desc='Multiplication',
        op_name="*",
        equiv="other * series",
        reverse='mul',
        series_examples=_mul_example_SERIES)

    def sub(self, other):
        return (self - other).rename(self.name)

    sub.__doc__ = _flex_doc_SERIES.format(
        desc='Subtraction',
        op_name="-",
        equiv="series - other",
        reverse='rsub',
        series_examples=_sub_example_SERIES)

    subtract = sub

    def rsub(self, other):
        return (other - self).rename(self.name)

    rsub.__doc__ = _flex_doc_SERIES.format(
        desc='Subtraction',
        op_name="-",
        equiv="other - series",
        reverse='sub',
        series_examples=_sub_example_SERIES)

    def mod(self, other):
        return (self % other).rename(self.name)

    mod.__doc__ = _flex_doc_SERIES.format(
        desc='Modulo',
        op_name='%',
        equiv='series % other',
        reverse='rmod',
        series_examples=_mod_example_SERIES)

    def rmod(self, other):
        return (other % self).rename(self.name)

    rmod.__doc__ = _flex_doc_SERIES.format(
        desc='Modulo',
        op_name='%',
        equiv='other % series',
        reverse='mod',
        series_examples=_mod_example_SERIES)

    def pow(self, other):
        return (self ** other).rename(self.name)

    pow.__doc__ = _flex_doc_SERIES.format(
        desc='Exponential power of series',
        op_name='**',
        equiv='series ** other',
        reverse='rpow',
        series_examples=_pow_example_SERIES)

    def rpow(self, other):
        return (other - self).rename(self.name)

    rpow.__doc__ = _flex_doc_SERIES.format(
        desc='Exponential power',
        op_name='**',
        equiv='other ** series',
        reverse='pow',
        series_examples=_pow_example_SERIES)

    def floordiv(self, other):
        return (self // other).rename(self.name)

    floordiv.__doc__ = _flex_doc_SERIES.format(
        desc='Integer division',
        op_name='//',
        equiv='series // other',
        reverse='rfloordiv',
        series_examples=_floordiv_example_SERIES)

    def rfloordiv(self, other):
        return (other - self).rename(self.name)

    rfloordiv.__doc__ = _flex_doc_SERIES.format(
        desc='Integer division',
        op_name='//',
        equiv='other // series',
        reverse='floordiv',
        series_examples=_floordiv_example_SERIES)

    # Comparison Operators
    def eq(self, other):
        """
        Compare if the current value is equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a == 1
        a     True
        b    False
        c    False
        d    False
        Name: (a = 1), dtype: bool

        >>> df.b.eq(1)
        a    True
        b    None
        c    True
        d    None
        Name: b, dtype: object
        """
        return (self == other).rename(self.name)

    equals = eq

    def gt(self, other):
        """
        Compare if the current value is greater than the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a > 1
        a    False
        b     True
        c     True
        d     True
        Name: (a > 1), dtype: bool


        >>> df.b.gt(1)
        a    False
        b     None
        c    False
        d     None
        Name: b, dtype: object
        """
        return (self > other).rename(self.name)

    def ge(self, other):
        """
        Compare if the current value is greater than or equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a >= 2
        a    False
        b     True
        c     True
        d     True
        Name: (a >= 2), dtype: bool

        >>> df.b.ge(2)
        a    False
        b     None
        c    False
        d     None
        Name: b, dtype: object
        """
        return (self >= other).rename(self.name)

    def lt(self, other):
        """
        Compare if the current value is less than the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a < 1
        a    False
        b    False
        c    False
        d    False
        Name: (a < 1), dtype: bool

        >>> df.b.lt(2)
        a    True
        b    None
        c    True
        d    None
        Name: b, dtype: object
        """
        return (self < other).rename(self.name)

    def le(self, other):
        """
        Compare if the current value is less than or equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a <= 2
        a     True
        b     True
        c    False
        d    False
        Name: (a <= 2), dtype: bool

        >>> df.b.le(2)
        a    True
        b    None
        c    True
        d    None
        Name: b, dtype: object
        """
        return (self <= other).rename(self.name)

    def ne(self, other):
        """
        Compare if the current value is not equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.a != 1
        a    False
        b     True
        c     True
        d     True
        Name: (NOT (a = 1)), dtype: bool

        >>> df.b.ne(1)
        a    False
        b     None
        c    False
        d     None
        Name: b, dtype: object
        """
        return (self != other).rename(self.name)

    # TODO: arg should support Series
    # TODO: NaN and None
    def map(self, arg):
        """
        Map values of Series according to input correspondence.

        Used for substituting each value in a Series with another value,
        that may be derived from a function, a ``dict``.

        .. note:: make sure the size of the dictionary is not huge because it could
            downgrade the performance or throw OutOfMemoryError due to a huge
            expression within Spark. Consider the input as a functions as an
            alternative instead in this case.

        Parameters
        ----------
        arg : function or dict
            Mapping correspondence.

        Returns
        -------
        Series
            Same index as caller.

        See Also
        --------
        Series.apply : For applying more complex functions on a Series.
        DataFrame.applymap : Apply a function elementwise on a whole DataFrame.

        Notes
        -----
        When ``arg`` is a dictionary, values in Series that are not in the
        dictionary (as keys) are converted to ``None``. However, if the
        dictionary is a ``dict`` subclass that defines ``__missing__`` (i.e.
        provides a method for default values), then this default is used
        rather than ``None``.

        Examples
        --------
        >>> s = ks.Series(['cat', 'dog', None, 'rabbit'])
        >>> s
        0       cat
        1       dog
        2      None
        3    rabbit
        Name: 0, dtype: object

        ``map`` accepts a ``dict``. Values that are not found
        in the ``dict`` are converted to ``None``, unless the dict has a default
        value (e.g. ``defaultdict``):

        >>> s.map({'cat': 'kitten', 'dog': 'puppy'})
        0    kitten
        1     puppy
        2      None
        3      None
        Name: 0, dtype: object

        It also accepts a function:

        >>> def format(x) -> str:
        ...     return 'I am a {}'.format(x)

        >>> s.map(format)
        0       I am a cat
        1       I am a dog
        2      I am a None
        3    I am a rabbit
        Name: 0, dtype: object
        """
        if isinstance(arg, dict):
            is_start = True
            # In case dictionary is empty.
            current = F.when(F.lit(False), F.lit(None).cast(self.spark_type))

            for to_replace, value in arg.items():
                if is_start:
                    current = F.when(self._scol == F.lit(to_replace), value)
                    is_start = False
                else:
                    current = current.when(self._scol == F.lit(to_replace), value)

            if hasattr(arg, "__missing__"):
                tmp_val = arg[np._NoValue]
                del arg[np._NoValue]  # Remove in case it's set in defaultdict.
                current = current.otherwise(F.lit(tmp_val))
            else:
                current = current.otherwise(F.lit(None).cast(self.spark_type))
            return Series(self._kdf._internal.copy(scol=current),
                          anchor=self._kdf).rename(self.name)
        else:
            return self.apply(arg)

    def astype(self, dtype) -> 'Series':
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
        return Series(self._kdf._internal.copy(scol=self._scol.cast(spark_type)), anchor=self._kdf)

    def getField(self, name):
        if not isinstance(self.schema, StructType):
            raise AttributeError("Not a struct: {}".format(self.schema))
        else:
            fnames = self.schema.fieldNames()
            if name not in fnames:
                raise AttributeError(
                    "Field {} not found, possible values are {}".format(name, ", ".join(fnames)))
            return Series(self._kdf._internal.copy(scol=self._scol.getField(name)),
                          anchor=self._kdf)

    def alias(self, name):
        """An alias for :meth:`Series.rename`."""
        return self.rename(name)

    @property
    def schema(self) -> StructType:
        """Return the underlying Spark DataFrame's schema."""
        return self.to_dataframe()._sdf.schema

    @property
    def shape(self):
        """Return a tuple of the shape of the underlying data."""
        return len(self),

    @property
    def ndim(self):
        """Returns number of dimensions of the Series."""
        return 1

    @property
    def name(self) -> str:
        """Return name of the Series."""
        return self._internal.data_columns[0]

    @name.setter
    def name(self, name):
        self.rename(name, inplace=True)

    # TODO: Functionality and documentation should be matched. Currently, changing index labels
    # taking dictionary and function to change index are not supported.
    def rename(self, index=None, **kwargs):
        """
        Alter Series name.

        Parameters
        ----------
        index : scalar
            Scalar will alter the ``Series.name`` attribute.

        inplace : bool, default False
            Whether to return a new Series. If True then value of copy is
            ignored.

        Returns
        -------
        Series
            Series with name altered.

        Examples
        --------

        >>> s = ks.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        Name: 0, dtype: int64

        >>> s.rename("my_name")  # scalar, changes Series.name
        0    1
        1    2
        2    3
        Name: my_name, dtype: int64
        """
        if index is None:
            return self
        scol = self._scol.alias(index)
        if kwargs.get('inplace', False):
            self._internal = self._internal.copy(scol=scol)
            return self
        else:
            return Series(self._kdf._internal.copy(scol=scol), anchor=self._kdf)

    @property
    def index(self):
        """The index (axis labels) Column of the Series.

        Currently not supported when the DataFrame has no index.

        See Also
        --------
        Index
        """
        return self._kdf.index

    @property
    def is_unique(self):
        """
        Return boolean if values in the object are unique

        Returns
        -------
        is_unique : boolean

        >>> ks.Series([1, 2, 3]).is_unique
        True
        >>> ks.Series([1, 2, 2]).is_unique
        False
        >>> ks.Series([1, 2, 3, None]).is_unique
        True
        """
        sdf = self._kdf._sdf.select(self._scol)
        col = self._scol

        # Here we check:
        #   1. the distinct count without nulls and count without nulls for non-null values
        #   2. count null values and see if null is a distinct value.
        #
        # This workaround is in order to calculate the distinct count including nulls in
        # single pass. Note that COUNT(DISTINCT expr) in Spark is designed to ignore nulls.
        return sdf.select(
            (F.count(col) == F.countDistinct(col)) &
            (F.count(F.when(col.isNull(), 1).otherwise(None)) <= 1)
        ).collect()[0][0]

    def reset_index(self, level=None, drop=False, name=None, inplace=False):
        """
        Generate a new DataFrame or Series with the index reset.

        This is useful when the index needs to be treated as a column,
        or when the index is meaningless and needs to be reset
        to the default before another operation.

        Parameters
        ----------
        level : int, str, tuple, or list, default optional
            For a Series with a MultiIndex, only remove the specified levels from the index.
            Removes all levels by default.
        drop : bool, default False
            Just reset the index, without inserting it as a column in the new DataFrame.
        name : object, optional
            The name to use for the column containing the original Series values.
            Uses self.name by default. This argument is ignored when drop is True.
        inplace : bool, default False
            Modify the Series in place (do not create a new object).

        Returns
        -------
        Series or DataFrame
            When `drop` is False (the default), a DataFrame is returned.
            The newly created columns will come first in the DataFrame,
            followed by the original Series values.
            When `drop` is True, a `Series` is returned.
            In either case, if ``inplace=True``, no value is returned.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4], name='foo',
        ...               index=pd.Index(['a', 'b', 'c', 'd'], name='idx'))

        Generate a DataFrame with default index.

        >>> s.reset_index()
          idx  foo
        0   a    1
        1   b    2
        2   c    3
        3   d    4

        To specify the name of the new column use `name`.

        >>> s.reset_index(name='values')
          idx  values
        0   a       1
        1   b       2
        2   c       3
        3   d       4

        To generate a new Series with the default set `drop` to True.

        >>> s.reset_index(drop=True)
        0    1
        1    2
        2    3
        3    4
        Name: foo, dtype: int64

        To update the Series in place, without generating a new one
        set `inplace` to True. Note that it also requires ``drop=True``.

        >>> s.reset_index(inplace=True, drop=True)
        >>> s
        0    1
        1    2
        2    3
        3    4
        Name: foo, dtype: int64
        """
        if inplace and not drop:
            raise TypeError('Cannot reset_index inplace on a Series to create a DataFrame')

        if name is not None:
            kdf = self.rename(name).to_dataframe()
        else:
            kdf = self.to_dataframe()
        kdf = kdf.reset_index(level=level, drop=drop)
        if drop:
            kseries = _col(kdf)
            if inplace:
                self._internal = kseries._internal
                self._kdf = kseries._kdf
            else:
                return kseries
        else:
            return kdf

    def to_frame(self, name=None) -> spark.DataFrame:
        """
        Convert Series to DataFrame.

        Parameters
        ----------
        name : object, default None
            The passed name should substitute for the series name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame representation of Series.

        Examples
        --------
        >>> s = ks.Series(["a", "b", "c"])
        >>> s.to_frame()
           0
        0  a
        1  b
        2  c

        >>> s = ks.Series(["a", "b", "c"], name="vals")
        >>> s.to_frame()
          vals
        0    a
        1    b
        2    c
        """
        renamed = self.rename(name)
        sdf = renamed._internal.spark_df
        internal = _InternalFrame(sdf=sdf,
                                  data_columns=[sdf.schema[-1].name],
                                  index_map=renamed._internal.index_map)
        return DataFrame(internal)

    to_dataframe = to_frame

    def to_string(self, buf=None, na_rep='NaN', float_format=None, header=True,
                  index=True, length=False, dtype=False, name=False,
                  max_rows=None):
        """
        Render a string representation of the Series.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory. If the input
                  is large, set max_rows parameter.

        Parameters
        ----------
        buf : StringIO-like, optional
            buffer to write to
        na_rep : string, optional
            string representation of NAN to use, default 'NaN'
        float_format : one-parameter function, optional
            formatter function to apply to columns' elements if they are floats
            default None
        header : boolean, default True
            Add the Series header (index name)
        index : bool, optional
            Add index (row) labels, default True
        length : boolean, default False
            Add the Series length
        dtype : boolean, default False
            Add the Series dtype
        name : boolean, default False
            Add the Series name if not None
        max_rows : int, optional
            Maximum number of rows to show before truncating. If None, show
            all.

        Returns
        -------
        formatted : string (if not buffer passed)

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)], columns=['dogs', 'cats'])
        >>> print(df['dogs'].to_string())
        0    0.2
        1    0.0
        2    0.6
        3    0.2

        >>> print(df['dogs'].to_string(max_rows=2))
        0    0.2
        1    0.0
        """
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        if max_rows is not None:
            kseries = self.head(max_rows)
        else:
            kseries = self

        return validate_arguments_and_invoke_function(
            kseries.to_pandas(), self.to_string, pd.Series.to_string, args)

    def to_clipboard(self, excel=True, sep=None, **kwargs):
        # Docstring defined below by reusing DataFrame.to_clipboard's.
        args = locals()
        kseries = self

        return validate_arguments_and_invoke_function(
            kseries.to_pandas(), self.to_clipboard, pd.Series.to_clipboard, args)

    to_clipboard.__doc__ = DataFrame.to_clipboard.__doc__

    def to_dict(self, into=dict):
        """
        Convert Series to {label -> value} dict or dict-like object.

        .. note:: This method should only be used if the resulting Pandas DataFrame is expected
            to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        into : class, default dict
            The collections.abc.Mapping subclass to use as the return
            object. Can be the actual class or an empty
            instance of the mapping type you want.  If you want a
            collections.defaultdict, you must pass it initialized.

        Returns
        -------
        collections.abc.Mapping
            Key-value representation of Series.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4])
        >>> s_dict = s.to_dict()
        >>> sorted(s_dict.items())
        [(0, 1), (1, 2), (2, 3), (3, 4)]

        >>> from collections import OrderedDict, defaultdict
        >>> s.to_dict(OrderedDict)
        OrderedDict([(0, 1), (1, 2), (2, 3), (3, 4)])

        >>> dd = defaultdict(list)
        >>> s.to_dict(dd)  # doctest: +ELLIPSIS
        defaultdict(<class 'list'>, {...})
        """
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries.to_pandas(), self.to_dict, pd.Series.to_dict, args)

    def to_latex(self, buf=None, columns=None, col_space=None, header=True, index=True,
                 na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True,
                 bold_rows=False, column_format=None, longtable=None, escape=None, encoding=None,
                 decimal='.', multicolumn=None, multicolumn_format=None, multirow=None):

        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries.to_pandas(), self.to_latex, pd.Series.to_latex, args)

    to_latex.__doc__ = DataFrame.to_latex.__doc__

    def to_pandas(self):
        """
        Return a pandas Series.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory. If the input
                  is large, set max_rows parameter.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)], columns=['dogs', 'cats'])
        >>> df['dogs'].to_pandas()
        0    0.2
        1    0.0
        2    0.6
        3    0.2
        Name: dogs, dtype: float64
        """
        return _col(self._internal.pandas_df.copy())

    # Alias to maintain backward compatibility with Spark
    toPandas = to_pandas

    def to_list(self):
        """
        Return a list of the values.

        These are each a scalar type, which is a Python scalar
        (for str, int, float) or a pandas scalar
        (for Timestamp/Timedelta/Interval/Period)

        .. note:: This method should only be used if the resulting list is expected
            to be small, as all the data is loaded into the driver's memory.

        """
        return self.to_pandas().to_list()

    tolist = to_list

    def fillna(self, value=None, axis=None, inplace=False):
        """Fill NA/NaN values.

        Parameters
        ----------
        value : scalar
            Value to use to fill holes.
        axis : {0 or `index`}
            1 and `columns` are not supported.
        inplace : boolean, default False
            Fill in place (do not create a new object)

        Returns
        -------
        Series
            Series with NA entries filled.

        Examples
        --------
        >>> s = ks.Series([np.nan, 2, 3, 4, np.nan, 6], name='x')
        >>> s
        0    NaN
        1    2.0
        2    3.0
        3    4.0
        4    NaN
        5    6.0
        Name: x, dtype: float64

        Replace all NaN elements with 0s.

        >>> s.fillna(0)
        0    0.0
        1    2.0
        2    3.0
        3    4.0
        4    0.0
        5    6.0
        Name: x, dtype: float64
        """

        kseries = _col(self.to_dataframe().fillna(value=value, axis=axis, inplace=False))
        if inplace:
            self._internal = kseries._internal
            self._kdf = kseries._kdf
        else:
            return kseries

    def dropna(self, axis=0, inplace=False, **kwargs):
        """
        Return a new Series with missing values removed.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            There is only one axis to drop values from.
        inplace : bool, default False
            If True, do operation inplace and return None.
        **kwargs
            Not in use.

        Returns
        -------
        Series
            Series with NA entries dropped from it.

        Examples
        --------
        >>> ser = ks.Series([1., 2., np.nan])
        >>> ser
        0    1.0
        1    2.0
        2    NaN
        Name: 0, dtype: float64

        Drop NA values from a Series.

        >>> ser.dropna()
        0    1.0
        1    2.0
        Name: 0, dtype: float64

        Keep the Series with valid entries in the same variable.

        >>> ser.dropna(inplace=True)
        >>> ser
        0    1.0
        1    2.0
        Name: 0, dtype: float64
        """
        # TODO: last two examples from Pandas produce different results.
        kseries = _col(self.to_dataframe().dropna(axis=axis, inplace=False))
        if inplace:
            self._internal = kseries._internal
            self._kdf = kseries._kdf
        else:
            return kseries

    def clip(self, lower: Union[float, int] = None, upper: Union[float, int] = None) -> 'Series':
        """
        Trim values at input threshold(s).

        Assigns values outside boundary to boundary values.

        Parameters
        ----------
        lower : float or int, default None
            Minimum threshold value. All values below this threshold will be set to it.
        upper : float or int, default None
            Maximum threshold value. All values above this threshold will be set to it.

        Returns
        -------
        Series
            Series with the values outside the clip boundaries replaced

        Examples
        --------
        >>> ks.Series([0, 2, 4]).clip(1, 3)
        0    1
        1    2
        2    3
        Name: 0, dtype: int64

        Notes
        -----
        One difference between this implementation and pandas is that running
        `pd.Series(['a', 'b']).clip(0, 1)` will crash with "TypeError: '<=' not supported between
        instances of 'str' and 'int'" while `ks.Series(['a', 'b']).clip(0, 1)` will output the
        original Series, simply ignoring the incompatible types.
        """
        return _col(self.to_dataframe().clip(lower, upper))

    def head(self, n=5):
        """
        Return the first n rows.

        This function returns the first n rows for the object based on position.
        It is useful for quickly testing if your object has the right type of data in it.

        Parameters
        ----------
        n : Integer, default =  5

        Returns
        -------
        The first n rows of the caller object.

        Examples
        --------
        >>> df = ks.DataFrame({'animal':['alligator', 'bee', 'falcon', 'lion']})
        >>> df.animal.head(2)  # doctest: +NORMALIZE_WHITESPACE
        0     alligator
        1     bee
        Name: animal, dtype: object
        """
        return _col(self.to_dataframe().head(n))

    # TODO: Categorical type isn't supported (due to PySpark's limitation) and
    # some doctests related with timestamps were not added.
    def unique(self):
        """
        Return unique values of Series object.

        Uniques are returned in order of appearance. Hash table-based unique,
        therefore does NOT sort.

        .. note:: This method returns newly creased Series whereas Pandas returns
                  the unique values as a NumPy array.

        Returns
        -------
        Returns the unique values as a Series.

        See Examples section.

        Examples
        --------
        >>> ks.Series([2, 1, 3, 3], name='A').unique()
        0    1
        1    3
        2    2
        Name: A, dtype: int64

        >>> ks.Series([pd.Timestamp('2016-01-01') for _ in range(3)]).unique()
        0   2016-01-01
        Name: 0, dtype: datetime64[ns]
        """
        sdf = self.to_dataframe()._sdf
        return _col(DataFrame(sdf.select(self._scol).distinct()))

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
        The number of unique values as an int.

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
        """
        return self.to_dataframe().nunique(dropna=dropna, approx=approx, rsd=rsd).iloc[0]

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
        """
        if bins is not None:
            raise NotImplementedError("value_counts currently does not support bins")

        if dropna:
            sdf_dropna = self._kdf._sdf.filter(self.notna()._scol)
        else:
            sdf_dropna = self._kdf._sdf
        sdf = sdf_dropna.groupby(self._scol).count()
        if sort:
            if ascending:
                sdf = sdf.orderBy(F.col('count'))
            else:
                sdf = sdf.orderBy(F.col('count').desc())

        if normalize:
            sum = sdf_dropna.count()
            sdf = sdf.withColumn('count', F.col('count') / F.lit(sum))

        index_name = 'index' if self.name != 'index' else 'level_0'
        sdf = sdf.select(sdf[self.name].alias(index_name), sdf['count'].alias(self.name))
        internal = _InternalFrame(sdf=sdf, data_columns=[self.name], index_map=[(index_name, None)])
        return _col(DataFrame(internal))

    def sort_values(self, ascending: bool = True, inplace: bool = False,
                    na_position: str = 'last') -> Union['Series', None]:
        """
        Sort by the values.

        Sort a Series in ascending or descending order by some criterion.

        Parameters
        ----------
        ascending : bool or list of bool, default True
             Sort ascending vs. descending. Specify list for multiple sort
             orders.  If this is a list of bools, must match the length of
             the by.
        inplace : bool, default False
             if True, perform operation in-place
        na_position : {'first', 'last'}, default 'last'
             `first` puts NaNs at the beginning, `last` puts NaNs at the end

        Returns
        -------
        sorted_obj : Series ordered by values.

        Examples
        --------
        >>> s = ks.Series([np.nan, 1, 3, 10, 5])
        >>> s
        0     NaN
        1     1.0
        2     3.0
        3    10.0
        4     5.0
        Name: 0, dtype: float64

        Sort values ascending order (default behaviour)

        >>> s.sort_values(ascending=True)
        1     1.0
        2     3.0
        4     5.0
        3    10.0
        0     NaN
        Name: 0, dtype: float64

        Sort values descending order

        >>> s.sort_values(ascending=False)
        3    10.0
        4     5.0
        2     3.0
        1     1.0
        0     NaN
        Name: 0, dtype: float64

        Sort values inplace

        >>> s.sort_values(ascending=False, inplace=True)
        >>> s
        3    10.0
        4     5.0
        2     3.0
        1     1.0
        0     NaN
        Name: 0, dtype: float64

        Sort values putting NAs first

        >>> s.sort_values(na_position='first')
        0     NaN
        1     1.0
        2     3.0
        4     5.0
        3    10.0
        Name: 0, dtype: float64

        Sort a series of strings

        >>> s = ks.Series(['z', 'b', 'd', 'a', 'c'])
        >>> s
        0    z
        1    b
        2    d
        3    a
        4    c
        Name: 0, dtype: object

        >>> s.sort_values()
        3    a
        1    b
        4    c
        2    d
        0    z
        Name: 0, dtype: object
        """
        kseries = _col(self.to_dataframe().sort_values(by=self.name, ascending=ascending,
                                                       na_position=na_position))
        if inplace:
            self._internal = kseries._internal
            self._kdf = kseries._kdf
            return None
        else:
            return kseries

    def sort_index(self, axis: int = 0, level: int = None, ascending: bool = True,
                   inplace: bool = False, kind: str = None, na_position: str = 'last') \
            -> Optional['Series']:
        """
        Sort object by labels (along an axis)

        Parameters
        ----------
        axis : index, columns to direct sorting. Currently, only axis = 0 is supported.
        level : int or level name or list of ints or list of level names
            if not None, sort on values in specified index level(s)
        ascending : boolean, default True
            Sort ascending vs. descending
        inplace : bool, default False
            if True, perform operation in-place
        kind : str, default None
            Koalas does not allow specifying the sorting algorithm at the moment, default None
        na_position : {‘first’, ‘last’}, default ‘last’
            first puts NaNs at the beginning, last puts NaNs at the end. Not implemented for
            MultiIndex.

        Returns
        -------
        sorted_obj : Series

        Examples
        --------
        >>> df = ks.Series([2, 1, np.nan], index=['b', 'a', np.nan])

        >>> df.sort_index()
        a      1.0
        b      2.0
        NaN    NaN
        Name: 0, dtype: float64

        >>> df.sort_index(ascending=False)
        b      2.0
        a      1.0
        NaN    NaN
        Name: 0, dtype: float64

        >>> df.sort_index(na_position='first')
        NaN    NaN
        a      1.0
        b      2.0
        Name: 0, dtype: float64

        >>> df.sort_index(inplace=True)
        >>> df
        a      1.0
        b      2.0
        NaN    NaN
        Name: 0, dtype: float64

        >>> ks.Series(range(4), index=[['b', 'b', 'a', 'a'], [1, 0, 1, 0]], name='0').sort_index()
        a  0    3
           1    2
        b  0    1
           1    0
        Name: 0, dtype: int64
        """
        if axis != 0:
            raise ValueError("No other axes than 0 are supported at the moment")
        if level is not None:
            raise ValueError("The 'axis' argument is not supported at the moment")
        if kind is not None:
            raise ValueError("Specifying the sorting algorithm is supported at the moment.")
        kseries = _col(self.to_dataframe().sort_values(by=self._internal.index_columns,
                                                       ascending=ascending,
                                                       na_position=na_position))
        if inplace:
            self._internal = kseries._internal
            self._kdf = kseries._kdf
            return None
        else:
            return kseries

    def add_prefix(self, prefix):
        """
        Prefix labels with string `prefix`.

        For Series, the row labels are prefixed.
        For DataFrame, the column labels are prefixed.

        Parameters
        ----------
        prefix : str
           The string to add before each label.

        Returns
        -------
        Series
           New Series with updated labels.

        See Also
        --------
        Series.add_suffix: Suffix column labels with string `suffix`.
        DataFrame.add_suffix: Suffix column labels with string `suffix`.
        DataFrame.add_prefix: Prefix column labels with string `prefix`.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        Name: 0, dtype: int64

        >>> s.add_prefix('item_')
        item_0    1
        item_1    2
        item_2    3
        item_3    4
        Name: 0, dtype: int64
        """
        assert isinstance(prefix, str)
        kdf = self.to_dataframe()
        internal = kdf._internal
        sdf = internal.sdf
        sdf = sdf.select([F.concat(F.lit(prefix), sdf[index_column]).alias(index_column)
                          for index_column in internal.index_columns] + internal.data_columns)
        kdf._internal = internal.copy(sdf=sdf)
        return Series(kdf._internal.copy(scol=self._scol), anchor=kdf)

    def add_suffix(self, suffix):
        """
        Suffix labels with string suffix.

        For Series, the row labels are suffixed.
        For DataFrame, the column labels are suffixed.

        Parameters
        ----------
        suffix : str
           The string to add after each label.

        Returns
        -------
        Series
           New Series with updated labels.

        See Also
        --------
        Series.add_prefix: Prefix row labels with string `prefix`.
        DataFrame.add_prefix: Prefix column labels with string `prefix`.
        DataFrame.add_suffix: Suffix column labels with string `suffix`.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4])
        >>> s
        0    1
        1    2
        2    3
        3    4
        Name: 0, dtype: int64

        >>> s.add_suffix('_item')
        0_item    1
        1_item    2
        2_item    3
        3_item    4
        Name: 0, dtype: int64
        """
        assert isinstance(suffix, str)
        kdf = self.to_dataframe()
        internal = kdf._internal
        sdf = internal.sdf
        sdf = sdf.select([F.concat(sdf[index_column], F.lit(suffix)).alias(index_column)
                          for index_column in internal.index_columns] + internal.data_columns)
        kdf._internal = internal.copy(sdf=sdf)
        return Series(kdf._internal.copy(scol=self._scol), anchor=kdf)

    def corr(self, other, method='pearson'):
        """
        Compute correlation with `other` Series, excluding missing values.

        Parameters
        ----------
        other : Series
        method : {'pearson', 'spearman'}
            * pearson : standard correlation coefficient
            * spearman : Spearman rank correlation

        Returns
        -------
        correlation : float

        Examples
        --------
        >>> df = ks.DataFrame({'s1': [.2, .0, .6, .2],
        ...                    's2': [.3, .6, .0, .1]})
        >>> s1 = df.s1
        >>> s2 = df.s2
        >>> s1.corr(s2, method='pearson')  # doctest: +ELLIPSIS
        -0.851064...

        >>> s1.corr(s2, method='spearman')  # doctest: +ELLIPSIS
        -0.948683...

        Notes
        -----
        There are behavior differences between Koalas and pandas.

        * the `method` argument only accepts 'pearson', 'spearman'
        * the data should not contain NaNs. Koalas will return an error.
        * Koalas doesn't support the following argument(s).

          * `min_periods` argument is not supported
        """
        # This implementation is suboptimal because it computes more than necessary,
        # but it should be a start
        df = self._kdf.assign(corr_arg1=self, corr_arg2=other)[["corr_arg1", "corr_arg2"]]
        c = df.corr(method=method)
        return c.loc["corr_arg1", "corr_arg2"]

    def nsmallest(self, n: int = 5) -> 'Series':
        """
        Return the smallest `n` elements.

        Parameters
        ----------
        n : int, default 5
            Return this many ascending sorted values.

        Returns
        -------
        Series
            The `n` smallest values in the Series, sorted in increasing order.

        See Also
        --------
        Series.nlargest: Get the `n` largest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values().head(n)`` for small `n` relative to
        the size of the ``Series`` object.
        In Koalas, thanks to Spark's lazy execution and query optimizer,
        the two would have same performance.

        Examples
        --------
        >>> data = [1, 2, 3, 4, np.nan ,6, 7, 8]
        >>> s = ks.Series(data)
        >>> s
        0    1.0
        1    2.0
        2    3.0
        3    4.0
        4    NaN
        5    6.0
        6    7.0
        7    8.0
        Name: 0, dtype: float64

        The `n` largest elements where ``n=5`` by default.

        >>> s.nsmallest()
        0    1.0
        1    2.0
        2    3.0
        3    4.0
        5    6.0
        Name: 0, dtype: float64

        >>> s.nsmallest(3)
        0    1.0
        1    2.0
        2    3.0
        Name: 0, dtype: float64
        """
        return _col(self._kdf.nsmallest(n=n, columns=self.name))

    def nlargest(self, n: int = 5) -> 'Series':
        """
        Return the largest `n` elements.

        Parameters
        ----------
        n : int, default 5

        Returns
        -------
        Series
            The `n` largest values in the Series, sorted in decreasing order.

        See Also
        --------
        Series.nsmallest: Get the `n` smallest elements.
        Series.sort_values: Sort Series by values.
        Series.head: Return the first `n` rows.

        Notes
        -----
        Faster than ``.sort_values(ascending=False).head(n)`` for small `n`
        relative to the size of the ``Series`` object.

        In Koalas, thanks to Spark's lazy execution and query optimizer,
        the two would have same performance.

        Examples
        --------
        >>> data = [1, 2, 3, 4, np.nan ,6, 7, 8]
        >>> s = ks.Series(data)
        >>> s
        0    1.0
        1    2.0
        2    3.0
        3    4.0
        4    NaN
        5    6.0
        6    7.0
        7    8.0
        Name: 0, dtype: float64

        The `n` largest elements where ``n=5`` by default.

        >>> s.nlargest()
        7    8.0
        6    7.0
        5    6.0
        3    4.0
        2    3.0
        Name: 0, dtype: float64

        >>> s.nlargest(n=3)
        7    8.0
        6    7.0
        5    6.0
        Name: 0, dtype: float64


        """
        return _col(self._kdf.nlargest(n=n, columns=self.name))

    def count(self):
        """
        Return number of non-NA/null observations in the Series.

        Returns
        -------
        nobs : int

        Examples
        --------
        Constructing DataFrame from a dictionary:

        >>> df = ks.DataFrame({"Person":
        ...                    ["John", "Myla", "Lewis", "John", "Myla"],
        ...                    "Age": [24., np.nan, 21., 33, 26]})

        Notice the uncounted NA values:

        >>> df['Person'].count()
        5

        >>> df['Age'].count()
        4
        """
        return self._reduce_for_stat_function(_Frame._count_expr)

    def append(self, to_append: 'Series', ignore_index: bool = False,
               verify_integrity: bool = False) -> 'Series':
        """
        Concatenate two or more Series.

        Parameters
        ----------
        to_append : Series or list/tuple of Series
        ignore_index : boolean, default False
            If True, do not use the index labels.
        verify_integrity : boolean, default False
            If True, raise Exception on creating index with duplicates

        Returns
        -------
        appended : Series

        Examples
        --------
        >>> s1 = ks.Series([1, 2, 3])
        >>> s2 = ks.Series([4, 5, 6])
        >>> s3 = ks.Series([4, 5, 6], index=[3,4,5])

        >>> s1.append(s2)
        0    1
        1    2
        2    3
        0    4
        1    5
        2    6
        Name: 0, dtype: int64

        >>> s1.append(s3)
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        Name: 0, dtype: int64

        With ignore_index set to True:

        >>> s1.append(s2, ignore_index=True)
        0    1
        1    2
        2    3
        3    4
        4    5
        5    6
        Name: 0, dtype: int64
        """
        return _col(self.to_dataframe().append(to_append.to_dataframe(), ignore_index,
                                               verify_integrity))

    def sample(self, n: Optional[int] = None, frac: Optional[float] = None, replace: bool = False,
               random_state: Optional[int] = None) -> 'Series':
        return _col(self.to_dataframe().sample(
            n=n, frac=frac, replace=replace, random_state=random_state))

    sample.__doc__ = DataFrame.sample.__doc__

    def hist(self, bins=10, **kwds):
        return self.plot.hist(bins, **kwds)

    hist.__doc__ = KoalasSeriesPlotMethods.hist.__doc__

    def apply(self, func, args=(), **kwds):
        """
        Invoke function on values of Series.

        Can be a Python function that only works on the Series.

        .. note:: unlike pandas, it is required for `func` to specify return type hint.

        Parameters
        ----------
        func : function
            Python function to apply. Note that type hint for return type is required.
        args : tuple
            Positional arguments passed to func after the series value.
        **kwds
            Additional keyword arguments passed to func.

        Returns
        -------
        Series

        Examples
        --------
        Create a Series with typical summer temperatures for each city.

        >>> s = ks.Series([20, 21, 12],
        ...               index=['London', 'New York', 'Helsinki'])
        >>> s
        London      20
        New York    21
        Helsinki    12
        Name: 0, dtype: int64


        Square the values by defining a function and passing it as an
        argument to ``apply()``.

        >>> def square(x) -> np.int64:
        ...     return x ** 2
        >>> s.apply(square)
        London      400
        New York    441
        Helsinki    144
        Name: 0, dtype: int64


        Define a custom function that needs additional positional
        arguments and pass these additional arguments using the
        ``args`` keyword

        >>> def subtract_custom_value(x, custom_value) -> np.int64:
        ...     return x - custom_value

        >>> s.apply(subtract_custom_value, args=(5,))
        London      15
        New York    16
        Helsinki     7
        Name: 0, dtype: int64


        Define a custom function that takes keyword arguments
        and pass these arguments to ``apply``

        >>> def add_custom_values(x, **kwargs) -> np.int64:
        ...     for month in kwargs:
        ...         x += kwargs[month]
        ...     return x

        >>> s.apply(add_custom_values, june=30, july=20, august=25)
        London      95
        New York    96
        Helsinki    87
        Name: 0, dtype: int64


        Use a function from the Numpy library

        >>> def numpy_log(col) -> np.float64:
        ...     return np.log(col)
        >>> s.apply(numpy_log)
        London      2.995732
        New York    3.044522
        Helsinki    2.484907
        Name: 0, dtype: float64
        """
        assert callable(func), "the first argument should be a callable function."
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        if return_sig is None:
            raise ValueError("Given function must have return type hint; however, not found.")

        apply_each = wraps(func)(lambda s, *a, **k: s.apply(func, args=a, **k))
        wrapped = ks.pandas_wraps(return_col=return_sig)(apply_each)
        return wrapped(self, *args, **kwds).rename(self.name)

    def transpose(self, *args, **kwargs):
        """
        Return the transpose, which is by definition self.

        Examples
        --------
        It returns the same object as the transpose of the given series object, which is by
        definition self.

        >>> s = ks.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        Name: 0, dtype: int64

        >>> s.transpose()
        0    1
        1    2
        2    3
        Name: 0, dtype: int64
        """
        return Series(self._kdf._internal.copy(), anchor=self._kdf)

    T = property(transpose)

    def transform(self, func, *args, **kwargs):
        """
        Call ``func`` producing the same type as `self` with transformed values
        and that has the same axis length as input.

        .. note:: unlike pandas, it is required for `func` to specify return type hint.

        Parameters
        ----------
        func : function or list
            A function or a list of functions to use for transforming the data.
        *args
            Positional arguments to pass to `func`.
        **kwargs
            Keyword arguments to pass to `func`.

        Returns
        -------
        An instance of the same type with `self` that must have the same length as input.

        See Also
        --------
        Series.apply : Invoke function on Series.

        Examples
        --------

        >>> s = ks.Series(range(3))
        >>> s
        0    0
        1    1
        2    2
        Name: 0, dtype: int64

        >>> def sqrt(x) -> float:
        ...    return np.sqrt(x)
        >>> s.transform(sqrt)
        0    0.000000
        1    1.000000
        2    1.414214
        Name: 0, dtype: float32

        Even though the resulting instance must have the same length as the
        input, it is possible to provide several input functions:

        >>> def exp(x) -> float:
        ...    return np.exp(x)
        >>> s.transform([sqrt, exp])
               sqrt       exp
        0  0.000000  1.000000
        1  1.000000  2.718282
        2  1.414214  7.389056

        """
        if isinstance(func, list):
            applied = []
            for f in func:
                applied.append(self.apply(f).rename(f.__name__))

            sdf = self._kdf._sdf.select(
                self._internal.index_columns + [c._scol for c in applied])

            internal = self.to_dataframe()._internal.copy(
                sdf=sdf, data_columns=[c.name for c in applied])

            return DataFrame(internal)
        else:
            return self.apply(func, args=args, **kwargs)

    def round(self, decimals=0):
        """
        Round each value in a Series to the given number of decimals.

        Parameters
        ----------
        decimals : int
            Number of decimal places to round to (default: 0).
            If decimals is negative, it specifies the number of
            positions to the left of the decimal point.

        Returns
        -------
        Series object

        See Also
        --------
        DataFrame.round

        Examples
        --------
        >>> df = ks.Series([0.028208, 0.038683, 0.877076], name='x')
        >>> df
        0    0.028208
        1    0.038683
        2    0.877076
        Name: x, dtype: float64

        >>> df.round(2)
        0    0.03
        1    0.04
        2    0.88
        Name: x, dtype: float64
        """
        if not isinstance(decimals, int):
            raise ValueError("decimals must be an integer")
        column_name = self.name
        scol = F.round(F.col(column_name), decimals)
        return Series(self._kdf._internal.copy(scol=scol), anchor=self._kdf).rename(column_name)

    # TODO: add 'interpolation' parameter.
    def quantile(self, q=0.5, accuracy=10000):
        """
        Return value at the given quantile.

        .. note:: Unlike pandas', the quantile in Koalas is an approximated quantile based upon
            approximate percentile computation because computing quantile across a large dataset
            is extremely expensive.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            0 <= q <= 1, the quantile(s) to compute.
        accuracy : int, optional
            Default accuracy of approximation. Larger value means better accuracy.
            The relative error can be deduced by 1.0 / accuracy.

        Returns
        -------
        float or Series
            If the current object is a Series and ``q`` is an array, a Series will be
            returned where the index is ``q`` and the values are the quantiles, otherwise
            a float will be returned.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4, 5])
        >>> s.quantile(.5)
        3

        >>> s.quantile([.25, .5, .75])
        0.25    2
        0.5     3
        0.75    4
        Name: 0, dtype: int64
        """
        if not isinstance(accuracy, int):
            raise ValueError("accuracy must be an integer; however, got [%s]" % type(accuracy))

        if isinstance(q, Iterable):
            q = list(q)

        for v in q if isinstance(q, list) else [q]:
            if not isinstance(v, float):
                raise ValueError(
                    "q must be a float of an array of floats; however, [%s] found." % type(v))
            if v < 0.0 or v > 1.0:
                raise ValueError(
                    "percentiles should all be in the interval [0, 1].")

        if isinstance(q, list):
            quantiles = q
            # TODO: avoid to use dataframe. After this, anchor will be lost.

            # First calculate the percentiles and map it to each `quantiles`
            # by creating each entry as a struct. So, it becomes an array of
            # structs as below:
            #
            # +--------------------------------+
            # | arrays                         |
            # +--------------------------------+
            # |[[0.25, 2], [0.5, 3], [0.75, 4]]|
            # +--------------------------------+
            sdf = self._kdf._sdf
            args = ", ".join(map(str, quantiles))
            percentile_col = F.expr(
                "approx_percentile(`%s`, array(%s), %s)" % (self.name, args, accuracy))
            sdf = sdf.select(percentile_col.alias("percentiles"))

            internal_index_column = "__index_level_0__"
            value_column = "value"
            cols = []
            for i, quantile in enumerate(quantiles):
                cols.append(F.struct(
                    F.lit("%s" % quantile).alias(internal_index_column),
                    F.expr("percentiles[%s]" % i).alias(value_column)))
            sdf = sdf.select(F.array(*cols).alias("arrays"))

            # And then, explode it and manually set the index.
            #
            # +-----------------+-----+
            # |__index_level_0__|value|
            # +-----------------+-----+
            # | 0.25            |    2|
            # |  0.5            |    3|
            # | 0.75            |    4|
            # +-----------------+-----+
            sdf = sdf.select(F.explode(F.col("arrays"))).selectExpr("col.*")

            internal = self._kdf._internal.copy(
                sdf=sdf,
                data_columns=[value_column],
                index_map=[(internal_index_column, None)])

            ser = DataFrame(internal)[value_column].rename(self.name)
            return ser
        else:
            return self._reduce_for_stat_function(
                lambda _: F.expr("approx_percentile(`%s`, %s, %s)" % (self.name, q, accuracy)))

    # TODO: add axis, numeric_only, pct, na_option parameter
    def rank(self, method='average', ascending=True):
        """
        Compute numerical data ranks (1 through n) along axis. Equal values are
        assigned a rank that is the average of the ranks of those values.

        .. note:: the current implementation of rank uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'first', 'dense'}
            * average: average rank of group
            * min: lowest rank in group
            * max: highest rank in group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups
        ascending : boolean, default True
            False for ranks by high (1) to low (N)

        Returns
        -------
        ranks : same type as caller

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 2, 3], 'B': [4, 3, 2, 1]}, columns= ['A', 'B'])
        >>> df
           A  B
        0  1  4
        1  2  3
        2  2  2
        3  3  1

        >>> df.rank().sort_index()
             A    B
        0  1.0  4.0
        1  2.5  3.0
        2  2.5  2.0
        3  4.0  1.0

        If method is set to 'min', it use lowest rank in group.

        >>> df.rank(method='min').sort_index()
             A    B
        0  1.0  4.0
        1  2.0  3.0
        2  2.0  2.0
        3  4.0  1.0

        If method is set to 'max', it use highest rank in group.

        >>> df.rank(method='max').sort_index()
             A    B
        0  1.0  4.0
        1  3.0  3.0
        2  3.0  2.0
        3  4.0  1.0

        If method is set to 'dense', it leaves no gaps in group.

        >>> df.rank(method='dense').sort_index()
             A    B
        0  1.0  4.0
        1  2.0  3.0
        2  2.0  2.0
        3  3.0  1.0
        """
        return _col(self.to_dataframe().rank(method=method, ascending=ascending))

    def describe(self, percentiles: Optional[List[float]] = None) -> 'Series':
        return _col(self.to_dataframe().describe(percentiles))

    describe.__doc__ = DataFrame.describe.__doc__

    def diff(self, periods=1):
        """
        First discrete difference of element.

        Calculates the difference of a Series element compared with another element in the
        DataFrame (default is the element in the same column of the previous row).

        .. note:: the current implementation of diff uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative values.

        Returns
        -------
        diffed : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 2, 3, 4, 5, 6],
        ...                    'b': [1, 1, 2, 3, 5, 8],
        ...                    'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        >>> df
           a  b   c
        0  1  1   1
        1  2  1   4
        2  3  2   9
        3  4  3  16
        4  5  5  25
        5  6  8  36

        >>> df.b.diff()
        0    NaN
        1    0.0
        2    1.0
        3    1.0
        4    2.0
        5    3.0
        Name: b, dtype: float64

        Difference with previous value

        >>> df.c.diff(periods=3)
        0     NaN
        1     NaN
        2     NaN
        3    15.0
        4    21.0
        5    27.0
        Name: c, dtype: float64

        Difference with following value

        >>> df.c.diff(periods=-1)
        0    -3.0
        1    -5.0
        2    -7.0
        3    -9.0
        4   -11.0
        5     NaN
        Name: c, dtype: float64
        """

        if len(self._internal.index_columns) == 0:
            raise ValueError("Index must be set.")

        if not isinstance(periods, int):
            raise ValueError('periods should be an int; however, got [%s]' % type(periods))

        col = self._scol
        window = Window.orderBy(self._internal.index_columns[0]).rowsBetween(-periods, -periods)
        return self._with_new_scol(col - F.lag(col, periods).over(window)).alias(self.name)

    def _cum(self, func, skipna):
        # This is used to cummin, cummax, cumsum, etc.
        if len(self._internal.index_columns) == 0:
            raise ValueError("Index must be set.")

        index_columns = self._internal.index_columns
        window = Window.orderBy(
            index_columns).rowsBetween(Window.unboundedPreceding, Window.currentRow)

        column_name = self.name

        if skipna:
            # There is a behavior difference between pandas and PySpark. In case of cummax,
            #
            # Input:
            #      A    B
            # 0  2.0  1.0
            # 1  5.0  NaN
            # 2  1.0  0.0
            # 3  2.0  4.0
            # 4  4.0  9.0
            #
            # pandas:
            #      A    B
            # 0  2.0  1.0
            # 1  5.0  NaN
            # 2  5.0  1.0
            # 3  5.0  4.0
            # 4  5.0  9.0
            #
            # PySpark:
            #      A    B
            # 0  2.0  1.0
            # 1  5.0  1.0
            # 2  5.0  1.0
            # 3  5.0  4.0
            # 4  5.0  9.0

            scol = F.when(
                # Manually sets nulls given the column defined above.
                F.col(column_name).isNull(), F.lit(None)
            ).otherwise(func(column_name).over(window))
        else:
            # Here, we use two Windows.
            # One for real data.
            # The other one for setting nulls after the first null it meets.
            #
            # There is a behavior difference between pandas and PySpark. In case of cummax,
            #
            # Input:
            #      A    B
            # 0  2.0  1.0
            # 1  5.0  NaN
            # 2  1.0  0.0
            # 3  2.0  4.0
            # 4  4.0  9.0
            #
            # pandas:
            #      A    B
            # 0  2.0  1.0
            # 1  5.0  NaN
            # 2  5.0  NaN
            # 3  5.0  NaN
            # 4  5.0  NaN
            #
            # PySpark:
            #      A    B
            # 0  2.0  1.0
            # 1  5.0  1.0
            # 2  5.0  1.0
            # 3  5.0  4.0
            # 4  5.0  9.0
            scol = F.when(
                # By going through with max, it sets True after the first time it meets null.
                F.max(F.col(column_name).isNull()).over(window),
                # Manually sets nulls given the column defined above.
                F.lit(None)
            ).otherwise(func(column_name).over(window))

        # cumprod uses exp(sum(log(...))) trick.
        if func.__name__ == "cumprod":
            scol = F.exp(scol)

        return Series(self._kdf._internal.copy(scol=scol), anchor=self._kdf).rename(column_name)

    # ----------------------------------------------------------------------
    # Accessor Methods
    # ----------------------------------------------------------------------
    dt = CachedAccessor("dt", DatetimeMethods)
    str = CachedAccessor("str", StringMethods)

    # ----------------------------------------------------------------------

    def _reduce_for_stat_function(self, sfun, numeric_only=None):
        """
        :param sfun: the stats function to be used for aggregation
        :param numeric_only: not used by this implementation, but passed down by stats functions
        """
        from inspect import signature
        num_args = len(signature(sfun).parameters)
        col_sdf = self._scol
        col_type = self.schema[self.name].dataType
        if isinstance(col_type, BooleanType) and sfun.__name__ not in ('min', 'max'):
            # Stat functions cannot be used with boolean values by default
            # Thus, cast to integer (true to 1 and false to 0)
            # Exclude the min and max methods though since those work with booleans
            col_sdf = col_sdf.cast('integer')
        if num_args == 1:
            # Only pass in the column if sfun accepts only one arg
            col_sdf = sfun(col_sdf)
        else:  # must be 2
            assert num_args == 2
            # Pass in both the column and its data type if sfun accepts two args
            col_sdf = sfun(col_sdf, col_type)
        return _unpack_scalar(self._kdf._sdf.select(col_sdf))

    def __len__(self):
        return len(self.to_dataframe())

    def __getitem__(self, key):
        return Series(self._scol.__getitem__(key), anchor=self._kdf, index=self._index_map)

    def __getattr__(self, item: str_type) -> Any:
        if item.startswith("__") or item.startswith("_pandas_") or item.startswith("_spark_"):
            raise AttributeError(item)
        if hasattr(_MissingPandasLikeSeries, item):
            property_or_func = getattr(_MissingPandasLikeSeries, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        return self.getField(item)

    def __str__(self):
        return self._pandas_orig_repr()

    def __repr__(self):
        pser = self.head(max_display_count + 1).to_pandas()
        pser_length = len(pser)
        repr_string = repr(pser.iloc[:max_display_count])
        if pser_length > max_display_count:
            rest, prev_footer = repr_string.rsplit("\n", 1)
            match = REPR_PATTERN.search(prev_footer)
            if match is not None:
                length = match.group("length")
                footer = ("\n{prev_footer}\nShowing only the first {length}"
                          .format(length=length, prev_footer=prev_footer))
                return rest + footer
        return repr_string

    def __dir__(self):
        if not isinstance(self.schema, StructType):
            fields = []
        else:
            fields = [f for f in self.schema.fieldNames() if ' ' not in f]
        return super(Series, self).__dir__() + fields

    def _pandas_orig_repr(self):
        # TODO: figure out how to reuse the original one.
        return 'Column<%s>' % self._scol._jc.toString().encode('utf8')


def _unpack_scalar(sdf):
    """
    Takes a dataframe that is supposed to contain a single row with a single scalar value,
    and returns this value.
    """
    l = sdf.head(2)
    assert len(l) == 1, (sdf, l)
    row = l[0]
    l2 = list(row.asDict().values())
    assert len(l2) == 1, (row, l2)
    return l2[0]


def _col(df):
    assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
    return df[df.columns[0]]
