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
from collections import Iterable, OrderedDict
from functools import partial, wraps, reduce
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from pandas.core.accessor import CachedAccessor
from pandas.io.formats.printing import pprint_thing
from pandas.api.types import is_list_like

from databricks.koalas.typedef import infer_return_type, SeriesType, ScalarType
from pyspark import sql as spark
from pyspark.sql import functions as F, Column
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    FloatType,
    StringType,
    StructType,
    LongType,
    IntegerType,
)
from pyspark.sql.window import Window

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.config import get_option, option_context
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.exceptions import SparkPandasIndexingError
from databricks.koalas.frame import DataFrame
from databricks.koalas.generic import _Frame
from databricks.koalas.internal import (
    _InternalFrame,
    NATURAL_ORDER_COLUMN_NAME,
    SPARK_DEFAULT_INDEX_NAME,
)
from databricks.koalas.missing.series import _MissingPandasLikeSeries
from databricks.koalas.plot import KoalasSeriesPlotMethods
from databricks.koalas.ml import corr
from databricks.koalas.utils import (
    validate_arguments_and_invoke_function,
    scol_for,
    combine_frames,
    name_like_string,
    validate_axis,
    validate_bool_kwarg,
    verify_temp_column_name,
)
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
>>> df = ks.DataFrame({'a': [2, 2, 4, np.nan],
...                    'b': [2, np.nan, 2, np.nan]},
...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])
>>> df
     a    b
a  2.0  2.0
b  2.0  NaN
c  4.0  2.0
d  NaN  NaN

>>> df.a.add(df.b)
a    4.0
b    NaN
c    6.0
d    NaN
Name: a, dtype: float64

>>> df.a.radd(df.b)
a    4.0
b    NaN
c    6.0
d    NaN
Name: a, dtype: float64
"""

_sub_example_SERIES = """
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

>>> df.a.subtract(df.b)
a    0.0
b    NaN
c    2.0
d    NaN
Name: a, dtype: float64

>>> df.a.rsub(df.b)
a    0.0
b    NaN
c   -2.0
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

>>> df.a.rmul(df.b)
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

>>> df.a.rdiv(df.b)
a    1.0
b    NaN
c    0.5
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

>>> df.a.rpow(df.b)
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

>>> df.a.rmod(df.b)
a    0.0
b    NaN
c    2.0
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

>>> df.a.rfloordiv(df.b)
a    1.0
b    NaN
c    0.0
d    NaN
Name: a, dtype: float64
"""

T = TypeVar("T")

# Needed to disambiguate Series.str and str type
str_type = str


class Series(_Frame, IndexOpsMixin, Generic[T]):
    """
    Koalas Series that corresponds to Pandas Series logically. This holds Spark Column
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

    def __init__(
        self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False, anchor=None
    ):
        if isinstance(data, _InternalFrame):
            assert dtype is None
            assert name is None
            assert not copy
            assert not fastpath
            IndexOpsMixin.__init__(self, data, anchor)
        else:
            assert anchor is None
            if isinstance(data, pd.Series):
                assert index is None
                assert dtype is None
                assert name is None
                assert not copy
                assert not fastpath
                s = data
            else:
                s = pd.Series(
                    data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath
                )
            kdf = DataFrame(s)
            IndexOpsMixin.__init__(
                self, kdf._internal.copy(spark_column=kdf._internal.data_spark_columns[0]), kdf
            )

    def _with_new_scol(self, scol: spark.Column) -> "Series":
        """
        Copy Koalas Series with the new Spark Column.

        :param scol: the new Spark Column
        :return: the copied Series
        """
        return Series(self._internal.copy(spark_column=scol), anchor=self._kdf)  # type: ignore

    @property
    def dtypes(self):
        """Return the dtype object of the underlying data.

        >>> s = ks.Series(list('abc'))
        >>> s.dtype == s.dtypes
        True
        """
        return self.dtype

    @property
    def axes(self):
        """
        Return a list of the row axis labels.

        Examples
        --------

        >>> kser = ks.Series([1, 2, 3])
        >>> kser.axes
        [Int64Index([0, 1, 2], dtype='int64')]
        """
        return [self.index]

    @property
    def spark_type(self):
        """ Returns the data type as defined by Spark, as a Spark DataType object."""
        return self._internal.spark_type_for(self._internal.column_labels[0])

    plot = CachedAccessor("plot", KoalasSeriesPlotMethods)

    # Arithmetic Operators
    def add(self, other):
        return (self + other).rename(self.name)

    add.__doc__ = _flex_doc_SERIES.format(
        desc="Addition",
        op_name="+",
        equiv="series + other",
        reverse="radd",
        series_examples=_add_example_SERIES,
    )

    def radd(self, other):
        return (other + self).rename(self.name)

    radd.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Addition",
        op_name="+",
        equiv="other + series",
        reverse="add",
        series_examples=_add_example_SERIES,
    )

    def div(self, other):
        return (self / other).rename(self.name)

    div.__doc__ = _flex_doc_SERIES.format(
        desc="Floating division",
        op_name="/",
        equiv="series / other",
        reverse="rdiv",
        series_examples=_div_example_SERIES,
    )

    divide = div

    def rdiv(self, other):
        return (other / self).rename(self.name)

    rdiv.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Floating division",
        op_name="/",
        equiv="other / series",
        reverse="div",
        series_examples=_div_example_SERIES,
    )

    def truediv(self, other):
        return (self / other).rename(self.name)

    truediv.__doc__ = _flex_doc_SERIES.format(
        desc="Floating division",
        op_name="/",
        equiv="series / other",
        reverse="rtruediv",
        series_examples=_div_example_SERIES,
    )

    def rtruediv(self, other):
        return (other / self).rename(self.name)

    rtruediv.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Floating division",
        op_name="/",
        equiv="other / series",
        reverse="truediv",
        series_examples=_div_example_SERIES,
    )

    def mul(self, other):
        return (self * other).rename(self.name)

    mul.__doc__ = _flex_doc_SERIES.format(
        desc="Multiplication",
        op_name="*",
        equiv="series * other",
        reverse="rmul",
        series_examples=_mul_example_SERIES,
    )

    multiply = mul

    def rmul(self, other):
        return (other * self).rename(self.name)

    rmul.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Multiplication",
        op_name="*",
        equiv="other * series",
        reverse="mul",
        series_examples=_mul_example_SERIES,
    )

    def sub(self, other):
        return (self - other).rename(self.name)

    sub.__doc__ = _flex_doc_SERIES.format(
        desc="Subtraction",
        op_name="-",
        equiv="series - other",
        reverse="rsub",
        series_examples=_sub_example_SERIES,
    )

    subtract = sub

    def rsub(self, other):
        return (other - self).rename(self.name)

    rsub.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Subtraction",
        op_name="-",
        equiv="other - series",
        reverse="sub",
        series_examples=_sub_example_SERIES,
    )

    def mod(self, other):
        return (self % other).rename(self.name)

    mod.__doc__ = _flex_doc_SERIES.format(
        desc="Modulo",
        op_name="%",
        equiv="series % other",
        reverse="rmod",
        series_examples=_mod_example_SERIES,
    )

    def rmod(self, other):
        return (other % self).rename(self.name)

    rmod.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Modulo",
        op_name="%",
        equiv="other % series",
        reverse="mod",
        series_examples=_mod_example_SERIES,
    )

    def pow(self, other):
        return (self ** other).rename(self.name)

    pow.__doc__ = _flex_doc_SERIES.format(
        desc="Exponential power of series",
        op_name="**",
        equiv="series ** other",
        reverse="rpow",
        series_examples=_pow_example_SERIES,
    )

    def rpow(self, other):
        return (other ** self).rename(self.name)

    rpow.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Exponential power",
        op_name="**",
        equiv="other ** series",
        reverse="pow",
        series_examples=_pow_example_SERIES,
    )

    def floordiv(self, other):
        return (self // other).rename(self.name)

    floordiv.__doc__ = _flex_doc_SERIES.format(
        desc="Integer division",
        op_name="//",
        equiv="series // other",
        reverse="rfloordiv",
        series_examples=_floordiv_example_SERIES,
    )

    def rfloordiv(self, other):
        return (other // self).rename(self.name)

    rfloordiv.__doc__ = _flex_doc_SERIES.format(
        desc="Reverse Integer division",
        op_name="//",
        equiv="other // series",
        reverse="floordiv",
        series_examples=_floordiv_example_SERIES,
    )

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
        Name: a, dtype: bool

        >>> df.b.eq(1)
        a     True
        b    False
        c     True
        d    False
        Name: b, dtype: bool
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
        Name: a, dtype: bool

        >>> df.b.gt(1)
        a    False
        b    False
        c    False
        d    False
        Name: b, dtype: bool
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
        Name: a, dtype: bool

        >>> df.b.ge(2)
        a    False
        b    False
        c    False
        d    False
        Name: b, dtype: bool
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
        Name: a, dtype: bool

        >>> df.b.lt(2)
        a     True
        b    False
        c     True
        d    False
        Name: b, dtype: bool
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
        Name: a, dtype: bool

        >>> df.b.le(2)
        a     True
        b    False
        c     True
        d    False
        Name: b, dtype: bool
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
        Name: a, dtype: bool

        >>> df.b.ne(1)
        a    False
        b     True
        c    False
        d     True
        Name: b, dtype: bool
        """
        return (self != other).rename(self.name)

    def between(self, left, right, inclusive=True):
        """
        Return boolean Series equivalent to left <= series <= right.
        This function returns a boolean vector containing `True` wherever the
        corresponding Series element is between the boundary values `left` and
        `right`. NA values are treated as `False`.

        Parameters
        ----------
        left : scalar or list-like
            Left boundary.
        right : scalar or list-like
            Right boundary.
        inclusive : bool, default True
            Include boundaries.

        Returns
        -------
        Series
            Series representing whether each element is between left and
            right (inclusive).

        See Also
        --------
        Series.gt : Greater than of series and other.
        Series.lt : Less than of series and other.

        Notes
        -----
        This function is equivalent to ``(left <= ser) & (ser <= right)``

        Examples
        --------
        >>> s = ks.Series([2, 0, 4, 8, np.nan])

        Boundary values are included by default:

        >>> s.between(1, 4)
        0     True
        1    False
        2     True
        3    False
        4    False
        Name: 0, dtype: bool

        With `inclusive` set to ``False`` boundary values are excluded:

        >>> s.between(1, 4, inclusive=False)
        0     True
        1    False
        2    False
        3    False
        4    False
        Name: 0, dtype: bool

        `left` and `right` can be any scalar value:

        >>> s = ks.Series(['Alice', 'Bob', 'Carol', 'Eve'])
        >>> s.between('Anna', 'Daniel')
        0    False
        1     True
        2     True
        3    False
        Name: 0, dtype: bool
        """
        if inclusive:
            lmask = self >= left
            rmask = self <= right
        else:
            lmask = self > left
            rmask = self < right

        return lmask & rmask

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
                    current = F.when(self.spark_column == F.lit(to_replace), value)
                    is_start = False
                else:
                    current = current.when(self.spark_column == F.lit(to_replace), value)

            if hasattr(arg, "__missing__"):
                tmp_val = arg[np._NoValue]
                del arg[np._NoValue]  # Remove in case it's set in defaultdict.
                current = current.otherwise(F.lit(tmp_val))
            else:
                current = current.otherwise(F.lit(None).cast(self.spark_type))
            return self._with_new_scol(current).rename(self.name)
        else:
            return self.apply(arg)

    def astype(self, dtype) -> "Series":
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
        if isinstance(spark_type, BooleanType):
            if isinstance(self.spark_type, StringType):
                scol = F.when(self.spark_column.isNull(), F.lit(False)).otherwise(
                    F.length(self.spark_column) > 0
                )
            elif isinstance(self.spark_type, (FloatType, DoubleType)):
                scol = F.when(
                    self.spark_column.isNull() | F.isnan(self.spark_column), F.lit(True)
                ).otherwise(self.spark_column.cast(spark_type))
            else:
                scol = F.when(self.spark_column.isNull(), F.lit(False)).otherwise(
                    self.spark_column.cast(spark_type)
                )
        else:
            scol = self.spark_column.cast(spark_type)
        return self._with_new_scol(scol)

    def getField(self, name):
        if not isinstance(self.spark_type, StructType):
            raise AttributeError("Not a struct: {}".format(self.spark_type))
        else:
            fnames = self.spark_type.fieldNames()
            if name not in fnames:
                raise AttributeError(
                    "Field {} not found, possible values are {}".format(name, ", ".join(fnames))
                )
            return self._with_new_scol(self.spark_column.getField(name))

    def alias(self, name):
        """An alias for :meth:`Series.rename`."""
        return self.rename(name)

    @property
    def shape(self):
        """Return a tuple of the shape of the underlying data."""
        return (len(self),)

    @property
    def name(self) -> Union[str, Tuple[str, ...]]:
        """Return name of the Series."""
        name = self._internal.column_labels[0]  # type: ignore
        if name is not None and len(name) == 1:
            return name[0]
        else:
            return name

    @name.setter
    def name(self, name: Union[str, Tuple[str, ...]]):
        self.rename(name, inplace=True)

    # TODO: Functionality and documentation should be matched. Currently, changing index labels
    # taking dictionary and function to change index are not supported.
    def rename(self, index: Union[str, Tuple[str, ...]] = None, **kwargs):
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
            scol = self.spark_column
        else:
            scol = self.spark_column.alias(name_like_string(index))
        internal = self._internal.copy(  # type: ignore
            spark_column=scol,
            column_labels=[index if index is None or isinstance(index, tuple) else (index,)],
        )
        if kwargs.get("inplace", False):
            self._internal = internal
            return self
        else:
            return Series(internal, anchor=self._kdf)

    @property
    def index(self):
        """The index (axis labels) Column of the Series.

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
        scol = self.spark_column

        # Here we check:
        #   1. the distinct count without nulls and count without nulls for non-null values
        #   2. count null values and see if null is a distinct value.
        #
        # This workaround is in order to calculate the distinct count including nulls in
        # single pass. Note that COUNT(DISTINCT expr) in Spark is designed to ignore nulls.
        return self._internal._sdf.select(
            (F.count(scol) == F.countDistinct(scol))
            & (F.count(F.when(scol.isNull(), 1).otherwise(None)) <= 1)
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
        inplace = validate_bool_kwarg(inplace, "inplace")
        if inplace and not drop:
            raise TypeError("Cannot reset_index inplace on a Series to create a DataFrame")

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

    def to_frame(self, name: Union[str, Tuple[str, ...]] = None) -> spark.DataFrame:
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
        if name is not None:
            renamed = self.rename(name)
        else:
            renamed = self
        sdf = renamed._internal.to_internal_spark_frame
        column_labels = None  # type: Optional[List[Tuple[str, ...]]]
        if renamed._internal.column_labels[0] is None:
            column_labels = [("0",)]
            column_label_names = None
        else:
            column_labels = renamed._internal.column_labels
            column_label_names = renamed._internal.column_label_names
        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=renamed._internal.index_map,
            column_labels=column_labels,
            data_spark_columns=[scol_for(sdf, sdf.columns[-1])],
            column_label_names=column_label_names,
        )
        return DataFrame(internal)

    to_dataframe = to_frame

    def to_string(
        self,
        buf=None,
        na_rep="NaN",
        float_format=None,
        header=True,
        index=True,
        length=False,
        dtype=False,
        name=False,
        max_rows=None,
    ):
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
            kseries._to_internal_pandas(), self.to_string, pd.Series.to_string, args
        )

    def to_clipboard(self, excel=True, sep=None, **kwargs):
        # Docstring defined below by reusing DataFrame.to_clipboard's.
        args = locals()
        kseries = self

        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), self.to_clipboard, pd.Series.to_clipboard, args
        )

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
            kseries._to_internal_pandas(), self.to_dict, pd.Series.to_dict, args
        )

    def to_latex(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        bold_rows=False,
        column_format=None,
        longtable=None,
        escape=None,
        encoding=None,
        decimal=".",
        multicolumn=None,
        multicolumn_format=None,
        multirow=None,
    ):

        args = locals()
        kseries = self
        return validate_arguments_and_invoke_function(
            kseries._to_internal_pandas(), self.to_latex, pd.Series.to_latex, args
        )

    to_latex.__doc__ = DataFrame.to_latex.__doc__

    def to_pandas(self):
        """
        Return a pandas Series.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory.

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
        return _col(self._internal.to_pandas_frame.copy())

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
        return self._to_internal_pandas().to_list()

    tolist = to_list

    def drop_duplicates(self, keep="first", inplace=False):
        """
        Return Series with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            Method to handle dropping duplicates:
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.
        inplace : bool, default ``False``
            If ``True``, performs operation inplace and returns None.

        Returns
        -------
        Series
            Series with duplicates dropped.

        Examples
        --------
        Generate a Series with duplicated entries.

        >>> s = ks.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'],
        ...               name='animal')
        >>> s.sort_index()
        0      lama
        1       cow
        2      lama
        3    beetle
        4      lama
        5     hippo
        Name: animal, dtype: object

        With the 'keep' parameter, the selection behaviour of duplicated values
        can be changed. The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> s.drop_duplicates().sort_index()
        0      lama
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object

        The value 'last' for parameter 'keep' keeps the last occurrence for
        each set of duplicated entries.

        >>> s.drop_duplicates(keep='last').sort_index()
        1       cow
        3    beetle
        4      lama
        5     hippo
        Name: animal, dtype: object

        The value ``False`` for parameter 'keep' discards all sets of
        duplicated entries. Setting the value of 'inplace' to ``True`` performs
        the operation inplace and returns ``None``.

        >>> s.drop_duplicates(keep=False, inplace=True)
        >>> s.sort_index()
        1       cow
        3    beetle
        5     hippo
        Name: animal, dtype: object
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        kseries = _col(self.to_frame().drop_duplicates(keep=keep))

        if inplace:
            self._internal = kseries._internal
            self._kdf = kseries._kdf
        else:
            return kseries

    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None):
        """Fill NA/NaN values.

        .. note:: the current implementation of 'method' parameter in fillna uses Spark's Window
            without specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        value : scalar, dict, Series
            Value to use to fill holes. alternately a dict/Series of values
            specifying which value to use for each column.
            DataFrame is not supported.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series pad / ffill: propagate last valid
            observation forward to next valid backfill / bfill:
            use NEXT valid observation to fill gap
        axis : {0 or `index`}
            1 and `columns` are not supported.
        inplace : boolean, default False
            Fill in place (do not create a new object)
        limit : int, default None
            If method is specified, this is the maximum number of consecutive NaN values to
            forward/backward fill. In other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. If method is not specified,
            this is the maximum number of entries along the entire axis where NaNs will be filled.
            Must be greater than 0 if not None

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

        We can also propagate non-null values forward or backward.

        >>> s.fillna(method='ffill')
        0    NaN
        1    2.0
        2    3.0
        3    4.0
        4    4.0
        5    6.0
        Name: x, dtype: float64

        >>> s = ks.Series([np.nan, 'a', 'b', 'c', np.nan], name='x')
        >>> s.fillna(method='ffill')
        0    None
        1       a
        2       b
        3       c
        4       c
        Name: x, dtype: object
        """
        return self._fillna(value, method, axis, inplace, limit)

    def _fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, part_cols=()):
        axis = validate_axis(axis)
        inplace = validate_bool_kwarg(inplace, "inplace")
        if axis != 0:
            raise NotImplementedError("fillna currently only works for axis=0 or axis='index'")
        if (value is None) and (method is None):
            raise ValueError("Must specify a fillna 'value' or 'method' parameter.")
        if (method is not None) and (method not in ["ffill", "pad", "backfill", "bfill"]):
            raise ValueError("Expecting 'pad', 'ffill', 'backfill' or 'bfill'.")
        if self.isnull().sum() == 0:
            if inplace:
                self._internal = self._internal.copy()
                self._kdf = self._kdf.copy()
            else:
                return self

        column_name = self.name
        scol = self.spark_column

        if value is not None:
            if not isinstance(value, (float, int, str, bool)):
                raise TypeError("Unsupported type %s" % type(value))
            if limit is not None:
                raise ValueError("limit parameter for value is not support now")
            scol = F.when(scol.isNull(), value).otherwise(scol)
        else:
            if method in ["ffill", "pad"]:
                func = F.last
                end = Window.currentRow - 1
                if limit is not None:
                    begin = Window.currentRow - limit
                else:
                    begin = Window.unboundedPreceding
            elif method in ["bfill", "backfill"]:
                func = F.first
                begin = Window.currentRow + 1
                if limit is not None:
                    end = Window.currentRow + limit
                else:
                    end = Window.unboundedFollowing

            window = (
                Window.partitionBy(*part_cols)
                .orderBy(NATURAL_ORDER_COLUMN_NAME)
                .rowsBetween(begin, end)
            )
            scol = F.when(scol.isNull(), func(scol, True).over(window)).otherwise(scol)
        kseries = self._with_new_scol(scol).rename(column_name)
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
        inplace = validate_bool_kwarg(inplace, "inplace")
        # TODO: last two examples from Pandas produce different results.
        kseries = _col(self.to_dataframe().dropna(axis=axis, inplace=False))
        if inplace:
            self._internal = kseries._internal
            self._kdf = kseries._kdf
        else:
            return kseries

    def clip(self, lower: Union[float, int] = None, upper: Union[float, int] = None) -> "Series":
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

    def drop(
        self,
        labels=None,
        index: Union[str, Tuple[str, ...], List[str], List[Tuple[str, ...]]] = None,
        level=None,
    ):
        """
        Return Series with specified index labels removed.

        Remove elements of a Series based on specifying the index labels.
        When using a multi-index, labels on different levels can be removed by specifying the level.

        Parameters
        ----------
        labels : single label or list-like
            Index labels to drop.
        index : None
            Redundant for application on Series, but index can be used instead of labels.
        level : int or level name, optional
            For MultiIndex, level for which the labels will be removed.

        Returns
        -------
        Series
            Series with specified index labels removed.

        See Also
        --------
        Series.dropna

        Examples
        --------
        >>> s = ks.Series(data=np.arange(3), index=['A', 'B', 'C'])
        >>> s
        A    0
        B    1
        C    2
        Name: 0, dtype: int64

        Drop single label A

        >>> s.drop('A')
        B    1
        C    2
        Name: 0, dtype: int64

        Drop labels B and C

        >>> s.drop(labels=['B', 'C'])
        A    0
        Name: 0, dtype: int64

        With 'index' rather than 'labels' returns exactly same result.

        >>> s.drop(index='A')
        B    1
        C    2
        Name: 0, dtype: int64

        >>> s.drop(index=['B', 'C'])
        A    0
        Name: 0, dtype: int64

        Also support for MultiIndex

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...               index=midx)
        >>> s
        lama    speed      45.0
                weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        Name: 0, dtype: float64

        >>> s.drop(labels='weight', level=1)
        lama    speed      45.0
                length      1.2
        cow     speed      30.0
                length      1.5
        falcon  speed     320.0
                length      0.3
        Name: 0, dtype: float64

        >>> s.drop(('lama', 'weight'))
        lama    speed      45.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        Name: 0, dtype: float64

        >>> s.drop([('lama', 'speed'), ('falcon', 'weight')])
        lama    weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                length      0.3
        Name: 0, dtype: float64
        """
        level_param = level
        if labels is not None:
            if index is not None:
                raise ValueError("Cannot specify both 'labels' and 'index'")
            return self.drop(index=labels, level=level)
        if index is not None:
            if not isinstance(index, (str, tuple, list)):
                raise ValueError("'index' type should be one of str, list, tuple")
            if level is None:
                level = 0
            if level >= len(self._internal.index_spark_columns):
                raise ValueError("'level' should be less than the number of indexes")

            if isinstance(index, str):
                index = [(index,)]  # type: ignore
            elif isinstance(index, tuple):
                index = [index]
            else:
                if not (
                    all((isinstance(idxes, str) for idxes in index))
                    or all((isinstance(idxes, tuple) for idxes in index))
                ):
                    raise ValueError(
                        "If the given index is a list, it "
                        "should only contains names as strings, "
                        "or a list of tuples that contain "
                        "index names as strings"
                    )
                new_index = []
                for idxes in index:
                    if isinstance(idxes, tuple):
                        new_index.append(idxes)
                    else:
                        new_index.append((idxes,))
                index = new_index

            drop_index_scols = []
            for idxes in index:
                try:
                    index_scols = [
                        self._internal.index_spark_columns[lvl] == idx
                        for lvl, idx in enumerate(idxes, level)
                    ]
                except IndexError:
                    if level_param is None:
                        raise KeyError(
                            "Key length ({}) exceeds index depth ({})".format(
                                len(self._internal.index_spark_columns), len(idxes)
                            )
                        )
                    else:
                        return self
                drop_index_scols.append(reduce(lambda x, y: x & y, index_scols))

            cond = ~reduce(lambda x, y: x | y, drop_index_scols)
            return _col(DataFrame(self._internal.with_filter(cond)))
        else:
            raise ValueError("Need to specify at least one of 'labels' or 'index'")

    def head(self, n: int = 5) -> "Series":
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

        See Also
        --------
        Index.unique
        groupby.SeriesGroupBy.unique

        Examples
        --------
        >>> kser = ks.Series([2, 1, 3, 3], name='A')
        >>> kser.unique().sort_values()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...  1
        ...  2
        ...  3
        Name: A, dtype: int64

        >>> ks.Series([pd.Timestamp('2016-01-01') for _ in range(3)]).unique()
        0   2016-01-01
        Name: 0, dtype: datetime64[ns]

        >>> kser.name = ('x', 'a')
        >>> kser.unique().sort_values()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...  1
        ...  2
        ...  3
        Name: (x, a), dtype: int64
        """
        sdf = self._internal.spark_frame.select(self.spark_column).distinct()
        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=None,
            column_labels=[self._internal.column_labels[0]],
            data_spark_columns=[scol_for(sdf, self._internal.data_spark_column_names[0])],
            column_label_names=self._internal.column_label_names,
        )
        return _col(DataFrame(internal))

    def sort_values(
        self, ascending: bool = True, inplace: bool = False, na_position: str = "last"
    ) -> Union["Series", None]:
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
        inplace = validate_bool_kwarg(inplace, "inplace")
        kseries = _col(
            self.to_dataframe().sort_values(
                by=self.name, ascending=ascending, na_position=na_position
            )
        )
        if inplace:
            self._internal = kseries._internal
            self._kdf = kseries._kdf
            return None
        else:
            return kseries

    def sort_index(
        self,
        axis: int = 0,
        level: Optional[Union[int, List[int]]] = None,
        ascending: bool = True,
        inplace: bool = False,
        kind: str = None,
        na_position: str = "last",
    ) -> Optional["Series"]:
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

        >>> df = ks.Series(range(4), index=[['b', 'b', 'a', 'a'], [1, 0, 1, 0]], name='0')

        >>> df.sort_index()
        a  0    3
           1    2
        b  0    1
           1    0
        Name: 0, dtype: int64

        >>> df.sort_index(level=1)  # doctest: +SKIP
        a  0    3
        b  0    1
        a  1    2
        b  1    0
        Name: 0, dtype: int64

        >>> df.sort_index(level=[1, 0])
        a  0    3
        b  0    1
        a  1    2
        b  1    0
        Name: 0, dtype: int64
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        kseries = _col(
            self.to_dataframe().sort_index(
                axis=axis, level=level, ascending=ascending, kind=kind, na_position=na_position
            )
        )
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
        sdf = internal.spark_frame
        sdf = sdf.select(
            [
                F.concat(F.lit(prefix), scol_for(sdf, index_column)).alias(index_column)
                for index_column in internal.index_spark_column_names
            ]
            + internal.data_spark_columns
        )
        kdf._internal = internal.with_new_sdf(sdf)
        return _col(kdf)

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
        sdf = internal.spark_frame
        sdf = sdf.select(
            [
                F.concat(scol_for(sdf, index_column), F.lit(suffix)).alias(index_column)
                for index_column in internal.index_spark_column_names
            ]
            + internal.data_spark_columns
        )
        kdf._internal = internal.with_new_sdf(sdf)
        return _col(kdf)

    def corr(self, other, method="pearson"):
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
        columns = ["__corr_arg1__", "__corr_arg2__"]
        kdf = self._kdf.assign(__corr_arg1__=self, __corr_arg2__=other)[columns]
        kdf.columns = columns
        c = corr(kdf, method=method)
        return c.loc[tuple(columns)]

    def nsmallest(self, n: int = 5) -> "Series":
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
        return _col(self.to_frame().nsmallest(n=n, columns=self.name))

    def nlargest(self, n: int = 5) -> "Series":
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
        return _col(self.to_frame().nlargest(n=n, columns=self.name))

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
        return self._reduce_for_stat_function(_Frame._count_expr, name="count")

    def append(
        self, to_append: "Series", ignore_index: bool = False, verify_integrity: bool = False
    ) -> "Series":
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
        return _col(
            self.to_dataframe().append(to_append.to_dataframe(), ignore_index, verify_integrity)
        )

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[int] = None,
    ) -> "Series":
        return _col(
            self.to_dataframe().sample(n=n, frac=frac, replace=replace, random_state=random_state)
        )

    sample.__doc__ = DataFrame.sample.__doc__

    def hist(self, bins=10, **kwds):
        return self.plot.hist(bins, **kwds)

    hist.__doc__ = KoalasSeriesPlotMethods.hist.__doc__

    def apply(self, func, args=(), **kwds):
        """
        Invoke function on values of Series.

        Can be a Python function that only works on the Series.

        .. note:: this API executes the function once to infer the type which is
             potentially expensive, for instance, when the dataset is created after
             aggregations or sorting.

             To avoid this, specify return type in ``func``, for instance, as below:

             >>> def square(x) -> np.int32:
             ...     return x ** 2

             Koalas uses return type hint and does not try to infer the type.

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

        See Also
        --------
        Series.aggregate : Only perform aggregating type operations.
        Series.transform : Only perform transforming type operations.
        DataFrame.apply : The equivalent function for DataFrame.

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


        You can omit the type hint and let Koalas infer its type.

        >>> s.apply(np.log)
        London      2.995732
        New York    3.044522
        Helsinki    2.484907
        Name: 0, dtype: float64

        """
        assert callable(func), "the first argument should be a callable function."
        try:
            spec = inspect.getfullargspec(func)
            return_sig = spec.annotations.get("return", None)
            should_infer_schema = return_sig is None
        except TypeError:
            # Falls back to schema inference if it fails to get signature.
            should_infer_schema = True

        apply_each = wraps(func)(lambda s: s.apply(func, args=args, **kwds))

        if should_infer_schema:
            # TODO: In this case, it avoids the shortcut for now (but only infers schema)
            #  because it returns a series from a different DataFrame and it has a different
            #  anchor. We should fix this to allow the shortcut or only allow to infer
            #  schema.
            limit = get_option("compute.shortcut_limit")
            pser = self.head(limit)._to_internal_pandas()
            transformed = pser.apply(func, *args, **kwds)
            kser = Series(transformed)
            return self._transform_batch(apply_each, kser.spark_type)
        else:
            sig_return = infer_return_type(func)
            if not isinstance(sig_return, ScalarType):
                raise ValueError(
                    "Expected the return type of this function to be of scalar type, "
                    "but found type {}".format(sig_return)
                )
            return_schema = sig_return.tpe
            return self._transform_batch(apply_each, return_schema)

    # TODO: not all arguments are implemented comparing to Pandas' for now.
    def aggregate(self, func: Union[str, List[str]]):
        """Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : str or a list of str
            function name(s) as string apply to series.

        Returns
        -------
        scalar, Series
            The return can be:
            - scalar : when Series.agg is called with single function
            - Series : when Series.agg is called with several functions

        Notes
        -----
        `agg` is an alias for `aggregate`. Use the alias.

        See Also
        --------
        Series.apply : Invoke function on a Series.
        Series.transform : Only perform transforming type operations.
        Series.groupby : Perform operations over groups.
        DataFrame.aggregate : The equivalent function for DataFrame.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4])
        >>> s.agg('min')
        1

        >>> s.agg(['min', 'max'])
        max    4
        min    1
        Name: 0, dtype: int64
        """
        if isinstance(func, list):
            return self.to_frame().agg(func)[self.name]
        elif isinstance(func, str):
            return getattr(self, func)()
        else:
            raise ValueError("func must be a string or list of strings")

    agg = aggregate

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
        return Series(self._internal.copy(), anchor=self._kdf)

    T = property(transpose)

    def transform(self, func, *args, **kwargs):
        """
        Call ``func`` producing the same type as `self` with transformed values
        and that has the same axis length as input.

        .. note:: this API executes the function once to infer the type which is
             potentially expensive, for instance, when the dataset is created after
             aggregations or sorting.

             To avoid this, specify return type in ``func``, for instance, as below:

             >>> def square(x) -> np.int32:
             ...     return x ** 2

             Koalas uses return type hint and does not try to infer the type.

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
        Series.aggregate : Only perform aggregating type operations.
        Series.apply : Invoke function on Series.
        DataFrame.transform : The equivalent function for DataFrame.

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

        You can omit the type hint and let Koalas infer its type.

        >>> s.transform([np.sqrt, np.exp])
               sqrt       exp
        0  0.000000  1.000000
        1  1.000000  2.718282
        2  1.414214  7.389056
        """
        if isinstance(func, list):
            applied = []
            for f in func:
                applied.append(self.apply(f, args=args, **kwargs).rename(f.__name__))

            internal = self._internal.with_new_columns(applied)
            return DataFrame(internal)
        else:
            return self.apply(func, args=args, **kwargs)

    def transform_batch(self, func) -> "ks.Series":
        """
        Transform the data with the function that takes pandas Series and outputs pandas Series.
        The pandas Series given to the function is of a batch used internally.

        .. note:: the `func` is unable to access to the whole input series. Koalas internally
            splits the input series into multiple batches and calls `func` with each batch multiple
            times. Therefore, operations such as global aggregations are impossible. See the example
            below.

            >>> # This case does not return the length of whole frame but of the batch internally
            ... # used.
            ... def length(pser) -> ks.Series[int]:
            ...     return pd.Series([len(pser)] * len(pser))
            ...
            >>> df = ks.DataFrame({'A': range(1000)})
            >>> df.A.transform_batch(length)  # doctest: +SKIP
                c0
            0   83
            1   83
            2   83
            ...

        .. note:: this API executes the function once to infer the type which is
            potentially expensive, for instance, when the dataset is created after
            aggregations or sorting.

            To avoid this, specify return type in ``func``, for instance, as below:

            >>> def plus_one(x) -> ks.Series[int]:
            ...    return x + 1

        Parameters
        ----------
        func : function
            Function to apply to each pandas frame.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.apply_batch : Similar but it takes pandas DataFrame as its internal batch.

        Examples
        --------
        >>> df = ks.DataFrame([(1, 2), (3, 4), (5, 6)], columns=['A', 'B'])
        >>> df
           A  B
        0  1  2
        1  3  4
        2  5  6

        >>> def plus_one_func(pser) -> ks.Series[np.int64]:
        ...     return pser + 1
        >>> df.A.transform_batch(plus_one_func)
        0    2
        1    4
        2    6
        Name: A, dtype: int64

        You can also omit the type hints so Koalas infers the return schema as below:

        >>> df.A.transform_batch(lambda pser: pser + 1)
        0    2
        1    4
        2    6
        Name: A, dtype: int64
        """

        assert callable(func), "the first argument should be a callable function."

        return_sig = None
        try:
            spec = inspect.getfullargspec(func)
            return_sig = spec.annotations.get("return", None)
        except TypeError:
            # Falls back to schema inference if it fails to get signature.
            pass

        return_schema = None
        if return_sig is not None:
            # Extract the signature arguments from this function.
            sig_return = infer_return_type(func)
            if not isinstance(sig_return, SeriesType):
                raise ValueError(
                    "Expected the return type of this function to be of type column,"
                    " but found type {}".format(sig_return)
                )
            return_schema = sig_return.tpe

        return self._transform_batch(func, return_schema)

    def _transform_batch(self, func, return_schema):
        if isinstance(func, np.ufunc):
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)

        if return_schema is None:
            # TODO: In this case, it avoids the shortcut for now (but only infers schema)
            #  because it returns a series from a different DataFrame and it has a different
            #  anchor. We should fix this to allow the shortcut or only allow to infer
            #  schema.
            limit = get_option("compute.shortcut_limit")
            pser = self.head(limit)._to_internal_pandas()
            transformed = pser.transform(func)
            kser = Series(transformed)
            spark_return_type = kser.spark_type
        else:
            spark_return_type = return_schema

        pudf = pandas_udf(func, returnType=spark_return_type, functionType=PandasUDFType.SCALAR)
        return self._with_new_scol(scol=pudf(self.spark_column)).rename(self.name)

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
        scol = F.round(self.spark_column, decimals)
        return self._with_new_scol(scol).rename(column_name)

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
                    "q must be a float of an array of floats; however, [%s] found." % type(v)
                )
            if v < 0.0 or v > 1.0:
                raise ValueError("percentiles should all be in the interval [0, 1].")

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
            sdf = self._internal._sdf
            args = ", ".join(map(str, quantiles))
            percentile_col = F.expr(
                "approx_percentile(`%s`, array(%s), %s)" % (self.name, args, accuracy)
            )
            sdf = sdf.select(percentile_col.alias("percentiles"))

            internal_index_column = SPARK_DEFAULT_INDEX_NAME
            value_column = "value"
            cols = []
            for i, quantile in enumerate(quantiles):
                cols.append(
                    F.struct(
                        F.lit("%s" % quantile).alias(internal_index_column),
                        F.expr("percentiles[%s]" % i).alias(value_column),
                    )
                )
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

            internal = _InternalFrame(
                spark_frame=sdf,
                index_map=OrderedDict({internal_index_column: None}),
                column_labels=None,
                data_spark_columns=[scol_for(sdf, value_column)],
                column_label_names=None,
            )

            return DataFrame(internal)[value_column].rename(self.name)
        else:
            return self._reduce_for_stat_function(
                lambda _: F.expr("approx_percentile(`%s`, %s, %s)" % (self.name, q, accuracy)),
                name="median",
            )

    # TODO: add axis, numeric_only, pct, na_option parameter
    def rank(self, method="average", ascending=True):
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
        >>> s = ks.Series([1, 2, 2, 3], name='A')
        >>> s
        0    1
        1    2
        2    2
        3    3
        Name: A, dtype: int64

        >>> s.rank()
        0    1.0
        1    2.5
        2    2.5
        3    4.0
        Name: A, dtype: float64

        If method is set to 'min', it use lowest rank in group.

        >>> s.rank(method='min')
        0    1.0
        1    2.0
        2    2.0
        3    4.0
        Name: A, dtype: float64

        If method is set to 'max', it use highest rank in group.

        >>> s.rank(method='max')
        0    1.0
        1    3.0
        2    3.0
        3    4.0
        Name: A, dtype: float64

        If method is set to 'first', it is assigned rank in order without groups.

        >>> s.rank(method='first')
        0    1.0
        1    2.0
        2    3.0
        3    4.0
        Name: A, dtype: float64

        If method is set to 'dense', it leaves no gaps in group.

        >>> s.rank(method='dense')
        0    1.0
        1    2.0
        2    2.0
        3    3.0
        Name: A, dtype: float64
        """
        return self._rank(method, ascending)

    def _rank(self, method="average", ascending=True, part_cols=()):
        if method not in ["average", "min", "max", "first", "dense"]:
            msg = "method must be one of 'average', 'min', 'max', 'first', 'dense'"
            raise ValueError(msg)

        if len(self._internal.index_spark_column_names) > 1:
            raise ValueError("rank do not support index now")

        if ascending:
            asc_func = lambda scol: scol.asc()
        else:
            asc_func = lambda scol: scol.desc()

        if method == "first":
            window = (
                Window.orderBy(
                    asc_func(self._internal.spark_column),
                    asc_func(F.col(NATURAL_ORDER_COLUMN_NAME)),
                )
                .partitionBy(*part_cols)
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
            scol = F.row_number().over(window)
        elif method == "dense":
            window = (
                Window.orderBy(asc_func(self._internal.spark_column))
                .partitionBy(*part_cols)
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
            scol = F.dense_rank().over(window)
        else:
            if method == "average":
                stat_func = F.mean
            elif method == "min":
                stat_func = F.min
            elif method == "max":
                stat_func = F.max
            window1 = (
                Window.orderBy(asc_func(self._internal.spark_column))
                .partitionBy(*part_cols)
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
            window2 = Window.partitionBy(
                [self._internal.spark_column] + list(part_cols)
            ).rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
            scol = stat_func(F.row_number().over(window1)).over(window2)
        kser = self._with_new_scol(scol).rename(self.name)
        return kser.astype(np.float64)

    def describe(self, percentiles: Optional[List[float]] = None) -> "Series":
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
        return self._diff(periods)

    def _diff(self, periods, part_cols=()):
        if not isinstance(periods, int):
            raise ValueError("periods should be an int; however, got [%s]" % type(periods))
        window = (
            Window.partitionBy(*part_cols)
            .orderBy(NATURAL_ORDER_COLUMN_NAME)
            .rowsBetween(-periods, -periods)
        )
        scol = self.spark_column - F.lag(self.spark_column, periods).over(window)
        return self._with_new_scol(scol).rename(self.name)

    def idxmax(self, skipna=True):
        """
        Return the row label of the maximum value.

        If multiple values equal the maximum, the first row label with that
        value is returned.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, the result
            will be NA.

        Returns
        -------
        Index
            Label of the maximum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        Series.idxmin : Return index *label* of the first occurrence
            of minimum of values.

        Examples
        --------
        >>> s = ks.Series(data=[1, None, 4, 3, 5],
        ...               index=['A', 'B', 'C', 'D', 'E'])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    3.0
        E    5.0
        Name: 0, dtype: float64

        >>> s.idxmax()
        'E'

        If `skipna` is False and there is an NA value in the data,
        the function returns ``nan``.

        >>> s.idxmax(skipna=False)
        nan

        In case of multi-index, you get a tuple:

        >>> index = pd.MultiIndex.from_arrays([
        ...     ['a', 'a', 'b', 'b'], ['c', 'd', 'e', 'f']], names=('first', 'second'))
        >>> s = ks.Series(data=[1, None, 4, 5], index=index)
        >>> s
        first  second
        a      c         1.0
               d         NaN
        b      e         4.0
               f         5.0
        Name: 0, dtype: float64

        >>> s.idxmax()
        ('b', 'f')

        If multiple values equal the maximum, the first row label with that
        value is returned.

        >>> s = ks.Series([1, 100, 1, 100, 1, 100], index=[10, 3, 5, 2, 1, 8])
        >>> s
        10      1
        3     100
        5       1
        2     100
        1       1
        8     100
        Name: 0, dtype: int64

        >>> s.idxmax()
        3
        """
        sdf = self._internal.spark_frame
        scol = self.spark_column
        index_scols = self._internal.index_spark_columns
        # desc_nulls_(last|first) is used via Py4J directly because
        # it's not supported in Spark 2.3.
        if skipna:
            sdf = sdf.orderBy(Column(scol._jc.desc_nulls_last()), NATURAL_ORDER_COLUMN_NAME)
        else:
            sdf = sdf.orderBy(Column(scol._jc.desc_nulls_first()), NATURAL_ORDER_COLUMN_NAME)
        results = sdf.select([scol] + index_scols).take(1)
        if len(results) == 0:
            raise ValueError("attempt to get idxmin of an empty sequence")
        if results[0][0] is None:
            # This will only happens when skipna is False because we will
            # place nulls first.
            return np.nan
        values = list(results[0][1:])
        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    def idxmin(self, skipna=True):
        """
        Return the row label of the minimum value.

        If multiple values equal the minimum, the first row label with that
        value is returned.

        Parameters
        ----------
        skipna : bool, default True
            Exclude NA/null values. If the entire Series is NA, the result
            will be NA.

        Returns
        -------
        Index
            Label of the minimum value.

        Raises
        ------
        ValueError
            If the Series is empty.

        See Also
        --------
        Series.idxmax : Return index *label* of the first occurrence
            of maximum of values.

        Notes
        -----
        This method is the Series version of ``ndarray.argmin``. This method
        returns the label of the minimum, while ``ndarray.argmin`` returns
        the position. To get the position, use ``series.values.argmin()``.

        Examples
        --------
        >>> s = ks.Series(data=[1, None, 4, 0],
        ...               index=['A', 'B', 'C', 'D'])
        >>> s
        A    1.0
        B    NaN
        C    4.0
        D    0.0
        Name: 0, dtype: float64

        >>> s.idxmin()
        'D'

        If `skipna` is False and there is an NA value in the data,
        the function returns ``nan``.

        >>> s.idxmin(skipna=False)
        nan

        In case of multi-index, you get a tuple:

        >>> index = pd.MultiIndex.from_arrays([
        ...     ['a', 'a', 'b', 'b'], ['c', 'd', 'e', 'f']], names=('first', 'second'))
        >>> s = ks.Series(data=[1, None, 4, 0], index=index)
        >>> s
        first  second
        a      c         1.0
               d         NaN
        b      e         4.0
               f         0.0
        Name: 0, dtype: float64

        >>> s.idxmin()
        ('b', 'f')

        If multiple values equal the minimum, the first row label with that
        value is returned.

        >>> s = ks.Series([1, 100, 1, 100, 1, 100], index=[10, 3, 5, 2, 1, 8])
        >>> s
        10      1
        3     100
        5       1
        2     100
        1       1
        8     100
        Name: 0, dtype: int64

        >>> s.idxmin()
        10
        """
        sdf = self._internal._sdf
        scol = self.spark_column
        index_scols = self._internal.index_spark_columns
        # asc_nulls_(last|first)is used via Py4J directly because
        # it's not supported in Spark 2.3.
        if skipna:
            sdf = sdf.orderBy(Column(scol._jc.asc_nulls_last()), NATURAL_ORDER_COLUMN_NAME)
        else:
            sdf = sdf.orderBy(Column(scol._jc.asc_nulls_first()), NATURAL_ORDER_COLUMN_NAME)
        results = sdf.select([scol] + index_scols).take(1)
        if len(results) == 0:
            raise ValueError("attempt to get idxmin of an empty sequence")
        if results[0][0] is None:
            # This will only happens when skipna is False because we will
            # place nulls first.
            return np.nan
        values = list(results[0][1:])
        if len(values) == 1:
            return values[0]
        else:
            return tuple(values)

    def pop(self, item):
        """
        Return item and drop from sereis.

        Parameters
        ----------
        item : str
            Label of index to be popped.

        Returns
        -------
        Series

        Examples
        --------
        >>> s = ks.Series(data=np.arange(3), index=['A', 'B', 'C'])
        >>> s
        A    0
        B    1
        C    2
        Name: 0, dtype: int64

        >>> s.pop('A')
        0

        >>> s
        B    1
        C    2
        Name: 0, dtype: int64

        >>> s = ks.Series(data=np.arange(3), index=['A', 'A', 'C'])
        >>> s
        A    0
        A    1
        C    2
        Name: 0, dtype: int64

        >>> s.pop('A')
        A    0
        A    1
        Name: 0, dtype: int64

        >>> s
        C    2
        Name: 0, dtype: int64

        Also support for MultiIndex

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...               index=midx)
        >>> s
        lama    speed      45.0
                weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        Name: 0, dtype: float64

        >>> s.pop('lama')
        speed      45.0
        weight    200.0
        length      1.2
        Name: 0, dtype: float64

        >>> s
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        Name: 0, dtype: float64

        Also support for MultiIndex with several indexs.

        >>> midx = pd.MultiIndex([['a', 'b', 'c'],
        ...                       ['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 0, 0, 0, 1, 1, 1],
        ...                       [0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 0, 2]]
        ...  )
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...              index=midx)
        >>> s
        a  lama    speed      45.0
                   weight    200.0
                   length      1.2
           cow     speed      30.0
                   weight    250.0
                   length      1.5
        b  falcon  speed     320.0
                   speed       1.0
                   length      0.3
        Name: 0, dtype: float64

        >>> s.pop(('a', 'lama'))
        speed      45.0
        weight    200.0
        length      1.2
        Name: 0, dtype: float64

        >>> s
        a  cow     speed      30.0
                   weight    250.0
                   length      1.5
        b  falcon  speed     320.0
                   speed       1.0
                   length      0.3
        Name: 0, dtype: float64

        >>> s.pop(('b', 'falcon', 'speed'))
        (b, falcon, speed)    320.0
        (b, falcon, speed)      1.0
        Name: 0, dtype: float64
        """
        if not isinstance(item, (str, tuple)):
            raise ValueError("'key' should be string or tuple that contains strings")
        if isinstance(item, str):
            item = (item,)
        if not all(isinstance(index, str) for index in item):
            raise ValueError(
                "'key' should have index names as only strings "
                "or a tuple that contain index names as only strings"
            )
        if len(self._internal._index_map) < len(item):
            raise KeyError(
                "Key length ({}) exceeds index depth ({})".format(
                    len(item), len(self._internal.index_map)
                )
            )

        cols = self._internal.index_spark_columns[len(item) :] + [
            self._internal.spark_column_for(self._internal.column_labels[0])
        ]
        rows = [self._internal.spark_columns[level] == index for level, index in enumerate(item)]
        sdf = self._internal.spark_frame.select(cols).filter(reduce(lambda x, y: x & y, rows))

        if len(self._internal._index_map) == len(item):
            # if spark_frame has one column and one data, return data only without frame
            pdf = sdf.limit(2).toPandas()
            length = len(pdf)
            if length == 1:
                self._internal = self.drop(item)._internal
                return pdf[self.name].iloc[0]

            self._internal = self.drop(item)._internal
            item_string = name_like_string(item)
            sdf = sdf.withColumn(SPARK_DEFAULT_INDEX_NAME, F.lit(str(item_string)))
            internal = _InternalFrame(
                spark_frame=sdf, index_map=OrderedDict({SPARK_DEFAULT_INDEX_NAME: None})
            )
            return _col(DataFrame(internal))

        internal = self._internal.copy(
            spark_frame=sdf,
            index_map=OrderedDict(list(self._internal._index_map.items())[len(item) :]),
        )

        self._internal = self.drop(item)._internal

        return _col(DataFrame(internal))

    def copy(self, deep=None) -> "Series":
        """
        Make a copy of this object's indices and data.

        Parameters
        ----------
        deep : None
            this parameter is not supported but just dummy parameter to match pandas.

        Returns
        -------
        copy : Series

        Examples
        --------
        >>> s = ks.Series([1, 2], index=["a", "b"])
        >>> s
        a    1
        b    2
        Name: 0, dtype: int64
        >>> s_copy = s.copy()
        >>> s_copy
        a    1
        b    2
        Name: 0, dtype: int64
        """
        return _col(DataFrame(self._internal.copy()))

    def mode(self, dropna=True) -> "Series":
        """
        Return the mode(s) of the dataset.

        Always returns Series even if only one value is returned.

        Parameters
        ----------
        dropna : bool, default True
            Don't consider counts of NaN/NaT.

        Returns
        -------
        Series
            Modes of the Series.

        Examples
        --------
        >>> s = ks.Series([0, 0, 1, 1, 1, np.nan, np.nan, np.nan])
        >>> s
        0    0.0
        1    0.0
        2    1.0
        3    1.0
        4    1.0
        5    NaN
        6    NaN
        7    NaN
        Name: 0, dtype: float64

        >>> s.mode()
        0    1.0
        Name: 0, dtype: float64

        If there are several same modes, all items are shown

        >>> s = ks.Series([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
        ...                np.nan, np.nan, np.nan])
        >>> s
        0     0.0
        1     0.0
        2     1.0
        3     1.0
        4     1.0
        5     2.0
        6     2.0
        7     2.0
        8     3.0
        9     3.0
        10    3.0
        11    NaN
        12    NaN
        13    NaN
        Name: 0, dtype: float64

        >>> s.mode().sort_values()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...  1.0
        ...  2.0
        ...  3.0
        Name: 0, dtype: float64

        With 'dropna' set to 'False', we can also see NaN in the result

        >>> s.mode(False).sort_values()  # doctest: +NORMALIZE_WHITESPACE, +ELLIPSIS
        <BLANKLINE>
        ...  1.0
        ...  2.0
        ...  3.0
        ...  NaN
        Name: 0, dtype: float64
        """
        ser_count = self.value_counts(dropna=dropna, sort=False)
        sdf_count = ser_count._internal.spark_frame
        most_value = ser_count.max()
        sdf_most_value = sdf_count.filter("count == {}".format(most_value))
        sdf = sdf_most_value.select(F.col(SPARK_DEFAULT_INDEX_NAME).alias("0"))
        internal = _InternalFrame(spark_frame=sdf, index_map=None)

        result = _col(DataFrame(internal))
        result.name = self.name

        return result

    def keys(self):
        """
        Return alias for index.

        Returns
        -------
        Index
            Index of the Series.

        Examples
        --------
        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> kser = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)

        >>> kser.keys()  # doctest: +SKIP
        MultiIndex([(  'lama',  'speed'),
                    (  'lama', 'weight'),
                    (  'lama', 'length'),
                    (   'cow',  'speed'),
                    (   'cow', 'weight'),
                    (   'cow', 'length'),
                    ('falcon',  'speed'),
                    ('falcon', 'weight'),
                    ('falcon', 'length')],
                   )
        """
        return self.index

    # TODO: 'regex', 'method' parameter
    def replace(self, to_replace=None, value=None, regex=False) -> "Series":
        """
        Replace values given in to_replace with value.
        Values of the Series are replaced with other values dynamically.

        Parameters
        ----------
        to_replace : str, list, dict, Series, int, float, or None
            How to find the values that will be replaced.
            * numeric, str:

                - numeric: numeric values equal to to_replace will be replaced with value
                - str: string exactly matching to_replace will be replaced with value

            * list of str or numeric:

                - if to_replace and value are both lists, they must be the same length.
                - str and numeric rules apply as above.

            * dict:

                - Dicts can be used to specify different replacement values for different
                  existing values.
                  For example, {'a': 'b', 'y': 'z'} replaces the value ‘a’ with ‘b’ and ‘y’
                  with ‘z’. To use a dict in this way the value parameter should be None.
                - For a DataFrame a dict can specify that different values should be replaced
                  in different columns. For example, {'a': 1, 'b': 'z'} looks for the value 1
                  in column ‘a’ and the value ‘z’ in column ‘b’ and replaces these values with
                  whatever is specified in value.
                  The value parameter should not be None in this case.
                  You can treat this as a special case of passing two lists except that you are
                  specifying the column to search in.

            See the examples section for examples of each of these.

        value : scalar, dict, list, str default None
            Value to replace any values matching to_replace with.
            For a DataFrame a dict of values can be used to specify which value to use
            for each column (columns not in the dict will not be filled).
            Regular expressions, strings and lists or dicts of such objects are also allowed.

        Returns
        -------
        Series
            Object after replacement.

        Examples
        --------

        Scalar `to_replace` and `value`

        >>> s = ks.Series([0, 1, 2, 3, 4])
        >>> s
        0    0
        1    1
        2    2
        3    3
        4    4
        Name: 0, dtype: int64

        >>> s.replace(0, 5)
        0    5
        1    1
        2    2
        3    3
        4    4
        Name: 0, dtype: int64

        List-like `to_replace`

        >>> s.replace([0, 4], 5000)
        0    5000
        1       1
        2       2
        3       3
        4    5000
        Name: 0, dtype: int64

        >>> s.replace([1, 2, 3], [10, 20, 30])
        0     0
        1    10
        2    20
        3    30
        4     4
        Name: 0, dtype: int64

        Dict-like `to_replace`

        >>> s.replace({1: 1000, 2: 2000, 3: 3000, 4: 4000})
        0       0
        1    1000
        2    2000
        3    3000
        4    4000
        Name: 0, dtype: int64

        Also support for MultiIndex

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...               index=midx)
        >>> s
        lama    speed      45.0
                weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        Name: 0, dtype: float64

        >>> s.replace(45, 450)
        lama    speed     450.0
                weight    200.0
                length      1.2
        cow     speed      30.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        Name: 0, dtype: float64

        >>> s.replace([45, 30, 320], 500)
        lama    speed     500.0
                weight    200.0
                length      1.2
        cow     speed     500.0
                weight    250.0
                length      1.5
        falcon  speed     500.0
                weight      1.0
                length      0.3
        Name: 0, dtype: float64

        >>> s.replace({45: 450, 30: 300})
        lama    speed     450.0
                weight    200.0
                length      1.2
        cow     speed     300.0
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        Name: 0, dtype: float64
        """
        if to_replace is None:
            return self
        if not isinstance(to_replace, (str, list, dict, int, float)):
            raise ValueError("'to_replace' should be one of str, list, dict, int, float")
        if regex:
            raise NotImplementedError("replace currently not support for regex")
        if isinstance(to_replace, list) and isinstance(value, list):
            if not len(to_replace) == len(value):
                raise ValueError(
                    "Replacement lists must match in length. Expecting {} got {}".format(
                        len(to_replace), len(value)
                    )
                )
            to_replace = {k: v for k, v in zip(to_replace, value)}
        if isinstance(to_replace, dict):
            is_start = True
            if len(to_replace) == 0:
                current = self.spark_column
            else:
                for to_replace_, value in to_replace.items():
                    if is_start:
                        current = F.when(self.spark_column == F.lit(to_replace_), value)
                        is_start = False
                    else:
                        current = current.when(self.spark_column == F.lit(to_replace_), value)
                current = current.otherwise(self.spark_column)
        else:
            current = F.when(self.spark_column.isin(to_replace), value).otherwise(self.spark_column)

        return self._with_new_scol(current)

    def update(self, other):
        """
        Modify Series in place using non-NA values from passed Series. Aligns on index.

        Parameters
        ----------
        other : Series

        Examples
        --------
        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> s = ks.Series([1, 2, 3])
        >>> s.update(ks.Series([4, 5, 6]))
        >>> s.sort_index()
        0    4
        1    5
        2    6
        Name: 0, dtype: int64

        >>> s = ks.Series(['a', 'b', 'c'])
        >>> s.update(ks.Series(['d', 'e'], index=[0, 2]))
        >>> s.sort_index()
        0    d
        1    b
        2    e
        Name: 0, dtype: object

        >>> s = ks.Series([1, 2, 3])
        >>> s.update(ks.Series([4, 5, 6, 7, 8]))
        >>> s.sort_index()
        0    4
        1    5
        2    6
        Name: 0, dtype: int64

        >>> s = ks.Series([1, 2, 3], index=[10, 11, 12])
        >>> s
        10    1
        11    2
        12    3
        Name: 0, dtype: int64

        >>> s.update(ks.Series([4, 5, 6]))
        >>> s.sort_index()
        10    1
        11    2
        12    3
        Name: 0, dtype: int64

        >>> s.update(ks.Series([4, 5, 6], index=[11, 12, 13]))
        >>> s.sort_index()
        10    1
        11    4
        12    5
        Name: 0, dtype: int64

        If ``other`` contains NaNs the corresponding values are not updated
        in the original Series.

        >>> s = ks.Series([1, 2, 3])
        >>> s.update(ks.Series([4, np.nan, 6]))
        >>> s.sort_index()
        0    4.0
        1    2.0
        2    6.0
        Name: 0, dtype: float64

        >>> reset_option("compute.ops_on_diff_frames")
        """
        if not isinstance(other, Series):
            raise ValueError("'other' must be a Series")

        index_scol_names = [index_map[0] for index_map in self._internal.index_map.items()]
        combined = combine_frames(self.to_frame(), other.to_frame(), how="leftouter")
        combined_sdf = combined._sdf
        this_col = "__this_%s" % str(
            self._internal.spark_column_name_for(self._internal.column_labels[0])
        )
        that_col = "__that_%s" % str(
            self._internal.spark_column_name_for(other._internal.column_labels[0])
        )
        cond = (
            F.when(scol_for(combined_sdf, that_col).isNotNull(), scol_for(combined_sdf, that_col))
            .otherwise(combined_sdf[this_col])
            .alias(str(self._internal.spark_column_name_for(self._internal.column_labels[0])))
        )
        internal = _InternalFrame(
            spark_frame=combined_sdf.select(index_scol_names + [cond]),
            index_map=self._internal.index_map,
            column_labels=self._internal.column_labels,
        )
        self_updated = _col(ks.DataFrame(internal))
        self._internal = self_updated._internal
        self._kdf = self_updated._kdf

    def where(self, cond, other=np.nan):
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : boolean Series
            Where cond is True, keep the original value. Where False,
            replace with corresponding value from other.
        other : scalar, Series
            Entries where cond is False are replaced with corresponding value from other.

        Returns
        -------
        Series

        Examples
        --------

        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> s1 = ks.Series([0, 1, 2, 3, 4])
        >>> s2 = ks.Series([100, 200, 300, 400, 500])
        >>> s1.where(s1 > 0).sort_index()
        0    NaN
        1    1.0
        2    2.0
        3    3.0
        4    4.0
        Name: 0, dtype: float64

        >>> s1.where(s1 > 1, 10).sort_index()
        0    10
        1    10
        2     2
        3     3
        4     4
        Name: 0, dtype: int64

        >>> s1.where(s1 > 1, s1 + 100).sort_index()
        0    100
        1    101
        2      2
        3      3
        4      4
        Name: 0, dtype: int64

        >>> s1.where(s1 > 1, s2).sort_index()
        0    100
        1    200
        2      2
        3      3
        4      4
        Name: 0, dtype: int64

        >>> reset_option("compute.ops_on_diff_frames")
        """
        assert isinstance(cond, Series)

        # We should check the DataFrame from both `cond` and `other`.
        should_try_ops_on_diff_frame = cond._kdf is not self._kdf or (
            isinstance(other, Series) and other._kdf is not self._kdf
        )

        if should_try_ops_on_diff_frame:
            # Try to perform it with 'compute.ops_on_diff_frame' option.
            kdf = self.to_frame()
            tmp_cond_col = verify_temp_column_name(kdf, "__tmp_cond_col__")
            tmp_other_col = verify_temp_column_name(kdf, "__tmp_other_col__")

            kdf[tmp_cond_col] = cond
            kdf[tmp_other_col] = other

            # above logic makes a Spark DataFrame looks like below:
            # +-----------------+---+----------------+-----------------+
            # |__index_level_0__|  0|__tmp_cond_col__|__tmp_other_col__|
            # +-----------------+---+----------------+-----------------+
            # |                0|  0|           false|              100|
            # |                1|  1|           false|              200|
            # |                3|  3|            true|              400|
            # |                2|  2|            true|              300|
            # |                4|  4|            true|              500|
            # +-----------------+---+----------------+-----------------+
            condition = (
                F.when(
                    kdf[tmp_cond_col].spark_column,
                    kdf[self._internal.column_labels[0]].spark_column,
                )
                .otherwise(kdf[tmp_other_col].spark_column)
                .alias(self._internal.data_spark_column_names[0])
            )

            internal = kdf._internal.with_new_columns(
                [condition], column_labels=self._internal.column_labels
            )
            return _col(DataFrame(internal))
        else:
            if isinstance(other, Series):
                other = other.spark_column
            condition = (
                F.when(cond.spark_column, self.spark_column)
                .otherwise(other)
                .alias(self._internal.data_spark_column_names[0])
            )
            return self._with_new_scol(condition)

    def mask(self, cond, other=np.nan):
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : boolean Series
            Where cond is False, keep the original value. Where True,
            replace with corresponding value from other.
        other : scalar, Series
            Entries where cond is True are replaced with corresponding value from other.

        Returns
        -------
        Series

        Examples
        --------

        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> s1 = ks.Series([0, 1, 2, 3, 4])
        >>> s2 = ks.Series([100, 200, 300, 400, 500])
        >>> s1.mask(s1 > 0).sort_index()
        0    0.0
        1    NaN
        2    NaN
        3    NaN
        4    NaN
        Name: 0, dtype: float64

        >>> s1.mask(s1 > 1, 10).sort_index()
        0     0
        1     1
        2    10
        3    10
        4    10
        Name: 0, dtype: int64

        >>> s1.mask(s1 > 1, s1 + 100).sort_index()
        0      0
        1      1
        2    102
        3    103
        4    104
        Name: 0, dtype: int64

        >>> s1.mask(s1 > 1, s2).sort_index()
        0      0
        1      1
        2    300
        3    400
        4    500
        Name: 0, dtype: int64

        >>> reset_option("compute.ops_on_diff_frames")
        """
        return self.where(~cond, other)

    def xs(self, key, level=None):
        """
        Return cross-section from the Series.

        This method takes a `key` argument to select data at a particular
        level of a MultiIndex.

        Parameters
        ----------
        key : label or tuple of label
            Label contained in the index, or partially in a MultiIndex.
        level : object, defaults to first n levels (n=1 or len(key))
            In case of a key partially contained in a MultiIndex, indicate
            which levels are used. Levels can be referred by label or position.

        Returns
        -------
        Series
            Cross-section from the original Series
            corresponding to the selected index levels.

        Examples
        --------
        >>> midx = pd.MultiIndex([['a', 'b', 'c'],
        ...                       ['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...               index=midx)
        >>> s
        a  lama    speed      45.0
                   weight    200.0
                   length      1.2
        b  cow     speed      30.0
                   weight    250.0
                   length      1.5
        c  falcon  speed     320.0
                   weight      1.0
                   length      0.3
        Name: 0, dtype: float64

        Get values at specified index

        >>> s.xs('a')
        lama  speed      45.0
              weight    200.0
              length      1.2
        Name: 0, dtype: float64

        Get values at several indexes

        >>> s.xs(('a', 'lama'))
        speed      45.0
        weight    200.0
        length      1.2
        Name: 0, dtype: float64

        Get values at specified index and level

        >>> s.xs('lama', level=1)
        a  speed      45.0
           weight    200.0
           length      1.2
        Name: 0, dtype: float64
        """
        if not isinstance(key, tuple):
            key = (key,)
        if level is None:
            level = 0

        cols = (
            self._internal.index_spark_columns[:level]
            + self._internal.index_spark_columns[level + len(key) :]
            + [self._internal.spark_column_for(self._internal.column_labels[0])]
        )
        rows = [self._internal.spark_columns[lvl] == index for lvl, index in enumerate(key, level)]
        sdf = self._internal.spark_frame.select(cols).where(reduce(lambda x, y: x & y, rows))

        if len(self._internal._index_map) == len(key):
            # if spark_frame has one column and one data, return data only without frame
            pdf = sdf.limit(2).toPandas()
            length = len(pdf)
            if length == 1:
                return pdf[self.name].iloc[0]

        index_cols = [
            col for col in sdf.columns if col not in self._internal.data_spark_column_names
        ]
        index_map_dict = dict(self._internal.index_map)
        internal = self._internal.copy(
            spark_frame=sdf,
            index_map=OrderedDict(
                (index_col, index_map_dict[index_col]) for index_col in index_cols
            ),
        )

        return _col(DataFrame(internal))

    def pct_change(self, periods=1):
        """
        Percentage change between the current and a prior element.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for forming percent change.

        Returns
        -------
        Series

        Examples
        --------

        >>> kser = ks.Series([90, 91, 85], index=[2, 4, 1])
        >>> kser
        2    90
        4    91
        1    85
        Name: 0, dtype: int64

        >>> kser.pct_change()
        2         NaN
        4    0.011111
        1   -0.065934
        Name: 0, dtype: float64

        >>> kser.sort_index().pct_change()
        1         NaN
        2    0.058824
        4    0.011111
        Name: 0, dtype: float64

        >>> kser.pct_change(periods=2)
        2         NaN
        4         NaN
        1   -0.055556
        Name: 0, dtype: float64
        """
        scol = self._internal.spark_column

        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-periods, -periods)
        prev_row = F.lag(scol, periods).over(window)

        return self._with_new_scol((scol - prev_row) / prev_row)

    def combine_first(self, other):
        """
        Combine Series values, choosing the calling Series's values first.

        Parameters
        ----------
        other : Series
            The value(s) to be combined with the `Series`.

        Returns
        -------
        Series
            The result of combining the Series with the other object.

        See Also
        --------
        Series.combine : Perform elementwise operation on two Series
            using a given function.

        Notes
        -----
        Result index will be the union of the two indexes.

        Examples
        --------
        >>> s1 = ks.Series([1, np.nan])
        >>> s2 = ks.Series([3, 4])
        >>> s1.combine_first(s2)
        0    1.0
        1    4.0
        Name: 0, dtype: float64
        """
        if not isinstance(other, ks.Series):
            raise ValueError("`combine_first` only allows `Series` for parameter `other`")
        if self._kdf is other._kdf:
            this = self.name
            that = other.name
            combined = self._kdf
        else:
            this = "__this_{}".format(self.name)
            that = "__that_{}".format(other.name)
            with option_context("compute.ops_on_diff_frames", True):
                combined = combine_frames(self.to_frame(), other)
        sdf = combined._sdf
        # If `self` has missing value, use value of `other`
        cond = F.when(sdf[this].isNull(), sdf[that]).otherwise(sdf[this])
        # If `self` and `other` come from same frame, the anchor should be kept
        if self._kdf is other._kdf:
            return self._with_new_scol(cond)
        index_scols = combined._internal.index_spark_columns
        sdf = sdf.select(*index_scols, cond.alias(self.name)).distinct()
        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=self._internal.index_map,
            column_labels=self._internal.column_labels,
            column_label_names=self._internal.column_label_names,
        )
        return _col(ks.DataFrame(internal))

    def dot(self, other):
        """
        Compute the dot product between the Series and the columns of other.

        This method computes the dot product between the Series and another
        one, or the Series and each columns of a DataFrame.

        It can also be called using `self @ other` in Python >= 3.5.

        .. note:: This API is slightly different from pandas when indexes from both
            are not aligned. To match with pandas', it requires to read the whole data for,
            for example, counting. pandas raises an exception; however, Koalas just proceeds
            and performs by ignoring mismatches with NaN permissively.

            >>> pdf1 = pd.Series([1, 2, 3], index=[0, 1, 2])
            >>> pdf2 = pd.Series([1, 2, 3], index=[0, 1, 3])
            >>> pdf1.dot(pdf2)  # doctest: +SKIP
            ...
            ValueError: matrices are not aligned

            >>> kdf1 = ks.Series([1, 2, 3], index=[0, 1, 2])
            >>> kdf2 = ks.Series([1, 2, 3], index=[0, 1, 3])
            >>> kdf1.dot(kdf2)  # doctest: +SKIP
            5

        Parameters
        ----------
        other : Series, DataFrame.
            The other object to compute the dot product with its columns.

        Returns
        -------
        scalar, Series
            Return the dot product of the Series and other if other is a
            Series, the Series of the dot product of Series and each rows of
            other if other is a DataFrame.

        Notes
        -----
        The Series and other has to share the same index if other is a Series
        or a DataFrame.

        Examples
        --------
        >>> s = ks.Series([0, 1, 2, 3])

        >>> s.dot(s)
        14

        >>> s @ s
        14
        """
        if isinstance(other, DataFrame):
            raise ValueError(
                "Series.dot() is currently not supported with DataFrame since "
                "it will cause expansive calculation as many as the number "
                "of columns of DataFrame"
            )
        if self._kdf is not other._kdf:
            if len(self.index) != len(other.index):
                raise ValueError("matrices are not aligned")
        if isinstance(other, Series):
            result = (self * other).sum()

        return result

    def __matmul__(self, other):
        """
        Matrix multiplication using binary `@` operator in Python>=3.5.
        """
        return self.dot(other)

    def repeat(self, repeats: int) -> "Series":
        """
        Repeat elements of a Series.

        Returns a new Series where each element of the current Series
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Series.

        Returns
        -------
        Series
            Newly created Series with repeated elements.

        See Also
        --------
        Index.repeat : Equivalent function for Index.

        Examples
        --------
        >>> s = ks.Series(['a', 'b', 'c'])
        >>> s
        0    a
        1    b
        2    c
        Name: 0, dtype: object
        >>> s.repeat(2)
        0    a
        1    b
        2    c
        0    a
        1    b
        2    c
        Name: 0, dtype: object
        >>> ks.Series([1, 2, 3]).repeat(0)
        Series([], Name: 0, dtype: int64)
        """
        if not isinstance(repeats, int):
            raise ValueError("`repeats` argument must be integer, but got {}".format(type(repeats)))
        elif repeats < 0:
            raise ValueError("negative dimensions are not allowed")

        kdf = self.to_frame()
        if repeats == 0:
            return _col(DataFrame(kdf._internal.with_filter(F.lit(False))))
        else:
            return _col(ks.concat([kdf] * repeats))

    def asof(self, where):
        """
        Return the last row(s) without any NaNs before `where`.

        The last row (for each element in `where`, if list) without any
        NaN is taken.

        If there is no good value, NaN is returned.

        .. note:: This API is dependent on :meth:`Index.is_monotonic_increasing`
            which can be expensive.

        Parameters
        ----------
        where : index or array-like of indices

        Returns
        -------
        scalar or Series

            The return can be:

            * scalar : when `self` is a Series and `where` is a scalar
            * Series: when `self` is a Series and `where` is an array-like

            Return scalar or Series

        Notes
        -----
        Indices are assumed to be sorted. Raises if this is not the case.

        Examples
        --------
        >>> s = ks.Series([1, 2, np.nan, 4], index=[10, 20, 30, 40])
        >>> s
        10    1.0
        20    2.0
        30    NaN
        40    4.0
        Name: 0, dtype: float64

        A scalar `where`.

        >>> s.asof(20)
        2.0

        For a sequence `where`, a Series is returned. The first value is
        NaN, because the first element of `where` is before the first
        index value.

        >>> s.asof([5, 20]).sort_index()
        5     NaN
        20    2.0
        Name: 0, dtype: float64

        Missing values are not considered. The following is ``2.0``, not
        NaN, even though NaN is at the index location for ``30``.

        >>> s.asof(30)
        2.0
        """
        should_return_series = True
        if isinstance(self.index, ks.MultiIndex):
            raise ValueError("asof is not supported for a MultiIndex")
        if isinstance(where, (ks.Index, ks.Series, ks.DataFrame)):
            raise ValueError("where cannot be an Index, Series or a DataFrame")
        if not self.index.is_monotonic_increasing:
            raise ValueError("asof requires a sorted index")
        if not is_list_like(where):
            should_return_series = False
            where = [where]
        sdf = self._internal._sdf
        index_scol = self._internal.index_spark_columns[0]
        cond = [F.max(F.when(index_scol <= index, self.spark_column)) for index in where]
        sdf = sdf.select(cond)
        if not should_return_series:
            result = sdf.head()[0]
            return result if result is not None else np.nan

        # The data is expected to be small so it's fine to transpose/use default index.
        with ks.option_context(
            "compute.default_index_type", "distributed", "compute.max_rows", None
        ):
            kdf = ks.DataFrame(sdf)
            kdf.columns = pd.Index(where)
            result_series = _col(kdf.transpose())

        result_series.name = self.name
        return result_series

    def _cum(self, func, skipna, part_cols=()):
        # This is used to cummin, cummax, cumsum, etc.

        window = (
            Window.orderBy(NATURAL_ORDER_COLUMN_NAME)
            .partitionBy(*part_cols)
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )

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
                self.spark_column.isNull(),
                F.lit(None),
            ).otherwise(func(self.spark_column).over(window))
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
                F.max(self.spark_column.isNull()).over(window),
                # Manually sets nulls given the column defined above.
                F.lit(None),
            ).otherwise(func(self.spark_column).over(window))

        return self._with_new_scol(scol).rename(self.name)

    def _cumprod(self, skipna, part_cols=()):
        from pyspark.sql.functions import pandas_udf

        def cumprod(scol):
            @pandas_udf(returnType=self.spark_type)
            def negative_check(s):
                assert len(s) == 0 or ((s > 0) | (s.isnull())).all(), (
                    "values should be bigger than 0: %s" % s
                )
                return s

            return F.sum(F.log(negative_check(scol)))

        kser = self._cum(cumprod, skipna, part_cols)
        return kser._with_new_scol(F.exp(kser.spark_column)).rename(self.name)

    # ----------------------------------------------------------------------
    # Accessor Methods
    # ----------------------------------------------------------------------
    dt = CachedAccessor("dt", DatetimeMethods)
    str = CachedAccessor("str", StringMethods)

    # ----------------------------------------------------------------------

    def _apply_series_op(self, op):
        return op(self)

    def _reduce_for_stat_function(self, sfun, name, axis=None, numeric_only=None):
        """
        Applies sfun to the column and returns a scalar

        Parameters
        ----------
        sfun : the stats function to be used for aggregation
        name : original pandas API name.
        axis : used only for sanity check because series only support index axis.
        numeric_only : not used by this implementation, but passed down by stats functions
        """
        from inspect import signature

        axis = validate_axis(axis)
        if axis == 1:
            raise ValueError("Series does not support columns axis.")
        num_args = len(signature(sfun).parameters)
        col_sdf = self.spark_column
        col_type = self.spark_type
        if isinstance(col_type, BooleanType) and sfun.__name__ not in ("min", "max"):
            # Stat functions cannot be used with boolean values by default
            # Thus, cast to integer (true to 1 and false to 0)
            # Exclude the min and max methods though since those work with booleans
            col_sdf = col_sdf.cast("integer")
        if num_args == 1:
            # Only pass in the column if sfun accepts only one arg
            col_sdf = sfun(col_sdf)
        else:  # must be 2
            assert num_args == 2
            # Pass in both the column and its data type if sfun accepts two args
            col_sdf = sfun(col_sdf, col_type)
        return _unpack_scalar(self._internal._sdf.select(col_sdf))

    def __len__(self):
        return len(self.to_dataframe())

    def __getitem__(self, key):
        try:
            if (isinstance(key, slice) and any(type(n) == int for n in [key.start, key.stop])) or (
                type(key) == int and not isinstance(self.index.spark_type, (IntegerType, LongType))
            ):
                # Seems like pandas Series always uses int as positional search when slicing
                # with ints, searches based on index values when the value is int.
                return self.iloc[key]
            return self.loc[key]
        except SparkPandasIndexingError:
            raise KeyError(
                "Key length ({}) exceeds index depth ({})".format(
                    len(key), len(self._internal.index_map)
                )
            )

    def __getattr__(self, item: str_type) -> Any:
        if item.startswith("__"):
            raise AttributeError(item)
        if hasattr(_MissingPandasLikeSeries, item):
            property_or_func = getattr(_MissingPandasLikeSeries, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        return self.getField(item)

    def _to_internal_pandas(self):
        """
        Return a pandas Series directly from _internal to avoid overhead of copy.

        This method is for internal use only.
        """
        return _col(self._internal.to_pandas_frame)

    def __repr__(self):
        max_display_count = get_option("display.max_rows")
        if max_display_count is None:
            return self._to_internal_pandas().to_string(name=self.name, dtype=self.dtype)

        pser = self.head(max_display_count + 1)._to_internal_pandas()
        pser_length = len(pser)
        pser = pser.iloc[:max_display_count]
        if pser_length > max_display_count:
            repr_string = pser.to_string(length=True)
            rest, prev_footer = repr_string.rsplit("\n", 1)
            match = REPR_PATTERN.search(prev_footer)
            if match is not None:
                length = match.group("length")
                name = str(self.dtype.name)
                footer = "\nName: {name}, dtype: {dtype}\nShowing only the first {length}".format(
                    length=length, name=self.name, dtype=pprint_thing(name)
                )
                return rest + footer
        return pser.to_string(name=self.name, dtype=self.dtype)

    def __dir__(self):
        if not isinstance(self.spark_type, StructType):
            fields = []
        else:
            fields = [f for f in self.spark_type.fieldNames() if " " not in f]
        return super(Series, self).__dir__() + fields

    def __iter__(self):
        return _MissingPandasLikeSeries.__iter__(self)

    def _equals(self, other: "Series") -> bool:
        return self.spark_column._jc.equals(other.spark_column._jc)


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
    """
    Takes a DataFrame and returns the first column of the DataFrame as a Series
    """
    assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
    if isinstance(df, DataFrame):
        return df._kser_for(df._internal.column_labels[0])
    else:
        return df[df.columns[0]]
