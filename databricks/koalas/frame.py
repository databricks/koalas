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
A wrapper class for Spark DataFrame to behave similar to pandas DataFrame.
"""
from distutils.version import LooseVersion
import re
import warnings
import inspect
from functools import partial, reduce
import sys
from typing import Any, Optional, List, Tuple, Union, Generic, TypeVar

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like, is_dict_like
if LooseVersion(pd.__version__) >= LooseVersion('0.24'):
    from pandas.core.dtypes.common import infer_dtype_from_object
else:
    from pandas.core.dtypes.common import _get_dtype_from_object as infer_dtype_from_object
from pandas.core.dtypes.inference import is_sequence
from pyspark import sql as spark
from pyspark.sql import functions as F, Column
from pyspark.sql.types import (BooleanType, ByteType, DecimalType, DoubleType, FloatType,
                               IntegerType, LongType, NumericType, ShortType, StructType)
from pyspark.sql.utils import AnalysisException
from pyspark.sql.window import Window

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.utils import validate_arguments_and_invoke_function
from databricks.koalas.generic import _Frame, max_display_count
from databricks.koalas.internal import _InternalFrame, IndexMap
from databricks.koalas.missing.frame import _MissingPandasLikeDataFrame
from databricks.koalas.ml import corr


# These regular expression patterns are complied and defined here to avoid to compile the same
# pattern every time it is used in _repr_ and _repr_html_ in DataFrame.
# Two patterns basically seek the footer string from Pandas'
REPR_PATTERN = re.compile(r"\n\n\[(?P<rows>[0-9]+) rows x (?P<columns>[0-9]+) columns\]$")
REPR_HTML_PATTERN = re.compile(
    r"\n\<p\>(?P<rows>[0-9]+) rows × (?P<columns>[0-9]+) columns\<\/p\>\n\<\/div\>$")


_flex_doc_FRAME = """
Get {desc} of dataframe and other, element-wise (binary operator `{op_name}`).

Equivalent to ``{equiv}``. With reverse version, `{reverse}`.

Among flexible wrappers (`add`, `sub`, `mul`, `div`) to
arithmetic operators: `+`, `-`, `*`, `/`, `//`.

Parameters
----------
other : scalar
    Any single data

Returns
-------
DataFrame
    Result of the arithmetic operation.

Examples
--------
>>> df = ks.DataFrame({{'angles': [0, 3, 4],
...                    'degrees': [360, 180, 360]}},
...                   index=['circle', 'triangle', 'rectangle'],
...                   columns=['angles', 'degrees'])
>>> df
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

Add a scalar with operator version which return the same
results.

>>> df + 1
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

>>> df.add(1)
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

Divide by constant with reverse version.

>>> df.div(10)
           angles  degrees
circle        0.0     36.0
triangle      0.3     18.0
rectangle     0.4     36.0

>>> df.rdiv(10)
             angles   degrees
circle          NaN  0.027778
triangle   3.333333  0.055556
rectangle  2.500000  0.027778

Subtract by constant.

>>> df - 1
           angles  degrees
circle         -1      359
triangle        2      179
rectangle       3      359

>>> df.sub(1)
           angles  degrees
circle         -1      359
triangle        2      179
rectangle       3      359

Multiply by constant.

>>> df * 1
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

>>> df.mul(1)
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

Divide by constant.

>>> df / 1
           angles  degrees
circle        0.0    360.0
triangle      3.0    180.0
rectangle     4.0    360.0

>>> df.div(1)
           angles  degrees
circle        0.0    360.0
triangle      3.0    180.0
rectangle     4.0    360.0

>>> df // 2
           angles  degrees
circle          0      180
triangle        1       90
rectangle       2      180

>>> df % 2
           angles  degrees
circle          0        0
triangle        1        0
rectangle       0        0

>>> df.pow(2)
           angles   degrees
circle        0.0  129600.0
triangle      9.0   32400.0
rectangle    16.0  129600.0
"""

T = TypeVar('T')


if (3, 5) <= sys.version_info < (3, 7):
    from typing import GenericMeta

    # This is a workaround to support variadic generic in DataFrame in Python 3.5+.
    # See https://github.com/python/typing/issues/193
    # We wrap the input params by a tuple to mimic variadic generic.
    old_getitem = GenericMeta.__getitem__  # type: ignore

    def new_getitem(self, params):
        if hasattr(self, "is_dataframe"):
            return old_getitem(self, Tuple[params])
        else:
            return old_getitem(self, params)

    GenericMeta.__getitem__ = new_getitem  # type: ignore


class DataFrame(_Frame, Generic[T]):
    """
    Koala DataFrame that corresponds to Pandas DataFrame logically. This holds Spark DataFrame
    internally.

    :ivar _internal: an internal immutable Frame to manage metadata.
    :type _internal: _InternalFrame

    Parameters
    ----------
    data : numpy ndarray (structured or homogeneous), dict, Pandas DataFrame, Spark DataFrame \
        or Koalas Series
        Dict can contain Series, arrays, constants, or list-like objects
        If data is a dict, argument order is maintained for Python 3.6
        and later.
        Note that if `data` is a Pandas DataFrame, a Spark DataFrame, and a Koalas Series,
        other arguments should not be used.
    index : Index or array-like
        Index to use for resulting frame. Will default to RangeIndex if
        no indexing information part of input data and no index provided
    columns : Index or array-like
        Column labels to use for resulting frame. Will default to
        RangeIndex (0, 1, 2, ..., n) if no column labels are provided
    dtype : dtype, default None
        Data type to force. Only a single dtype is allowed. If None, infer
    copy : boolean, default False
        Copy data from inputs. Only affects DataFrame / 2d ndarray input

    Examples
    --------
    Constructing DataFrame from a dictionary.

    >>> d = {'col1': [1, 2], 'col2': [3, 4]}
    >>> df = ks.DataFrame(data=d, columns=['col1', 'col2'])
    >>> df
       col1  col2
    0     1     3
    1     2     4

    Constructing DataFrame from Pandas DataFrame

    >>> df = ks.DataFrame(pd.DataFrame(data=d, columns=['col1', 'col2']))
    >>> df
       col1  col2
    0     1     3
    1     2     4

    Notice that the inferred dtype is int64.

    >>> df.dtypes
    col1    int64
    col2    int64
    dtype: object

    To enforce a single dtype:

    >>> df = ks.DataFrame(data=d, dtype=np.int8)
    >>> df.dtypes
    col1    int8
    col2    int8
    dtype: object

    Constructing DataFrame from numpy ndarray:

    >>> df2 = ks.DataFrame(np.random.randint(low=0, high=10, size=(5, 5)),
    ...                    columns=['a', 'b', 'c', 'd', 'e'])
    >>> df2  # doctest: +SKIP
       a  b  c  d  e
    0  3  1  4  9  8
    1  4  8  4  8  4
    2  7  6  5  6  7
    3  8  7  9  1  0
    4  2  5  4  3  9
    """
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        if isinstance(data, _InternalFrame):
            assert index is None
            assert columns is None
            assert dtype is None
            assert not copy
            super(DataFrame, self).__init__(data)
        elif isinstance(data, spark.DataFrame):
            assert index is None
            assert columns is None
            assert dtype is None
            assert not copy
            super(DataFrame, self).__init__(_InternalFrame(data))
        elif isinstance(data, ks.Series):
            assert index is None
            assert columns is None
            assert dtype is None
            assert not copy
            data = data.to_dataframe()
            super(DataFrame, self).__init__(data._internal)
        else:
            if isinstance(data, pd.DataFrame):
                assert index is None
                assert columns is None
                assert dtype is None
                assert not copy
                pdf = data
            else:
                pdf = pd.DataFrame(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
            super(DataFrame, self).__init__(_InternalFrame.from_pandas(pdf))

    @property
    def _sdf(self) -> spark.DataFrame:
        return self._internal.sdf

    def _reduce_for_stat_function(self, sfun, numeric_only=False):
        """
        Applies sfun to each column and returns a pd.Series where the number of rows equal the
        number of columns.

        Parameters
        ----------
        sfun : either an 1-arg function that takes a Column and returns a Column, or
        a 2-arg function that takes a Column and its DataType and returns a Column.
        numeric_only : boolean, default False
            If True, sfun is applied on numeric columns (including booleans) only.
        """
        from inspect import signature
        exprs = []
        num_args = len(signature(sfun).parameters)
        for col in self.columns:
            col_sdf = self._sdf[col]
            col_type = self._sdf.schema[col].dataType

            is_numeric_or_boolean = isinstance(col_type, (NumericType, BooleanType))
            min_or_max = sfun.__name__ in ('min', 'max')
            keep_column = not numeric_only or is_numeric_or_boolean or min_or_max

            if keep_column:
                if isinstance(col_type, BooleanType) and not min_or_max:
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
                exprs.append(col_sdf.alias(col))

        sdf = self._sdf.select(*exprs)
        pdf = sdf.toPandas()
        assert len(pdf) == 1, (sdf, pdf)
        row = pdf.iloc[0]
        row.name = None
        return row  # Return first row as a Series

    # Arithmetic Operators
    def _map_series_op(self, op, other):
        if isinstance(other, DataFrame) or is_sequence(other):
            raise ValueError(
                "%s with another DataFrame or a sequence is currently not supported; "
                "however, got %s." % (op, type(other)))

        applied = []
        for column in self._internal.data_columns:
            applied.append(getattr(self[column], op)(other))

        sdf = self._sdf.select(
            self._internal.index_columns + [c._scol for c in applied])
        internal = self._internal.copy(sdf=sdf, data_columns=[c.name for c in applied])
        return DataFrame(internal)

    def __add__(self, other):
        return self._map_series_op("add", other)

    def __radd__(self, other):
        return self._map_series_op("radd", other)

    def __div__(self, other):
        return self._map_series_op("div", other)

    def __rdiv__(self, other):
        return self._map_series_op("rdiv", other)

    def __truediv__(self, other):
        return self._map_series_op("truediv", other)

    def __rtruediv__(self, other):
        return self._map_series_op("rtruediv", other)

    def __mul__(self, other):
        return self._map_series_op("mul", other)

    def __rmul__(self, other):
        return self._map_series_op("rmul", other)

    def __sub__(self, other):
        return self._map_series_op("sub", other)

    def __rsub__(self, other):
        return self._map_series_op("rsub", other)

    def __pow__(self, other):
        return self._map_series_op("pow", other)

    def __rpow__(self, other):
        return self._map_series_op("rpow", other)

    def __mod__(self, other):
        return self._map_series_op("mod", other)

    def __rmod__(self, other):
        return self._map_series_op("rmod", other)

    def __floordiv__(self, other):
        return self._map_series_op("floordiv", other)

    def __rfloordiv__(self, other):
        return self._map_series_op("rfloordiv", other)

    def add(self, other):
        return self + other

    add.__doc__ = _flex_doc_FRAME.format(
        desc='Addition',
        op_name='+',
        equiv='dataframe + other',
        reverse='radd')

    def radd(self, other):
        return other + self

    radd.__doc__ = _flex_doc_FRAME.format(
        desc='Addition',
        op_name="+",
        equiv="other + dataframe",
        reverse='add')

    def div(self, other):
        return self / other

    div.__doc__ = _flex_doc_FRAME.format(
        desc='Floating division',
        op_name="/",
        equiv="dataframe / other",
        reverse='rdiv')

    divide = div

    def rdiv(self, other):
        return other / self

    rdiv.__doc__ = _flex_doc_FRAME.format(
        desc='Floating division',
        op_name="/",
        equiv="other / dataframe",
        reverse='div')

    def truediv(self, other):
        return self / other

    truediv.__doc__ = _flex_doc_FRAME.format(
        desc='Floating division',
        op_name="/",
        equiv="dataframe / other",
        reverse='rtruediv')

    def rtruediv(self, other):
        return other / self

    rtruediv.__doc__ = _flex_doc_FRAME.format(
        desc='Floating division',
        op_name="/",
        equiv="other / dataframe",
        reverse='truediv')

    def mul(self, other):
        return self * other

    mul.__doc__ = _flex_doc_FRAME.format(
        desc='Multiplication',
        op_name="*",
        equiv="dataframe * other",
        reverse='rmul')

    multiply = mul

    def rmul(self, other):
        return other * self

    rmul.__doc__ = _flex_doc_FRAME.format(
        desc='Multiplication',
        op_name="*",
        equiv="other * dataframe",
        reverse='mul')

    def sub(self, other):
        return self - other

    sub.__doc__ = _flex_doc_FRAME.format(
        desc='Subtraction',
        op_name="-",
        equiv="dataframe - other",
        reverse='rsub')

    subtract = sub

    def rsub(self, other):
        return other - self

    rsub.__doc__ = _flex_doc_FRAME.format(
        desc='Subtraction',
        op_name="-",
        equiv="other - dataframe",
        reverse='sub')

    def mod(self, other):
        return self % other

    mod.__doc__ = _flex_doc_FRAME.format(
        desc='Modulo',
        op_name='%',
        equiv='dataframe % other',
        reverse='rmod')

    def rmod(self, other):
        return other % self

    rmod.__doc__ = _flex_doc_FRAME.format(
        desc='Modulo',
        op_name='%',
        equiv='other % dataframe',
        reverse='mod')

    def pow(self, other):
        return self ** other

    pow.__doc__ = _flex_doc_FRAME.format(
        desc='Exponential power of series',
        op_name='**',
        equiv='dataframe ** other',
        reverse='rpow')

    def rpow(self, other):
        return other - self

    rpow.__doc__ = _flex_doc_FRAME.format(
        desc='Exponential power',
        op_name='**',
        equiv='other ** dataframe',
        reverse='pow')

    def floordiv(self, other):
        return self // other

    floordiv.__doc__ = _flex_doc_FRAME.format(
        desc='Integer division',
        op_name='//',
        equiv='dataframe // other',
        reverse='rfloordiv')

    def rfloordiv(self, other):
        return other - self

    rfloordiv.__doc__ = _flex_doc_FRAME.format(
        desc='Integer division',
        op_name='//',
        equiv='other // dataframe',
        reverse='floordiv')

    # Comparison Operators
    def __eq__(self, other):
        return self._map_series_op("eq", other)

    def __ne__(self, other):
        return self._map_series_op("ne", other)

    def __lt__(self, other):
        return self._map_series_op("lt", other)

    def __le__(self, other):
        return self._map_series_op("le", other)

    def __ge__(self, other):
        return self._map_series_op("ge", other)

    def __gt__(self, other):
        return self._map_series_op("gt", other)

    def eq(self, other):
        """
        Compare if the current value is equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.eq(1)
               a     b
        a   True  True
        b  False  None
        c  False  True
        d  False  None
        """
        return self == other

    equals = eq

    def gt(self, other):
        """
        Compare if the current value is greater than the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.gt(2)
               a      b
        a  False  False
        b  False   None
        c   True  False
        d   True   None
        """
        return self > other

    def ge(self, other):
        """
        Compare if the current value is greater than or equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.ge(1)
              a     b
        a  True  True
        b  True  None
        c  True  True
        d  True  None
        """
        return self >= other

    def lt(self, other):
        """
        Compare if the current value is less than the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.lt(1)
               a      b
        a  False  False
        b  False   None
        c  False  False
        d  False   None
        """
        return self < other

    def le(self, other):
        """
        Compare if the current value is less than or equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.le(2)
               a     b
        a   True  True
        b   True  None
        c  False  True
        d  False  None
        """
        return self <= other

    def ne(self, other):
        """
        Compare if the current value is not equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.ne(1)
               a      b
        a  False  False
        b   True   None
        c   True  False
        d   True   None
        """
        return self != other

    def applymap(self, func):
        """
        Apply a function to a Dataframe elementwise.

        This method applies a function that accepts and returns a scalar
        to every element of a DataFrame.

        .. note:: unlike pandas, it is required for `func` to specify return type hint.
            See https://docs.python.org/3/library/typing.html. For instance, as below:

            >>> def function() -> int:
            ...     return 1

        Parameters
        ----------
        func : callable
            Python function, returns a single value from a single value.

        Returns
        -------
        DataFrame
            Transformed DataFrame.

        Examples
        --------
        >>> df = ks.DataFrame([[1, 2.12], [3.356, 4.567]])
        >>> df
               0      1
        0  1.000  2.120
        1  3.356  4.567

        >>> def str_len(x) -> int:
        ...     return len(str(x))
        >>> df.applymap(str_len)
           0  1
        0  3  4
        1  5  5

        >>> def power(x) -> float:
        ...     return x ** 2
        >>> df.applymap(power)
                   0          1
        0   1.000000   4.494400
        1  11.262736  20.857489
        """

        applied = []
        for column in self._internal.data_columns:
            applied.append(self[column].apply(func))

        sdf = self._sdf.select(
            self._internal.index_columns + [c._scol for c in applied])

        internal = self._internal.copy(sdf=sdf, data_columns=[c.name for c in applied])
        return DataFrame(internal)

    def corr(self, method='pearson'):
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method : {'pearson', 'spearman'}
            * pearson : standard correlation coefficient
            * spearman : Spearman rank correlation

        Returns
        -------
        y : pandas.DataFrame

        See Also
        --------
        Series.corr

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df.corr('pearson')
                  dogs      cats
        dogs  1.000000 -0.851064
        cats -0.851064  1.000000

        >>> df.corr('spearman')
                  dogs      cats
        dogs  1.000000 -0.948683
        cats -0.948683  1.000000

        Notes
        -----
        There are behavior differences between Koalas and pandas.

        * the `method` argument only accepts 'pearson', 'spearman'
        * the data should not contain NaNs. Koalas will return an error.
        * Koalas doesn't support the following argument(s).

          * `min_periods` argument is not supported
        """
        return corr(self, method)

    def iteritems(self):
        """
        Iterator over (column name, Series) pairs.

        Iterates over the DataFrame columns, returning a tuple with
        the column name and the content as a Series.

        Returns
        -------
        label : object
            The column names for the DataFrame being iterated over.
        content : Series
            The column entries belonging to each label, as a Series.

        Examples
        --------
        >>> df = ks.DataFrame({'species': ['bear', 'bear', 'marsupial'],
        ...                    'population': [1864, 22000, 80000]},
        ...                   index=['panda', 'polar', 'koala'],
        ...                   columns=['species', 'population'])
        >>> df
                 species  population
        panda       bear        1864
        polar       bear       22000
        koala  marsupial       80000

        >>> for label, content in df.iteritems():
        ...    print('label:', label)
        ...    print('content:', content.to_string())
        ...
        label: species
        content: panda         bear
        polar         bear
        koala    marsupial
        label: population
        content: panda     1864
        polar    22000
        koala    80000
        """
        cols = list(self.columns)
        return list((col_name, self[col_name]) for col_name in cols)

    def to_clipboard(self, excel=True, sep=None, **kwargs):
        """
        Copy object to the system clipboard.

        Write a text representation of object to the system clipboard.
        This can be pasted into Excel, for example.

        .. note:: This method should only be used if the resulting DataFrame is expected
            to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        excel : bool, default True
            - True, use the provided separator, writing in a csv format for
              allowing easy pasting into excel.
            - False, write a string representation of the object to the
              clipboard.

        sep : str, default ``'\\t'``
            Field delimiter.
        **kwargs
            These parameters will be passed to DataFrame.to_csv.

        Notes
        -----
        Requirements for your platform.

          - Linux : `xclip`, or `xsel` (with `gtk` or `PyQt4` modules)
          - Windows : none
          - OS X : none

        See Also
        --------
        read_clipboard : Read text from clipboard.

        Examples
        --------
        Copy the contents of a DataFrame to the clipboard.

        >>> df = ks.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])  # doctest: +SKIP
        >>> df.to_clipboard(sep=',')  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # ,A,B,C
        ... # 0,1,2,3
        ... # 1,4,5,6

        We can omit the the index by passing the keyword `index` and setting
        it to false.

        >>> df.to_clipboard(sep=',', index=False)  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # A,B,C
        ... # 1,2,3
        ... # 4,5,6

        This function also works for Series:

        >>> df = ks.Series([1, 2, 3, 4, 5, 6, 7], name='x')  # doctest: +SKIP
        >>> df.to_clipboard(sep=',')  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # 0, 1
        ... # 1, 2
        ... # 2, 3
        ... # 3, 4
        ... # 4, 5
        ... # 5, 6
        ... # 6, 7
        """

        args = locals()
        kdf = self
        return validate_arguments_and_invoke_function(
            kdf.to_pandas(), self.to_clipboard, pd.DataFrame.to_clipboard, args)

    def to_html(self, buf=None, columns=None, col_space=None, header=True, index=True,
                na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True,
                justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal='.',
                bold_rows=True, classes=None, escape=True, notebook=False, border=None,
                table_id=None, render_links=False):
        """
        Render a DataFrame as an HTML table.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory. If the input
                  is large, set max_rows parameter.

        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        columns : sequence, optional, default None
            The subset of columns to write. Writes all columns by default.
        col_space : int, optional
            The minimum width of each column.
        header : bool, optional
            Write out the column names. If a list of strings is given, it
            is assumed to be aliases for the column names
        index : bool, optional, default True
            Whether to print index (row) labels.
        na_rep : str, optional, default 'NaN'
            String representation of NAN to use.
        formatters : list or dict of one-param. functions, optional
            Formatter functions to apply to columns' elements by position or
            name.
            The result of each function must be a unicode string.
            List must be of length equal to the number of columns.
        float_format : one-parameter function, optional, default None
            Formatter function to apply to columns' elements if they are
            floats. The result of this function must be a unicode string.
        sparsify : bool, optional, default True
            Set to False for a DataFrame with a hierarchical index to print
            every multiindex key at each row.
        index_names : bool, optional, default True
            Prints the names of the indexes.
        justify : str, default None
            How to justify the column labels. If None uses the option from
            the print configuration (controlled by set_option), 'right' out
            of the box. Valid values are

            * left
            * right
            * center
            * justify
            * justify-all
            * start
            * end
            * inherit
            * match-parent
            * initial
            * unset.
        max_rows : int, optional
            Maximum number of rows to display in the console.
        max_cols : int, optional
            Maximum number of columns to display in the console.
        show_dimensions : bool, default False
            Display DataFrame dimensions (number of rows by number of columns).
        decimal : str, default '.'
            Character recognized as decimal separator, e.g. ',' in Europe.
        bold_rows : bool, default True
            Make the row labels bold in the output.
        classes : str or list or tuple, default None
            CSS class(es) to apply to the resulting html table.
        escape : bool, default True
            Convert the characters <, >, and & to HTML-safe sequences.
        notebook : {True, False}, default False
            Whether the generated HTML is for IPython Notebook.
        border : int
            A ``border=border`` attribute is included in the opening
            `<table>` tag. Default ``pd.options.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links (only works with Pandas 0.24+).

        Returns
        -------
        str (or unicode, depending on data and options)
            String representation of the dataframe.

        See Also
        --------
        to_string : Convert DataFrame to a string.
        """
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        if max_rows is not None:
            kdf = self.head(max_rows)
        else:
            kdf = self

        return validate_arguments_and_invoke_function(
            kdf.to_pandas(), self.to_html, pd.DataFrame.to_html, args)

    def to_string(self, buf=None, columns=None, col_space=None, header=True,
                  index=True, na_rep='NaN', formatters=None, float_format=None,
                  sparsify=None, index_names=True, justify=None,
                  max_rows=None, max_cols=None, show_dimensions=False,
                  decimal='.', line_width=None):
        """
        Render a DataFrame to a console-friendly tabular output.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory. If the input
                  is large, set max_rows parameter.

        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        columns : sequence, optional, default None
            The subset of columns to write. Writes all columns by default.
        col_space : int, optional
            The minimum width of each column.
        header : bool, optional
            Write out the column names. If a list of strings is given, it
            is assumed to be aliases for the column names
        index : bool, optional, default True
            Whether to print index (row) labels.
        na_rep : str, optional, default 'NaN'
            String representation of NAN to use.
        formatters : list or dict of one-param. functions, optional
            Formatter functions to apply to columns' elements by position or
            name.
            The result of each function must be a unicode string.
            List must be of length equal to the number of columns.
        float_format : one-parameter function, optional, default None
            Formatter function to apply to columns' elements if they are
            floats. The result of this function must be a unicode string.
        sparsify : bool, optional, default True
            Set to False for a DataFrame with a hierarchical index to print
            every multiindex key at each row.
        index_names : bool, optional, default True
            Prints the names of the indexes.
        justify : str, default None
            How to justify the column labels. If None uses the option from
            the print configuration (controlled by set_option), 'right' out
            of the box. Valid values are

            * left
            * right
            * center
            * justify
            * justify-all
            * start
            * end
            * inherit
            * match-parent
            * initial
            * unset.
        max_rows : int, optional
            Maximum number of rows to display in the console.
        max_cols : int, optional
            Maximum number of columns to display in the console.
        show_dimensions : bool, default False
            Display DataFrame dimensions (number of rows by number of columns).
        decimal : str, default '.'
            Character recognized as decimal separator, e.g. ',' in Europe.
        line_width : int, optional
            Width to wrap a line in characters.

        Returns
        -------
        str (or unicode, depending on data and options)
            String representation of the dataframe.

        See Also
        --------
        to_html : Convert DataFrame to HTML.

        Examples
        --------
        >>> df = ks.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}, columns=['col1', 'col2'])
        >>> print(df.to_string())
           col1  col2
        0     1     4
        1     2     5
        2     3     6

        >>> print(df.to_string(max_rows=2))
           col1  col2
        0     1     4
        1     2     5
        """
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        if max_rows is not None:
            kdf = self.head(max_rows)
        else:
            kdf = self

        return validate_arguments_and_invoke_function(
            kdf.to_pandas(), self.to_string, pd.DataFrame.to_string, args)

    def to_dict(self, orient='dict', into=dict):
        """
        Convert the DataFrame to a dictionary.

        The type of the key-value pairs can be customized with the parameters
        (see below).

        .. note:: This method should only be used if the resulting Pandas DataFrame is expected
            to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        orient : str {'dict', 'list', 'series', 'split', 'records', 'index'}
            Determines the type of the values of the dictionary.

            - 'dict' (default) : dict like {column -> {index -> value}}
            - 'list' : dict like {column -> [values]}
            - 'series' : dict like {column -> Series(values)}
            - 'split' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
            - 'records' : list like
              [{column -> value}, ... , {column -> value}]
            - 'index' : dict like {index -> {column -> value}}

            Abbreviations are allowed. `s` indicates `series` and `sp`
            indicates `split`.

        into : class, default dict
            The collections.abc.Mapping subclass used for all Mappings
            in the return value.  Can be the actual class or an empty
            instance of the mapping type you want.  If you want a
            collections.defaultdict, you must pass it initialized.

        Returns
        -------
        dict, list or collections.abc.Mapping
            Return a collections.abc.Mapping object representing the DataFrame.
            The resulting transformation depends on the `orient` parameter.

        Examples
        --------
        >>> df = ks.DataFrame({'col1': [1, 2],
        ...                    'col2': [0.5, 0.75]},
        ...                   index=['row1', 'row2'],
        ...                   columns=['col1', 'col2'])
        >>> df
              col1  col2
        row1     1  0.50
        row2     2  0.75

        >>> df_dict = df.to_dict()
        >>> sorted([(key, sorted(values.items())) for key, values in df_dict.items()])
        [('col1', [('row1', 1), ('row2', 2)]), ('col2', [('row1', 0.5), ('row2', 0.75)])]

        You can specify the return orientation.

        >>> df_dict = df.to_dict('series')
        >>> sorted(df_dict.items())
        [('col1', row1    1
        row2    2
        Name: col1, dtype: int64), ('col2', row1    0.50
        row2    0.75
        Name: col2, dtype: float64)]

        >>> df_dict = df.to_dict('split')
        >>> sorted(df_dict.items())  # doctest: +ELLIPSIS
        [('columns', ['col1', 'col2']), ('data', [[1..., 0.75]]), ('index', ['row1', 'row2'])]

        >>> df_dict = df.to_dict('records')
        >>> [sorted(values.items()) for values in df_dict]  # doctest: +ELLIPSIS
        [[('col1', 1...), ('col2', 0.5)], [('col1', 2...), ('col2', 0.75)]]

        >>> df_dict = df.to_dict('index')
        >>> sorted([(key, sorted(values.items())) for key, values in df_dict.items()])
        [('row1', [('col1', 1), ('col2', 0.5)]), ('row2', [('col1', 2), ('col2', 0.75)])]

        You can also specify the mapping type.

        >>> from collections import OrderedDict, defaultdict
        >>> df.to_dict(into=OrderedDict)
        OrderedDict([('col1', OrderedDict([('row1', 1), ('row2', 2)])), \
('col2', OrderedDict([('row1', 0.5), ('row2', 0.75)]))])

        If you want a `defaultdict`, you need to initialize it:

        >>> dd = defaultdict(list)
        >>> df.to_dict('records', into=dd)  # doctest: +ELLIPSIS
        [defaultdict(<class 'list'>, {'col..., 'col...}), \
defaultdict(<class 'list'>, {'col..., 'col...})]
        """
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        kdf = self
        return validate_arguments_and_invoke_function(
            kdf.to_pandas(), self.to_dict, pd.DataFrame.to_dict, args)

    def to_latex(self, buf=None, columns=None, col_space=None, header=True, index=True,
                 na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True,
                 bold_rows=False, column_format=None, longtable=None, escape=None, encoding=None,
                 decimal='.', multicolumn=None, multicolumn_format=None, multirow=None):
        r"""
        Render an object to a LaTeX tabular environment table.

        Render an object to a tabular environment table. You can splice this into a LaTeX
        document. Requires usepackage{booktabs}.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory. If the input
                  is large, consider alternative formats.

        Parameters
        ----------
        buf : file descriptor or None
            Buffer to write to. If None, the output is returned as a string.
        columns : list of label, optional
            The subset of columns to write. Writes all columns by default.
        col_space : int, optional
            The minimum width of each column.
        header : bool or list of str, default True
            Write out the column names. If a list of strings is given, it is assumed to be aliases
            for the column names.
        index : bool, default True
            Write row names (index).
        na_rep : str, default ‘NaN’
            Missing data representation.
        formatters : list of functions or dict of {str: function}, optional
            Formatter functions to apply to columns’ elements by position or name. The result of
            each function must be a unicode string. List must be of length equal to the number of
            columns.
        float_format : str, optional
            Format string for floating point numbers.
        sparsify : bool, optional
            Set to False for a DataFrame with a hierarchical index to print every multiindex key at
            each row. By default, the value will be read from the config module.
        index_names : bool, default True
            Prints the names of the indexes.
        bold_rows : bool, default False
            Make the row labels bold in the output.
        column_format : str, optional
            The columns format as specified in LaTeX table format e.g. ‘rcl’ for 3 columns. By
            default, ‘l’ will be used for all columns except columns of numbers, which default
            to ‘r’.
        longtable : bool, optional
            By default, the value will be read from the pandas config module. Use a longtable
            environment instead of tabular. Requires adding a usepackage{longtable} to your LaTeX
            preamble.
        escape : bool, optional
            By default, the value will be read from the pandas config module. When set to False
            prevents from escaping latex special characters in column names.
        encoding : str, optional
            A string representing the encoding to use in the output file, defaults to ‘ascii’ on
            Python 2 and ‘utf-8’ on Python 3.
        decimal : str, default ‘.’
            Character recognized as decimal separator, e.g. ‘,’ in Europe.
        multicolumn : bool, default True
            Use multicolumn to enhance MultiIndex columns. The default will be read from the config
            module.
        multicolumn_format : str, default ‘l’
            The alignment for multicolumns, similar to column_format The default will be read from
            the config module.
        multirow : bool, default False
            Use multirow to enhance MultiIndex rows. Requires adding a usepackage{multirow} to your
            LaTeX preamble. Will print centered labels (instead of top-aligned) across the contained
            rows, separating groups via clines. The default will be read from the pandas config
            module.

        Returns
        -------
        str or None
            If buf is None, returns the resulting LateX format as a string. Otherwise returns None.

        See Also
        --------
        DataFrame.to_string : Render a DataFrame to a console-friendly
            tabular output.
        DataFrame.to_html : Render a DataFrame as an HTML table.


        Examples
        --------
        >>> df = ks.DataFrame({'name': ['Raphael', 'Donatello'],
        ...                    'mask': ['red', 'purple'],
        ...                    'weapon': ['sai', 'bo staff']},
        ...                   columns=['name', 'mask', 'weapon'])
        >>> df.to_latex(index=False) # doctest: +NORMALIZE_WHITESPACE
        '\\begin{tabular}{lll}\n\\toprule\n name & mask & weapon
        \\\\\n\\midrule\n Raphael & red & sai \\\\\n Donatello &
        purple & bo staff \\\\\n\\bottomrule\n\\end{tabular}\n'
        """

        args = locals()
        kdf = self
        return validate_arguments_and_invoke_function(
            kdf.to_pandas(), self.to_latex, pd.DataFrame.to_latex, args)

    # TODO: enable doctests once we drop Spark 2.3.x (due to type coercion logic
    #  when creating arrays)
    def transpose(self, limit: Optional[int] = 1000):
        """
        Transpose index and columns.

        Reflect the DataFrame over its main diagonal by writing rows as columns
        and vice-versa. The property :attr:`.T` is an accessor to the method
        :meth:`transpose`.

        .. note:: This method is based on an expensive operation due to the nature
            of big data. Internally it needs to generate each row for each value, and
            then group twice - it is a huge operation. To prevent misusage, this method
            has the default limit of input length, 1000 and raises a ValueError.

                >>> ks.DataFrame({'a': range(1001)}).transpose()  # doctest: +NORMALIZE_WHITESPACE
                Traceback (most recent call last):
                  ...
                ValueError: Current DataFrame has more then the given limit 1000 rows.
                Please use df.transpose(limit=<maximum number of rows>) to retrieve more than
                1000 rows. Note that, before changing the given 'limit', this operation is
                considerably expensive.

        Parameters
        ----------
        limit : int, optional
            This parameter sets the limit of the current DataFrame. Set `None` to unlimit
            the input length. When the limit is set, it is executed by the shortcut by collecting
            the data into driver side, and then using pandas API. If the limit is unset,
            the operation is executed by PySpark. Default is 1000.

        Returns
        -------
        DataFrame
            The transposed DataFrame.

        Notes
        -----
        Transposing a DataFrame with mixed dtypes will result in a homogeneous
        DataFrame with the coerced dtype. For instance, if int and float have
        to be placed in same column, it becomes float. If type coercion is not
        possible, it fails.

        Also, note that the values in index should be unique because they become
        unique column names.

        In addition, if Spark 2.3 is used, the types should always be exactly same.

        Examples
        --------
        **Square DataFrame with homogeneous dtype**

        >>> d1 = {'col1': [1, 2], 'col2': [3, 4]}
        >>> df1 = ks.DataFrame(data=d1, columns=['col1', 'col2'])
        >>> df1
           col1  col2
        0     1     3
        1     2     4

        >>> df1_transposed = df1.T.sort_index()  # doctest: +SKIP
        >>> df1_transposed  # doctest: +SKIP
              0  1
        col1  1  2
        col2  3  4

        When the dtype is homogeneous in the original DataFrame, we get a
        transposed DataFrame with the same dtype:

        >>> df1.dtypes
        col1    int64
        col2    int64
        dtype: object
        >>> df1_transposed.dtypes  # doctest: +SKIP
        0    int64
        1    int64
        dtype: object

        **Non-square DataFrame with mixed dtypes**

        >>> d2 = {'score': [9.5, 8],
        ...       'kids': [0, 0],
        ...       'age': [12, 22]}
        >>> df2 = ks.DataFrame(data=d2, columns=['score', 'kids', 'age'])
        >>> df2
           score  kids  age
        0    9.5     0   12
        1    8.0     0   22

        >>> df2_transposed = df2.T.sort_index()  # doctest: +SKIP
        >>> df2_transposed  # doctest: +SKIP
                  0     1
        age    12.0  22.0
        kids    0.0   0.0
        score   9.5   8.0

        When the DataFrame has mixed dtypes, we get a transposed DataFrame with
        the coerced dtype:

        >>> df2.dtypes
        score    float64
        kids       int64
        age        int64
        dtype: object

        >>> df2_transposed.dtypes  # doctest: +SKIP
        0    float64
        1    float64
        dtype: object
        """
        if len(self._internal.index_columns) != 1:
            raise ValueError("Single index must be set to transpose the current DataFrame.")
        if limit is not None:
            pdf = self.head(limit + 1).to_pandas()
            if len(pdf) > limit:
                raise ValueError(
                    "Current DataFrame has more then the given limit %s rows. Please use "
                    "df.transpose(limit=<maximum number of rows>) to retrieve more than %s rows. "
                    "Note that, before changing the given 'limit', this operation is considerably "
                    "expensive." % (limit, limit))
            return DataFrame(pdf.transpose())

        index_columns = self._internal.index_columns
        index_column = index_columns[0]
        data_columns = self._internal.data_columns
        sdf = self._sdf

        # Explode the data to be pairs.
        #
        # For instance, if the current input DataFrame is as below:
        #
        # +-----+---+---+---+
        # |index| x1| x2| x3|
        # +-----+---+---+---+
        # |   y1|  1|  0|  0|
        # |   y2|  0| 50|  0|
        # |   y3|  3|  2|  1|
        # +-----+---+---+---+
        #
        # Output of `exploded_df` becomes as below:
        #
        # +-----+---+-----+
        # |index|key|value|
        # +-----+---+-----+
        # |   y1| x1|    1|
        # |   y1| x2|    0|
        # |   y1| x3|    0|
        # |   y2| x1|    0|
        # |   y2| x2|   50|
        # |   y2| x3|    0|
        # |   y3| x1|    3|
        # |   y3| x2|    2|
        # |   y3| x3|    1|
        # +-----+---+-----+
        pairs = F.explode(F.array(*[
            F.struct(
                F.lit(column).alias("key"),
                F.col(column).alias("value")
            ) for column in data_columns]))

        exploded_df = sdf.withColumn("pairs", pairs).select(
            [F.col(index_column), F.col("pairs.key"), F.col("pairs.value")])

        # After that, executes pivot with key and its index column.
        # Note that index column should contain unique values since column names
        # should be unique.
        pivoted_df = exploded_df.groupBy(F.col("key")).pivot(index_column)

        # New index column is always single index.
        internal_index_column = "__index_level_0__"
        transposed_df = pivoted_df.agg(
            F.first(F.col("value"))).withColumnRenamed("key", internal_index_column)

        new_data_columns = filter(lambda x: x != internal_index_column, transposed_df.columns)

        internal = self._internal.copy(
            sdf=transposed_df,
            data_columns=list(new_data_columns),
            index_map=[(internal_index_column, None)])

        return DataFrame(internal)

    T = property(transpose)

    def transform(self, func):
        """
        Call ``func`` on self producing a Series with transformed values
        and that has the same length as its input.

        .. note:: unlike pandas, it is required for ``func`` to specify return type hint.

        .. note:: the series within ``func`` is actually a pandas series, and
            the length of each series is not guaranteed.

        Parameters
        ----------
        func : function
            Function to use for transforming the data. It must work when pandas Series
            is passed.

        Returns
        -------
        DataFrame
            A DataFrame that must have the same length as self.

        Raises
        ------
        Exception : If the returned DataFrame has a different length than self.

        Examples
        --------
        >>> df = ks.DataFrame({'A': range(3), 'B': range(1, 4)})
        >>> df
           A  B
        0  0  1
        1  1  2
        2  2  3

        >>> def square(x) -> ks.Series[np.int32]:
        ...     return x ** 2
        >>> df.transform(square)
           A  B
        0  0  1
        1  1  4
        2  4  9
        """
        assert callable(func), "the first argument should be a callable function."
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        if return_sig is None:
            raise ValueError("Given function must have return type hint; however, not found.")

        wrapped = ks.pandas_wraps(func)
        applied = []
        for column in self._internal.data_columns:
            applied.append(wrapped(self[column]).rename(column))

        sdf = self._sdf.select(
            self._internal.index_columns + [c._scol for c in applied])
        internal = self._internal.copy(sdf=sdf)

        return DataFrame(internal)

    @property
    def index(self):
        """The index (row labels) Column of the DataFrame.

        Currently not supported when the DataFrame has no index.

        See Also
        --------
        Index
        """
        from databricks.koalas.indexes import Index, MultiIndex
        if len(self._internal.index_map) == 0:
            return None
        elif len(self._internal.index_map) == 1:
            return Index(self)
        else:
            return MultiIndex(self)

    @property
    def empty(self):
        """
        Returns true if the current DataFrame is empty. Otherwise, returns false.

        Examples
        --------
        >>> ks.range(10).empty
        False

        >>> ks.range(0).empty
        True

        >>> ks.DataFrame({}, index=list('abc')).empty
        True
        """
        return len(self._internal.data_columns) == 0 or self._sdf.rdd.isEmpty()

    def set_index(self, keys, drop=True, append=False, inplace=False):
        """Set the DataFrame index (row labels) using one or more existing columns.

        Set the DataFrame index (row labels) using one or more existing
        columns or arrays (of the correct length). The index can replace the
        existing index or expand on it.

        Parameters
        ----------
        keys : label or array-like or list of labels/arrays
            This parameter can be either a single column key, a single array of
            the same length as the calling DataFrame, or a list containing an
            arbitrary combination of column keys and arrays. Here, "array"
            encompasses :class:`Series`, :class:`Index` and ``np.ndarray``.
        drop : bool, default True
            Delete columns to be used as the new index.
        append : bool, default False
            Whether to append columns to existing index.
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).

        Returns
        -------
        DataFrame
            Changed row labels.

        See Also
        --------
        DataFrame.reset_index : Opposite of set_index.

        Examples
        --------
        >>> df = ks.DataFrame({'month': [1, 4, 7, 10],
        ...                    'year': [2012, 2014, 2013, 2014],
        ...                    'sale': [55, 40, 84, 31]},
        ...                   columns=['month', 'year', 'sale'])
        >>> df
           month  year  sale
        0      1  2012    55
        1      4  2014    40
        2      7  2013    84
        3     10  2014    31

        Set the index to become the 'month' column:

        >>> df.set_index('month')  # doctest: +NORMALIZE_WHITESPACE
               year  sale
        month
        1      2012    55
        4      2014    40
        7      2013    84
        10     2014    31

        Create a MultiIndex using columns 'year' and 'month':

        >>> df.set_index(['year', 'month'])  # doctest: +NORMALIZE_WHITESPACE
                    sale
        year  month
        2012  1     55
        2014  4     40
        2013  7     84
        2014  10    31
        """
        if isinstance(keys, str):
            keys = [keys]
        else:
            keys = list(keys)
        for key in keys:
            if key not in self.columns:
                raise KeyError(key)

        if drop:
            data_columns = [column for column in self._internal.data_columns if column not in keys]
        else:
            data_columns = self._internal.data_columns
        if append:
            index_map = self._internal.index_map + [(column, column) for column in keys]
        else:
            index_map = [(column, column) for column in keys]

        index_columns = set(column for column, _ in index_map)
        columns = [column for column, _ in index_map] + \
                  [column for column in data_columns if column not in index_columns]

        # Sync Spark's columns as well.
        sdf = self._sdf.select(['`{}`'.format(name) for name in columns])

        internal = _InternalFrame(sdf=sdf, index_map=index_map, data_columns=data_columns)

        if inplace:
            self._internal = internal
        else:
            return DataFrame(internal)

    def reset_index(self, level=None, drop=False, inplace=False):
        """Reset the index, or a level of it.

        For DataFrame with multi-level index, return new DataFrame with labeling information in
        the columns under the index names, defaulting to 'level_0', 'level_1', etc. if any are None.
        For a standard index, the index name will be used (if set), otherwise a default 'index' or
        'level_0' (if 'index' is already taken) will be used.

        Parameters
        ----------
        level : int, str, tuple, or list, default None
            Only remove the given levels from the index. Removes all levels by
            default.
        drop : bool, default False
            Do not try to insert index into dataframe columns. This resets
            the index to the default integer index.
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).

        Returns
        -------
        DataFrame
            DataFrame with the new index.

        See Also
        --------
        DataFrame.set_index : Opposite of reset_index.

        Examples
        --------
        >>> df = ks.DataFrame([('bird', 389.0),
        ...                    ('bird', 24.0),
        ...                    ('mammal', 80.5),
        ...                    ('mammal', np.nan)],
        ...                   index=['falcon', 'parrot', 'lion', 'monkey'],
        ...                   columns=('class', 'max_speed'))
        >>> df
                 class  max_speed
        falcon    bird      389.0
        parrot    bird       24.0
        lion    mammal       80.5
        monkey  mammal        NaN

        When we reset the index, the old index is added as a column. Unlike pandas, Koalas
        does not automatically add a sequential index. The following 0, 1, 2, 3 are only
        there when we display the DataFrame.

        >>> df.reset_index()
            index   class  max_speed
        0  falcon    bird      389.0
        1  parrot    bird       24.0
        2    lion  mammal       80.5
        3  monkey  mammal        NaN

        We can use the `drop` parameter to avoid the old index being added as
        a column:

        >>> df.reset_index(drop=True)
            class  max_speed
        0    bird      389.0
        1    bird       24.0
        2  mammal       80.5
        3  mammal        NaN
        """
        # TODO: add example of MultiIndex back. See https://github.com/databricks/koalas/issues/301
        if len(self._internal.index_map) == 0:
            raise NotImplementedError('Can\'t reset index because there is no index.')

        multi_index = len(self._internal.index_map) > 1

        def rename(index):
            if multi_index:
                return 'level_{}'.format(index)
            else:
                if 'index' not in self._internal.data_columns:
                    return 'index'
                else:
                    return 'level_{}'.format(index)

        if level is None:
            new_index_map = [(column, name if name is not None else rename(i))
                             for i, (column, name) in enumerate(self._internal.index_map)]
            index_map = []
        else:
            if isinstance(level, (int, str)):
                level = [level]
            level = list(level)

            if all(isinstance(l, int) for l in level):
                for lev in level:
                    if lev >= len(self._internal.index_map):
                        raise IndexError('Too many levels: Index has only {} level, not {}'
                                         .format(len(self._internal.index_map), lev + 1))
                idx = level
            elif all(isinstance(lev, str) for lev in level):
                idx = []
                for l in level:
                    try:
                        i = self._internal.index_columns.index(l)
                        idx.append(i)
                    except ValueError:
                        if multi_index:
                            raise KeyError('Level unknown not found')
                        else:
                            raise KeyError('Level unknown must be same as name ({})'
                                           .format(self._internal.index_columns[0]))
            else:
                raise ValueError('Level should be all int or all string.')
            idx.sort()

            new_index_map = []
            index_map = self._internal.index_map.copy()
            for i in idx:
                info = self._internal.index_map[i]
                index_column, index_name = info
                new_index_map.append(
                    (index_column,
                     index_name if index_name is not None else rename(index_name)))
                index_map.remove(info)

        if drop:
            new_index_map = []

        internal = self._internal.copy(
            data_columns=[column for column, _ in new_index_map] + self._internal.data_columns,
            index_map=index_map)
        columns = [name for _, name in new_index_map] + self._internal.data_columns
        if inplace:
            self._internal = internal
            self.columns = columns
        else:
            kdf = DataFrame(internal)
            kdf.columns = columns
            return kdf

    def isnull(self):
        """
        Detects missing values for items in the current Dataframe.

        Return a boolean same-sized Dataframe indicating if the values are NA.
        NA values, such as None or numpy.NaN, gets mapped to True values.
        Everything else gets mapped to False values.

        See Also
        --------
        Dataframe.notnull

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, None), (.6, None), (.2, .1)])
        >>> df.isnull()
               0      1
        0  False  False
        1  False   True
        2  False   True
        3  False  False

        >>> df = ks.DataFrame([[None, 'bee', None], ['dog', None, 'fly']])
        >>> df.isnull()
               0      1      2
        0   True  False   True
        1  False   True  False
        """
        kdf = self.copy()
        for name, ks in kdf.iteritems():
            kdf[name] = ks.isnull()
        return kdf

    isna = isnull

    def notnull(self):
        """
        Detects non-missing values for items in the current Dataframe.

        This function takes a dataframe and indicates whether it's
        values are valid (not missing, which is ``NaN`` in numeric
        datatypes, ``None`` or ``NaN`` in objects and ``NaT`` in datetimelike).

        See Also
        --------
        Dataframe.isnull

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, None), (.6, None), (.2, .1)])
        >>> df.notnull()
              0      1
        0  True   True
        1  True  False
        2  True  False
        3  True   True

        >>> df = ks.DataFrame([['ant', 'bee', 'cat'], ['dog', None, 'fly']])
        >>> df.notnull()
              0      1     2
        0  True   True  True
        1  True  False  True
        """
        kdf = self.copy()
        for name, ks in kdf.iteritems():
            kdf[name] = ks.notnull()
        return kdf

    notna = notnull

    # TODO: add frep and axis parameter
    def shift(self, periods=1, fill_value=None):
        """
        Shift DataFrame by desired number of periods.

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
        Copy of input DataFrame, shifted.

        Examples
        --------
        >>> df = ks.DataFrame({'Col1': [10, 20, 15, 30, 45],
        ...                    'Col2': [13, 23, 18, 33, 48],
        ...                    'Col3': [17, 27, 22, 37, 52]},
        ...                   columns=['Col1', 'Col2', 'Col3'])

        >>> df.shift(periods=3)
           Col1  Col2  Col3
        0   NaN   NaN   NaN
        1   NaN   NaN   NaN
        2   NaN   NaN   NaN
        3  10.0  13.0  17.0
        4  20.0  23.0  27.0

        >>> df.shift(periods=3, fill_value=0)
           Col1  Col2  Col3
        0     0     0     0
        1     0     0     0
        2     0     0     0
        3    10    13    17
        4    20    23    27

        """
        applied = []
        for column in self._internal.data_columns:
            applied.append(self[column].shift(periods, fill_value))

        sdf = self._sdf.select(
            self._internal.index_columns + [c._scol for c in applied])
        internal = self._internal.copy(sdf=sdf, data_columns=[c.name for c in applied])
        return DataFrame(internal)

    # TODO: add axis parameter
    def diff(self, periods=1):
        """
        First discrete difference of element.

        Calculates the difference of a DataFrame element compared with another element in the
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

        >>> df.diff()
             a    b     c
        0  NaN  NaN   NaN
        1  1.0  0.0   3.0
        2  1.0  1.0   5.0
        3  1.0  1.0   7.0
        4  1.0  2.0   9.0
        5  1.0  3.0  11.0

        Difference with previous column

        >>> df.diff(periods=3)
             a    b     c
        0  NaN  NaN   NaN
        1  NaN  NaN   NaN
        2  NaN  NaN   NaN
        3  3.0  2.0  15.0
        4  3.0  4.0  21.0
        5  3.0  6.0  27.0

        Difference with following row

        >>> df.diff(periods=-1)
             a    b     c
        0 -1.0  0.0  -3.0
        1 -1.0 -1.0  -5.0
        2 -1.0 -1.0  -7.0
        3 -1.0 -2.0  -9.0
        4 -1.0 -3.0 -11.0
        5  NaN  NaN   NaN
        """
        applied = []
        for column in self._internal.data_columns:
            applied.append(self[column].diff(periods))
        sdf = self._sdf.select(
            self._internal.index_columns + [c._scol for c in applied])
        internal = self._internal.copy(sdf=sdf, data_columns=[c.name for c in applied])
        return DataFrame(internal)

    def nunique(self, axis: int = 0, dropna: bool = True, approx: bool = False,
                rsd: float = 0.05) -> pd.Series:
        """
        Return number of unique elements in the object.

        Excludes NA values by default.

        Parameters
        ----------
        axis : int, default 0
            Can only be set to 0 at the moment.
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
        The number of unique values per column as a pandas Series.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 3], 'B': [np.nan, 3, np.nan]})
        >>> df.nunique()
        A    3
        B    1
        Name: 0, dtype: int64

        >>> df.nunique(dropna=False)
        A    3
        B    2
        Name: 0, dtype: int64

        On big data, we recommend using the approximate algorithm to speed up this function.
        The result will be very close to the exact unique count.

        >>> df.nunique(approx=True)
        A    3
        B    1
        Name: 0, dtype: int64
        """
        if axis != 0:
            raise ValueError("The 'nunique' method only works with axis=0 at the moment")
        count_fn = partial(F.approx_count_distinct, rsd=rsd) if approx else F.countDistinct
        if dropna:
            res = self._sdf.select([count_fn(Column(c))
                                   .alias(c)
                                    for c in self.columns])
        else:
            res = self._sdf.select([(count_fn(Column(c))
                                     # If the count of null values in a column is at least 1,
                                     # increase the total count by 1 else 0. This is like adding
                                     # self.isnull().sum().clip(upper=1) but can be computed in a
                                     # single Spark job when pulling it into the select statement.
                                     + F.when(F.count(F.when(F.col(c).isNull(), 1).otherwise(None))
                                              >= 1, 1).otherwise(0))
                                   .alias(c)
                                    for c in self.columns])
        return res.toPandas().T.iloc[:, 0]

    def round(self, decimals=0):
        """
        Round a DataFrame to a variable number of decimal places.

        Parameters
        ----------
        decimals : int, dict, Series
            Number of decimal places to round each column to. If an int is
            given, round each column to the same number of places.
            Otherwise dict and Series round to variable numbers of places.
            Column names should be in the keys if `decimals` is a
            dict-like, or in the index if `decimals` is a Series. Any
            columns not included in `decimals` will be left as is. Elements
            of `decimals` which are not columns of the input will be
            ignored.

        Returns
        -------
        DataFrame

        See Also
        --------
        Series.round

        Examples
        --------
        >>> df = ks.DataFrame({'A':[0.028208, 0.038683, 0.877076],
        ...                    'B':[0.992815, 0.645646, 0.149370],
        ...                    'C':[0.173891, 0.577595, 0.491027]},
        ...                    columns=['A', 'B', 'C'],
        ...                    index=['first', 'second', 'third'])
        >>> df
                       A         B         C
        first   0.028208  0.992815  0.173891
        second  0.038683  0.645646  0.577595
        third   0.877076  0.149370  0.491027

        >>> df.round(2)
                   A     B     C
        first   0.03  0.99  0.17
        second  0.04  0.65  0.58
        third   0.88  0.15  0.49

        >>> df.round({'A': 1, 'C': 2})
                  A         B     C
        first   0.0  0.992815  0.17
        second  0.0  0.645646  0.58
        third   0.9  0.149370  0.49

        >>> decimals = ks.Series([1, 0, 2], index=['A', 'B', 'C'])
        >>> df.round(decimals)
                  A    B     C
        first   0.0  1.0  0.17
        second  0.0  1.0  0.58
        third   0.9  0.0  0.49
        """
        if isinstance(decimals, ks.Series):
            decimals_list = [kv for kv in decimals.to_pandas().items()]
        elif isinstance(decimals, dict):
            decimals_list = [(k, v) for k, v in decimals.items()]
        elif isinstance(decimals, int):
            decimals_list = [(v, decimals) for v in self._internal.data_columns]
        else:
            raise ValueError("decimals must be an integer, a dict-like or a Series")

        sdf = self._sdf
        for decimal in decimals_list:
            sdf = sdf.withColumn(decimal[0], F.round(decimal[0], decimal[1]))
        return DataFrame(self._internal.copy(sdf=sdf))

    def to_koalas(self):
        """
        Converts the existing DataFrame into a Koalas DataFrame.

        This method is monkey-patched into Spark's DataFrame and can be used
        to convert a Spark DataFrame into a Koalas DataFrame. If running on
        an existing Koalas DataFrame, the method returns itself.

        If a Koalas DataFrame is converted to a Spark DataFrame and then back
        to Koalas, it will lose the index information and the original index
        will be turned into a normal column.

        See Also
        --------
        DataFrame.to_spark

        Examples
        --------
        >>> df = ks.DataFrame({'col1': [1, 2], 'col2': [3, 4]}, columns=['col1', 'col2'])
        >>> df
           col1  col2
        0     1     3
        1     2     4

        >>> spark_df = df.to_spark()
        >>> spark_df
        DataFrame[__index_level_0__: bigint, col1: bigint, col2: bigint]

        >>> kdf = spark_df.to_koalas()
        >>> kdf
           __index_level_0__  col1  col2
        0                  0     1     3
        1                  1     2     4

        Calling to_koalas on a Koalas DataFrame simply returns itself.

        >>> df.to_koalas()
           col1  col2
        0     1     3
        1     2     4
        """
        if isinstance(self, DataFrame):
            return self
        else:
            return DataFrame(self)

    def cache(self):
        """
        Yields and caches the current DataFrame.

        The Koalas DataFrame is yielded as a protected resource and its corresponding
        data is cached which gets uncached after execution goes of the context.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df
           dogs  cats
        0   0.2   0.3
        1   0.0   0.6
        2   0.6   0.0
        3   0.2   0.1

        >>> with df.cache() as cached_df:
        ...     print(cached_df.count())
        ...
        dogs    4
        cats    4
        dtype: int64

        >>> df = df.cache()
        >>> df.to_pandas().mean(axis=1)
        0    0.25
        1    0.30
        2    0.30
        3    0.15
        dtype: float64

        To uncache the dataframe, use `unpersist` function

        >>> df.unpersist()
        """
        return _CachedDataFrame(self._internal)

    def to_table(self, name: str, format: Optional[str] = None, mode: str = 'error',
                 partition_cols: Union[str, List[str], None] = None,
                 **options):
        """
        Write the DataFrame into a Spark table.

        Parameters
        ----------
        name : str, required
            Table name in Spark.
        format : string, optional
            Specifies the output data source format. Some common ones are:

            - 'delta'
            - 'parquet'
            - 'orc'
            - 'json'
            - 'csv'

        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default 'error'.
            Specifies the behavior of the save operation when the table exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        options
            Additional options passed directly to Spark.

        See Also
        --------
        read_table
        DataFrame.to_spark_io
        DataFrame.to_parquet

        Examples
        --------
        >>> df = ks.DataFrame(dict(
        ...    date=list(pd.date_range('2012-1-1 12:00:00', periods=3, freq='M')),
        ...    country=['KR', 'US', 'JP'],
        ...    code=[1, 2 ,3]), columns=['date', 'country', 'code'])
        >>> df
                         date country  code
        0 2012-01-31 12:00:00      KR     1
        1 2012-02-29 12:00:00      US     2
        2 2012-03-31 12:00:00      JP     3

        >>> df.to_table('%s.my_table' % db, partition_cols='date')
        """
        self._sdf.write.saveAsTable(name=name, format=format, mode=mode,
                                    partitionBy=partition_cols, options=options)

    def to_delta(self, path: str, mode: str = 'error',
                 partition_cols: Union[str, List[str], None] = None, **options):
        """
        Write the DataFrame out as a Delta Lake table.

        Parameters
        ----------
        path : str, required
            Path to write to.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default 'error'.
            Specifies the behavior of the save operation when the destination exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        options : dict
            All other options passed directly into Delta Lake.

        See Also
        --------
        read_delta
        DataFrame.to_parquet
        DataFrame.to_table
        DataFrame.to_spark_io

        Examples
        --------

        >>> df = ks.DataFrame(dict(
        ...    date=list(pd.date_range('2012-1-1 12:00:00', periods=3, freq='M')),
        ...    country=['KR', 'US', 'JP'],
        ...    code=[1, 2 ,3]), columns=['date', 'country', 'code'])
        >>> df
                         date country  code
        0 2012-01-31 12:00:00      KR     1
        1 2012-02-29 12:00:00      US     2
        2 2012-03-31 12:00:00      JP     3

        Create a new Delta Lake table, partitioned by one column:

        >>> df.to_delta('%s/to_delta/foo' % path, partition_cols='date')

        Partitioned by two columns:

        >>> df.to_delta('%s/to_delta/bar' % path, partition_cols=['date', 'country'])

        Overwrite an existing table's partitions, using the 'replaceWhere' capability in Delta:

        >>> df.to_delta('%s/to_delta/bar' % path,
        ...             mode='overwrite', replaceWhere='date >= "2019-01-01"')
        """
        self.to_spark_io(
            path=path, mode=mode, format="delta", partition_cols=partition_cols, options=options)

    def to_parquet(self, path: str, mode: str = 'error',
                   partition_cols: Union[str, List[str], None] = None,
                   compression: Optional[str] = None):
        """
        Write the DataFrame out as a Parquet file or directory.

        Parameters
        ----------
        path : str, required
            Path to write to.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default 'error'.
            Specifies the behavior of the save operation when the destination exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        compression : str {'none', 'uncompressed', 'snappy', 'gzip', 'lzo', 'brotli', 'lz4', 'zstd'}
            Compression codec to use when saving to file. If None is set, it uses the
            value specified in `spark.sql.parquet.compression.codec`.

        See Also
        --------
        read_parquet
        DataFrame.to_delta
        DataFrame.to_table
        DataFrame.to_spark_io

        Examples
        --------
        >>> df = ks.DataFrame(dict(
        ...    date=list(pd.date_range('2012-1-1 12:00:00', periods=3, freq='M')),
        ...    country=['KR', 'US', 'JP'],
        ...    code=[1, 2 ,3]), columns=['date', 'country', 'code'])
        >>> df
                         date country  code
        0 2012-01-31 12:00:00      KR     1
        1 2012-02-29 12:00:00      US     2
        2 2012-03-31 12:00:00      JP     3

        >>> df.to_parquet('%s/to_parquet/foo.parquet' % path, partition_cols='date')

        >>> df.to_parquet(
        ...     '%s/to_parquet/foo.parquet' % path,
        ...     mode = 'overwrite',
        ...     partition_cols=['date', 'country'])
        """
        self._sdf.write.parquet(path=path, mode=mode, partitionBy=partition_cols,
                                compression=compression)

    def to_spark_io(self, path: Optional[str] = None, format: Optional[str] = None,
                    mode: str = 'error', partition_cols: Union[str, List[str], None] = None,
                    **options):
        """Write the DataFrame out to a Spark data source.

        Parameters
        ----------
        path : string, optional
            Path to the data source.
        format : string, optional
            Specifies the output data source format. Some common ones are:

            - 'delta'
            - 'parquet'
            - 'orc'
            - 'json'
            - 'csv'
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default 'error'.
            Specifies the behavior of the save operation when data already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.
        partition_cols : str or list of str, optional
            Names of partitioning columns
        options : dict
            All other options passed directly into Spark's data source.

        See Also
        --------
        read_spark_io
        DataFrame.to_delta
        DataFrame.to_parquet
        DataFrame.to_table

        Examples
        --------
        >>> df = ks.DataFrame(dict(
        ...    date=list(pd.date_range('2012-1-1 12:00:00', periods=3, freq='M')),
        ...    country=['KR', 'US', 'JP'],
        ...    code=[1, 2 ,3]), columns=['date', 'country', 'code'])
        >>> df
                         date country  code
        0 2012-01-31 12:00:00      KR     1
        1 2012-02-29 12:00:00      US     2
        2 2012-03-31 12:00:00      JP     3

        >>> df.to_spark_io(path='%s/to_spark_io/foo.json' % path, format='json')
        """
        self._sdf.write.save(path=path, format=format, mode=mode, partitionBy=partition_cols,
                             options=options)

    def to_spark(self):
        """
        Return the current DataFrame as a Spark DataFrame.

        See Also
        --------
        DataFrame.to_koalas
        """
        return self._internal.spark_df

    def to_pandas(self):
        """
        Return a Pandas DataFrame.

        .. note:: This method should only be used if the resulting Pandas DataFrame is expected
            to be small, as all the data is loaded into the driver's memory.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df.to_pandas()
           dogs  cats
        0   0.2   0.3
        1   0.0   0.6
        2   0.6   0.0
        3   0.2   0.1
        """
        return self._internal.pandas_df.copy()

    # Alias to maintain backward compatibility with Spark
    toPandas = to_pandas

    def assign(self, **kwargs):
        """
        Assign new columns to a DataFrame.

        Returns a new object with all original columns in addition to new ones.
        Existing columns that are re-assigned will be overwritten.

        Parameters
        ----------
        **kwargs : dict of {str: callable or Series}
            The column names are keywords. If the values are
            callable, they are computed on the DataFrame and
            assigned to the new columns. The callable must not
            change input DataFrame (though Koalas doesn't check it).
            If the values are not callable, (e.g. a Series or a literal),
            they are simply assigned.

        Returns
        -------
        DataFrame
            A new DataFrame with the new columns in addition to
            all the existing columns.

        Examples
        --------
        >>> df = ks.DataFrame({'temp_c': [17.0, 25.0]},
        ...                   index=['Portland', 'Berkeley'])
        >>> df
                  temp_c
        Portland    17.0
        Berkeley    25.0

        Where the value is a callable, evaluated on `df`:

        >>> df.assign(temp_f=lambda x: x.temp_c * 9 / 5 + 32)
                  temp_c  temp_f
        Portland    17.0    62.6
        Berkeley    25.0    77.0

        Alternatively, the same behavior can be achieved by directly
        referencing an existing Series or sequence and you can also
        create multiple columns within the same assign.

        >>> assigned = df.assign(temp_f=df['temp_c'] * 9 / 5 + 32,
        ...                      temp_k=df['temp_c'] + 273.15)
        >>> assigned[['temp_c', 'temp_f', 'temp_k']]
                  temp_c  temp_f  temp_k
        Portland    17.0    62.6  290.15
        Berkeley    25.0    77.0  298.15

        Notes
        -----
        Assigning multiple columns within the same ``assign`` is possible
        but you cannot refer to newly created or modified columns. This
        feature is supported in pandas for Python 3.6 and later but not in
        Koalas. In Koalas, all items are computed first, and then assigned.
        """
        from databricks.koalas.series import Series
        for k, v in kwargs.items():
            if not (isinstance(v, (Series, spark.Column)) or
                    callable(v) or pd.api.types.is_scalar(v)):
                raise TypeError("Column assignment doesn't support type "
                                "{0}".format(type(v).__name__))
            if callable(v):
                kwargs[k] = v(self)

        pairs = list(kwargs.items())
        sdf = self._sdf
        for (name, c) in pairs:
            if isinstance(c, Series):
                sdf = sdf.withColumn(name, c._scol)
            elif isinstance(c, Column):
                sdf = sdf.withColumn(name, c)
            else:
                sdf = sdf.withColumn(name, F.lit(c))

        data_columns = set(self._internal.data_columns)
        internal = self._internal.copy(
            sdf=sdf,
            data_columns=(self._internal.data_columns +
                          [name for name, _ in pairs if name not in data_columns]))
        return DataFrame(internal)

    @staticmethod
    def from_records(data: Union[np.array, List[tuple], dict, pd.DataFrame],
                     index: Union[str, list, np.array] = None, exclude: list = None,
                     columns: list = None, coerce_float: bool = False, nrows: int = None) \
            -> 'DataFrame':
        """
        Convert structured or record ndarray to DataFrame.

        Parameters
        ----------
        data : ndarray (structured dtype), list of tuples, dict, or DataFrame
        index : string, list of fields, array-like
            Field of array to use as the index, alternately a specific set of input labels to use
        exclude : sequence, default None
            Columns or fields to exclude
        columns : sequence, default None
            Column names to use. If the passed data do not have names associated with them, this
            argument provides names for the columns. Otherwise this argument indicates the order of
            the columns in the result (any names not found in the data will become all-NA columns)
        coerce_float : boolean, default False
            Attempt to convert values of non-string, non-numeric objects (like decimal.Decimal) to
            floating point, useful for SQL result sets
        nrows : int, default None
            Number of rows to read if data is an iterator

        Returns
        -------
        df : DataFrame

        Examples
        --------
        Use dict as input

        >>> ks.DataFrame.from_records({'A': [1, 2, 3]})
           A
        0  1
        1  2
        2  3

        Use list of tuples as input

        >>> ks.DataFrame.from_records([(1, 2), (3, 4)])
           0  1
        0  1  2
        1  3  4

        Use NumPy array as input

        >>> ks.DataFrame.from_records(np.eye(3))
             0    1    2
        0  1.0  0.0  0.0
        1  0.0  1.0  0.0
        2  0.0  0.0  1.0
        """
        return DataFrame(pd.DataFrame.from_records(data, index, exclude, columns, coerce_float,
                                                   nrows))

    def to_records(self, index=True, convert_datetime64=None,
                   column_dtypes=None, index_dtypes=None):
        """
        Convert DataFrame to a NumPy record array.

        Index will be included as the first field of the record array if
        requested.

        .. note:: This method should only be used if the resulting NumPy ndarray is
            expected to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        index : bool, default True
            Include index in resulting record array, stored in 'index'
            field or using the index label, if set.
        convert_datetime64 : bool, default None
            Whether to convert the index to datetime.datetime if it is a
            DatetimeIndex.
        column_dtypes : str, type, dict, default None
            If a string or type, the data type to store all columns. If
            a dictionary, a mapping of column names and indices (zero-indexed)
            to specific data types.
        index_dtypes : str, type, dict, default None
            If a string or type, the data type to store all index levels. If
            a dictionary, a mapping of index level names and indices
            (zero-indexed) to specific data types.
            This mapping is applied only if `index=True`.

        Returns
        -------
        numpy.recarray
            NumPy ndarray with the DataFrame labels as fields and each row
            of the DataFrame as entries.

        See Also
        --------
        DataFrame.from_records: Convert structured or record ndarray
            to DataFrame.
        numpy.recarray: An ndarray that allows field access using
            attributes, analogous to typed columns in a
            spreadsheet.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2], 'B': [0.5, 0.75]},
        ...                   index=['a', 'b'])
        >>> df
           A     B
        a  1  0.50
        b  2  0.75

        >>> df.to_records() # doctest: +SKIP
        rec.array([('a', 1, 0.5 ), ('b', 2, 0.75)],
                  dtype=[('index', 'O'), ('A', '<i8'), ('B', '<f8')])

        The index can be excluded from the record array:

        >>> df.to_records(index=False) # doctest: +SKIP
        rec.array([(1, 0.5 ), (2, 0.75)],
                  dtype=[('A', '<i8'), ('B', '<f8')])

        Specification of dtype for columns is new in Pandas 0.24.0.
        Data types can be specified for the columns:

        >>> df.to_records(column_dtypes={"A": "int32"}) # doctest: +SKIP
        rec.array([('a', 1, 0.5 ), ('b', 2, 0.75)],
                  dtype=[('index', 'O'), ('A', '<i4'), ('B', '<f8')])

        Specification of dtype for index is new in Pandas 0.24.0.
        Data types can also be specified for the index:

        >>> df.to_records(index_dtypes="<S2") # doctest: +SKIP
        rec.array([(b'a', 1, 0.5 ), (b'b', 2, 0.75)],
                  dtype=[('index', 'S2'), ('A', '<i8'), ('B', '<f8')])
        """
        args = locals()
        kdf = self

        return validate_arguments_and_invoke_function(
            kdf.to_pandas(), self.to_records, pd.DataFrame.to_records, args)

    def copy(self) -> 'DataFrame':
        """
        Make a copy of this object's indices and data.

        Returns
        -------
        copy : DataFrame
        """
        return DataFrame(self._internal.copy())

    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
        """
        Remove missing values.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Determine if rows or columns which contain missing values are
            removed.

            * 0, or 'index' : Drop rows which contain missing values.
        how : {'any', 'all'}, default 'any'
            Determine if row or column is removed from DataFrame, when we have
            at least one NA or all NA.

            * 'any' : If any NA values are present, drop that row or column.
            * 'all' : If all values are NA, drop that row or column.

        thresh : int, optional
            Require that many non-NA values.
        subset : array-like, optional
            Labels along other axis to consider, e.g. if you are dropping rows
            these would be a list of columns to include.
        inplace : bool, default False
            If True, do operation inplace and return None.

        Returns
        -------
        DataFrame
            DataFrame with NA entries dropped from it.

        See Also
        --------
        DataFrame.drop : Drop specified labels from columns.
        DataFrame.isnull: Indicate missing values.
        DataFrame.notnull : Indicate existing (non-missing) values.

        Examples
        --------
        >>> df = ks.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
        ...                    "toy": [None, 'Batmobile', 'Bullwhip'],
        ...                    "born": [None, "1940-04-25", None]},
        ...                   columns=['name', 'toy', 'born'])
        >>> df
               name        toy        born
        0    Alfred       None        None
        1    Batman  Batmobile  1940-04-25
        2  Catwoman   Bullwhip        None

        Drop the rows where at least one element is missing.

        >>> df.dropna()
             name        toy        born
        1  Batman  Batmobile  1940-04-25

        Drop the rows where all elements are missing.

        >>> df.dropna(how='all')
               name        toy        born
        0    Alfred       None        None
        1    Batman  Batmobile  1940-04-25
        2  Catwoman   Bullwhip        None

        Keep only the rows with at least 2 non-NA values.

        >>> df.dropna(thresh=2)
               name        toy        born
        1    Batman  Batmobile  1940-04-25
        2  Catwoman   Bullwhip        None

        Define in which columns to look for missing values.

        >>> df.dropna(subset=['name', 'born'])
             name        toy        born
        1  Batman  Batmobile  1940-04-25

        Keep the DataFrame with valid entries in the same variable.

        >>> df.dropna(inplace=True)
        >>> df
             name        toy        born
        1  Batman  Batmobile  1940-04-25
        """
        if axis == 0 or axis == 'index':
            if subset is not None:
                if isinstance(subset, str):
                    columns = [subset]
                else:
                    columns = list(subset)
                invalids = [column for column in columns
                            if column not in self._internal.data_columns]
                if len(invalids) > 0:
                    raise KeyError(invalids)
            else:
                columns = list(self.columns)

            cnt = reduce(lambda x, y: x + y,
                         [F.when(self[column].notna()._scol, 1).otherwise(0)
                          for column in columns],
                         F.lit(0))
            if thresh is not None:
                pred = cnt >= F.lit(int(thresh))
            elif how == 'any':
                pred = cnt == F.lit(len(columns))
            elif how == 'all':
                pred = cnt > F.lit(0)
            else:
                if how is not None:
                    raise ValueError('invalid how option: {h}'.format(h=how))
                else:
                    raise TypeError('must specify how or thresh')

            sdf = self._sdf.filter(pred)
            internal = self._internal.copy(sdf=sdf)
            if inplace:
                self._internal = internal
            else:
                return DataFrame(internal)

        else:
            raise NotImplementedError("dropna currently only works for axis=0 or axis='index'")

    def fillna(self, value=None, axis=None, inplace=False):
        """Fill NA/NaN values.

        Parameters
        ----------
        value : scalar, dict, Series
            Value to use to fill holes. alternately a dict/Series of values
            specifying which value to use for each column.
            DataFrame is not supported.
        axis : {0 or `index`}
            1 and `columns` are not supported.
        inplace : boolean, default False
            Fill in place (do not create a new object)

        Returns
        -------
        DataFrame
            DataFrame with NA entries filled.

        Examples
        --------
        >>> df = ks.DataFrame({
        ...     'A': [None, 3, None, None],
        ...     'B': [2, 4, None, 3],
        ...     'C': [None, None, None, 1],
        ...     'D': [0, 1, 5, 4]
        ...     },
        ...     columns=['A', 'B', 'C', 'D'])
        >>> df
             A    B    C  D
        0  NaN  2.0  NaN  0
        1  3.0  4.0  NaN  1
        2  NaN  NaN  NaN  5
        3  NaN  3.0  1.0  4

        Replace all NaN elements with 0s.

        >>> df.fillna(0)
             A    B    C  D
        0  0.0  2.0  0.0  0
        1  3.0  4.0  0.0  1
        2  0.0  0.0  0.0  5
        3  0.0  3.0  1.0  4

        Replace all NaN elements in column 'A', 'B', 'C', and 'D', with 0, 1,
        2, and 3 respectively.

        >>> values = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        >>> df.fillna(value=values)
             A    B    C  D
        0  0.0  2.0  2.0  0
        1  3.0  4.0  2.0  1
        2  0.0  1.0  2.0  5
        3  0.0  3.0  1.0  4
        """
        if axis is None:
            axis = 0
        if not (axis == 0 or axis == "index"):
            raise NotImplementedError("fillna currently only works for axis=0 or axis='index'")

        if value is None:
            raise ValueError('Currently must specify value')
        if not isinstance(value, (float, int, str, bool, dict, pd.Series)):
            raise TypeError("Unsupported type %s" % type(value))
        if isinstance(value, pd.Series):
            value = value.to_dict()
        if isinstance(value, dict):
            for v in value.values():
                if not isinstance(v, (float, int, str, bool)):
                    raise TypeError("Unsupported type %s" % type(v))

        sdf = self._sdf.fillna(value)
        internal = self._internal.copy(sdf=sdf)
        if inplace:
            self._internal = internal
        else:
            return DataFrame(internal)

    def replace(self, to_replace=None, value=None, subset=None, inplace=False,
                limit=None, regex=False, method='pad'):
        """
        Returns a new DataFrame replacing a value with another value.

        Parameters
        ----------
        to_replace : int, float, string, or list
            Value to be replaced. If the value is a dict, then value is ignored and
            to_replace must be a mapping from column name (string) to replacement value.
            The value to be replaced must be an int, float, or string.
        value : int, float, string, or list
            Value to use to replace holes. The replacement value must be an int, float,
            or string. If value is a list, value should be of the same length with to_replace.
        subset : string, list
            Optional list of column names to consider. Columns specified in subset that
            do not have matching data type are ignored. For example, if value is a string,
            and subset contains a non-string column, then the non-string column is simply ignored.
        inplace : boolean, default False
            Fill in place (do not create a new object)

        Returns
        -------
        DataFrame
            Object after replacement.

        Examples
        --------
        >>> df = ks.DataFrame({"name": ['Ironman', 'Captain America', 'Thor', 'Hulk'],
        ...                    "weapon": ['Mark-45', 'Shield', 'Mjolnir', 'Smash']},
        ...                   columns=['name', 'weapon'])
        >>> df
                      name   weapon
        0          Ironman  Mark-45
        1  Captain America   Shield
        2             Thor  Mjolnir
        3             Hulk    Smash

        Scalar `to_replace` and `value`

        >>> df.replace('Ironman', 'War-Machine')
                      name   weapon
        0      War-Machine  Mark-45
        1  Captain America   Shield
        2             Thor  Mjolnir
        3             Hulk    Smash

        List like `to_replace` and `value`

        >>> df.replace(['Ironman', 'Captain America'], ['Rescue', 'Hawkeye'], inplace=True)
        >>> df
              name   weapon
        0   Rescue  Mark-45
        1  Hawkeye   Shield
        2     Thor  Mjolnir
        3     Hulk    Smash

        Replacing value by specifying column

        >>> df.replace('Mjolnir', 'Stormbuster', subset='weapon')
              name       weapon
        0   Rescue      Mark-45
        1  Hawkeye       Shield
        2     Thor  Stormbuster
        3     Hulk        Smash

        Dict like `to_replace`

        >>> df = ks.DataFrame({'A': [0, 1, 2, 3, 4],
        ...                    'B': [5, 6, 7, 8, 9],
        ...                    'C': ['a', 'b', 'c', 'd', 'e']},
        ...                   columns=['A', 'B', 'C'])

        >>> df.replace({'A': {0: 100, 4: 400}})
             A  B  C
        0  100  5  a
        1    1  6  b
        2    2  7  c
        3    3  8  d
        4  400  9  e

        >>> df.replace({'A': 0, 'B': 5}, 100)
             A    B  C
        0  100  100  a
        1    1    6  b
        2    2    7  c
        3    3    8  d
        4    4    9  e

        Notes
        -----
        One difference between this implementation and pandas is that it is necessary
        to specify the column name when you are passing dictionary in `to_replace`
        parameter. Calling `replace` on its index such as `df.replace({0: 10, 1: 100})` will
        throw an error. Instead specify column-name like `df.replace({'A': {0: 10, 1: 100}})`.
        """
        if method != 'pad':
            raise NotImplementedError("replace currently works only for method='pad")
        if limit is not None:
            raise NotImplementedError("replace currently works only when limit=None")
        if regex is not False:
            raise NotImplementedError("replace currently doesn't supports regex")

        if value is not None and not isinstance(value, (int, float, str, list, dict)):
            raise TypeError("Unsupported type {}".format(type(value)))
        if to_replace is not None and not isinstance(to_replace, (int, float, str, list, dict)):
            raise TypeError("Unsupported type {}".format(type(to_replace)))

        if isinstance(value, list) and isinstance(to_replace, list):
            if len(value) != len(to_replace):
                raise ValueError('Length of to_replace and value must be same')

        sdf = self._sdf.select(self._internal.data_columns)
        if isinstance(to_replace, dict) and value is None and \
                (not any(isinstance(i, dict) for i in to_replace.values())):
            sdf = sdf.replace(to_replace, value, subset)
        elif isinstance(to_replace, dict):
            for df_column, replacement in to_replace.items():
                if isinstance(replacement, dict):
                    sdf = sdf.replace(replacement, subset=df_column)
                else:
                    sdf = sdf.withColumn(df_column, F.when(F.col(df_column) == replacement, value)
                                         .otherwise(F.col(df_column)))

        else:
            sdf = sdf.replace(to_replace, value, subset)

        kdf = DataFrame(sdf)
        if inplace:
            self._internal = kdf._internal
        else:
            return kdf

    def clip(self, lower: Union[float, int] = None, upper: Union[float, int] = None) \
            -> 'DataFrame':
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
        DataFrame
            DataFrame with the values outside the clip boundaries replaced.

        Examples
        --------
        >>> ks.DataFrame({'A': [0, 2, 4]}).clip(1, 3)
           A
        0  1
        1  2
        2  3

        Notes
        -----
        One difference between this implementation and pandas is that running
        pd.DataFrame({'A': ['a', 'b']}).clip(0, 1) will crash with "TypeError: '<=' not supported
        between instances of 'str' and 'int'" while ks.DataFrame({'A': ['a', 'b']}).clip(0, 1)
        will output the original DataFrame, simply ignoring the incompatible types.
        """
        if is_list_like(lower) or is_list_like(upper):
            raise ValueError("List-like value are not supported for 'lower' and 'upper' at the " +
                             "moment")

        if lower is None and upper is None:
            return self

        sdf = self._sdf

        numeric_types = (DecimalType, DoubleType, FloatType, ByteType, IntegerType, LongType,
                         ShortType)
        numeric_columns = [c for c in self.columns
                           if isinstance(sdf.schema[c].dataType, numeric_types)]
        nonnumeric_columns = [c for c in self.columns
                              if not isinstance(sdf.schema[c].dataType, numeric_types)]

        if lower is not None:
            sdf = sdf.select(*[F.when(F.col(c) < lower, lower).otherwise(F.col(c)).alias(c)
                               for c in numeric_columns] + nonnumeric_columns)
        if upper is not None:
            sdf = sdf.select(*[F.when(F.col(c) > upper, upper).otherwise(F.col(c)).alias(c)
                               for c in numeric_columns] + nonnumeric_columns)

        # Restore initial column order
        sdf = sdf.select(list(self.columns))

        return ks.DataFrame(sdf)

    def head(self, n=5):
        """
        Return the first `n` rows.

        This function returns the first `n` rows for the object based
        on position. It is useful for quickly testing if your object
        has the right type of data in it.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        obj_head : same type as caller
            The first `n` rows of the caller object.

        Examples
        --------
        >>> df = ks.DataFrame({'animal':['alligator', 'bee', 'falcon', 'lion',
        ...                    'monkey', 'parrot', 'shark', 'whale', 'zebra']})
        >>> df
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey
        5     parrot
        6      shark
        7      whale
        8      zebra

        Viewing the first 5 lines

        >>> df.head()
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey

        Viewing the first `n` lines (three in this case)

        >>> df.head(3)
              animal
        0  alligator
        1        bee
        2     falcon
        """

        return DataFrame(self._internal.copy(sdf=self._sdf.limit(n)))

    def pivot_table(self, values=None, index=None, columns=None,
                    aggfunc='mean', fill_value=None):
        """
        Create a spreadsheet-style pivot table as a DataFrame. The levels in
        the pivot table will be stored in MultiIndex objects (hierarchical
        indexes) on the index and columns of the result DataFrame.

        Parameters
        ----------
        values : column to aggregate.
            They should be either a list of one column or a string. A list of columns
            is not supported yet.
        index : column (string) or list of columns
            If an array is passed, it must be the same length as the data.
            The list should contain string.
        columns : column
            Columns used in the pivot operation. Only one column is supported and
            it should be a string.
        aggfunc : function (string), dict, default mean
            If dict is passed, the resulting pivot table will have
            columns concatenated by "_" where the first part is the value
            of columns and the second part is the column name in values
            If dict is passed, the key is column to aggregate and value
            is function or list of functions.
        fill_value : scalar, default None
            Value to replace missing values with.

        Returns
        -------
        table : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
        ...                          "bar", "bar", "bar", "bar"],
        ...                    "B": ["one", "one", "one", "two", "two",
        ...                          "one", "one", "two", "two"],
        ...                    "C": ["small", "large", "large", "small",
        ...                          "small", "large", "small", "small",
        ...                          "large"],
        ...                    "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
        ...                    "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]},
        ...                   columns=['A', 'B', 'C', 'D', 'E'])
        >>> df
             A    B      C  D  E
        0  foo  one  small  1  2
        1  foo  one  large  2  4
        2  foo  one  large  2  5
        3  foo  two  small  3  5
        4  foo  two  small  3  6
        5  bar  one  large  4  6
        6  bar  one  small  5  8
        7  bar  two  small  6  9
        8  bar  two  large  7  9

        This first example aggregates values by taking the sum.

        >>> table = df.pivot_table(values='D', index=['A', 'B'],
        ...                         columns='C', aggfunc='sum')
        >>> table  # doctest: +NORMALIZE_WHITESPACE
                 large  small
        A   B
        foo one    4.0      1
            two    NaN      6
        bar two    7.0      6
            one    4.0      5

        We can also fill missing values using the `fill_value` parameter.

        >>> table = df.pivot_table(values='D', index=['A', 'B'],
        ...                         columns='C', aggfunc='sum', fill_value=0)
        >>> table  # doctest: +NORMALIZE_WHITESPACE
                 large  small
        A   B
        foo one      4      1
            two      0      6
        bar two      7      6
            one      4      5

        We can also calculate multiple types of aggregations for any given
        value column.

        >>> table = df.pivot_table(values = ['D'], index =['C'],
        ...                         columns="A", aggfunc={'D':'mean'})
        >>> table  # doctest: +NORMALIZE_WHITESPACE
               bar       foo
        C
        small  5.5  2.333333
        large  5.5  2.000000
        """
        if not isinstance(columns, str):
            raise ValueError("columns should be string.")

        if not isinstance(values, str) and not isinstance(values, list):
            raise ValueError('values should be string or list of one column.')

        if not isinstance(aggfunc, str) and (not isinstance(aggfunc, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in aggfunc.items())):
            raise ValueError("aggfunc must be a dict mapping from column name (string) "
                             "to aggregate functions (string).")

        if isinstance(aggfunc, dict) and index is None:
            raise NotImplementedError("pivot_table doesn't support aggfunc"
                                      " as dict and without index.")

        if isinstance(values, list) and len(values) > 1:
            raise NotImplementedError('Values as list of columns is not implemented yet.')

        if isinstance(aggfunc, str):
            agg_cols = [F.expr('{1}({0}) as {0}'.format(values, aggfunc))]

        elif isinstance(aggfunc, dict):
            agg_cols = [F.expr('{1}({0}) as {0}'.format(key, value))
                        for key, value in aggfunc.items()]
            agg_columns = [key for key, value in aggfunc.items()]

            if set(agg_columns) != set(values):
                raise ValueError("Columns in aggfunc must be the same as values.")

        if index is None:
            sdf = self._sdf.groupBy().pivot(pivot_col=columns).agg(*agg_cols)

        elif isinstance(index, list):
            sdf = self._sdf.groupBy(index).pivot(pivot_col=columns).agg(*agg_cols)
        else:
            raise ValueError("index should be a None or a list of columns.")

        if fill_value is not None and isinstance(fill_value, (int, float)):
            sdf = sdf.fillna(fill_value)

        if index is not None:
            return DataFrame(sdf).set_index(index)
        else:
            if isinstance(values, list):
                index_values = values[-1]
            else:
                index_values = values

            return DataFrame(sdf.withColumn(columns, F.lit(index_values))).set_index(columns)

    def pivot(self, index=None, columns=None, values=None):
        """
        Return reshaped DataFrame organized by given index / column values.

        Reshape data (produce a "pivot" table) based on column values. Uses
        unique values from specified `index` / `columns` to form axes of the
        resulting DataFrame. This function does not support data
        aggregation.

        Parameters
        ----------
        index : string, optional
            Column to use to make new frame's index. If None, uses
            existing index.
        columns : string
            Column to use to make new frame's columns.
        values : string, object or a list of the previous
            Column(s) to use for populating new frame's values.

        Returns
        -------
        DataFrame
            Returns reshaped DataFrame.

        See Also
        --------
        DataFrame.pivot_table : Generalization of pivot that can handle
            duplicate values for one index/column pair.

        Examples
        --------
        >>> df = ks.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
        ...                            'two'],
        ...                    'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
        ...                    'baz': [1, 2, 3, 4, 5, 6],
        ...                    'zoo': ['x', 'y', 'z', 'q', 'w', 't']},
        ...                   columns=['foo', 'bar', 'baz', 'zoo'])
        >>> df
           foo bar  baz zoo
        0  one   A    1   x
        1  one   B    2   y
        2  one   C    3   z
        3  two   A    4   q
        4  two   B    5   w
        5  two   C    6   t

        >>> df.pivot(index='foo', columns='bar', values='baz').sort_index()
        ... # doctest: +NORMALIZE_WHITESPACE
             A  B  C
        foo
        one  1  2  3
        two  4  5  6

        >>> df.pivot(columns='bar', values='baz').sort_index()
        ... # doctest: +NORMALIZE_WHITESPACE
             A    B    C
        0  1.0  NaN  NaN
        1  NaN  2.0  NaN
        2  NaN  NaN  3.0
        3  4.0  NaN  NaN
        4  NaN  5.0  NaN
        5  NaN  NaN  6.0

        Notice that, unlike pandas raises an ValueError when duplicated values are found,
        Koalas' pivot still works with its first value it meets during operation because pivot
        is an expensive operation and it is preferred to permissively execute over failing fast
        when processing large data.

        >>> df = ks.DataFrame({"foo": ['one', 'one', 'two', 'two'],
        ...                    "bar": ['A', 'A', 'B', 'C'],
        ...                    "baz": [1, 2, 3, 4]}, columns=['foo', 'bar', 'baz'])
        >>> df
           foo bar  baz
        0  one   A    1
        1  one   A    2
        2  two   B    3
        3  two   C    4

        >>> df.pivot(index='foo', columns='bar', values='baz').sort_index()
        ... # doctest: +NORMALIZE_WHITESPACE
               A    B    C
        foo
        one  1.0  NaN  NaN
        two  NaN  3.0  4.0
        """
        if columns is None:
            raise ValueError("columns should be set.")

        if values is None:
            raise ValueError("values should be set.")

        should_use_existing_index = index is not None
        if should_use_existing_index:
            index = [index]
        else:
            index = self._internal.index_columns

        df = self.pivot_table(
            index=index, columns=columns, values=values, aggfunc='first')

        if should_use_existing_index:
            return df
        else:
            index_columns = df._internal.index_columns
            # Note that the existing indexing column won't exist in the pivoted DataFrame.
            internal = df._internal.copy(
                index_map=[(index_column, None) for index_column in index_columns])
            return DataFrame(internal)

    @property
    def columns(self):
        """The column labels of the DataFrame."""
        return pd.Index(self._internal.data_columns)

    @columns.setter
    def columns(self, names):
        old_names = self._internal.data_columns
        if len(old_names) != len(names):
            raise ValueError(
                "Length mismatch: Expected axis has %d elements, new values have %d elements"
                % (len(old_names), len(names)))
        sdf = self._sdf.select(self._internal.index_columns +
                               [self[old_name]._scol.alias(new_name)
                                for (old_name, new_name) in zip(old_names, names)])
        self._internal = self._internal.copy(sdf=sdf, data_columns=names)

    @property
    def dtypes(self):
        """Return the dtypes in the DataFrame.

        This returns a Series with the data type of each column. The result's index is the original
        DataFrame's columns. Columns with mixed types are stored with the object dtype.

        Returns
        -------
        pd.Series
            The data type of each column.

        Examples
        --------
        >>> df = ks.DataFrame({'a': list('abc'),
        ...                    'b': list(range(1, 4)),
        ...                    'c': np.arange(3, 6).astype('i1'),
        ...                    'd': np.arange(4.0, 7.0, dtype='float64'),
        ...                    'e': [True, False, True],
        ...                    'f': pd.date_range('20130101', periods=3)},
        ...                   columns=['a', 'b', 'c', 'd', 'e', 'f'])
        >>> df.dtypes
        a            object
        b             int64
        c              int8
        d           float64
        e              bool
        f    datetime64[ns]
        dtype: object
        """
        return pd.Series([self[col].dtype for col in self._internal.data_columns],
                         index=self._internal.data_columns)

    def select_dtypes(self, include=None, exclude=None):
        """
        Return a subset of the DataFrame's columns based on the column dtypes.

        Parameters
        ----------
        include, exclude : scalar or list-like
            A selection of dtypes or strings to be included/excluded. At least
            one of these parameters must be supplied. It also takes Spark SQL
            DDL type strings, for instance, 'string' and 'date'.

        Returns
        -------
        DataFrame
            The subset of the frame including the dtypes in ``include`` and
            excluding the dtypes in ``exclude``.

        Raises
        ------
        ValueError
            * If both of ``include`` and ``exclude`` are empty

                >>> df = pd.DataFrame({'a': [1, 2] * 3,
                ...                    'b': [True, False] * 3,
                ...                    'c': [1.0, 2.0] * 3})
                >>> df.select_dtypes()
                Traceback (most recent call last):
                ...
                ValueError: at least one of include or exclude must be nonempty

            * If ``include`` and ``exclude`` have overlapping elements

                >>> df = pd.DataFrame({'a': [1, 2] * 3,
                ...                    'b': [True, False] * 3,
                ...                    'c': [1.0, 2.0] * 3})
                >>> df.select_dtypes(include='a', exclude='a')
                Traceback (most recent call last):
                ...
                TypeError: string dtypes are not allowed, use 'object' instead

        Notes
        -----
        * To select datetimes, use ``np.datetime64``, ``'datetime'`` or
          ``'datetime64'``

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 2] * 3,
        ...                    'b': [True, False] * 3,
        ...                    'c': [1.0, 2.0] * 3,
        ...                    'd': ['a', 'b'] * 3}, columns=['a', 'b', 'c', 'd'])
        >>> df
           a      b    c  d
        0  1   True  1.0  a
        1  2  False  2.0  b
        2  1   True  1.0  a
        3  2  False  2.0  b
        4  1   True  1.0  a
        5  2  False  2.0  b

        >>> df.select_dtypes(include='bool')
               b
        0   True
        1  False
        2   True
        3  False
        4   True
        5  False

        >>> df.select_dtypes(include=['float64'], exclude=['int'])
             c
        0  1.0
        1  2.0
        2  1.0
        3  2.0
        4  1.0
        5  2.0

        >>> df.select_dtypes(exclude=['int'])
               b    c  d
        0   True  1.0  a
        1  False  2.0  b
        2   True  1.0  a
        3  False  2.0  b
        4   True  1.0  a
        5  False  2.0  b

        Spark SQL DDL type strings can be used as well.

        >>> df.select_dtypes(exclude=['string'])
           a      b    c
        0  1   True  1.0
        1  2  False  2.0
        2  1   True  1.0
        3  2  False  2.0
        4  1   True  1.0
        5  2  False  2.0
        """
        from pyspark.sql.types import _parse_datatype_string

        if not is_list_like(include):
            include = (include,) if include is not None else ()
        if not is_list_like(exclude):
            exclude = (exclude,) if exclude is not None else ()

        if not any((include, exclude)):
            raise ValueError('at least one of include or exclude must be '
                             'nonempty')

        # can't both include AND exclude!
        if set(include).intersection(set(exclude)):
            raise ValueError('include and exclude overlap on {inc_ex}'.format(
                inc_ex=set(include).intersection(set(exclude))))

        # Handle Spark types
        columns = []
        include_spark_type = []
        for inc in include:
            try:
                include_spark_type.append(_parse_datatype_string(inc))
            except:
                pass

        exclude_spark_type = []
        for exc in exclude:
            try:
                exclude_spark_type.append(_parse_datatype_string(exc))
            except:
                pass

        # Handle Pandas types
        include_numpy_type = []
        for inc in include:
            try:
                include_numpy_type.append(infer_dtype_from_object(inc))
            except:
                pass

        exclude_numpy_type = []
        for exc in exclude:
            try:
                exclude_numpy_type.append(infer_dtype_from_object(exc))
            except:
                pass

        for col in self._internal.data_columns:
            if len(include) > 0:
                should_include = (
                    infer_dtype_from_object(self[col].dtype.name) in include_numpy_type or
                    self._sdf.schema[col].dataType in include_spark_type)
            else:
                should_include = not (
                    infer_dtype_from_object(self[col].dtype.name) in exclude_numpy_type or
                    self._sdf.schema[col].dataType in exclude_spark_type)

            if should_include:
                columns += col

        return DataFrame(self._internal.copy(
            sdf=self._sdf.select(self._internal.index_columns + columns), data_columns=columns))

    def count(self):
        """
        Count non-NA cells for each column.

        The values `None`, `NaN` are considered NA.

        Returns
        -------
        pandas.Series

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.shape: Number of DataFrame rows and columns (including NA
            elements).
        DataFrame.isna: Boolean same-sized DataFrame showing places of NA
            elements.

        Examples
        --------
        Constructing DataFrame from a dictionary:

        >>> df = ks.DataFrame({"Person":
        ...                    ["John", "Myla", "Lewis", "John", "Myla"],
        ...                    "Age": [24., np.nan, 21., 33, 26],
        ...                    "Single": [False, True, True, True, False]},
        ...                   columns=["Person", "Age", "Single"])
        >>> df
          Person   Age  Single
        0   John  24.0   False
        1   Myla   NaN    True
        2  Lewis  21.0    True
        3   John  33.0    True
        4   Myla  26.0   False

        Notice the uncounted NA values:

        >>> df.count()
        Person    5
        Age       4
        Single    5
        dtype: int64
        """
        return self._reduce_for_stat_function(_Frame._count_expr, numeric_only=False)

    def drop(self, labels=None, axis=1, columns: Union[str, List[str]] = None):
        """
        Drop specified labels from columns.

        Remove columns by specifying label names and axis=1 or columns.
        When specifying both labels and columns, only labels will be dropped.
        Removing rows is yet to be implemented.

        Parameters
        ----------
        labels : single label or list-like
            Column labels to drop.
        axis : {1 or 'columns'}, default 1
            .. dropna currently only works for axis=1 'columns'
               axis=0 is yet to be implemented.
        columns : single label or list-like
            Alternative to specifying axis (``labels, axis=1``
            is equivalent to ``columns=labels``).

        Returns
        -------
        dropped : DataFrame

        See Also
        --------
        Series.dropna

        Examples
        --------
        >>> df = ks.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6], 'w': [7, 8]},
        ...                   columns=['x', 'y', 'z', 'w'])
        >>> df
           x  y  z  w
        0  1  3  5  7
        1  2  4  6  8

        >>> df.drop('x', axis=1)
           y  z  w
        0  3  5  7
        1  4  6  8

        >>> df.drop(['y', 'z'], axis=1)
           x  w
        0  1  7
        1  2  8

        >>> df.drop(columns=['y', 'z'])
           x  w
        0  1  7
        1  2  8

        Notes
        -----
        Currently only axis = 1 is supported in this function,
        axis = 0 is yet to be implemented.
        """
        if labels is not None:
            axis = self._validate_axis(axis)
            if axis == 1:
                return self.drop(columns=labels)
            raise NotImplementedError("Drop currently only works for axis=1")
        elif columns is not None:
            if isinstance(columns, str):
                columns = [columns]
            sdf = self._sdf.drop(*columns)
            internal = self._internal.copy(
                sdf=sdf,
                data_columns=[column for column in self.columns if column not in columns])
            return DataFrame(internal)
        else:
            raise ValueError("Need to specify at least one of 'labels' or 'columns'")

    def get(self, key, default=None):
        """
        Get item from object for given key (DataFrame column, Panel slice,
        etc.). Returns default value if not found.

        Parameters
        ----------
        key : object

        Returns
        -------
        value : same type as items contained in object

        Examples
        --------
        >>> df = ks.DataFrame({'x':range(3), 'y':['a','b','b'], 'z':['a','b','b']},
        ...                   columns=['x', 'y', 'z'])
        >>> df
           x  y  z
        0  0  a  a
        1  1  b  b
        2  2  b  b

        >>> df.get('x')
        0    0
        1    1
        2    2
        Name: x, dtype: int64

        >>> df.get(['x', 'y'])
           x  y
        0  0  a
        1  1  b
        2  2  b
        """
        try:
            return self._pd_getitem(key)
        except (KeyError, ValueError, IndexError):
            return default

    def sort_values(self, by: Union[str, List[str]], ascending: Union[bool, List[bool]] = True,
                    inplace: bool = False, na_position: str = 'last') -> Optional['DataFrame']:
        """
        Sort by the values along either axis.

        Parameters
        ----------
        by : str or list of str
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
        sorted_obj : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame({
        ...     'col1': ['A', 'B', None, 'D', 'C'],
        ...     'col2': [2, 9, 8, 7, 4],
        ...     'col3': [0, 9, 4, 2, 3],
        ...   },
        ...   columns=['col1', 'col2', 'col3'])
        >>> df
           col1  col2  col3
        0     A     2     0
        1     B     9     9
        2  None     8     4
        3     D     7     2
        4     C     4     3

        Sort by col1

        >>> df.sort_values(by=['col1'])
           col1  col2  col3
        0     A     2     0
        1     B     9     9
        4     C     4     3
        3     D     7     2
        2  None     8     4

        Sort Descending

        >>> df.sort_values(by='col1', ascending=False)
           col1  col2  col3
        3     D     7     2
        4     C     4     3
        1     B     9     9
        0     A     2     0
        2  None     8     4

        Sort by multiple columns

        >>> df = ks.DataFrame({
        ...     'col1': ['A', 'A', 'B', None, 'D', 'C'],
        ...     'col2': [2, 1, 9, 8, 7, 4],
        ...     'col3': [0, 1, 9, 4, 2, 3],
        ...   },
        ...   columns=['col1', 'col2', 'col3'])
        >>> df.sort_values(by=['col1', 'col2'])
           col1  col2  col3
        1     A     1     1
        0     A     2     0
        2     B     9     9
        5     C     4     3
        4     D     7     2
        3  None     8     4
        """
        if isinstance(by, str):
            by = [by]
        if isinstance(ascending, bool):
            ascending = [ascending] * len(by)
        if len(ascending) != len(by):
            raise ValueError('Length of ascending ({}) != length of by ({})'
                             .format(len(ascending), len(by)))
        if na_position not in ('first', 'last'):
            raise ValueError("invalid na_position: '{}'".format(na_position))

        # Mapper: Get a spark column function for (ascending, na_position) combination
        # Note that 'asc_nulls_first' and friends were added as of Spark 2.4, see SPARK-23847.
        mapper = {
            (True, 'first'): lambda x: Column(getattr(x._jc, "asc_nulls_first")()),
            (True, 'last'): lambda x: Column(getattr(x._jc, "asc_nulls_last")()),
            (False, 'first'): lambda x: Column(getattr(x._jc, "desc_nulls_first")()),
            (False, 'last'): lambda x: Column(getattr(x._jc, "desc_nulls_last")()),
        }
        by = [mapper[(asc, na_position)](self[colname]._scol)
              for colname, asc in zip(by, ascending)]
        kdf = DataFrame(self._internal.copy(sdf=self._sdf.sort(*by)))  # type: ks.DataFrame
        if inplace:
            self._internal = kdf._internal
            return None
        else:
            return kdf

    def sort_index(self, axis: int = 0, level: int = None, ascending: bool = True,
                   inplace: bool = False, kind: str = None, na_position: str = 'last') \
            -> Optional['DataFrame']:
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
        sorted_obj : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame({'A': [2, 1, np.nan]}, index=['b', 'a', np.nan])

        >>> df.sort_index()
               A
        a    1.0
        b    2.0
        NaN  NaN

        >>> df.sort_index(ascending=False)
               A
        b    2.0
        a    1.0
        NaN  NaN

        >>> df.sort_index(na_position='first')
               A
        NaN  NaN
        a    1.0
        b    2.0

        >>> df.sort_index(inplace=True)
        >>> df
               A
        a    1.0
        b    2.0
        NaN  NaN


        >>> ks.DataFrame({'A': range(4), 'B': range(4)[::-1]},
        ...              index=[['b', 'b', 'a', 'a'], [1, 0, 1, 0]]).sort_index()
             A  B
        a 0  3  0
          1  2  1
        b 0  1  2
          1  0  3
        """
        if axis != 0:
            raise ValueError("No other axes than 0 are supported at the moment")
        if level is not None:
            raise ValueError("The 'axis' argument is not supported at the moment")
        if kind is not None:
            raise ValueError("Specifying the sorting algorithm is supported at the moment.")
        return self.sort_values(by=self._internal.index_columns, ascending=ascending,
                                inplace=inplace, na_position=na_position)

    # TODO:  add keep = First
    def nlargest(self, n: int, columns: 'Any') -> 'DataFrame':
        """
        Return the first `n` rows ordered by `columns` in descending order.

        Return the first `n` rows with the largest values in `columns`, in
        descending order. The columns that are not specified are returned as
        well, but not used for ordering.

        This method is equivalent to
        ``df.sort_values(columns, ascending=False).head(n)``, but more
        performant in Pandas.
        In Koalas, thanks to Spark's lazy execution and query optimizer,
        the two would have same performance.

        Parameters
        ----------
        n : int
            Number of rows to return.
        columns : label or list of labels
            Column label(s) to order by.

        Returns
        -------
        DataFrame
            The first `n` rows ordered by the given columns in descending
            order.

        See Also
        --------
        DataFrame.nsmallest : Return the first `n` rows ordered by `columns` in
            ascending order.
        DataFrame.sort_values : Sort DataFrame by the values.
        DataFrame.head : Return the first `n` rows without re-ordering.

        Notes
        -----

        This function cannot be used with all column types. For example, when
        specifying columns with `object` or `category` dtypes, ``TypeError`` is
        raised.

        Examples
        --------
        >>> df = ks.DataFrame({'X': [1, 2, 3, 5, 6, 7, np.nan],
        ...                    'Y': [6, 7, 8, 9, 10, 11, 12]})
        >>> df
             X   Y
        0  1.0   6
        1  2.0   7
        2  3.0   8
        3  5.0   9
        4  6.0  10
        5  7.0  11
        6  NaN  12

        In the following example, we will use ``nlargest`` to select the three
        rows having the largest values in column "population".

        >>> df.nlargest(n=3, columns='X')
             X   Y
        5  7.0  11
        4  6.0  10
        3  5.0   9

        >>> df.nlargest(n=3, columns=['Y', 'X'])
             X   Y
        6  NaN  12
        5  7.0  11
        4  6.0  10

        """
        kdf = self.sort_values(by=columns, ascending=False)  # type: Optional[DataFrame]
        assert kdf is not None
        return kdf.head(n=n)

    # TODO: add keep = First
    def nsmallest(self, n: int, columns: 'Any') -> 'DataFrame':
        """
        Return the first `n` rows ordered by `columns` in ascending order.

        Return the first `n` rows with the smallest values in `columns`, in
        ascending order. The columns that are not specified are returned as
        well, but not used for ordering.

        This method is equivalent to ``df.sort_values(columns, ascending=True).head(n)``,
        but more performant. In Koalas, thanks to Spark's lazy execution and query optimizer,
        the two would have same performance.

        Parameters
        ----------
        n : int
            Number of items to retrieve.
        columns : list or str
            Column name or names to order by.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.nlargest : Return the first `n` rows ordered by `columns` in
            descending order.
        DataFrame.sort_values : Sort DataFrame by the values.
        DataFrame.head : Return the first `n` rows without re-ordering.

        Examples
        --------
        >>> df = ks.DataFrame({'X': [1, 2, 3, 5, 6, 7, np.nan],
        ...                    'Y': [6, 7, 8, 9, 10, 11, 12]})
        >>> df
             X   Y
        0  1.0   6
        1  2.0   7
        2  3.0   8
        3  5.0   9
        4  6.0  10
        5  7.0  11
        6  NaN  12

        In the following example, we will use ``nsmallest`` to select the
        three rows having the smallest values in column "a".

        >>> df.nsmallest(n=3, columns='X') # doctest: +NORMALIZE_WHITESPACE
             X   Y
        0  1.0   6
        1  2.0   7
        2  3.0   8

        To order by the largest values in column "a" and then "c", we can
        specify multiple columns like in the next example.

        >>> df.nsmallest(n=3, columns=['Y', 'X']) # doctest: +NORMALIZE_WHITESPACE
             X   Y
        0  1.0   6
        1  2.0   7
        2  3.0   8
        """
        kdf = self.sort_values(by=columns, ascending=True)  # type: Optional[DataFrame]
        assert kdf is not None
        return kdf.head(n=n)

    def isin(self, values):
        """
        Whether each element in the DataFrame is contained in values.

        Parameters
        ----------
        values : iterable or dict
           The sequence of values to test. If values is a dict,
           the keys must be the column names, which must match.
           Series and DataFrame are not supported.

        Returns
        -------
        DataFrame
            DataFrame of booleans showing whether each element in the DataFrame
            is contained in values.

        Examples
        --------
        >>> df = ks.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]},
        ...                   index=['falcon', 'dog'],
        ...                   columns=['num_legs', 'num_wings'])
        >>> df
                num_legs  num_wings
        falcon         2          2
        dog            4          0

        When ``values`` is a list check whether every value in the DataFrame
        is present in the list (which animals have 0 or 2 legs or wings)

        >>> df.isin([0, 2])
                num_legs  num_wings
        falcon      True       True
        dog        False       True

        When ``values`` is a dict, we can pass values to check for each
        column separately:

        >>> df.isin({'num_wings': [0, 3]})
                num_legs  num_wings
        falcon     False      False
        dog        False       True
        """
        if isinstance(values, (pd.DataFrame, pd.Series)):
            raise NotImplementedError("DataFrame and Series are not supported")
        if isinstance(values, dict) and not set(values.keys()).issubset(self.columns):
            raise AttributeError(
                "'DataFrame' object has no attribute %s"
                % (set(values.keys()).difference(self.columns)))

        _select_columns = self._internal.index_columns.copy()
        if isinstance(values, dict):
            for col in self.columns:
                if col in values:
                    _select_columns.append(self._sdf[col].isin(values[col]).alias(col))
                else:
                    _select_columns.append(F.lit(False).alias(col))
        elif is_list_like(values):
            _select_columns += [
                self._sdf[col].isin(list(values)).alias(col) for col in self.columns]
        else:
            raise TypeError('Values should be iterable, Series, DataFrame or dict.')

        return DataFrame(self._internal.copy(sdf=self._sdf.select(_select_columns)))

    @property
    def shape(self):
        """
        Return a tuple representing the dimensionality of the DataFrame.

        Examples
        --------
        >>> df = ks.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.shape
        (2, 2)

        >>> df = ks.DataFrame({'col1': [1, 2], 'col2': [3, 4],
        ...                    'col3': [5, 6]})
        >>> df.shape
        (2, 3)
        """
        return len(self), len(self.columns)

    def merge(self, right: 'DataFrame', how: str = 'inner',
              on: Optional[Union[str, List[str]]] = None,
              left_on: Optional[Union[str, List[str]]] = None,
              right_on: Optional[Union[str, List[str]]] = None,
              left_index: bool = False, right_index: bool = False,
              suffixes: Tuple[str, str] = ('_x', '_y')) -> 'DataFrame':
        """
        Merge DataFrame objects with a database-style join.

        The index of the resulting DataFrame will be one of the following:
            - 0...n if no index is used for merging
            - Index of the left DataFrame if merged only on the index of the right DataFrame
            - Index of the right DataFrame if merged only on the index of the left DataFrame
            - All involved indices if merged using the indices of both DataFrames
                e.g. if `left` with indices (a, x) and `right` with indices (b, x), the result will
                be an index (x, a, b)

        Parameters
        ----------
        right: Object to merge with.
        how: Type of merge to be performed.
            {'left', 'right', 'outer', 'inner'}, default 'inner'

            left: use only keys from left frame, similar to a SQL left outer join; preserve key
                order.
            right: use only keys from right frame, similar to a SQL right outer join; preserve key
                order.
            outer: use union of keys from both frames, similar to a SQL full outer join; sort keys
                lexicographically.
            inner: use intersection of keys from both frames, similar to a SQL inner join;
                preserve the order of the left keys.
        on: Column or index level names to join on. These must be found in both DataFrames. If on
            is None and not merging on indexes then this defaults to the intersection of the
            columns in both DataFrames.
        left_on: Column or index level names to join on in the left DataFrame. Can also
            be an array or list of arrays of the length of the left DataFrame.
            These arrays are treated as if they are columns.
        right_on: Column or index level names to join on in the right DataFrame. Can also
            be an array or list of arrays of the length of the right DataFrame.
            These arrays are treated as if they are columns.
        left_index: Use the index from the left DataFrame as the join key(s). If it is a
            MultiIndex, the number of keys in the other DataFrame (either the index or a number of
            columns) must match the number of levels.
        right_index: Use the index from the right DataFrame as the join key. Same caveats as
            left_index.
        suffixes: Suffix to apply to overlapping column names in the left and right side,
            respectively.

        Returns
        -------
        DataFrame
            A DataFrame of the two merged objects.

        Examples
        --------
        >>> df1 = ks.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
        ...                     'value': [1, 2, 3, 5]},
        ...                    columns=['lkey', 'value'])
        >>> df2 = ks.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
        ...                     'value': [5, 6, 7, 8]},
        ...                    columns=['rkey', 'value'])
        >>> df1
          lkey  value
        0  foo      1
        1  bar      2
        2  baz      3
        3  foo      5
        >>> df2
          rkey  value
        0  foo      5
        1  bar      6
        2  baz      7
        3  foo      8

        Merge df1 and df2 on the lkey and rkey columns. The value columns have
        the default suffixes, _x and _y, appended.

        >>> merged = df1.merge(df2, left_on='lkey', right_on='rkey')
        >>> merged.sort_values(by=['lkey', 'value_x', 'rkey', 'value_y'])
          lkey  value_x rkey  value_y
        0  bar        2  bar        6
        1  baz        3  baz        7
        2  foo        1  foo        5
        3  foo        1  foo        8
        4  foo        5  foo        5
        5  foo        5  foo        8

        >>> left_kdf = ks.DataFrame({'A': [1, 2]})
        >>> right_kdf = ks.DataFrame({'B': ['x', 'y']}, index=[1, 2])

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True)
           A  B
        1  2  x

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True, how='left')
           A     B
        0  1  None
        1  2     x

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True, how='right')
             A  B
        1  2.0  x
        2  NaN  y

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True, how='outer')
             A     B
        0  1.0  None
        1  2.0     x
        2  NaN     y

        Notes
        -----
        As described in #263, joining string columns currently returns None for missing values
            instead of NaN.
        """
        _to_list = lambda o: o if o is None or is_list_like(o) else [o]

        if on:
            if left_on or right_on:
                raise ValueError('Can only pass argument "on" OR "left_on" and "right_on", '
                                 'not a combination of both.')
            left_keys = _to_list(on)
            right_keys = _to_list(on)
        else:
            # TODO: need special handling for multi-index.
            if left_index:
                left_keys = self._internal.index_columns
            else:
                left_keys = _to_list(left_on)
            if right_index:
                right_keys = right._internal.index_columns
            else:
                right_keys = _to_list(right_on)

            if left_keys and not right_keys:
                raise ValueError('Must pass right_on or right_index=True')
            if right_keys and not left_keys:
                raise ValueError('Must pass left_on or left_index=True')
            if not left_keys and not right_keys:
                common = list(self.columns.intersection(right.columns))
                if len(common) == 0:
                    raise ValueError(
                        'No common columns to perform merge on. Merge options: '
                        'left_on=None, right_on=None, left_index=False, right_index=False')
                left_keys = common
                right_keys = common
            if len(left_keys) != len(right_keys):  # type: ignore
                raise ValueError('len(left_keys) must equal len(right_keys)')

        if how == 'full':
            warnings.warn("Warning: While Koalas will accept 'full', you should use 'outer' " +
                          "instead to be compatible with the pandas merge API", UserWarning)
        if how == 'outer':
            # 'outer' in pandas equals 'full' in Spark
            how = 'full'
        if how not in ('inner', 'left', 'right', 'full'):
            raise ValueError("The 'how' parameter has to be amongst the following values: ",
                             "['inner', 'left', 'right', 'outer']")

        left_table = self._sdf.alias('left_table')
        right_table = right._sdf.alias('right_table')

        left_key_columns = [left_table[col] for col in left_keys]  # type: ignore
        right_key_columns = [right_table[col] for col in right_keys]  # type: ignore

        join_condition = reduce(lambda x, y: x & y,
                                [lkey == rkey for lkey, rkey
                                 in zip(left_key_columns, right_key_columns)])

        joined_table = left_table.join(right_table, join_condition, how=how)

        # Unpack suffixes tuple for convenience
        left_suffix = suffixes[0]
        right_suffix = suffixes[1]

        # Append suffixes to columns with the same name to avoid conflicts later
        duplicate_columns = (set(self._internal.data_columns)
                             & set(right._internal.data_columns))

        left_index_columns = set(self._internal.index_columns)
        right_index_columns = set(right._internal.index_columns)

        exprs = []
        for col in left_table.columns:
            if col in left_index_columns:
                continue
            scol = left_table[col]
            if col in duplicate_columns:
                if col in left_keys and col in right_keys:
                    pass
                else:
                    col = col + left_suffix
                    scol = scol.alias(col)
            exprs.append(scol)
        for col in right_table.columns:
            if col in right_index_columns:
                continue
            scol = right_table[col]
            if col in duplicate_columns:
                if col in left_keys and col in right_keys:
                    continue
                else:
                    col = col + right_suffix
                    scol = scol.alias(col)
            exprs.append(scol)

        # Retain indices if they are used for joining
        if left_index:
            if right_index:
                exprs.extend(['left_table.%s' % col for col in left_index_columns])
                exprs.extend(['right_table.%s' % col for col in right_index_columns])
                index_map = self._internal.index_map + [idx for idx in right._internal.index_map
                                                        if idx not in self._internal.index_map]
            else:
                exprs.extend(['right_table.%s' % col for col in right_index_columns])
                index_map = right._internal.index_map
        elif right_index:
            exprs.extend(['left_table.%s' % col for col in left_index_columns])
            index_map = self._internal.index_map
        else:
            index_map = []

        selected_columns = joined_table.select(*exprs)

        # Merge left and right indices after the join by replacing missing values in the left index
        # with values from the right index and dropping
        if (how == 'right' or how == 'full') and right_index:
            for left_index_col, right_index_col in zip(self._internal.index_columns,
                                                       right._internal.index_columns):
                selected_columns = selected_columns.withColumn(
                    'left_table.' + left_index_col,
                    F.when(F.col('left_table.%s' % left_index_col).isNotNull(),
                           F.col('left_table.%s' % left_index_col))
                    .otherwise(F.col('right_table.%s' % right_index_col))
                ).withColumnRenamed(
                    'left_table.%s' % left_index_col, left_index_col
                ).drop(F.col('left_table.%s' % left_index_col))
        if not (left_index and not right_index):
            selected_columns = selected_columns.drop(*[F.col('right_table.%s' % right_index_col)
                                                       for right_index_col in right_index_columns
                                                       if right_index_col in left_index_columns])

        if index_map:
            data_columns = [c for c in selected_columns.columns
                            if c not in [idx[0] for idx in index_map]]
            internal = _InternalFrame(
                sdf=selected_columns, data_columns=data_columns, index_map=index_map)
            return DataFrame(internal)
        else:
            return DataFrame(selected_columns)

    def join(self, right: 'DataFrame', on: Optional[Union[str, List[str]]] = None,
             how: str = 'left', lsuffix: str = '', rsuffix: str = '') -> 'DataFrame':
        """
        Join columns of another DataFrame.

        Join columns with `right` DataFrame either on index or on a key column. Efficiently join
        multiple DataFrame objects by index at once by passing a list.

        Parameters
        ----------
        right: DataFrame, Series
        on: str, list of str, or array-like, optional
            Column or index level name(s) in the caller to join on the index in `right`, otherwise
            joins index-on-index. If multiple values given, the `right` DataFrame must have a
            MultiIndex. Can pass an array as the join key if it is not already contained in the
            calling DataFrame. Like an Excel VLOOKUP operation.
        how: {'left', 'right', 'outer', 'inner'}, default 'left'
            How to handle the operation of the two objects.

            * left: use `left` frame’s index (or column if on is specified).
            * right: use `right`’s index.
            * outer: form union of `left` frame’s index (or column if on is specified) with
              right’s index, and sort it. lexicographically.
            * inner: form intersection of `left` frame’s index (or column if on is specified)
              with `right`’s index, preserving the order of the `left`’s one.
        lsuffix : str, default ''
            Suffix to use from left frame's overlapping columns.
        rsuffix : str, default ''
            Suffix to use from `right` frame's overlapping columns.

        Returns
        -------
        DataFrame
            A dataframe containing columns from both the `left` and `right`.

        See Also
        --------
        DataFrame.merge: For column(s)-on-columns(s) operations.

        Notes
        -----
        Parameters on, lsuffix, and rsuffix are not supported when passing a list of DataFrame
        objects.

        Examples
        --------
        >>> kdf1 = ks.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
        ...                      'A': ['A0', 'A1', 'A2', 'A3']},
        ...                     columns=['key', 'A'])
        >>> kdf2 = ks.DataFrame({'key': ['K0', 'K1', 'K2'],
        ...                      'B': ['B0', 'B1', 'B2']},
        ...                     columns=['key', 'B'])
        >>> kdf1
          key   A
        0  K0  A0
        1  K1  A1
        2  K2  A2
        3  K3  A3
        >>> kdf2
          key   B
        0  K0  B0
        1  K1  B1
        2  K2  B2

        Join DataFrames using their indexes.

        >>> join_kdf = kdf1.join(kdf2, lsuffix='_left', rsuffix='_right')
        >>> join_kdf.sort_values(by=join_kdf.columns)
          key_left   A key_right     B
        0       K0  A0        K0    B0
        1       K1  A1        K1    B1
        2       K2  A2        K2    B2
        3       K3  A3      None  None

        If we want to join using the key columns, we need to set key to be the index in both df and
        right. The joined DataFrame will have key as its index.

        >>> join_kdf = kdf1.set_index('key').join(kdf2.set_index('key'))
        >>> join_kdf.sort_values(by=join_kdf.columns) # doctest: +NORMALIZE_WHITESPACE
              A     B
        key
        K0   A0    B0
        K1   A1    B1
        K2   A2    B2
        K3   A3  None

        Another option to join using the key columns is to use the on parameter. DataFrame.join
        always uses right’s index but we can use any column in df. This method preserves the
        original DataFrame’s index in the result.

        >>> join_kdf = kdf1.join(kdf2.set_index('key'), on='key')
        >>> join_kdf.sort_values(by=join_kdf.columns)
          key   A     B
        0  K0  A0    B0
        1  K1  A1    B1
        2  K2  A2    B2
        3  K3  A3  None
        """
        if on:
            self = self.set_index(on)
            join_kdf = self.merge(right, left_index=True, right_index=True, how=how,
                                  suffixes=(lsuffix, rsuffix)).reset_index()
        else:
            join_kdf = self.merge(right, left_index=True, right_index=True, how=how,
                                  suffixes=(lsuffix, rsuffix))
        return join_kdf

    def append(self, other: 'DataFrame', ignore_index: bool = False,
               verify_integrity: bool = False, sort: bool = False) -> 'DataFrame':
        """
        Append rows of other to the end of caller, returning a new object.

        Columns in other that are not in the caller are added as new columns.

        Parameters
        ----------
        other : DataFrame or Series/dict-like object, or list of these
            The data to append.

        ignore_index : boolean, default False
            If True, do not use the index labels.

        verify_integrity : boolean, default False
            If True, raise ValueError on creating index with duplicates.

        sort : boolean, default False
            Currently not supported.

        Returns
        -------
        appended : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame([[1, 2], [3, 4]], columns=list('AB'))

        >>> df.append(df)
           A  B
        0  1  2
        1  3  4
        0  1  2
        1  3  4

        >>> df.append(df, ignore_index=True)
           A  B
        0  1  2
        1  3  4
        2  1  2
        3  3  4
        """
        if isinstance(other, ks.Series):
            raise ValueError("DataFrames.append() does not support appending Series to DataFrames")
        if sort:
            raise ValueError("The 'sort' parameter is currently not supported")

        if not ignore_index:
            index_columns = self._internal.index_columns
            if len(index_columns) != len(other._internal.index_columns):
                raise ValueError("Both DataFrames have to have the same number of index levels")

            if verify_integrity and len(index_columns) > 0:
                if (self._sdf.select(index_columns)
                        .intersect(other._sdf.select(other._internal.index_columns))
                        .count()) > 0:
                    raise ValueError("Indices have overlapping values")

        # Lazy import to avoid circular dependency issues
        from databricks.koalas.namespace import concat
        return concat([self, other], ignore_index=ignore_index)

    # TODO: add 'filter_func' and 'errors' parameter
    def update(self, other: 'DataFrame', join: str = 'left', overwrite: bool = True):
        """
        Modify in place using non-NA values from another DataFrame.
        Aligns on indices. There is no return value.

        Parameters
        ----------
        other : DataFrame, or Series
        join : 'left', default 'left'
            Only left join is implemented, keeping the index and columns of the original object.
        overwrite : bool, default True
            How to handle non-NA values for overlapping keys:

            * True: overwrite original DataFrame's values with values from `other`.
            * False: only update values that are NA in the original DataFrame.

        Returns
        -------
        None : method directly changes calling object

        See Also
        --------
        DataFrame.merge : For column(s)-on-columns(s) operations.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 3], 'B': [400, 500, 600]}, columns=['A', 'B'])
        >>> new_df = ks.DataFrame({'B': [4, 5, 6], 'C': [7, 8, 9]}, columns=['B', 'C'])
        >>> df.update(new_df)
        >>> df
           A  B
        0  1  4
        1  2  5
        2  3  6

        The DataFrame's length does not increase as a result of the update,
        only values at matching index/column labels are updated.

        >>> df = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']}, columns=['A', 'B'])
        >>> new_df = ks.DataFrame({'B': ['d', 'e', 'f', 'g', 'h', 'i']}, columns=['B'])
        >>> df.update(new_df)
        >>> df
           A  B
        0  a  d
        1  b  e
        2  c  f

        For Series, it's name attribute must be set.

        >>> df = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']}, columns=['A', 'B'])
        >>> new_column = ks.Series(['d', 'e'], name='B', index=[0, 2])
        >>> df.update(new_column)
        >>> df
           A  B
        0  a  d
        1  b  y
        2  c  e

        If `other` contains None the corresponding values are not updated in the original dataframe.

        >>> df = ks.DataFrame({'A': [1, 2, 3], 'B': [400, 500, 600]}, columns=['A', 'B'])
        >>> new_df = ks.DataFrame({'B': [4, None, 6]}, columns=['B'])
        >>> df.update(new_df)
        >>> df
           A      B
        0  1    4.0
        1  2  500.0
        2  3    6.0
        """
        if join != 'left':
            raise NotImplementedError("Only left join is supported")

        if isinstance(other, ks.Series):
            other = DataFrame(other)

        update_columns = list(set(self._internal.data_columns)
                              .intersection(set(other._internal.data_columns)))
        update_sdf = self.join(other[update_columns], rsuffix='_new')._sdf

        for column_name in update_columns:
            old_col = update_sdf[column_name]
            new_col = update_sdf[column_name + '_new']
            if overwrite:
                update_sdf = update_sdf.withColumn(column_name, F.when(new_col.isNull(), old_col)
                                                   .otherwise(new_col))
            else:
                update_sdf = update_sdf.withColumn(column_name, F.when(old_col.isNull(), new_col)
                                                   .otherwise(old_col))
        internal = self._internal.copy(sdf=update_sdf.select(self._internal.columns))
        self._internal = internal

    def sample(self, n: Optional[int] = None, frac: Optional[float] = None, replace: bool = False,
               random_state: Optional[int] = None) -> 'DataFrame':
        """
        Return a random sample of items from an axis of object.

        Please call this function using named argument by specifing the ``frac`` argument.

        You can use `random_state` for reproducibility. However, note that different from pandas,
        specifying a seed in Koalas/Spark does not guarantee the sampled rows will be fixed. The
        result set depends on not only the seed, but also how the data is distributed across
        machines and to some extent network randomness when shuffle operations are involved. Even
        in the simplest case, the result set will depend on the system's CPU core count.

        Parameters
        ----------
        n : int, optional
            Number of items to return. This is currently NOT supported. Use frac instead.
        frac : float, optional
            Fraction of axis items to return.
        replace : bool, default False
            Sample with or without replacement.
        random_state : int, optional
            Seed for the random number generator (if int).

        Returns
        -------
        Series or DataFrame
            A new object of same type as caller containing the sampled items.

        Examples
        --------
        >>> df = ks.DataFrame({'num_legs': [2, 4, 8, 0],
        ...                    'num_wings': [2, 0, 0, 0],
        ...                    'num_specimen_seen': [10, 2, 1, 8]},
        ...                   index=['falcon', 'dog', 'spider', 'fish'],
        ...                   columns=['num_legs', 'num_wings', 'num_specimen_seen'])
        >>> df  # doctest: +SKIP
                num_legs  num_wings  num_specimen_seen
        falcon         2          2                 10
        dog            4          0                  2
        spider         8          0                  1
        fish           0          0                  8

        A random 25% sample of the ``DataFrame``.
        Note that we use `random_state` to ensure the reproducibility of
        the examples.

        >>> df.sample(frac=0.25, random_state=1)  # doctest: +SKIP
                num_legs  num_wings  num_specimen_seen
        falcon         2          2                 10
        fish           0          0                  8

        Extract 25% random elements from the ``Series`` ``df['num_legs']``, with replacement,
        so the same items could appear more than once.

        >>> df['num_legs'].sample(frac=0.4, replace=True, random_state=1)  # doctest: +SKIP
        falcon    2
        spider    8
        spider    8
        Name: num_legs, dtype: int64

        Specifying the exact number of items to return is not supported at the moment.

        >>> df.sample(n=5)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
            ...
        NotImplementedError: Function sample currently does not support specifying ...
        """
        # Note: we don't run any of the doctests because the result can change depending on the
        # system's core count.
        if n is not None:
            raise NotImplementedError("Function sample currently does not support specifying "
                                      "exact number of items to return. Use frac instead.")

        if frac is None:
            raise ValueError("frac must be specified.")

        sdf = self._sdf.sample(withReplacement=replace, fraction=frac, seed=random_state)
        return DataFrame(self._internal.copy(sdf=sdf))

    def astype(self, dtype) -> 'DataFrame':
        """
        Cast a pandas object to a specified dtype ``dtype``.

        Parameters
        ----------
        dtype : data type, or dict of column name -> data type
            Use a numpy.dtype or Python type to cast entire pandas object to
            the same type. Alternatively, use {col: dtype, ...}, where col is a
            column label and dtype is a numpy.dtype or Python type to cast one
            or more of the DataFrame's columns to column-specific types.

        Returns
        -------
        casted : same type as caller

        See Also
        --------
        to_datetime : Convert argument to datetime.

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 2, 3], 'b': [1, 2, 3]}, dtype='int64')
        >>> df
           a  b
        0  1  1
        1  2  2
        2  3  3

        Convert to float type:

        >>> df.astype('float')
             a    b
        0  1.0  1.0
        1  2.0  2.0
        2  3.0  3.0

        Convert to int64 type back:

        >>> df.astype('int64')
           a  b
        0  1  1
        1  2  2
        2  3  3

        Convert column a to float type:

        >>> df.astype({'a': float})
             a  b
        0  1.0  1
        1  2.0  2
        2  3.0  3

        """
        results = []
        if is_dict_like(dtype):
            for col_name in dtype.keys():
                if col_name not in self.columns:
                    raise KeyError('Only a column name can be used for the '
                                   'key in a dtype mappings argument.')
            for col_name, col in self.iteritems():
                if col_name in dtype:
                    results.append(col.astype(dtype=dtype[col_name]))
                else:
                    results.append(col)
        else:
            for col_name, col in self.iteritems():
                results.append(col.astype(dtype=dtype))
        sdf = self._sdf.select(
            self._internal.index_columns + list(map(lambda ser: ser._scol, results)))
        return DataFrame(self._internal.copy(sdf=sdf))

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
        DataFrame
           New DataFrame with updated labels.

        See Also
        --------
        Series.add_prefix: Prefix row labels with string `prefix`.
        Series.add_suffix: Suffix row labels with string `suffix`.
        DataFrame.add_suffix: Suffix column labels with string `suffix`.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]}, columns=['A', 'B'])
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6

        >>> df.add_prefix('col_')
           col_A  col_B
        0      1      3
        1      2      4
        2      3      5
        3      4      6
        """
        assert isinstance(prefix, str)
        data_columns = self._internal.data_columns

        sdf = self._sdf.select(self._internal.index_columns +
                               [self[name]._scol.alias(prefix + name)
                                for name in data_columns])
        internal = self._internal.copy(
            sdf=sdf, data_columns=[prefix + name for name in data_columns])
        return DataFrame(internal)

    def add_suffix(self, suffix):
        """
        Suffix labels with string `suffix`.

        For Series, the row labels are suffixed.
        For DataFrame, the column labels are suffixed.

        Parameters
        ----------
        suffix : str
           The string to add before each label.

        Returns
        -------
        DataFrame
           New DataFrame with updated labels.

        See Also
        --------
        Series.add_prefix: Prefix row labels with string `prefix`.
        Series.add_suffix: Suffix row labels with string `suffix`.
        DataFrame.add_prefix: Prefix column labels with string `prefix`.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]}, columns=['A', 'B'])
        >>> df
           A  B
        0  1  3
        1  2  4
        2  3  5
        3  4  6

        >>> df.add_suffix('_col')
           A_col  B_col
        0      1      3
        1      2      4
        2      3      5
        3      4      6
        """
        assert isinstance(suffix, str)
        data_columns = self._internal.data_columns

        sdf = self._sdf.select(self._internal.index_columns +
                               [self[name]._scol.alias(name + suffix)
                                for name in data_columns])
        internal = self._internal.copy(
            sdf=sdf, data_columns=[name + suffix for name in data_columns])
        return DataFrame(internal)

    # TODO: include, and exclude should be implemented.
    def describe(self, percentiles: Optional[List[float]] = None) -> 'DataFrame':
        """
        Generate descriptive statistics that summarize the central tendency,
        dispersion and shape of a dataset's distribution, excluding
        ``NaN`` values.

        Analyzes both numeric and object series, as well
        as ``DataFrame`` column sets of mixed data types. The output
        will vary depending on what is provided. Refer to the notes
        below for more detail.

        Parameters
        ----------
        percentiles : list of ``float`` in range [0.0, 1.0], default [0.25, 0.5, 0.75]
            A list of percentiles to be computed.

        Returns
        -------
        Series or DataFrame
            Summary statistics of the Series or Dataframe provided.

        See Also
        --------
        DataFrame.count: Count number of non-NA/null observations.
        DataFrame.max: Maximum of the values in the object.
        DataFrame.min: Minimum of the values in the object.
        DataFrame.mean: Mean of the values.
        DataFrame.std: Standard deviation of the obersvations.

        Notes
        -----
        For numeric data, the result's index will include ``count``,
        ``mean``, ``std``, ``min``, ``25%``, ``50%``, ``75%``, ``max``.

        Currently only numeric data is supported.

        Examples
        --------
        Describing a numeric ``Series``.

        >>> s = ks.Series([1, 2, 3])
        >>> s.describe()
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        25%      1.0
        50%      2.0
        75%      3.0
        max      3.0
        Name: 0, dtype: float64

        Describing a ``DataFrame``. Only numeric fields are returned.

        >>> df = ks.DataFrame({'numeric1': [1, 2, 3],
        ...                    'numeric2': [4.0, 5.0, 6.0],
        ...                    'object': ['a', 'b', 'c']
        ...                   },
        ...                   columns=['numeric1', 'numeric2', 'object'])
        >>> df.describe()
               numeric1  numeric2
        count       3.0       3.0
        mean        2.0       5.0
        std         1.0       1.0
        min         1.0       4.0
        25%         1.0       4.0
        50%         2.0       5.0
        75%         3.0       6.0
        max         3.0       6.0

        Describing a ``DataFrame`` and selecting custom percentiles.

        >>> df = ks.DataFrame({'numeric1': [1, 2, 3],
        ...                    'numeric2': [4.0, 5.0, 6.0]
        ...                   },
        ...                   columns=['numeric1', 'numeric2'])
        >>> df.describe(percentiles = [0.85, 0.15])
               numeric1  numeric2
        count       3.0       3.0
        mean        2.0       5.0
        std         1.0       1.0
        min         1.0       4.0
        15%         1.0       4.0
        50%         2.0       5.0
        85%         3.0       6.0
        max         3.0       6.0

        Describing a column from a ``DataFrame`` by accessing it as
        an attribute.

        >>> df.numeric1.describe()
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        25%      1.0
        50%      2.0
        75%      3.0
        max      3.0
        Name: numeric1, dtype: float64

        Describing a column from a ``DataFrame`` by accessing it as
        an attribute and selecting custom percentiles.

        >>> df.numeric1.describe(percentiles = [0.85, 0.15])
        count    3.0
        mean     2.0
        std      1.0
        min      1.0
        15%      1.0
        50%      2.0
        85%      3.0
        max      3.0
        Name: numeric1, dtype: float64
        """
        exprs = []
        data_columns = []
        for col in self.columns:
            kseries = self[col]
            spark_type = kseries.spark_type
            if isinstance(spark_type, DoubleType) or isinstance(spark_type, FloatType):
                exprs.append(F.nanvl(kseries._scol, F.lit(None)).alias(kseries.name))
                data_columns.append(kseries.name)
            elif isinstance(spark_type, NumericType):
                exprs.append(kseries._scol)
                data_columns.append(kseries.name)

        if len(exprs) == 0:
            raise ValueError("Cannot describe a DataFrame without columns")

        if percentiles is not None:
            if any((p < 0.0) or (p > 1.0) for p in percentiles):
                raise ValueError("Percentiles should all be in the interval [0, 1]")
            # appending 50% if not in percentiles already
            percentiles = (percentiles + [0.5]) if 0.5 not in percentiles else percentiles
        else:
            percentiles = [0.25, 0.5, 0.75]

        formatted_perc = ["{:.0%}".format(p) for p in sorted(percentiles)]
        stats = ["count", "mean", "stddev", "min", *formatted_perc, "max"]

        sdf = self._sdf.select(*exprs).summary(stats)

        internal = _InternalFrame(sdf=sdf.replace("stddev", "std", subset='summary'),
                                  data_columns=data_columns,
                                  index_map=[('summary', None)])
        return DataFrame(internal).astype('float64')

    def _cum(self, func, skipna: bool):
        # This is used for cummin, cummax, cumxum, etc.
        if func == F.min:
            func = "cummin"
        elif func == F.max:
            func = "cummax"
        elif func == F.sum:
            func = "cumsum"
        elif func.__name__ == "cumprod":
            func = "cumprod"

        if len(self._internal.index_columns) == 0:
            raise ValueError("Index must be set.")

        applied = []
        for column in self._internal.data_columns:
            applied.append(getattr(self[column], func)(skipna))

        sdf = self._sdf.select(
            self._internal.index_columns + [c._scol for c in applied])
        internal = self._internal.copy(sdf=sdf, data_columns=[c.name for c in applied])
        return DataFrame(internal)

    # TODO: implements 'keep' parameters
    def drop_duplicates(self, subset=None, inplace=False):
        """
        Return DataFrame with duplicate rows removed, optionally only
        considering certain columns.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns
        inplace : boolean, default False
            Whether to drop duplicates in place or to return a copy

        Returns
        -------
        DataFrame

        >>> df = ks.DataFrame(
        ...     {'a': [1, 2, 2, 2, 3], 'b': ['a', 'a', 'a', 'c', 'd']}, columns = ['a', 'b'])
        >>> df
           a  b
        0  1  a
        1  2  a
        2  2  a
        3  2  c
        4  3  d

        >>> df.drop_duplicates().sort_values(['a', 'b'])
           a  b
        0  1  a
        1  2  a
        3  2  c
        4  3  d

        >>> df.drop_duplicates('a').sort_values(['a', 'b'])
           a  b
        0  1  a
        1  2  a
        4  3  d

        >>> df.drop_duplicates(['a', 'b']).sort_values(['a', 'b'])
           a  b
        0  1  a
        1  2  a
        3  2  c
        4  3  d
        """
        if subset is None:
            subset = self._internal.data_columns
        elif not isinstance(subset, list):
            subset = [subset]

        sdf = self._sdf.drop_duplicates(subset=subset)
        internal = self._internal.copy(sdf=sdf)
        if inplace:
            self._internal = internal
        else:
            return DataFrame(internal)

    def reindex(self, labels: Optional[Any] = None, index: Optional[Any] = None,
                columns: Optional[Any] = None, axis: Optional[Union[int, str]] = None,
                copy: Optional[bool] = True, fill_value: Optional[Any] = None) -> 'DataFrame':
        """
        Conform DataFrame to new index with optional filling logic, placing
        NA/NaN in locations having no value in the previous index. A new object
        is produced unless the new index is equivalent to the current one and
        ``copy=False``.

        Parameters
        ----------
        labels: array-like, optional
            New labels / index to conform the axis specified by ‘axis’ to.
        index, columns: array-like, optional
            New labels / index to conform to, should be specified using keywords.
            Preferably an Index object to avoid duplicating data
        axis: int or str, optional
            Axis to target. Can be either the axis name (‘index’, ‘columns’) or
            number (0, 1).
        copy : bool, default True
            Return a new object, even if the passed indexes are the same.
        fill_value : scalar, default np.NaN
            Value to use for missing values. Defaults to NaN, but can be any
            "compatible" value.

        Returns
        -------
        DataFrame with changed index.

        See Also
        --------
        DataFrame.set_index : Set row labels.
        DataFrame.reset_index : Remove row labels or move them to new columns.

        Examples
        --------

        ``DataFrame.reindex`` supports two calling conventions

        * ``(index=index_labels, columns=column_labels, ...)``
        * ``(labels, axis={'index', 'columns'}, ...)``

        We *highly* recommend using keyword arguments to clarify your
        intent.

        Create a dataframe with some fictional data.

        >>> index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
        >>> df = ks.DataFrame({
        ...      'http_status': [200, 200, 404, 404, 301],
        ...      'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
        ...       index=index)
        >>> df
                   http_status  response_time
        Firefox            200           0.04
        Chrome             200           0.02
        Safari             404           0.07
        IE10               404           0.08
        Konqueror          301           1.00

        Create a new index and reindex the dataframe. By default
        values in the new index that do not have corresponding
        records in the dataframe are assigned ``NaN``.

        >>> new_index= ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10',
        ...             'Chrome']
        >>> df.reindex(new_index).sort_index()
        ... # doctest: +NORMALIZE_WHITESPACE
                       http_status  response_time
        Chrome               200.0           0.02
        Comodo Dragon          NaN            NaN
        IE10                 404.0           0.08
        Iceweasel              NaN            NaN
        Safari               404.0           0.07

        We can fill in the missing values by passing a value to
        the keyword ``fill_value``.

        >>> df.reindex(new_index, fill_value=0, copy=False).sort_index()
        ... # doctest: +NORMALIZE_WHITESPACE
                       http_status  response_time
        Chrome                 200           0.02
        Comodo Dragon            0           0.00
        IE10                   404           0.08
        Iceweasel                0           0.00
        Safari                 404           0.07

        We can also reindex the columns.

        >>> df.reindex(columns=['http_status', 'user_agent']).sort_index()
        ... # doctest: +NORMALIZE_WHITESPACE
                       http_status  user_agent
        Chrome                 200         NaN
        Comodo Dragon            0         NaN
        IE10                   404         NaN
        Iceweasel                0         NaN
        Safari                 404         NaN

        Or we can use "axis-style" keyword arguments

        >>> df.reindex(['http_status', 'user_agent'], axis="columns").sort_index()
        ... # doctest: +NORMALIZE_WHITESPACE
                      http_status  user_agent
        Chrome                 200         NaN
        Comodo Dragon            0         NaN
        IE10                   404         NaN
        Iceweasel                0         NaN
        Safari                 404         NaN

        To further illustrate the filling functionality in
        ``reindex``, we will create a dataframe with a
        monotonically increasing index (for example, a sequence
        of dates).

        >>> date_index = pd.date_range('1/1/2010', periods=6, freq='D')
        >>> df2 = ks.DataFrame({"prices": [100, 101, np.nan, 100, 89, 88]},
        ...                    index=date_index)
        >>> df2.sort_index()  # doctest: +NORMALIZE_WHITESPACE
                    prices
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0

        Suppose we decide to expand the dataframe to cover a wider
        date range.

        >>> date_index2 = pd.date_range('12/29/2009', periods=10, freq='D')
        >>> df2.reindex(date_index2).sort_index()  # doctest: +NORMALIZE_WHITESPACE
                    prices
        2009-12-29     NaN
        2009-12-30     NaN
        2009-12-31     NaN
        2010-01-01   100.0
        2010-01-02   101.0
        2010-01-03     NaN
        2010-01-04   100.0
        2010-01-05    89.0
        2010-01-06    88.0
        2010-01-07     NaN
        """
        if axis is not None and (index is not None or columns is not None):
            raise TypeError("Cannot specify both 'axis' and any of 'index' or 'columns'.")

        if labels is not None:
            if axis in ('index', 0, None):
                index = labels
            elif axis in ('columns', 1):
                columns = labels
            else:
                raise ValueError("No axis named %s for object type %s." % (axis, type(axis)))

        if index is not None and not is_list_like(index):
            raise TypeError("Index must be called with a collection of some kind, "
                            "%s was passed" % type(index))

        if columns is not None and not is_list_like(columns):
            raise TypeError("Columns must be called with a collection of some kind, "
                            "%s was passed" % type(columns))

        df = self.copy()

        if index is not None:
            df = DataFrame(df._reindex_index(index))

        if columns is not None:
            df = DataFrame(df._reindex_columns(columns))

        # Process missing values.
        if fill_value is not None:
            df = df.fillna(fill_value)

        # Copy
        if copy:
            return df.copy()
        else:
            self._internal = df._internal
            return self

    def _reindex_index(self, index):
        # When axis is index, we can mimic pandas' by a right outer join.
        index_column = self._internal.index_columns
        assert len(index_column) <= 1, "Index should be single column or not set."

        if len(index_column) == 1:
            kser = ks.Series(list(index))
            index_column = index_column[0]
            labels = kser._kdf._sdf.select(kser._scol.alias(index_column))
        else:
            index_column = None
            labels = ks.Series(index).to_frame()._sdf

        joined_df = self._sdf.join(labels, on=index_column, how="right")
        new_data_columns = filter(lambda x: x not in index_column, joined_df.columns)
        if index_column is not None:
            index_map = [(index_column, None)]  # type: List[IndexMap]
            internal = self._internal.copy(
                sdf=joined_df,
                data_columns=list(new_data_columns),
                index_map=index_map)
        else:
            internal = self._internal.copy(
                sdf=joined_df,
                data_columns=list(new_data_columns))
        return internal

    def _reindex_columns(self, columns):
        label_columns = list(columns)
        null_columns = [
            F.lit(np.nan).alias(label_column) for label_column
            in label_columns if label_column not in self.columns]

        # Concatenate all fields
        sdf = self._sdf.select(
            self._internal.index_columns +
            list(map(F.col, self.columns)) +
            null_columns)

        # Only select label_columns (with index columns)
        sdf = sdf.select(self._internal.index_columns + label_columns)
        return self._internal.copy(
            sdf=sdf,
            data_columns=label_columns)

    def melt(self, id_vars=None, value_vars=None, var_name='variable',
             value_name='value'):
        """
        Unpivot a DataFrame from wide format to long format, optionally
        leaving identifier variables set.

        This function is useful to massage a DataFrame into a format where one
        or more columns are identifier variables (`id_vars`), while all other
        columns, considered measured variables (`value_vars`), are "unpivoted" to
        the row axis, leaving just two non-identifier columns, 'variable' and
        'value'.

        Parameters
        ----------
        frame : DataFrame
        id_vars : tuple, list, or ndarray, optional
            Column(s) to use as identifier variables.
        value_vars : tuple, list, or ndarray, optional
            Column(s) to unpivot. If not specified, uses all columns that
            are not set as `id_vars`.
        var_name : scalar, default 'variable'
            Name to use for the 'variable' column.
        value_name : scalar, default 'value'
            Name to use for the 'value' column.

        Returns
        -------
        DataFrame
            Unpivoted DataFrame.

        Examples
        --------
        >>> df = ks.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
        ...                    'B': {0: 1, 1: 3, 2: 5},
        ...                    'C': {0: 2, 1: 4, 2: 6}})
        >>> df
           A  B  C
        0  a  1  2
        1  b  3  4
        2  c  5  6

        >>> ks.melt(df)
          variable value
        0        A     a
        1        B     1
        2        C     2
        3        A     b
        4        B     3
        5        C     4
        6        A     c
        7        B     5
        8        C     6

        >>> df.melt(id_vars='A')
           A variable  value
        0  a        B      1
        1  a        C      2
        2  b        B      3
        3  b        C      4
        4  c        B      5
        5  c        C      6

        >>> ks.melt(df, id_vars=['A', 'B'])
           A  B variable  value
        0  a  1        C      2
        1  b  3        C      4
        2  c  5        C      6

        >>> df.melt(id_vars=['A'], value_vars=['C'])
           A variable  value
        0  a        C      2
        1  b        C      4
        2  c        C      6

        The names of 'variable' and 'value' columns can be customized:

        >>> ks.melt(df, id_vars=['A'], value_vars=['B'],
        ...         var_name='myVarname', value_name='myValname')
           A myVarname  myValname
        0  a         B          1
        1  b         B          3
        2  c         B          5
        """
        if id_vars is None:
            id_vars = []
        if not isinstance(id_vars, (list, tuple, np.ndarray)):
            id_vars = list(id_vars)

        data_columns = self._internal.data_columns

        if value_vars is None:
            value_vars = []
        if not isinstance(value_vars, (list, tuple, np.ndarray)):
            value_vars = list(value_vars)
        if len(value_vars) == 0:
            value_vars = data_columns

        data_columns = [data_column for data_column in data_columns if data_column not in id_vars]
        sdf = self._sdf

        pairs = F.explode(F.array(*[
            F.struct(*(
                [F.lit(column).alias(var_name)] +
                [F.col(column).alias(value_name)])
            ) for column in data_columns if column in value_vars]))

        columns = (id_vars +
                   [F.col("pairs.%s" % var_name), F.col("pairs.%s" % value_name)])
        exploded_df = sdf.withColumn("pairs", pairs).select(columns)

        return DataFrame(exploded_df)

    # TODO: axis, skipna, and many arguments should be implemented.
    def all(self, axis: Union[int, str] = 0) -> bool:
        """
        Return whether all elements are True.

        Returns True unless there is at least one element within a series that is
        False or equivalent (e.g. zero or empty)

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Indicate which axis or axes should be reduced.

            * 0 / 'index' : reduce the index, return a Series whose index is the
              original column labels.

        Examples
        --------
        Create a dataframe from a dictionary.

        >>> df = ks.DataFrame({
        ...    'col1': [True, True, True],
        ...    'col2': [True, False, False],
        ...    'col3': [0, 0, 0],
        ...    'col4': [1, 2, 3],
        ...    'col5': [True, True, None],
        ...    'col6': [True, False, None]},
        ...    columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])

        Default behaviour checks if column-wise values all return a boolean.

        >>> df.all()
        col1     True
        col2    False
        col3    False
        col4     True
        col5     True
        col6    False
        Name: all, dtype: bool

        Returns
        -------
        Series
        """

        if axis not in [0, 'index']:
            raise ValueError('axis should be either 0 or "index" currently.')

        applied = []
        data_columns = self._internal.data_columns
        for column in data_columns:
            col = self[column]._scol
            all_col = F.min(F.coalesce(col.cast('boolean'), F.lit(True)))
            applied.append(F.when(all_col.isNull(), True).otherwise(all_col))

        # TODO: there is a similar logic to transpose in, for instance,
        #  DataFrame.any, Series.quantile. Maybe we should deduplicate it.
        sdf = self._sdf
        internal_index_column = "__index_level_0__"
        value_column = "value"
        cols = []
        for data_column, applied_col in zip(data_columns, applied):
            cols.append(F.struct(
                F.lit(data_column).alias(internal_index_column),
                applied_col.alias(value_column)))

        sdf = sdf.select(
            F.array(*cols).alias("arrays")
        ).select(F.explode(F.col("arrays")))

        sdf = sdf.selectExpr("col.*")

        internal = self._internal.copy(
            sdf=sdf,
            data_columns=[value_column],
            index_map=[(internal_index_column, None)])

        ser = DataFrame(internal)[value_column].rename("all")
        return ser

    # TODO: axis, skipna, and many arguments should be implemented.
    def any(self, axis: Union[int, str] = 0) -> bool:
        """
        Return whether any element is True.

        Returns False unless there is at least one element within a series that is
        True or equivalent (e.g. non-zero or non-empty).

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Indicate which axis or axes should be reduced.

            * 0 / 'index' : reduce the index, return a Series whose index is the
              original column labels.

        Examples
        --------
        Create a dataframe from a dictionary.

        >>> df = ks.DataFrame({
        ...    'col1': [False, False, False],
        ...    'col2': [True, False, False],
        ...    'col3': [0, 0, 1],
        ...    'col4': [0, 1, 2],
        ...    'col5': [False, False, None],
        ...    'col6': [True, False, None]},
        ...    columns=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])

        Default behaviour checks if column-wise values all return a boolean.

        >>> df.any()
        col1    False
        col2     True
        col3     True
        col4     True
        col5    False
        col6     True
        Name: any, dtype: bool

        Returns
        -------
        Series
        """

        if axis not in [0, 'index']:
            raise ValueError('axis should be either 0 or "index" currently.')

        applied = []
        data_columns = self._internal.data_columns
        for column in data_columns:
            col = self[column]._scol
            all_col = F.max(F.coalesce(col.cast('boolean'), F.lit(False)))
            applied.append(F.when(all_col.isNull(), False).otherwise(all_col))

        # TODO: there is a similar logic to transpose in, for instance,
        #  DataFrame.all, Series.quantile. Maybe we should deduplicate it.
        sdf = self._sdf
        internal_index_column = "__index_level_0__"
        value_column = "value"
        cols = []
        for data_column, applied_col in zip(data_columns, applied):
            cols.append(F.struct(
                F.lit(data_column).alias(internal_index_column),
                applied_col.alias(value_column)))

        sdf = sdf.select(
            F.array(*cols).alias("arrays")
        ).select(F.explode(F.col("arrays")))

        sdf = sdf.selectExpr("col.*")

        internal = self._internal.copy(
            sdf=sdf,
            data_columns=[value_column],
            index_map=[(internal_index_column, None)])

        ser = DataFrame(internal)[value_column].rename("any")
        return ser

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
        if method not in ['average', 'min', 'max', 'first', 'dense']:
            msg = "method must be one of 'average', 'min', 'max', 'first', 'dense'"
            raise ValueError(msg)

        if ascending:
            asc_func = spark.functions.asc
        else:
            asc_func = spark.functions.desc

        index_column = self._internal.index_columns[0]
        data_columns = self._internal.data_columns
        sdf = self._sdf

        for column_name in data_columns:
            if method == 'first':
                window = Window.orderBy(asc_func(column_name), asc_func(index_column))\
                    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
                sdf = sdf.withColumn(column_name, F.row_number().over(window))
            elif method == 'dense':
                window = Window.orderBy(asc_func(column_name))\
                    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
                sdf = sdf.withColumn(column_name, F.dense_rank().over(window))
            else:
                if method == 'average':
                    stat_func = F.mean
                elif method == 'min':
                    stat_func = F.min
                elif method == 'max':
                    stat_func = F.max
                window = Window.orderBy(asc_func(column_name))\
                    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
                sdf = sdf.withColumn('rank', F.row_number().over(window))
                window = Window.partitionBy(column_name)\
                    .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
                sdf = sdf.withColumn(column_name, stat_func(F.col('rank')).over(window))

        return DataFrame(self._internal.copy(sdf=sdf.select(self._internal.columns)))\
            .astype(np.float64)

    def _pd_getitem(self, key):
        from databricks.koalas.series import Series
        if key is None:
            raise KeyError("none key")
        if isinstance(key, str):
            try:
                return Series(self._internal.copy(scol=self._sdf.__getitem__(key)), anchor=self)
            except AnalysisException:
                raise KeyError(key)
        if np.isscalar(key) or isinstance(key, (tuple, str)):
            raise NotImplementedError(key)
        elif isinstance(key, slice):
            return self.loc[key]

        if isinstance(key, (pd.Series, np.ndarray, pd.Index)):
            raise NotImplementedError(key)
        if isinstance(key, list):
            return self.loc[:, key]
        if isinstance(key, DataFrame):
            # TODO Should not implement alignment, too dangerous?
            return Series(self._internal.copy(scol=self._sdf.__getitem__(key)), anchor=self)
        if isinstance(key, Series):
            # TODO Should not implement alignment, too dangerous?
            # It is assumed to be only a filter, otherwise .loc should be used.
            bcol = key._scol.cast("boolean")
            return DataFrame(self._internal.copy(sdf=self._sdf.filter(bcol)))
        raise NotImplementedError(key)

    def __repr__(self):
        pdf = self.head(max_display_count + 1).to_pandas()
        pdf_length = len(pdf)
        repr_string = repr(pdf.iloc[:max_display_count])
        if pdf_length > max_display_count:
            match = REPR_PATTERN.search(repr_string)
            if match is not None:
                nrows = match.group("rows")
                ncols = match.group("columns")
                footer = ("\n\n[Showing only the first {nrows} rows x {ncols} columns]"
                          .format(nrows=nrows, ncols=ncols))
                return REPR_PATTERN.sub(footer, repr_string)
        return repr_string

    def _repr_html_(self):
        pdf = self.head(max_display_count + 1).to_pandas()
        pdf_length = len(pdf)
        repr_html = pdf[:max_display_count]._repr_html_()
        if pdf_length > max_display_count:
            match = REPR_HTML_PATTERN.search(repr_html)
            if match is not None:
                nrows = match.group("rows")
                ncols = match.group("columns")
                by = chr(215)
                footer = ('\n<p>Showing only the first {rows} rows {by} {cols} columns</p>\n</div>'
                          .format(rows=nrows,
                                  by=by,
                                  cols=ncols))
                return REPR_HTML_PATTERN.sub(footer, repr_html)
        return repr_html

    def __getitem__(self, key):
        return self._pd_getitem(key)

    def __setitem__(self, key, value):
        from databricks.koalas.series import Series
        # For now, we don't support realignment against different dataframes.
        # This is too expensive in Spark.
        # Are we assigning against a column?
        if isinstance(value, Series):
            assert value._kdf is self, \
                "Cannot combine column argument because it comes from a different dataframe"
        if isinstance(key, (tuple, list)):
            assert isinstance(value.schema, StructType)
            field_names = value.schema.fieldNames()
            kdf = self.assign(**{k: value[c] for k, c in zip(key, field_names)})
        else:
            kdf = self.assign(**{key: value})

        self._internal = kdf._internal

    def __getattr__(self, key: str) -> Any:
        from databricks.koalas.series import Series
        if key.startswith("__") or key.startswith("_pandas_") or key.startswith("_spark_"):
            raise AttributeError(key)
        if hasattr(_MissingPandasLikeDataFrame, key):
            property_or_func = getattr(_MissingPandasLikeDataFrame, key)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        return Series(self._internal.copy(scol=self._sdf.__getattr__(key)), anchor=self)

    def __len__(self):
        return self._sdf.count()

    def __dir__(self):
        fields = [f for f in self._sdf.schema.fieldNames() if ' ' not in f]
        return super(DataFrame, self).__dir__() + fields

    @classmethod
    def _validate_axis(cls, axis=0):
        if axis not in (0, 1, 'index', 'columns', None):
            raise ValueError('No axis named {0}'.format(axis))
        # convert to numeric axis
        return {None: 0, 'index': 0, 'columns': 1}.get(axis, axis)

    if sys.version_info >= (3, 7):
        def __class_getitem__(cls, params):
            # This is a workaround to support variadic generic in DataFrame in Python 3.7.
            # See https://github.com/python/typing/issues/193
            # we always wraps the given type hints by a tuple to mimic the variadic generic.
            return super(cls, DataFrame).__class_getitem__(Tuple[params])
    elif (3, 5) <= sys.version_info < (3, 7):
        # This is a workaround to support variadic generic in DataFrame in Python 3.5+
        # The implementation is in its metaclass so this flag is needed to distinguish
        # Koalas DataFrame.
        is_dataframe = None


def _reduce_spark_multi(sdf, aggs):
    """
    Performs a reduction on a dataframe, the functions being known sql aggregate functions.
    """
    assert isinstance(sdf, spark.DataFrame)
    sdf0 = sdf.agg(*aggs)
    l = sdf0.head(2)
    assert len(l) == 1, (sdf, l)
    row = l[0]
    l2 = list(row)
    assert len(l2) == len(aggs), (row, l2)
    return l2


class _CachedDataFrame(DataFrame):
    """
    Cached Koalas DataFrame, which corresponds to Pandas DataFrame logically, but internally
    it caches the corresponding Spark DataFrame.
    """
    def __init__(self, internal):
        self._cached = internal._sdf.cache()
        super(_CachedDataFrame, self).__init__(internal)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.unpersist()

    def unpersist(self):
        """
        The `unpersist` function is used to uncache the Koalas DataFrame when it
        is not used with `with` statement.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df = df.cache()

        To uncache the dataframe, use `unpersist` function

        >>> df.unpersist()
        """
        if self._cached.is_cached:
            self._cached.unpersist()
