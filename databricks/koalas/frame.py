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
from collections import OrderedDict, defaultdict
from distutils.version import LooseVersion
import re
import warnings
import inspect
import json
from functools import partial, reduce
import sys
from itertools import zip_longest
from typing import Any, Optional, List, Tuple, Union, Generic, TypeVar, Iterable, Dict, Callable

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like, is_dict_like, is_scalar

if LooseVersion(pd.__version__) >= LooseVersion("0.24"):
    from pandas.core.dtypes.common import infer_dtype_from_object
else:
    from pandas.core.dtypes.common import _get_dtype_from_object as infer_dtype_from_object
from pandas.core.accessor import CachedAccessor
from pandas.core.dtypes.inference import is_sequence
import pyspark
from pyspark import StorageLevel
from pyspark import sql as spark
from pyspark.sql import functions as F, Column
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.readwriter import OptionUtils
from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NumericType,
    ShortType,
    StructType,
    StructField,
)
from pyspark.sql.window import Window

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.config import option_context, get_option
from databricks.koalas.utils import (
    validate_arguments_and_invoke_function,
    align_diff_frames,
    validate_bool_kwarg,
    column_labels_level,
    name_like_string,
    scol_for,
    validate_axis,
    verify_temp_column_name,
)
from databricks.koalas.generic import _Frame
from databricks.koalas.internal import (
    _InternalFrame,
    HIDDEN_COLUMNS,
    NATURAL_ORDER_COLUMN_NAME,
    SPARK_INDEX_NAME_FORMAT,
    SPARK_DEFAULT_INDEX_NAME,
)
from databricks.koalas.missing.frame import _MissingPandasLikeDataFrame
from databricks.koalas.ml import corr
from databricks.koalas.typedef import infer_return_type, as_spark_type
from databricks.koalas.plot import KoalasFramePlotMethods

# These regular expression patterns are complied and defined here to avoid to compile the same
# pattern every time it is used in _repr_ and _repr_html_ in DataFrame.
# Two patterns basically seek the footer string from Pandas'
REPR_PATTERN = re.compile(r"\n\n\[(?P<rows>[0-9]+) rows x (?P<columns>[0-9]+) columns\]$")
REPR_HTML_PATTERN = re.compile(
    r"\n\<p\>(?P<rows>[0-9]+) rows × (?P<columns>[0-9]+) columns\<\/p\>\n\<\/div\>$"
)


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
results. Also reverse version.

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

>>> df.add(df)
           angles  degrees
circle          0      720
triangle        6      360
rectangle       8      720

>>> df.radd(1)
           angles  degrees
circle          1      361
triangle        4      181
rectangle       5      361

Divide and true divide by constant with reverse version.

>>> df / 10
           angles  degrees
circle        0.0     36.0
triangle      0.3     18.0
rectangle     0.4     36.0

>>> df.div(10)
           angles  degrees
circle        0.0     36.0
triangle      0.3     18.0
rectangle     0.4     36.0

>>> df.rdiv(10)
             angles   degrees
circle          inf  0.027778
triangle   3.333333  0.055556
rectangle  2.500000  0.027778

>>> df.truediv(10)
           angles  degrees
circle        0.0     36.0
triangle      0.3     18.0
rectangle     0.4     36.0

>>> df.rtruediv(10)
             angles   degrees
circle          inf  0.027778
triangle   3.333333  0.055556
rectangle  2.500000  0.027778

Subtract by constant with reverse version.

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

>>> df.rsub(1)
           angles  degrees
circle          1     -359
triangle       -2     -179
rectangle      -3     -359

Multiply by constant with reverse version.

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

>>> df.rmul(1)
           angles  degrees
circle          0      360
triangle        3      180
rectangle       4      360

Floor Divide by constant with reverse version.

>>> df // 10
           angles  degrees
circle        0.0     36.0
triangle      0.0     18.0
rectangle     0.0     36.0

>>> df.floordiv(10)
           angles  degrees
circle        0.0     36.0
triangle      0.0     18.0
rectangle     0.0     36.0

>>> df.rfloordiv(10)  # doctest: +SKIP
           angles  degrees
circle        inf      0.0
triangle      3.0      0.0
rectangle     2.0      0.0

Mod by constant with reverse version.

>>> df % 2
           angles  degrees
circle          0        0
triangle        1        0
rectangle       0        0

>>> df.mod(2)
           angles  degrees
circle          0        0
triangle        1        0
rectangle       0        0

>>> df.rmod(2)
           angles  degrees
circle        NaN        2
triangle      2.0        2
rectangle     2.0        2

Power by constant with reverse version.

>>> df ** 2
           angles   degrees
circle        0.0  129600.0
triangle      9.0   32400.0
rectangle    16.0  129600.0

>>> df.pow(2)
           angles   degrees
circle        0.0  129600.0
triangle      9.0   32400.0
rectangle    16.0  129600.0

>>> df.rpow(2)
           angles        degrees
circle        1.0  2.348543e+108
triangle      8.0   1.532496e+54
rectangle    16.0  2.348543e+108
"""

T = TypeVar("T")


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
    Koalas DataFrame that corresponds to Pandas DataFrame logically. This holds Spark DataFrame
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
            super(DataFrame, self).__init__(_InternalFrame(spark_frame=data, index_map=None))
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
        return self._internal.spark_frame

    @property
    def ndim(self):
        """
        Return an int representing the number of array dimensions.

        return 2 for DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=['cobra', 'viper', None],
        ...                   columns=['max_speed', 'shield'])
        >>> df
               max_speed  shield
        cobra          1       2
        viper          4       5
        NaN            7       8
        >>> df.ndim
        2
        """
        return 2

    @property
    def axes(self):
        """
        Return a list representing the axes of the DataFrame.

        It has the row axis labels and column axis labels as the only members.
        They are returned in that order.

        Examples
        --------

        >>> df = ks.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.axes
        [Int64Index([0, 1], dtype='int64'), Index(['col1', 'col2'], dtype='object')]
        """
        return [self.index, self.columns]

    def _reduce_for_stat_function(self, sfun, name, axis=None, numeric_only=True):
        """
        Applies sfun to each column and returns a pd.Series where the number of rows equal the
        number of columns.

        Parameters
        ----------
        sfun : either an 1-arg function that takes a Column and returns a Column, or
            a 2-arg function that takes a Column and its DataType and returns a Column.
            axis: used only for sanity check because series only support index axis.
        name : original pandas API name.
        axis : axis to apply. 0 or 1, or 'index' or 'columns.
        numeric_only : bool, default True
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility. Only 'DataFrame.count' uses this parameter
            currently.
        """
        from inspect import signature
        from databricks.koalas import Series
        from databricks.koalas.series import _col

        if name not in ("count", "min", "max") and not numeric_only:
            raise ValueError("Disabling 'numeric_only' parameter is not supported.")

        axis = validate_axis(axis)
        if axis == 0:
            exprs = []
            new_column_labels = []
            num_args = len(signature(sfun).parameters)
            for label in self._internal.column_labels:
                col_sdf = self._internal.spark_column_for(label)
                col_type = self._internal.spark_type_for(label)

                is_numeric_or_boolean = isinstance(col_type, (NumericType, BooleanType))
                min_or_max = sfun.__name__ in ("min", "max")
                keep_column = not numeric_only or is_numeric_or_boolean or min_or_max

                if keep_column:
                    if isinstance(col_type, BooleanType) and not min_or_max:
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
                    exprs.append(col_sdf.alias(name_like_string(label)))
                    new_column_labels.append(label)

            sdf = self._sdf.select(*exprs)

            # The data is expected to be small so it's fine to transpose/use default index.
            with ks.option_context(
                "compute.default_index_type", "distributed", "compute.max_rows", None
            ):
                kdf = DataFrame(sdf)
                internal = _InternalFrame(
                    kdf._internal.spark_frame,
                    index_map=kdf._internal.index_map,
                    column_labels=new_column_labels,
                    column_label_names=self._internal.column_label_names,
                )

                return _col(DataFrame(internal).transpose())

        elif axis == 1:
            # Here we execute with the first 1000 to get the return type.
            # If the records were less than 1000, it uses pandas API directly for a shortcut.
            limit = get_option("compute.shortcut_limit")
            pdf = self.head(limit + 1)._to_internal_pandas()
            pser = getattr(pdf, name)(axis=axis, numeric_only=numeric_only)
            if len(pdf) <= limit:
                return Series(pser)

            @pandas_udf(returnType=as_spark_type(pser.dtype.type))
            def calculate_columns_axis(*cols):
                return getattr(pd.concat(cols, axis=1), name)(axis=axis, numeric_only=numeric_only)

            df = self._sdf.select(
                calculate_columns_axis(*self._internal.data_spark_columns).alias("0")
            )
            return DataFrame(df)["0"]

        else:
            raise ValueError("No axis named %s for object type %s." % (axis, type(axis)))

    def _kser_for(self, label):
        """
        Create Series with a proper column label.

        The given label must be verified to exist in `_InternalFrame.column_labels`.

        For example, in some method, self is like:

        >>> self = ks.range(3)

        `self._kser_for(label)` can be used with `_InternalFrame.column_labels`:

        >>> self._kser_for(self._internal.column_labels[0])
        0    0
        1    1
        2    2
        Name: id, dtype: int64

        `self._kser_for(label)` must not be used directly with user inputs.
        In that case, `self[label]` should be used instead, which checks the label exists or not:

        >>> self['id']
        0    0
        1    1
        2    2
        Name: id, dtype: int64
        """
        from databricks.koalas.series import Series

        return Series(
            self._internal.copy(
                spark_column=self._internal.spark_column_for(label), column_labels=[label]
            ),
            anchor=self,
        )

    def _apply_series_op(self, op):
        applied = []
        for label in self._internal.column_labels:
            applied.append(op(self._kser_for(label)))
        internal = self._internal.with_new_columns(applied)
        return DataFrame(internal)

    # Arithmetic Operators
    def _map_series_op(self, op, other):
        from databricks.koalas.base import IndexOpsMixin

        if not isinstance(other, DataFrame) and (
            isinstance(other, IndexOpsMixin) or is_sequence(other)
        ):
            raise ValueError(
                "%s with a sequence is currently not supported; "
                "however, got %s." % (op, type(other))
            )

        if isinstance(other, DataFrame) and self is not other:
            if self._internal.column_labels_level != other._internal.column_labels_level:
                raise ValueError("cannot join with no overlapping index names")

            # Different DataFrames
            def apply_op(kdf, this_column_labels, that_column_labels):
                for this_label, that_label in zip(this_column_labels, that_column_labels):
                    yield (
                        getattr(kdf._kser_for(this_label), op)(kdf._kser_for(that_label)),
                        this_label,
                    )

            return align_diff_frames(apply_op, self, other, fillna=True, how="full")
        else:
            # DataFrame and Series
            if isinstance(other, DataFrame):
                return self._apply_series_op(lambda kser: getattr(kser, op)(other[kser.name]))
            else:
                return self._apply_series_op(lambda kser: getattr(kser, op)(other))

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

    # create accessor for plot
    plot = CachedAccessor("plot", KoalasFramePlotMethods)

    def hist(self, bins=10, **kwds):
        return self.plot.hist(bins, **kwds)

    hist.__doc__ = KoalasFramePlotMethods.hist.__doc__

    def kde(self, bw_method=None, ind=None, **kwds):
        return self.plot.kde(bw_method, ind, **kwds)

    kde.__doc__ = KoalasFramePlotMethods.kde.__doc__

    add.__doc__ = _flex_doc_FRAME.format(
        desc="Addition", op_name="+", equiv="dataframe + other", reverse="radd"
    )

    def radd(self, other):
        return other + self

    radd.__doc__ = _flex_doc_FRAME.format(
        desc="Addition", op_name="+", equiv="other + dataframe", reverse="add"
    )

    def div(self, other):
        return self / other

    div.__doc__ = _flex_doc_FRAME.format(
        desc="Floating division", op_name="/", equiv="dataframe / other", reverse="rdiv"
    )

    divide = div

    def rdiv(self, other):
        return other / self

    rdiv.__doc__ = _flex_doc_FRAME.format(
        desc="Floating division", op_name="/", equiv="other / dataframe", reverse="div"
    )

    def truediv(self, other):
        return self / other

    truediv.__doc__ = _flex_doc_FRAME.format(
        desc="Floating division", op_name="/", equiv="dataframe / other", reverse="rtruediv"
    )

    def rtruediv(self, other):
        return other / self

    rtruediv.__doc__ = _flex_doc_FRAME.format(
        desc="Floating division", op_name="/", equiv="other / dataframe", reverse="truediv"
    )

    def mul(self, other):
        return self * other

    mul.__doc__ = _flex_doc_FRAME.format(
        desc="Multiplication", op_name="*", equiv="dataframe * other", reverse="rmul"
    )

    multiply = mul

    def rmul(self, other):
        return other * self

    rmul.__doc__ = _flex_doc_FRAME.format(
        desc="Multiplication", op_name="*", equiv="other * dataframe", reverse="mul"
    )

    def sub(self, other):
        return self - other

    sub.__doc__ = _flex_doc_FRAME.format(
        desc="Subtraction", op_name="-", equiv="dataframe - other", reverse="rsub"
    )

    subtract = sub

    def rsub(self, other):
        return other - self

    rsub.__doc__ = _flex_doc_FRAME.format(
        desc="Subtraction", op_name="-", equiv="other - dataframe", reverse="sub"
    )

    def mod(self, other):
        return self % other

    mod.__doc__ = _flex_doc_FRAME.format(
        desc="Modulo", op_name="%", equiv="dataframe % other", reverse="rmod"
    )

    def rmod(self, other):
        return other % self

    rmod.__doc__ = _flex_doc_FRAME.format(
        desc="Modulo", op_name="%", equiv="other % dataframe", reverse="mod"
    )

    def pow(self, other):
        return self ** other

    pow.__doc__ = _flex_doc_FRAME.format(
        desc="Exponential power of series", op_name="**", equiv="dataframe ** other", reverse="rpow"
    )

    def rpow(self, other):
        return other ** self

    rpow.__doc__ = _flex_doc_FRAME.format(
        desc="Exponential power", op_name="**", equiv="other ** dataframe", reverse="pow"
    )

    def floordiv(self, other):
        return self // other

    floordiv.__doc__ = _flex_doc_FRAME.format(
        desc="Integer division", op_name="//", equiv="dataframe // other", reverse="rfloordiv"
    )

    def rfloordiv(self, other):
        return other // self

    rfloordiv.__doc__ = _flex_doc_FRAME.format(
        desc="Integer division", op_name="//", equiv="other // dataframe", reverse="floordiv"
    )

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
               a      b
        a   True   True
        b  False  False
        c  False   True
        d  False  False
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
        b  False  False
        c   True  False
        d   True  False
        """
        return self > other

    def ge(self, other):
        """
        Compare if the current value is greater than or equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.ge(1)
              a      b
        a  True   True
        b  True  False
        c  True   True
        d  True  False
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
        b  False  False
        c  False  False
        d  False  False
        """
        return self < other

    def le(self, other):
        """
        Compare if the current value is less than or equal to the other.

        >>> df = ks.DataFrame({'a': [1, 2, 3, 4],
        ...                    'b': [1, np.nan, 1, np.nan]},
        ...                   index=['a', 'b', 'c', 'd'], columns=['a', 'b'])

        >>> df.le(2)
               a      b
        a   True   True
        b   True  False
        c  False   True
        d  False  False
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
        b   True   True
        c   True  False
        d   True   True
        """
        return self != other

    def applymap(self, func):
        """
        Apply a function to a Dataframe elementwise.

        This method applies a function that accepts and returns a scalar
        to every element of a DataFrame.

        .. note:: this API executes the function once to infer the type which is
             potentially expensive, for instance, when the dataset is created after
             aggregations or sorting.

             To avoid this, specify return type in ``func``, for instance, as below:

             >>> def square(x) -> np.int32:
             ...     return x ** 2

             Koalas uses return type hint and does not try to infer the type.

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

        You can omit the type hint and let Koalas infer its type.

        >>> df.applymap(lambda x: x ** 2)
                   0          1
        0   1.000000   4.494400
        1  11.262736  20.857489
        """

        # TODO: We can implement shortcut theoretically since it creates new DataFrame
        #  anyway and we don't have to worry about operations on different DataFrames.
        return self._apply_series_op(lambda kser: kser.apply(func))

    # TODO: not all arguments are implemented comparing to Pandas' for now.
    def aggregate(self, func: Union[List[str], Dict[str, List[str]]]):
        """Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : dict or a list
             a dict mapping from column name (string) to
             aggregate functions (list of strings).
             If a list is given, the aggregation is performed against
             all columns.

        Returns
        -------
        DataFrame

        Notes
        -----
        `agg` is an alias for `aggregate`. Use the alias.

        See Also
        --------
        DataFrame.apply : Invoke function on DataFrame.
        DataFrame.transform : Only perform transforming type operations.
        DataFrame.groupby : Perform operations over groups.
        Series.aggregate : The equivalent function for Series.

        Examples
        --------
        >>> df = ks.DataFrame([[1, 2, 3],
        ...                    [4, 5, 6],
        ...                    [7, 8, 9],
        ...                    [np.nan, np.nan, np.nan]],
        ...                   columns=['A', 'B', 'C'])

        >>> df
             A    B    C
        0  1.0  2.0  3.0
        1  4.0  5.0  6.0
        2  7.0  8.0  9.0
        3  NaN  NaN  NaN

        Aggregate these functions over the rows.

        >>> df.agg(['sum', 'min'])[['A', 'B', 'C']]
                A     B     C
        min   1.0   2.0   3.0
        sum  12.0  15.0  18.0

        Different aggregations per column.

        >>> df.agg({'A' : ['sum', 'min'], 'B' : ['min', 'max']})[['A', 'B']]
                A    B
        max   NaN  8.0
        min   1.0  2.0
        sum  12.0  NaN
        """
        from databricks.koalas.groupby import GroupBy

        if isinstance(func, list):
            if all((isinstance(f, str) for f in func)):
                func = dict([(column, func) for column in self.columns])
            else:
                raise ValueError(
                    "If the given function is a list, it "
                    "should only contains function names as strings."
                )

        if not isinstance(func, dict) or not all(
            isinstance(key, str)
            and (
                isinstance(value, str)
                or isinstance(value, list)
                and all(isinstance(v, str) for v in value)
            )
            for key, value in func.items()
        ):
            raise ValueError(
                "aggs must be a dict mapping from column name (string) to aggregate "
                "functions (list of strings)."
            )

        kdf = DataFrame(GroupBy._spark_groupby(self, func))  # type: DataFrame

        # The codes below basically converts:
        #
        #           A         B
        #         sum  min  min  max
        #     0  12.0  1.0  2.0  8.0
        #
        # to:
        #             A    B
        #     max   NaN  8.0
        #     min   1.0  2.0
        #     sum  12.0  NaN
        #
        # Aggregated output is usually pretty much small. So it is fine to directly use pandas API.
        pdf = kdf.to_pandas().stack()
        pdf.index = pdf.index.droplevel()
        pdf.columns.names = [None]
        pdf.index.names = [None]

        return DataFrame(pdf[list(func.keys())])

    agg = aggregate

    def corr(self, method="pearson"):
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
        return ks.from_pandas(corr(self, method))

    def iteritems(self) -> Iterable:
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
        return [
            (label if len(label) > 1 else label[0], self._kser_for(label))
            for label in self._internal.column_labels
        ]

    def iterrows(self):
        """
        Iterate over DataFrame rows as (index, Series) pairs.

        Yields
        ------
        index : label or tuple of label
            The index of the row. A tuple for a `MultiIndex`.
        data : pandas.Series
            The data of the row as a Series.

        it : generator
            A generator that iterates over the rows of the frame.

        Notes
        -----

        1. Because ``iterrows`` returns a Series for each row,
           it does **not** preserve dtypes across the rows (dtypes are
           preserved across columns for DataFrames). For example,

           >>> df = ks.DataFrame([[1, 1.5]], columns=['int', 'float'])
           >>> row = next(df.iterrows())[1]
           >>> row
           int      1.0
           float    1.5
           Name: 0, dtype: float64
           >>> print(row['int'].dtype)
           float64
           >>> print(df['int'].dtype)
           int64

           To preserve dtypes while iterating over the rows, it is better
           to use :meth:`itertuples` which returns namedtuples of the values
           and which is generally faster than ``iterrows``.

        2. You should **never modify** something you are iterating over.
           This is not guaranteed to work in all cases. Depending on the
           data types, the iterator returns a copy and not a view, and writing
           to it will have no effect.
        """

        columns = self.columns
        internal_index_columns = self._internal.index_spark_column_names
        internal_data_columns = self._internal.data_spark_column_names

        def extract_kv_from_spark_row(row):
            k = (
                row[internal_index_columns[0]]
                if len(internal_index_columns) == 1
                else tuple(row[c] for c in internal_index_columns)
            )
            v = [row[c] for c in internal_data_columns]
            return k, v

        for k, v in map(extract_kv_from_spark_row, self._sdf.toLocalIterator()):
            s = pd.Series(v, index=columns, name=k)
            yield k, s

    def items(self) -> Iterable:
        """This is an alias of ``iteritems``."""
        return self.iteritems()

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
            kdf._to_internal_pandas(), self.to_clipboard, pd.DataFrame.to_clipboard, args
        )

    def to_html(
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
        justify=None,
        max_rows=None,
        max_cols=None,
        show_dimensions=False,
        decimal=".",
        bold_rows=True,
        classes=None,
        escape=True,
        notebook=False,
        border=None,
        table_id=None,
        render_links=False,
    ):
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
            kdf._to_internal_pandas(), self.to_html, pd.DataFrame.to_html, args
        )

    def to_string(
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
        justify=None,
        max_rows=None,
        max_cols=None,
        show_dimensions=False,
        decimal=".",
        line_width=None,
    ):
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
            kdf._to_internal_pandas(), self.to_string, pd.DataFrame.to_string, args
        )

    def to_dict(self, orient="dict", into=dict):
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
            kdf._to_internal_pandas(), self.to_dict, pd.DataFrame.to_dict, args
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
            kdf._to_internal_pandas(), self.to_latex, pd.DataFrame.to_latex, args
        )

    def to_markdown(self, buf=None, mode=None, max_rows=None):
        """
        Print DataFrame in Markdown-friendly format.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory. If the input
                  is large, set max_rows parameter.

        Parameters
        ----------
        buf : writable buffer, defaults to sys.stdout
            Where to send the output. By default, the output is printed to
            sys.stdout. Pass a writable buffer if you need to further process
            the output.
        mode : str, optional
            Mode in which file is opened.
        **kwargs
            These parameters will be passed to `tabulate`.

        Returns
        -------
        str
            DataFrame in Markdown-friendly format.

        Examples
        --------
        >>> df = ks.DataFrame(
        ...     data={"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]}
        ... )
        >>> print(df.to_markdown())  # doctest: +SKIP
        |    | animal_1   | animal_2   |
        |---:|:-----------|:-----------|
        |  0 | elk        | dog        |
        |  1 | pig        | quetzal    |
        """
        # `to_markdown` is supported in pandas >= 1.0.0 since it's newly added in pandas 1.0.0.
        if LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            raise NotImplementedError(
                "`to_markdown()` only supported in Kaoals with pandas >= 1.0.0"
            )
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        if max_rows is not None:
            kdf = self.head(max_rows)
        else:
            kdf = self
        return validate_arguments_and_invoke_function(
            kdf._to_internal_pandas(), self.to_markdown, pd.DataFrame.to_markdown, args
        )

    # TODO: enable doctests once we drop Spark 2.3.x (due to type coercion logic
    #  when creating arrays)
    def transpose(self):
        """
        Transpose index and columns.

        Reflect the DataFrame over its main diagonal by writing rows as columns
        and vice-versa. The property :attr:`.T` is an accessor to the method
        :meth:`transpose`.

        .. note:: This method is based on an expensive operation due to the nature
            of big data. Internally it needs to generate each row for each value, and
            then group twice - it is a huge operation. To prevent misusage, this method
            has the 'compute.max_rows' default limit of input length, and raises a ValueError.

                >>> from databricks.koalas.config import option_context
                >>> with option_context('compute.max_rows', 1000):  # doctest: +NORMALIZE_WHITESPACE
                ...     ks.DataFrame({'a': range(1001)}).transpose()
                Traceback (most recent call last):
                  ...
                ValueError: Current DataFrame has more then the given limit 1000 rows.
                Please set 'compute.max_rows' by using 'databricks.koalas.config.set_option'
                to retrieve to retrieve more than 1000 rows. Note that, before changing the
                'compute.max_rows', this operation is considerably expensive.

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
        max_compute_count = get_option("compute.max_rows")
        if max_compute_count is not None:
            pdf = self.head(max_compute_count + 1)._to_internal_pandas()
            if len(pdf) > max_compute_count:
                raise ValueError(
                    "Current DataFrame has more then the given limit {0} rows. "
                    "Please set 'compute.max_rows' by using 'databricks.koalas.config.set_option' "
                    "to retrieve to retrieve more than {0} rows. Note that, before changing the "
                    "'compute.max_rows', this operation is considerably expensive.".format(
                        max_compute_count
                    )
                )
            return DataFrame(pdf.transpose())

        # Explode the data to be pairs.
        #
        # For instance, if the current input DataFrame is as below:
        #
        # +------+------+------+------+------+
        # |index1|index2|(a,x1)|(a,x2)|(b,x3)|
        # +------+------+------+------+------+
        # |    y1|    z1|     1|     0|     0|
        # |    y2|    z2|     0|    50|     0|
        # |    y3|    z3|     3|     2|     1|
        # +------+------+------+------+------+
        #
        # Output of `exploded_df` becomes as below:
        #
        # +-----------------+-----------------+-----------------+-----+
        # |            index|__index_level_0__|__index_level_1__|value|
        # +-----------------+-----------------+-----------------+-----+
        # |{"a":["y1","z1"]}|                a|               x1|    1|
        # |{"a":["y1","z1"]}|                a|               x2|    0|
        # |{"a":["y1","z1"]}|                b|               x3|    0|
        # |{"a":["y2","z2"]}|                a|               x1|    0|
        # |{"a":["y2","z2"]}|                a|               x2|   50|
        # |{"a":["y2","z2"]}|                b|               x3|    0|
        # |{"a":["y3","z3"]}|                a|               x1|    3|
        # |{"a":["y3","z3"]}|                a|               x2|    2|
        # |{"a":["y3","z3"]}|                b|               x3|    1|
        # +-----------------+-----------------+-----------------+-----+
        pairs = F.explode(
            F.array(
                *[
                    F.struct(
                        [
                            F.lit(col).alias(SPARK_INDEX_NAME_FORMAT(i))
                            for i, col in enumerate(label)
                        ]
                        + [self._internal.spark_column_for(label).alias("value")]
                    )
                    for label in self._internal.column_labels
                ]
            )
        )

        exploded_df = self._sdf.withColumn("pairs", pairs).select(
            [
                F.to_json(
                    F.struct(
                        F.array(
                            [scol.cast("string") for scol in self._internal.index_spark_columns]
                        ).alias("a")
                    )
                ).alias("index"),
                F.col("pairs.*"),
            ]
        )

        # After that, executes pivot with key and its index column.
        # Note that index column should contain unique values since column names
        # should be unique.
        internal_index_columns = [
            SPARK_INDEX_NAME_FORMAT(i) for i in range(self._internal.column_labels_level)
        ]
        pivoted_df = exploded_df.groupBy(internal_index_columns).pivot("index")

        transposed_df = pivoted_df.agg(F.first(F.col("value")))

        new_data_columns = list(
            filter(lambda x: x not in internal_index_columns, transposed_df.columns)
        )

        internal = self._internal.copy(
            spark_frame=transposed_df,
            index_map=OrderedDict((col, None) for col in internal_index_columns),
            column_labels=[tuple(json.loads(col)["a"]) for col in new_data_columns],
            data_spark_columns=[scol_for(transposed_df, col) for col in new_data_columns],
            column_label_names=None,
        )

        return DataFrame(internal)

    T = property(transpose)

    def apply_batch(self, func):
        """
        Apply a function that takes pandas DataFrame and outputs pandas DataFrame. The pandas
        DataFrame given to the function is of a batch used internally.

        .. note:: the `func` is unable to access to the whole input frame. Koalas internally
            splits the input series into multiple batches and calls `func` with each batch multiple
            times. Therefore, operations such as global aggregations are impossible. See the example
            below.

            >>> # This case does not return the length of whole frame but of the batch internally
            ... # used.
            ... def length(pdf) -> ks.DataFrame[int]:
            ...     return pd.DataFrame([len(pdf)])
            ...
            >>> df = ks.DataFrame({'A': range(1000)})
            >>> df.apply_batch(length)  # doctest: +SKIP
                c0
            0   83
            1   83
            2   83
            ...
            10  83
            11  83

        .. note:: this API executes the function once to infer the type which is
            potentially expensive, for instance, when the dataset is created after
            aggregations or sorting.

            To avoid this, specify return type in ``func``, for instance, as below:

            >>> def plus_one(x) -> ks.DataFrame[float, float]:
            ...    return x + 1

            If the return type is specified, the output column names become
            `c0, c1, c2 ... cn`. These names are positionally mapped to the returned
            DataFrame in ``func``. See examples below.


        Parameters
        ----------
        func : function
            Function to apply to each pandas frame.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.apply: For row/columnwise operations.
        DataFrame.applymap: For elementwise operations.
        DataFrame.aggregate: Only perform aggregating type operations.
        DataFrame.transform: Only perform transforming type operations.
        Series.transform_batch: transform the search as each pandas chunks.

        Examples
        --------
        >>> df = ks.DataFrame([(1, 2), (3, 4), (5, 6)], columns=['A', 'B'])
        >>> df
           A  B
        0  1  2
        1  3  4
        2  5  6

        >>> def query_func(pdf) -> ks.DataFrame[int, int]:
        ...     return pdf.query('A == 1')
        >>> df.apply_batch(query_func)
           c0  c1
        0   1   2

        You can also omit the type hints so Koalas infers the return schema as below:

        >>> df.apply_batch(lambda pdf: pdf.query('A == 1'))
           A  B
        0  1  2
        """
        # TODO: codes here partially duplicate `DataFrame.apply`. Can we deduplicate?

        from databricks.koalas.groupby import GroupBy

        if isinstance(func, np.ufunc):
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)

        assert callable(func), "the first argument should be a callable function."

        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        should_infer_schema = return_sig is None
        should_use_map_in_pandas = LooseVersion(pyspark.__version__) >= "3.0"

        if should_infer_schema:
            # Here we execute with the first 1000 to get the return type.
            # If the records were less than 1000, it uses pandas API directly for a shortcut.
            limit = get_option("compute.shortcut_limit")
            pdf = self.head(limit + 1)._to_internal_pandas()
            applied = func(pdf)
            if not isinstance(applied, pd.DataFrame):
                raise ValueError(
                    "The given function should return a frame; however, "
                    "the return type was %s." % type(applied)
                )
            kdf = ks.DataFrame(applied)
            if len(pdf) <= limit:
                return kdf

            return_schema = kdf._internal.to_internal_spark_frame.schema
            if should_use_map_in_pandas:
                output_func = GroupBy._make_pandas_df_builder_func(
                    self, func, return_schema, retain_index=True
                )
                sdf = self._internal.to_internal_spark_frame.mapInPandas(
                    lambda iterator: map(output_func, iterator), schema=return_schema
                )
            else:
                sdf = GroupBy._spark_group_map_apply(
                    self, func, (F.spark_partition_id(),), return_schema, retain_index=True
                )

            # If schema is inferred, we can restore indexes too.
            internal = kdf._internal.with_new_sdf(sdf)
        else:
            return_schema = infer_return_type(func).tpe
            is_return_dataframe = getattr(return_sig, "__origin__", None) == ks.DataFrame
            if not is_return_dataframe:
                raise TypeError(
                    "The given function should specify a frame as its type "
                    "hints; however, the return type was %s." % return_sig
                )

            if should_use_map_in_pandas:
                output_func = GroupBy._make_pandas_df_builder_func(
                    self, func, return_schema, retain_index=False
                )
                sdf = self._internal.to_internal_spark_frame.mapInPandas(
                    lambda iterator: map(output_func, iterator), schema=return_schema
                )
            else:
                sdf = GroupBy._spark_group_map_apply(
                    self, func, (F.spark_partition_id(),), return_schema, retain_index=False
                )

            # Otherwise, it loses index.
            internal = _InternalFrame(spark_frame=sdf, index_map=None)

        return DataFrame(internal)

    def map_in_pandas(self, func):
        warnings.warn(
            "map_in_pandas is deprecated as of DataFrame.apply_batch. "
            "Please use the API instead.",
            DeprecationWarning,
        )
        return self.apply_batch(func)

    map_in_pandas.__doc__ = apply_batch.__doc__

    def apply(self, func, axis=0):
        """
        Apply a function along an axis of the DataFrame.

        Objects passed to the function are Series objects whose index is
        either the DataFrame's index (``axis=0``) or the DataFrame's columns
        (``axis=1``).

        .. note:: when `axis` is 0 or 'index', the `func` is unable to access
            to the whole input series. Koalas internally splits the input series into multiple
            batches and calls `func` with each batch multiple times. Therefore, operations
            such as global aggregations are impossible. See the example below.

            >>> # This case does not return the length of whole series but of the batch internally
            ... # used.
            ... def length(s) -> int:
            ...    return len(s)
            ...
            >>> df = ks.DataFrame({'A': range(1000)})
            >>> df.apply(length, axis=0)  # doctest: +SKIP
            0     83
            1     83
            2     83
            ...
            10    83
            11    83
            Name: 0, dtype: int32

        .. note:: this API executes the function once to infer the type which is
            potentially expensive, for instance, when the dataset is created after
            aggregations or sorting.

            To avoid this, specify the return type as `Series` or scalar value in ``func``,
            for instance, as below:

            >>> def square(s) -> ks.Series[np.int32]:
            ...     return s ** 2

            Koalas uses return type hint and does not try to infer the type.

            In case when axis is 1, it requires to specify `DataFrame` or scalar value
            with type hints as below:

            >>> def plus_one(x) -> ks.DataFrame[float, float]:
            ...    return x + 1

            If the return type is specified as `DataFrame`, the output column names become
            `c0, c1, c2 ... cn`. These names are positionally mapped to the returned
            DataFrame in ``func``. See examples below.

            However, this way switches the index type to default index type in the output
            because the type hint cannot express the index type at this moment. Use
            `reset_index()` to keep index as a workaround.

        Parameters
        ----------
        func : function
            Function to apply to each column or row.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis along which the function is applied:

            * 0 or 'index': apply function to each column.
            * 1 or 'columns': apply function to each row.

        Returns
        -------
        Series or DataFrame
            Result of applying ``func`` along the given axis of the
            DataFrame.

        See Also
        --------
        DataFrame.applymap : For elementwise operations.
        DataFrame.aggregate : Only perform aggregating type operations.
        DataFrame.transform : Only perform transforming type operations.
        Series.apply : The equivalent function for Series.

        Examples
        --------
        >>> df = ks.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
        >>> df
           A  B
        0  4  9
        1  4  9
        2  4  9

        Using a numpy universal function (in this case the same as
        ``np.sqrt(df)``):

        >>> def sqrt(x) -> ks.Series[float]:
        ...    return np.sqrt(x)
        ...
        >>> df.apply(sqrt, axis=0)
             A    B
        0  2.0  3.0
        1  2.0  3.0
        2  2.0  3.0

        You can omit the type hint and let Koalas infer its type.

        >>> df.apply(np.sqrt, axis=0)
             A    B
        0  2.0  3.0
        1  2.0  3.0
        2  2.0  3.0

        When `axis` is 1 or 'columns', it applies the function for each row.

        >>> def summation(x) -> np.int64:
        ...    return np.sum(x)
        ...
        >>> df.apply(summation, axis=1)
        0    13
        1    13
        2    13
        Name: 0, dtype: int64

        Likewise, you can omit the type hint and let Koalas infer its type.

        >>> df.apply(np.sum, axis=1)
        0    13
        1    13
        2    13
        Name: 0, dtype: int64

        Returning a list-like will result in a Series

        >>> df.apply(lambda x: [1, 2], axis=1)
        0    [1, 2]
        1    [1, 2]
        2    [1, 2]
        Name: 0, dtype: object

        In order to specify the types when `axis` is '1', it should use DataFrame[...]
        annotation. In this case, the column names are automatically generated.

        >>> def identify(x) -> ks.DataFrame[np.int64, np.int64]:
        ...     return x
        ...
        >>> df.apply(identify, axis=1)
           c0  c1
        0   4   9
        1   4   9
        2   4   9
        """
        from databricks.koalas.groupby import GroupBy
        from databricks.koalas.series import _col

        if isinstance(func, np.ufunc):
            f = func
            func = lambda *args, **kwargs: f(*args, **kwargs)

        assert callable(func), "the first argument should be a callable function."

        axis = validate_axis(axis)
        should_return_series = False
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        should_infer_schema = return_sig is None

        def apply_func(pdf):
            pdf_or_pser = pdf.apply(func, axis=axis)
            if isinstance(pdf_or_pser, pd.Series):
                return pdf_or_pser.to_frame()
            else:
                return pdf_or_pser

        if should_infer_schema:
            # Here we execute with the first 1000 to get the return type.
            # If the records were less than 1000, it uses pandas API directly for a shortcut.
            limit = get_option("compute.shortcut_limit")
            pdf = self.head(limit + 1)._to_internal_pandas()
            applied = pdf.apply(func, axis=axis)
            kser_or_kdf = ks.from_pandas(applied)
            if len(pdf) <= limit:
                return kser_or_kdf

            kdf = kser_or_kdf
            if isinstance(kser_or_kdf, ks.Series):
                should_return_series = True
                kdf = kser_or_kdf.to_frame()

            return_schema = kdf._internal._sdf.drop(*HIDDEN_COLUMNS).schema

            sdf = GroupBy._spark_group_map_apply(
                self, apply_func, (F.spark_partition_id(),), return_schema, retain_index=True
            )

            # If schema is inferred, we can restore indexes too.
            internal = kdf._internal.with_new_sdf(sdf)
        else:
            return_schema = infer_return_type(func).tpe
            require_index_axis = getattr(return_sig, "__origin__", None) == ks.Series
            require_column_axis = getattr(return_sig, "__origin__", None) == ks.DataFrame
            if require_index_axis:
                if axis != 0:
                    raise TypeError(
                        "The given function should specify a scalar or a series as its type "
                        "hints when axis is 0 or 'index'; however, the return type "
                        "was %s" % return_sig
                    )
                fields_types = zip(self.columns, [return_schema] * len(self.columns))
                return_schema = StructType([StructField(c, t) for c, t in fields_types])
            elif require_column_axis:
                if axis != 1:
                    raise TypeError(
                        "The given function should specify a scalar or a frame as its type "
                        "hints when axis is 1 or 'column'; however, the return type "
                        "was %s" % return_sig
                    )
            else:
                # any axis is fine.
                should_return_series = True
                return_schema = StructType([StructField("0", return_schema)])

            sdf = GroupBy._spark_group_map_apply(
                self, apply_func, (F.spark_partition_id(),), return_schema, retain_index=False
            )

            # Otherwise, it loses index.
            internal = _InternalFrame(spark_frame=sdf, index_map=None)

        result = DataFrame(internal)
        if should_return_series:
            return _col(result)
        else:
            return result

    def transform(self, func):
        """
        Call ``func`` on self producing a Series with transformed values
        and that has the same length as its input.

        .. note:: this API executes the function once to infer the type which is
             potentially expensive, for instance, when the dataset is created after
             aggregations or sorting.

             To avoid this, specify return type in ``func``, for instance, as below:

             >>> def square(x) -> ks.Series[np.int32]:
             ...     return x ** 2

             Koalas uses return type hint and does not try to infer the type.

        .. note:: the series within ``func`` is actually multiple pandas series as the
            segments of the whole Koalas series; therefore, the length of each series
            is not guaranteed. As an example, an aggregation against each series
            does work as a global aggregation but an aggregation of each segment. See
            below:

            >>> def func(x) -> ks.Series[np.int32]:
            ...     return x + sum(x)

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

        See Also
        --------
        DataFrame.aggregate : Only perform aggregating type operations.
        DataFrame.apply : Invoke function on DataFrame.
        Series.transform : The equivalent function for Series.

        Examples
        --------
        >>> df = ks.DataFrame({'A': range(3), 'B': range(1, 4)}, columns=['A', 'B'])
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

        You can omit the type hint and let Koalas infer its type.

        >>> df.transform(lambda x: x ** 2)
           A  B
        0  0  1
        1  1  4
        2  4  9

        For multi-index columns:

        >>> df.columns = [('X', 'A'), ('X', 'B')]
        >>> df.transform(square)  # doctest: +NORMALIZE_WHITESPACE
           X
           A  B
        0  0  1
        1  1  4
        2  4  9

        >>> df.transform(lambda x: x ** 2)  # doctest: +NORMALIZE_WHITESPACE
           X
           A  B
        0  0  1
        1  1  4
        2  4  9
        """
        assert callable(func), "the first argument should be a callable function."
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        should_infer_schema = return_sig is None

        if should_infer_schema:
            # Here we execute with the first 1000 to get the return type.
            # If the records were less than 1000, it uses pandas API directly for a shortcut.
            limit = get_option("compute.shortcut_limit")
            pdf = self.head(limit + 1)._to_internal_pandas()
            transformed = pdf.transform(func)
            kdf = DataFrame(transformed)
            if len(pdf) <= limit:
                return kdf

            applied = []
            for input_label, output_label in zip(
                self._internal.column_labels, kdf._internal.column_labels
            ):
                pudf = pandas_udf(
                    func,
                    returnType=kdf._internal.spark_type_for(output_label),
                    functionType=PandasUDFType.SCALAR,
                )
                kser = self._kser_for(input_label)
                applied.append(
                    kser._with_new_scol(scol=pudf(kser.spark_column)).rename(input_label)
                )

            internal = self._internal.with_new_columns(applied)
            return DataFrame(internal)
        else:
            return self._apply_series_op(lambda kser: kser.transform_batch(func))

    def transform_batch(self, func):
        """
        Transform chunks with a function that takes pandas DataFrame and outputs pandas DataFrame.
        The pandas DataFrame given to the function is of a batch used internally. The length of
        each input and output should be the same.

        .. note:: the `func` is unable to access to the whole input frame. Koalas internally
            splits the input series into multiple batches and calls `func` with each batch multiple
            times. Therefore, operations such as global aggregations are impossible. See the example
            below.

            >>> # This case does not return the length of whole frame but of the batch internally
            ... # used.
            ... def length(pdf) -> ks.DataFrame[int]:
            ...     return pd.DataFrame([len(pdf)] * len(pdf))
            ...
            >>> df = ks.DataFrame({'A': range(1000)})
            >>> df.transform_batch(length)  # doctest: +SKIP
                c0
            0   83
            1   83
            2   83
            ...

        .. note:: this API executes the function once to infer the type which is
            potentially expensive, for instance, when the dataset is created after
            aggregations or sorting.

            To avoid this, specify return type in ``func``, for instance, as below:

            >>> def plus_one(x) -> ks.DataFrame[float, float]:
            ...    return x + 1

            If the return type is specified, the output column names become
            `c0, c1, c2 ... cn`. These names are positionally mapped to the returned
            DataFrame in ``func``. See examples below.


        Parameters
        ----------
        func : function
            Function to transform each pandas frame.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.apply_batch: For row/columnwise operations.
        Series.transform_batch: transform the search as each pandas chunks.

        Examples
        --------
        >>> df = ks.DataFrame([(1, 2), (3, 4), (5, 6)], columns=['A', 'B'])
        >>> df
           A  B
        0  1  2
        1  3  4
        2  5  6

        >>> def plus_one_func(pdf) -> ks.DataFrame[int, int]:
        ...     return pdf + 1
        >>> df.transform_batch(plus_one_func)
           c0  c1
        0   2   3
        1   4   5
        2   6   7

        >>> def plus_one_func(pdf) -> ks.Series[int]:
        ...     return pdf.B + 1
        >>> df.transform_batch(plus_one_func)
        0    3
        1    5
        2    7
        Name: 0, dtype: int32

        You can also omit the type hints so Koalas infers the return schema as below:

        >>> df.transform_batch(lambda pdf: pdf + 1)
           A  B
        0  2  3
        1  4  5
        2  6  7

        Note that you should not transform the index. The index information will not change.

        >>> df.transform_batch(lambda pdf: pdf.B + 1)
        0    3
        1    5
        2    7
        Name: B, dtype: int64
        """
        from databricks.koalas.groupby import GroupBy
        from databricks.koalas import Series

        assert callable(func), "the first argument should be a callable function."
        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        should_infer_schema = return_sig is None

        names = self._internal.to_internal_spark_frame.schema.names
        should_by_pass = LooseVersion(pyspark.__version__) >= "3.0"

        def pandas_concat(series):
            # The input can only be a DataFrame for struct from Spark 3.0.
            # This works around to make the input as a frame. See SPARK-27240
            pdf = pd.concat(series, axis=1)
            pdf = pdf.rename(columns=dict(zip(pdf.columns, names)))
            return pdf

        def pandas_extract(pdf, name):
            # This is for output to work around a DataFrame for struct
            # from Spark 3.0.  See SPARK-23836
            return pdf[name]

        def pandas_series_func(f):
            ff = f
            return lambda *series: ff(pandas_concat(series))

        def pandas_frame_func(f):
            ff = f
            return lambda *series: pandas_extract(ff(pandas_concat(series)), field.name)

        if should_infer_schema:
            # Here we execute with the first 1000 to get the return type.
            # If the records were less than 1000, it uses pandas API directly for a shortcut.
            limit = get_option("compute.shortcut_limit")
            pdf = self.head(limit + 1)._to_internal_pandas()
            transformed = func(pdf)
            if not isinstance(transformed, (pd.DataFrame, pd.Series)):
                raise ValueError(
                    "The given function should return a frame; however, "
                    "the return type was %s." % type(transformed)
                )
            if len(transformed) != len(pdf):
                raise ValueError("transform_batch cannot produce aggregated results")
            kdf_or_kser = ks.from_pandas(transformed)

            if isinstance(kdf_or_kser, ks.Series):
                kser = kdf_or_kser
                pudf = pandas_udf(
                    func if should_by_pass else pandas_series_func(func),
                    returnType=kser.spark_type,
                    functionType=PandasUDFType.SCALAR,
                )
                columns = self._internal.spark_columns
                # TODO: Index will be lost in this case.
                internal = self._internal.copy(
                    spark_column=pudf(F.struct(*columns)) if should_by_pass else pudf(*columns),
                    column_labels=kser._internal.column_labels,
                    column_label_names=kser._internal.column_label_names,
                )
                return Series(internal, anchor=self)
            else:
                kdf = kdf_or_kser
                if len(pdf) <= limit:
                    # only do the short cut when it returns a frame to avoid
                    # operations on different dataframes in case of series.
                    return kdf

                return_schema = kdf._internal.to_internal_spark_frame.schema
                # Force nullability.
                return_schema = StructType(
                    [StructField(field.name, field.dataType) for field in return_schema.fields]
                )
                output_func = GroupBy._make_pandas_df_builder_func(
                    self, func, return_schema, retain_index=True
                )
                columns = self._internal.spark_columns
                if should_by_pass:
                    pudf = pandas_udf(
                        output_func, returnType=return_schema, functionType=PandasUDFType.SCALAR
                    )
                    temp_struct_column = verify_temp_column_name(
                        self._internal.spark_frame, "__temp_struct__"
                    )
                    applied = pudf(F.struct(*columns)).alias(temp_struct_column)
                    sdf = self._internal.spark_frame.select(applied)
                    sdf = sdf.selectExpr("%s.*" % temp_struct_column)
                else:
                    applied = []
                    for field in return_schema.fields:
                        applied.append(
                            pandas_udf(
                                pandas_frame_func(output_func),
                                returnType=field.dataType,
                                functionType=PandasUDFType.SCALAR,
                            )(*columns).alias(field.name)
                        )
                    sdf = self._internal.spark_frame.select(*applied)
                return DataFrame(kdf._internal.with_new_sdf(sdf))
        else:
            return_schema = infer_return_type(func).tpe
            is_return_dataframe = getattr(return_sig, "__origin__", None) == ks.DataFrame
            is_return_series = getattr(return_sig, "__origin__", None) == ks.Series
            if not is_return_dataframe and not is_return_series:
                raise TypeError(
                    "The given function should specify a frame or seires as its type "
                    "hints; however, the return type was %s." % return_sig
                )
            if is_return_series:
                pudf = pandas_udf(
                    func if should_by_pass else pandas_series_func(func),
                    returnType=return_schema,
                    functionType=PandasUDFType.SCALAR,
                )
                columns = self._internal.spark_columns
                internal = self._internal.copy(
                    spark_column=pudf(F.struct(*columns)) if should_by_pass else pudf(*columns),
                    column_labels=[("0",)],
                    column_label_names=None,
                )
                return Series(internal, anchor=self)
            else:
                output_func = GroupBy._make_pandas_df_builder_func(
                    self, func, return_schema, retain_index=False
                )
                columns = self._internal.spark_columns

                if should_by_pass:
                    pudf = pandas_udf(
                        output_func, returnType=return_schema, functionType=PandasUDFType.SCALAR
                    )
                    temp_struct_column = verify_temp_column_name(
                        self._internal.spark_frame, "__temp_struct__"
                    )
                    applied = pudf(F.struct(*columns)).alias(temp_struct_column)
                    sdf = self._internal.spark_frame.select(applied)
                    sdf = sdf.selectExpr("%s.*" % temp_struct_column)
                else:
                    applied = []
                    for field in return_schema.fields:
                        applied.append(
                            pandas_udf(
                                pandas_frame_func(output_func),
                                returnType=field.dataType,
                                functionType=PandasUDFType.SCALAR,
                            )(*columns).alias(field.name)
                        )
                    sdf = self._internal.spark_frame.select(*applied)
                return DataFrame(sdf)

    def pop(self, item):
        """
        Return item and drop from frame. Raise KeyError if not found.
        Parameters
        ----------
        item : str
            Label of column to be popped.
        Returns
        -------
        Series
        Examples
        --------
        >>> df = ks.DataFrame([('falcon', 'bird', 389.0),
        ...                    ('parrot', 'bird', 24.0),
        ...                    ('lion', 'mammal', 80.5),
        ...                    ('monkey','mammal', np.nan)],
        ...                   columns=('name', 'class', 'max_speed'))
        >>> df
             name   class  max_speed
        0  falcon    bird      389.0
        1  parrot    bird       24.0
        2    lion  mammal       80.5
        3  monkey  mammal        NaN
        >>> df.pop('class')
        0      bird
        1      bird
        2    mammal
        3    mammal
        Name: class, dtype: object
        >>> df
             name  max_speed
        0  falcon      389.0
        1  parrot       24.0
        2    lion       80.5
        3  monkey        NaN

        Also support for MultiIndex

        >>> df = ks.DataFrame([('falcon', 'bird', 389.0),
        ...                    ('parrot', 'bird', 24.0),
        ...                    ('lion', 'mammal', 80.5),
        ...                    ('monkey','mammal', np.nan)],
        ...                   columns=('name', 'class', 'max_speed'))
        >>> columns = [('a', 'name'), ('a', 'class'), ('b', 'max_speed')]
        >>> df.columns = pd.MultiIndex.from_tuples(columns)
        >>> df
                a                 b
             name   class max_speed
        0  falcon    bird     389.0
        1  parrot    bird      24.0
        2    lion  mammal      80.5
        3  monkey  mammal       NaN
        >>> df.pop('a')
             name   class
        0  falcon    bird
        1  parrot    bird
        2    lion  mammal
        3  monkey  mammal
        >>> df
                  b
          max_speed
        0     389.0
        1      24.0
        2      80.5
        3       NaN
        """
        result = self[item]
        self._internal = self.drop(item)._internal

        return result

    # TODO: add axis parameter can work when '1' or 'columns'
    def xs(self, key, axis=0, level=None):
        """
        Return cross-section from the DataFrame.

        This method takes a `key` argument to select data at a particular
        level of a MultiIndex.

        Parameters
        ----------
        key : label or tuple of label
            Label contained in the index, or partially in a MultiIndex.
        axis : 0 or 'index', default 0
            Axis to retrieve cross-section on.
            currently only support 0 or 'index'
        level : object, defaults to first n levels (n=1 or len(key))
            In case of a key partially contained in a MultiIndex, indicate
            which levels are used. Levels can be referred by label or position.

        Returns
        -------
        DataFrame
            Cross-section from the original DataFrame
            corresponding to the selected index levels.

        See Also
        --------
        DataFrame.loc : Access a group of rows and columns
            by label(s) or a boolean array.
        DataFrame.iloc : Purely integer-location based indexing
            for selection by position.

        Examples
        --------
        >>> d = {'num_legs': [4, 4, 2, 2],
        ...      'num_wings': [0, 0, 2, 2],
        ...      'class': ['mammal', 'mammal', 'mammal', 'bird'],
        ...      'animal': ['cat', 'dog', 'bat', 'penguin'],
        ...      'locomotion': ['walks', 'walks', 'flies', 'walks']}
        >>> df = ks.DataFrame(data=d)
        >>> df = df.set_index(['class', 'animal', 'locomotion'])
        >>> df  # doctest: +NORMALIZE_WHITESPACE
                                   num_legs  num_wings
        class  animal  locomotion
        mammal cat     walks              4          0
               dog     walks              4          0
               bat     flies              2          2
        bird   penguin walks              2          2

        Get values at specified index

        >>> df.xs('mammal')  # doctest: +NORMALIZE_WHITESPACE
                           num_legs  num_wings
        animal locomotion
        cat    walks              4          0
        dog    walks              4          0
        bat    flies              2          2

        Get values at several indexes

        >>> df.xs(('mammal', 'dog'))  # doctest: +NORMALIZE_WHITESPACE
                    num_legs  num_wings
        locomotion
        walks              4          0

        Get values at specified index and level

        >>> df.xs('cat', level=1)  # doctest: +NORMALIZE_WHITESPACE
                           num_legs  num_wings
        class  locomotion
        mammal walks              4          0
        """
        from databricks.koalas.series import _col

        if not isinstance(key, (str, tuple)):
            raise ValueError("'key' should be string or tuple that contains strings")
        if not all(isinstance(index, str) for index in key):
            raise ValueError(
                "'key' should have index names as only strings "
                "or a tuple that contain index names as only strings"
            )

        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        if isinstance(key, str):
            key = (key,)
        if len(key) > len(self._internal.index_spark_columns):
            raise KeyError(
                "Key length ({}) exceeds index depth ({})".format(
                    len(key), len(self._internal.index_spark_columns)
                )
            )
        if level is None:
            level = 0

        scols = (
            self._internal.spark_columns[:level] + self._internal.spark_columns[level + len(key) :]
        )
        rows = [self._internal.spark_columns[lvl] == index for lvl, index in enumerate(key, level)]

        sdf = (
            self._sdf.select(scols + list(HIDDEN_COLUMNS))
            .drop(NATURAL_ORDER_COLUMN_NAME)
            .filter(reduce(lambda x, y: x & y, rows))
        )

        if len(key) == len(self._internal.index_spark_columns):
            result = _col(DataFrame(_InternalFrame(spark_frame=sdf, index_map=None)).T)
            result.name = key
        else:
            new_index_map = OrderedDict(
                list(self._internal.index_map.items())[:level]
                + list(self._internal.index_map.items())[level + len(key) :]
            )
            internal = self._internal.copy(spark_frame=sdf, index_map=new_index_map,)
            result = DataFrame(internal)

        return result

    def where(self, cond, other=np.nan):
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : boolean DataFrame
            Where cond is True, keep the original value. Where False,
            replace with corresponding value from other.
        other : scalar, DataFrame
            Entries where cond is False are replaced with corresponding value from other.

        Returns
        -------
        DataFrame

        Examples
        --------

        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> df1 = ks.DataFrame({'A': [0, 1, 2, 3, 4], 'B':[100, 200, 300, 400, 500]})
        >>> df2 = ks.DataFrame({'A': [0, -1, -2, -3, -4], 'B':[-100, -200, -300, -400, -500]})
        >>> df1
           A    B
        0  0  100
        1  1  200
        2  2  300
        3  3  400
        4  4  500
        >>> df2
           A    B
        0  0 -100
        1 -1 -200
        2 -2 -300
        3 -3 -400
        4 -4 -500

        >>> df1.where(df1 > 0).sort_index()
             A      B
        0  NaN  100.0
        1  1.0  200.0
        2  2.0  300.0
        3  3.0  400.0
        4  4.0  500.0

        >>> df1.where(df1 > 1, 10).sort_index()
            A    B
        0  10  100
        1  10  200
        2   2  300
        3   3  400
        4   4  500

        >>> df1.where(df1 > 1, df1 + 100).sort_index()
             A    B
        0  100  100
        1  101  200
        2    2  300
        3    3  400
        4    4  500

        >>> df1.where(df1 > 1, df2).sort_index()
           A    B
        0  0  100
        1 -1  200
        2  2  300
        3  3  400
        4  4  500

        When the column name of cond is different from self, it treats all values are False

        >>> cond = ks.DataFrame({'C': [0, -1, -2, -3, -4], 'D':[4, 3, 2, 1, 0]}) % 3 == 0
        >>> cond
               C      D
        0   True  False
        1  False   True
        2  False  False
        3   True  False
        4  False   True

        >>> df1.where(cond).sort_index()
            A   B
        0 NaN NaN
        1 NaN NaN
        2 NaN NaN
        3 NaN NaN
        4 NaN NaN

        When the type of cond is Series, it just check boolean regardless of column name

        >>> cond = ks.Series([1, 2]) > 1
        >>> cond
        0    False
        1     True
        Name: 0, dtype: bool

        >>> df1.where(cond).sort_index()
             A      B
        0  NaN    NaN
        1  1.0  200.0
        2  NaN    NaN
        3  NaN    NaN
        4  NaN    NaN

        >>> reset_option("compute.ops_on_diff_frames")
        """
        from databricks.koalas.series import Series

        tmp_cond_col_name = "__tmp_cond_col_{}__".format
        tmp_other_col_name = "__tmp_other_col_{}__".format

        kdf = self.copy()

        tmp_cond_col_names = [
            tmp_cond_col_name(name_like_string(label)) for label in self._internal.column_labels
        ]
        if isinstance(cond, DataFrame):
            cond = cond[
                [
                    (
                        cond._internal.spark_column_for(label)
                        if label in cond._internal.column_labels
                        else F.lit(False)
                    ).alias(name)
                    for label, name in zip(self._internal.column_labels, tmp_cond_col_names)
                ]
            ]
            kdf[tmp_cond_col_names] = cond
        elif isinstance(cond, Series):
            cond = cond.to_frame()
            cond = cond[
                [cond._internal.data_spark_columns[0].alias(name) for name in tmp_cond_col_names]
            ]
            kdf[tmp_cond_col_names] = cond
        else:
            raise ValueError("type of cond must be a DataFrame or Series")

        tmp_other_col_names = [
            tmp_other_col_name(name_like_string(label)) for label in self._internal.column_labels
        ]
        if isinstance(other, DataFrame):
            other = other[
                [
                    (
                        other._internal.spark_column_for(label)
                        if label in other._internal.column_labels
                        else F.lit(np.nan)
                    ).alias(name)
                    for label, name in zip(self._internal.column_labels, tmp_other_col_names)
                ]
            ]
            kdf[tmp_other_col_names] = other
        elif isinstance(other, Series):
            other = other.to_frame()
            other = other[
                [other._internal.data_spark_columns[0].alias(name) for name in tmp_other_col_names]
            ]
            kdf[tmp_other_col_names] = other
        else:
            for label in self._internal.column_labels:
                kdf[tmp_other_col_name(name_like_string(label))] = other

        # above logic make spark dataframe looks like below:
        # +-----------------+---+---+------------------+-------------------+------------------+--...
        # |__index_level_0__|  A|  B|__tmp_cond_col_A__|__tmp_other_col_A__|__tmp_cond_col_B__|__...
        # +-----------------+---+---+------------------+-------------------+------------------+--...
        # |                0|  0|100|              true|                  0|             false|  ...
        # |                1|  1|200|             false|                 -1|             false|  ...
        # |                3|  3|400|              true|                 -3|             false|  ...
        # |                2|  2|300|             false|                 -2|              true|  ...
        # |                4|  4|500|             false|                 -4|             false|  ...
        # +-----------------+---+---+------------------+-------------------+------------------+--...

        data_spark_columns = []
        for label in self._internal.column_labels:
            data_spark_columns.append(
                F.when(
                    kdf[tmp_cond_col_name(name_like_string(label))].spark_column,
                    kdf._internal.spark_column_for(label),
                )
                .otherwise(kdf[tmp_other_col_name(name_like_string(label))].spark_column)
                .alias(kdf._internal.spark_column_name_for(label))
            )

        return DataFrame(
            kdf._internal.with_new_columns(
                data_spark_columns, column_labels=self._internal.column_labels
            )
        )

    def mask(self, cond, other=np.nan):
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : boolean DataFrame
            Where cond is False, keep the original value. Where True,
            replace with corresponding value from other.
        other : scalar, DataFrame
            Entries where cond is True are replaced with corresponding value from other.

        Returns
        -------
        DataFrame

        Examples
        --------

        >>> from databricks.koalas.config import set_option, reset_option
        >>> set_option("compute.ops_on_diff_frames", True)
        >>> df1 = ks.DataFrame({'A': [0, 1, 2, 3, 4], 'B':[100, 200, 300, 400, 500]})
        >>> df2 = ks.DataFrame({'A': [0, -1, -2, -3, -4], 'B':[-100, -200, -300, -400, -500]})
        >>> df1
           A    B
        0  0  100
        1  1  200
        2  2  300
        3  3  400
        4  4  500
        >>> df2
           A    B
        0  0 -100
        1 -1 -200
        2 -2 -300
        3 -3 -400
        4 -4 -500

        >>> df1.mask(df1 > 0).sort_index()
             A   B
        0  0.0 NaN
        1  NaN NaN
        2  NaN NaN
        3  NaN NaN
        4  NaN NaN

        >>> df1.mask(df1 > 1, 10).sort_index()
            A   B
        0   0  10
        1   1  10
        2  10  10
        3  10  10
        4  10  10

        >>> df1.mask(df1 > 1, df1 + 100).sort_index()
             A    B
        0    0  200
        1    1  300
        2  102  400
        3  103  500
        4  104  600

        >>> df1.mask(df1 > 1, df2).sort_index()
           A    B
        0  0 -100
        1  1 -200
        2 -2 -300
        3 -3 -400
        4 -4 -500

        >>> reset_option("compute.ops_on_diff_frames")
        """
        from databricks.koalas.series import Series

        if not isinstance(cond, (DataFrame, Series)):
            raise ValueError("type of cond must be a DataFrame or Series")

        cond_inversed = cond._apply_series_op(lambda kser: ~kser)
        return self.where(cond_inversed, other)

    @property
    def index(self):
        """The index (row labels) Column of the DataFrame.

        Currently not supported when the DataFrame has no index.

        See Also
        --------
        Index
        """
        from databricks.koalas.indexes import Index, MultiIndex

        if len(self._internal.index_map) == 1:
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
        return len(self._internal.column_labels) == 0 or self._sdf.rdd.isEmpty()

    @property
    def style(self):
        """
        Property returning a Styler object containing methods for
        building a styled HTML representation fo the DataFrame.

        .. note:: currently it collects top 1000 rows and return its
            pandas `pandas.io.formats.style.Styler` instance.

        Examples
        --------
        >>> ks.range(1001).style  # doctest: +ELLIPSIS
        <pandas.io.formats.style.Styler object at ...>
        """
        max_results = get_option("compute.max_rows")
        pdf = self.head(max_results + 1).to_pandas()
        if len(pdf) > max_results:
            warnings.warn("'style' property will only use top %s rows." % max_results, UserWarning)
        return pdf.head(max_results).style

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
        inplace = validate_bool_kwarg(inplace, "inplace")
        if isinstance(keys, (str, tuple)):
            keys = [keys]
        else:
            keys = list(keys)
        columns = set(self.columns)
        for key in keys:
            if key not in columns:
                raise KeyError(key)
        keys = [key if isinstance(key, tuple) else (key,) for key in keys]

        if drop:
            column_labels = [label for label in self._internal.column_labels if label not in keys]
        else:
            column_labels = self._internal.column_labels
        if append:
            index_map = OrderedDict(
                list(self._internal.index_map.items())
                + [(self._internal.spark_column_name_for(label), label) for label in keys]
            )
        else:
            index_map = OrderedDict(
                (self._internal.spark_column_name_for(label), label) for label in keys
            )

        internal = self._internal.copy(
            index_map=index_map,
            column_labels=column_labels,
            data_spark_columns=[self._internal.spark_column_for(label) for label in column_labels],
        )

        if inplace:
            self._internal = internal
        else:
            return DataFrame(internal)

    def reset_index(self, level=None, drop=False, inplace=False, col_level=0, col_fill=""):
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
        col_level : int or str, default 0
            If the columns have multiple levels, determines which level the
            labels are inserted into. By default it is inserted into the first
            level.
        col_fill : object, default ''
            If the columns have multiple levels, determines how the other
            levels are named. If None then the index name is repeated.

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

        You can also use `reset_index` with `MultiIndex`.

        >>> index = pd.MultiIndex.from_tuples([('bird', 'falcon'),
        ...                                    ('bird', 'parrot'),
        ...                                    ('mammal', 'lion'),
        ...                                    ('mammal', 'monkey')],
        ...                                   names=['class', 'name'])
        >>> columns = pd.MultiIndex.from_tuples([('speed', 'max'),
        ...                                      ('species', 'type')])
        >>> df = ks.DataFrame([(389.0, 'fly'),
        ...                    ( 24.0, 'fly'),
        ...                    ( 80.5, 'run'),
        ...                    (np.nan, 'jump')],
        ...                   index=index,
        ...                   columns=columns)
        >>> df  # doctest: +NORMALIZE_WHITESPACE
                       speed species
                         max    type
        class  name
        bird   falcon  389.0     fly
               parrot   24.0     fly
        mammal lion     80.5     run
               monkey    NaN    jump

        If the index has multiple levels, we can reset a subset of them:

        >>> df.reset_index(level='class')  # doctest: +NORMALIZE_WHITESPACE
                 class  speed species
                          max    type
        name
        falcon    bird  389.0     fly
        parrot    bird   24.0     fly
        lion    mammal   80.5     run
        monkey  mammal    NaN    jump

        If we are not dropping the index, by default, it is placed in the top
        level. We can place it in another level:

        >>> df.reset_index(level='class', col_level=1)  # doctest: +NORMALIZE_WHITESPACE
                        speed species
                 class    max    type
        name
        falcon    bird  389.0     fly
        parrot    bird   24.0     fly
        lion    mammal   80.5     run
        monkey  mammal    NaN    jump

        When the index is inserted under another level, we can specify under
        which one with the parameter `col_fill`:

        >>> df.reset_index(level='class', col_level=1,
        ...                col_fill='species')  # doctest: +NORMALIZE_WHITESPACE
                      species  speed species
                        class    max    type
        name
        falcon           bird  389.0     fly
        parrot           bird   24.0     fly
        lion           mammal   80.5     run
        monkey         mammal    NaN    jump

        If we specify a nonexistent level for `col_fill`, it is created:

        >>> df.reset_index(level='class', col_level=1,
        ...                col_fill='genus')  # doctest: +NORMALIZE_WHITESPACE
                        genus  speed species
                        class    max    type
        name
        falcon           bird  389.0     fly
        parrot           bird   24.0     fly
        lion           mammal   80.5     run
        monkey         mammal    NaN    jump
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        multi_index = len(self._internal.index_map) > 1

        def rename(index):
            if multi_index:
                return ("level_{}".format(index),)
            else:
                if ("index",) not in self._internal.column_labels:
                    return ("index",)
                else:
                    return ("level_{}".format(index),)

        if level is None:
            new_index_map = [
                (column, name if name is not None else rename(i))
                for i, (column, name) in enumerate(self._internal.index_map.items())
            ]
            index_map = []
        else:
            if isinstance(level, (int, str)):
                level = [level]
            level = list(level)

            if all(isinstance(l, int) for l in level):
                for lev in level:
                    if lev >= len(self._internal.index_map):
                        raise IndexError(
                            "Too many levels: Index has only {} level, not {}".format(
                                len(self._internal.index_map), lev + 1
                            )
                        )
                idx = level
            elif all(isinstance(lev, str) for lev in level):
                idx = []
                for l in level:
                    try:
                        i = self._internal.index_names.index((l,))
                        idx.append(i)
                    except ValueError:
                        if multi_index:
                            raise KeyError("Level unknown not found")
                        else:
                            raise KeyError(
                                "Level unknown must be same as name ({})".format(
                                    name_like_string(self._internal.index_names[0])
                                )
                            )
            else:
                raise ValueError("Level should be all int or all string.")
            idx.sort()

            new_index_map = []
            index_map_items = list(self._internal.index_map.items())
            new_index_map_items = index_map_items.copy()
            for i in idx:
                info = index_map_items[i]
                index_column, index_name = info
                new_index_map.append(
                    (index_column, index_name if index_name is not None else rename(i))
                )
                new_index_map_items.remove(info)

            index_map = OrderedDict(new_index_map_items)

        new_data_scols = [
            scol_for(self._sdf, column).alias(name_like_string(name))
            for column, name in new_index_map
        ]

        if len(index_map) > 0:
            index_scols = [scol_for(self._sdf, column) for column in index_map]
            sdf = self._sdf.select(
                index_scols
                + new_data_scols
                + self._internal.data_spark_columns
                + list(HIDDEN_COLUMNS)
            )
        else:
            sdf = self._sdf.select(
                new_data_scols + self._internal.data_spark_columns + list(HIDDEN_COLUMNS)
            )

            # Now, new internal Spark columns are named as same as index name.
            new_index_map = [(column, name) for column, name in new_index_map]

            sdf = _InternalFrame.attach_default_index(sdf)
            index_map = OrderedDict({SPARK_DEFAULT_INDEX_NAME: None})

        if drop:
            new_index_map = []

        if self._internal.column_labels_level > 1:
            column_depth = len(self._internal.column_labels[0])
            if col_level >= column_depth:
                raise IndexError(
                    "Too many levels: Index has only {} levels, not {}".format(
                        column_depth, col_level + 1
                    )
                )
            if any(col_level + len(name) > column_depth for _, name in new_index_map):
                raise ValueError("Item must have length equal to number of levels.")
            column_labels = [
                tuple(
                    ([col_fill] * col_level)
                    + list(name)
                    + ([col_fill] * (column_depth - (len(name) + col_level)))
                )
                for _, name in new_index_map
            ] + self._internal.column_labels
        else:
            column_labels = [name for _, name in new_index_map] + self._internal.column_labels

        internal = self._internal.copy(
            spark_frame=sdf,
            index_map=index_map,
            column_labels=column_labels,
            data_spark_columns=(
                [scol_for(sdf, name_like_string(name)) for _, name in new_index_map]
                + [scol_for(sdf, col) for col in self._internal.data_spark_column_names]
            ),
        )

        if inplace:
            self._internal = internal
        else:
            return DataFrame(internal)

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
        return self._apply_series_op(lambda kser: kser.isnull())

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
        return self._apply_series_op(lambda kser: kser.notnull())

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
        return self._apply_series_op(lambda kser: kser.shift(periods, fill_value))

    # TODO: axis should support 1 or 'columns' either at this moment
    def diff(self, periods: int = 1, axis: Union[int, str] = 0):
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
        axis : int, default 0 or 'index'
            Can only be set to 0 at the moment.

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
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')

        return self._apply_series_op(lambda kser: kser.diff(periods))

    # TODO: axis should support 1 or 'columns' either at this moment
    def nunique(
        self,
        axis: Union[int, str] = 0,
        dropna: bool = True,
        approx: bool = False,
        rsd: float = 0.05,
    ) -> "ks.Series":
        """
        Return number of unique elements in the object.

        Excludes NA values by default.

        Parameters
        ----------
        axis : int, default 0 or 'index'
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
        The number of unique values per column as a Koalas Series.

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
        from databricks.koalas.series import _col

        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        sdf = self._sdf.select(
            [
                self._kser_for(label)._nunique(dropna, approx, rsd)
                for label in self._internal.column_labels
            ]
        )

        # The data is expected to be small so it's fine to transpose/use default index.
        with ks.option_context(
            "compute.default_index_type", "distributed", "compute.max_rows", None
        ):
            kdf = DataFrame(sdf)  # type: ks.DataFrame
            internal = _InternalFrame(
                spark_frame=kdf._internal.spark_frame,
                index_map=kdf._internal.index_map,
                column_labels=self._internal.column_labels,
                column_label_names=self._internal.column_label_names,
            )

            return _col(DataFrame(internal).transpose())

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

            .. note:: If `decimals` is a Series, it is expected to be small,
                as all the data is loaded into the driver's memory.

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
            decimals = {
                k if isinstance(k, tuple) else (k,): v
                for k, v in decimals._to_internal_pandas().items()
            }
        elif isinstance(decimals, dict):
            decimals = {k if isinstance(k, tuple) else (k,): v for k, v in decimals.items()}
        elif isinstance(decimals, int):
            decimals = {k: decimals for k in self._internal.column_labels}
        else:
            raise ValueError("decimals must be an integer, a dict-like or a Series")

        def op(kser):
            label = kser._internal.column_labels[0]
            if label in decimals:
                return F.round(kser.spark_column, decimals[label]).alias(
                    kser._internal.data_spark_column_names[0]
                )
            else:
                return kser

        return self._apply_series_op(op)

    def _mark_duplicates(self, subset=None, keep="first"):
        if subset is None:
            subset = self._internal.column_labels
        else:
            if isinstance(subset, str):
                subset = [(subset,)]
            elif isinstance(subset, tuple):
                subset = [subset]
            else:
                subset = [sub if isinstance(sub, tuple) else (sub,) for sub in subset]
            diff = set(subset).difference(set(self._internal.column_labels))
            if len(diff) > 0:
                raise KeyError(", ".join([str(d) if len(d) > 1 else d[0] for d in diff]))
        group_cols = [self._internal.spark_column_name_for(label) for label in subset]

        sdf = self._sdf

        column = verify_temp_column_name(sdf, "__duplicated__")

        if keep == "first" or keep == "last":
            if keep == "first":
                ord_func = spark.functions.asc
            else:
                ord_func = spark.functions.desc
            window = (
                Window.partitionBy(group_cols)
                .orderBy(ord_func(NATURAL_ORDER_COLUMN_NAME))
                .rowsBetween(Window.unboundedPreceding, Window.currentRow)
            )
            sdf = sdf.withColumn(column, F.row_number().over(window) > 1)
        elif not keep:
            window = Window.partitionBy(group_cols).rowsBetween(
                Window.unboundedPreceding, Window.unboundedFollowing
            )
            sdf = sdf.withColumn(column, F.count("*").over(window) > 1)
        else:
            raise ValueError("'keep' only supports 'first', 'last' and False")
        return sdf, column

    def duplicated(self, subset=None, keep="first"):
        """
        Return boolean Series denoting duplicate rows, optionally only considering certain columns.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates,
            by default use all of the columns
        keep : {'first', 'last', False}, default 'first'
           - ``first`` : Mark duplicates as ``True`` except for the first occurrence.
           - ``last`` : Mark duplicates as ``True`` except for the last occurrence.
           - False : Mark all duplicates as ``True``.

        Returns
        -------
        duplicated : Series

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 1, 3], 'b': [1, 1, 1, 4], 'c': [1, 1, 1, 5]},
        ...                   columns = ['a', 'b', 'c'])
        >>> df
           a  b  c
        0  1  1  1
        1  1  1  1
        2  1  1  1
        3  3  4  5

        >>> df.duplicated().sort_index()
        0    False
        1     True
        2     True
        3    False
        Name: 0, dtype: bool

        Mark duplicates as ``True`` except for the last occurrence.

        >>> df.duplicated(keep='last').sort_index()
        0     True
        1     True
        2    False
        3    False
        Name: 0, dtype: bool

        Mark all duplicates as ``True``.

        >>> df.duplicated(keep=False).sort_index()
        0     True
        1     True
        2     True
        3    False
        Name: 0, dtype: bool
        """
        from databricks.koalas.series import _col

        sdf, column = self._mark_duplicates(subset, keep)
        column_label = ("0",)

        sdf = sdf.select(
            self._internal.index_spark_columns
            + [scol_for(sdf, column).alias(name_like_string(column_label))]
        )
        return _col(
            DataFrame(
                _InternalFrame(
                    spark_frame=sdf,
                    index_map=self._internal.index_map,
                    column_labels=[column_label],
                    data_spark_columns=[scol_for(sdf, name_like_string(column_label))],
                )
            )
        )

    def to_koalas(self, index_col: Optional[Union[str, List[str]]] = None):
        """
        Converts the existing DataFrame into a Koalas DataFrame.

        This method is monkey-patched into Spark's DataFrame and can be used
        to convert a Spark DataFrame into a Koalas DataFrame. If running on
        an existing Koalas DataFrame, the method returns itself.

        If a Koalas DataFrame is converted to a Spark DataFrame and then back
        to Koalas, it will lose the index information and the original index
        will be turned into a normal column.

        Parameters
        ----------
        index_col: str or list of str, optional, default: None
            Index column of table in Spark.

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
        DataFrame[col1: bigint, col2: bigint]

        >>> kdf = spark_df.to_koalas()
        >>> kdf
           col1  col2
        0     1     3
        1     2     4

        We can specify the index columns.

        >>> kdf = spark_df.to_koalas(index_col='col1')
        >>> kdf  # doctest: +NORMALIZE_WHITESPACE
              col2
        col1
        1        3
        2        4

        Calling to_koalas on a Koalas DataFrame simply returns itself.

        >>> df.to_koalas()
           col1  col2
        0     1     3
        1     2     4
        """
        if isinstance(self, DataFrame):
            return self
        else:
            assert isinstance(self, spark.DataFrame), type(self)
            from databricks.koalas.namespace import _get_index_map

            index_map = _get_index_map(self, index_col)
            internal = _InternalFrame(spark_frame=self, index_map=index_map)
            return DataFrame(internal)

    def cache(self):
        """
        Yields and caches the current DataFrame.

        The Koalas DataFrame is yielded as a protected resource and its corresponding
        data is cached which gets uncached after execution goes of the context.

        If you want to specify the StorageLevel manually, use :meth:`DataFrame.persist`

        See Also
        --------
        DataFrame.persist

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
        Name: 0, dtype: int64

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

    def persist(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Yields and caches the current DataFrame with a specific StorageLevel.
        If a StogeLevel is not given, the `MEMORY_AND_DISK` level is used by default like PySpark.

        The Koalas DataFrame is yielded as a protected resource and its corresponding
        data is cached which gets uncached after execution goes of the context.

        See Also
        --------
        DataFrame.cache

        Examples
        --------
        >>> import pyspark
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df
           dogs  cats
        0   0.2   0.3
        1   0.0   0.6
        2   0.6   0.0
        3   0.2   0.1

        Set the StorageLevel to `MEMORY_ONLY`.

        >>> with df.persist(pyspark.StorageLevel.MEMORY_ONLY) as cached_df:
        ...     print(cached_df.storage_level)
        ...     print(cached_df.count())
        ...
        Memory Serialized 1x Replicated
        dogs    4
        cats    4
        Name: 0, dtype: int64

        Set the StorageLevel to `DISK_ONLY`.

        >>> with df.persist(pyspark.StorageLevel.DISK_ONLY) as cached_df:
        ...     print(cached_df.storage_level)
        ...     print(cached_df.count())
        ...
        Disk Serialized 1x Replicated
        dogs    4
        cats    4
        Name: 0, dtype: int64

        If a StorageLevel is not given, it uses `MEMORY_AND_DISK` by default.

        >>> with df.persist() as cached_df:
        ...     print(cached_df.storage_level)
        ...     print(cached_df.count())
        ...
        Disk Memory Serialized 1x Replicated
        dogs    4
        cats    4
        Name: 0, dtype: int64

        >>> df = df.persist()
        >>> df.to_pandas().mean(axis=1)
        0    0.25
        1    0.30
        2    0.30
        3    0.15
        dtype: float64

        To uncache the dataframe, use `unpersist` function

        >>> df.unpersist()
        """
        return _CachedDataFrame(self._internal, storage_level=storage_level)

    def hint(self, name: str, *parameters) -> "DataFrame":
        """
        Specifies some hint on the current DataFrame.

        Parameters
        ----------
        name : A name of the hint.
        parameters : Optional parameters.

        Returns
        -------
        ret : DataFrame with the hint.

        See Also
        --------
        broadcast : Marks a DataFrame as small enough for use in broadcast joins.

        Examples
        --------
        >>> df1 = ks.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
        ...                     'value': [1, 2, 3, 5]},
        ...                    columns=['lkey', 'value'])
        >>> df2 = ks.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
        ...                     'value': [5, 6, 7, 8]},
        ...                    columns=['rkey', 'value'])
        >>> merged = df1.merge(df2.hint("broadcast"), left_on='lkey', right_on='rkey')
        >>> merged.explain()  # doctest: +ELLIPSIS
        == Physical Plan ==
        ...
        ...BroadcastHashJoin...
        ...
        """
        return DataFrame(self._internal.with_new_sdf(self._sdf.hint(name, *parameters)))

    def to_table(
        self,
        name: str,
        format: Optional[str] = None,
        mode: str = "overwrite",
        partition_cols: Union[str, List[str], None] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options
    ):
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

        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default
            'overwrite'. Specifies the behavior of the save operation when the table exists
            already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
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
        if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
            options = options.get("options")  # type: ignore

        self.to_spark(index_col=index_col).write.saveAsTable(
            name=name, format=format, mode=mode, partitionBy=partition_cols, **options
        )

    def to_delta(
        self,
        path: str,
        mode: str = "overwrite",
        partition_cols: Union[str, List[str], None] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options
    ):
        """
        Write the DataFrame out as a Delta Lake table.

        Parameters
        ----------
        path : str, required
            Path to write to.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default
            'overwrite'. Specifies the behavior of the save operation when the destination
            exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
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
        ...             mode='overwrite', replaceWhere='date >= "2012-01-01"')
        """
        if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
            options = options.get("options")  # type: ignore

        self.to_spark_io(
            path=path,
            mode=mode,
            format="delta",
            partition_cols=partition_cols,
            index_col=index_col,
            **options
        )

    def to_parquet(
        self,
        path: str,
        mode: str = "overwrite",
        partition_cols: Union[str, List[str], None] = None,
        compression: Optional[str] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options
    ):
        """
        Write the DataFrame out as a Parquet file or directory.

        Parameters
        ----------
        path : str, required
            Path to write to.
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'},
            default 'overwrite'. Specifies the behavior of the save operation when the
            destination exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        compression : str {'none', 'uncompressed', 'snappy', 'gzip', 'lzo', 'brotli', 'lz4', 'zstd'}
            Compression codec to use when saving to file. If None is set, it uses the
            value specified in `spark.sql.parquet.compression.codec`.
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        options : dict
            All other options passed directly into Spark's data source.

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
        if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
            options = options.get("options")  # type: ignore

        builder = self.to_spark(index_col=index_col).write.mode(mode)
        OptionUtils._set_opts(
            builder, mode=mode, partitionBy=partition_cols, compression=compression
        )
        builder.options(**options).format("parquet").save(path)

    def to_spark_io(
        self,
        path: Optional[str] = None,
        format: Optional[str] = None,
        mode: str = "overwrite",
        partition_cols: Union[str, List[str], None] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options
    ):
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
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'}, default
            'overwrite'. Specifies the behavior of the save operation when data already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.
        partition_cols : str or list of str, optional
            Names of partitioning columns
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
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
        if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
            options = options.get("options")  # type: ignore

        self.to_spark(index_col=index_col).write.save(
            path=path, format=format, mode=mode, partitionBy=partition_cols, **options
        )

    def to_spark(self, index_col: Optional[Union[str, List[str]]] = None):
        """
        Return the current DataFrame as a Spark DataFrame.

        Parameters
        ----------
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.

        See Also
        --------
        DataFrame.to_koalas

        Examples
        --------
        By default, this method loses the index as below.

        >>> df = ks.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        >>> df.to_spark().show()  # doctest: +NORMALIZE_WHITESPACE
        +---+---+---+
        |  a|  b|  c|
        +---+---+---+
        |  1|  4|  7|
        |  2|  5|  8|
        |  3|  6|  9|
        +---+---+---+

        If `index_col` is set, it keeps the index column as specified.

        >>> df.to_spark(index_col="index").show()  # doctest: +NORMALIZE_WHITESPACE
        +-----+---+---+---+
        |index|  a|  b|  c|
        +-----+---+---+---+
        |    0|  1|  4|  7|
        |    1|  2|  5|  8|
        |    2|  3|  6|  9|
        +-----+---+---+---+

        Keeping index column is useful when you want to call some Spark APIs and
        convert it back to Koalas DataFrame without creating a default index, which
        can affect performance.

        >>> spark_df = df.to_spark(index_col="index")
        >>> spark_df = spark_df.filter("a == 2")
        >>> spark_df.to_koalas(index_col="index")  # doctest: +NORMALIZE_WHITESPACE
               a  b  c
        index
        1      2  5  8

        In case of multi-index, specify a list to `index_col`.

        >>> new_df = df.set_index("a", append=True)
        >>> new_spark_df = new_df.to_spark(index_col=["index_1", "index_2"])
        >>> new_spark_df.show()  # doctest: +NORMALIZE_WHITESPACE
        +-------+-------+---+---+
        |index_1|index_2|  b|  c|
        +-------+-------+---+---+
        |      0|      1|  4|  7|
        |      1|      2|  5|  8|
        |      2|      3|  6|  9|
        +-------+-------+---+---+

        Likewise, can be converted to back to Koalas DataFrame.

        >>> new_spark_df.to_koalas(
        ...     index_col=["index_1", "index_2"])  # doctest: +NORMALIZE_WHITESPACE
                         b  c
        index_1 index_2
        0       1        4  7
        1       2        5  8
        2       3        6  9
        """
        if index_col is None:
            return self._internal.to_external_spark_frame
        else:
            if isinstance(index_col, str):
                index_col = [index_col]

            data_column_names = []
            data_columns = []
            data_columns_column_labels = zip(
                self._internal.data_spark_column_names, self._internal.column_labels
            )
            # TODO: this code is similar with _InternalFrame.to_new_spark_frame. Might have to
            #  deduplicate.
            for i, (column, label) in enumerate(data_columns_column_labels):
                scol = self._internal.spark_column_for(label)
                name = str(i) if label is None else name_like_string(label)
                data_column_names.append(name)
                if column != name:
                    scol = scol.alias(name)
                data_columns.append(scol)

            old_index_scols = self._internal.index_spark_columns

            if len(index_col) != len(old_index_scols):
                raise ValueError(
                    "length of index columns is %s; however, the length of the given "
                    "'index_col' is %s." % (len(old_index_scols), len(index_col))
                )

            if any(col in data_column_names for col in index_col):
                raise ValueError("'index_col' cannot be overlapped with other columns.")

            sdf = self._internal.to_internal_spark_frame
            new_index_scols = [
                index_scol.alias(col) for index_scol, col in zip(old_index_scols, index_col)
            ]
            return sdf.select(new_index_scols + data_columns)

    def to_pandas(self):
        """
        Return a pandas DataFrame.

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
        return self._internal.to_pandas_frame.copy()

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
        return self._assign(kwargs)

    def _assign(self, kwargs):
        assert isinstance(kwargs, dict)
        from databricks.koalas.series import Series

        for k, v in kwargs.items():
            if not (isinstance(v, (Series, spark.Column)) or callable(v) or is_scalar(v)):
                raise TypeError(
                    "Column assignment doesn't support type " "{0}".format(type(v).__name__)
                )
            if callable(v):
                kwargs[k] = v(self)

        pairs = {
            (k if isinstance(k, tuple) else (k,)): (
                v.spark_column
                if isinstance(v, Series)
                else v
                if isinstance(v, spark.Column)
                else F.lit(v)
            )
            for k, v in kwargs.items()
        }

        scols = []
        for label in self._internal.column_labels:
            for i in range(len(label)):
                if label[: len(label) - i] in pairs:
                    name = self._internal.spark_column_name_for(label)
                    scol = pairs[label[: len(label) - i]].alias(name)
                    break
            else:
                scol = self._internal.spark_column_for(label)
            scols.append(scol)

        column_labels = self._internal.column_labels.copy()
        for label, scol in pairs.items():
            if label not in set(i[: len(label)] for i in self._internal.column_labels):
                scols.append(scol.alias(name_like_string(label)))
                column_labels.append(label)

        level = self._internal.column_labels_level
        column_labels = [
            tuple(list(label) + ([""] * (level - len(label)))) for label in column_labels
        ]

        internal = self._internal.with_new_columns(scols, column_labels=column_labels)
        return DataFrame(internal)

    @staticmethod
    def from_records(
        data: Union[np.array, List[tuple], dict, pd.DataFrame],
        index: Union[str, list, np.array] = None,
        exclude: list = None,
        columns: list = None,
        coerce_float: bool = False,
        nrows: int = None,
    ) -> "DataFrame":
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
        return DataFrame(
            pd.DataFrame.from_records(data, index, exclude, columns, coerce_float, nrows)
        )

    def to_records(self, index=True, column_dtypes=None, index_dtypes=None):
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
            kdf._to_internal_pandas(), self.to_records, pd.DataFrame.to_records, args
        )

    def copy(self, deep=None) -> "DataFrame":
        """
        Make a copy of this object's indices and data.

        Parameters
        ----------
        deep : None
            this parameter is not supported but just dummy parameter to match pandas.

        Returns
        -------
        copy : DataFrame

        Examples
        --------
        >>> df = ks.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6], 'w': [7, 8]},
        ...                   columns=['x', 'y', 'z', 'w'])
        >>> df
           x  y  z  w
        0  1  3  5  7
        1  2  4  6  8
        >>> df_copy = df.copy()
        >>> df_copy
           x  y  z  w
        0  1  3  5  7
        1  2  4  6  8
        """
        return DataFrame(self._internal.copy())

    def dropna(self, axis=0, how="any", thresh=None, subset=None, inplace=False):
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
        axis = validate_axis(axis)
        inplace = validate_bool_kwarg(inplace, "inplace")
        if axis == 0:
            if subset is not None:
                if isinstance(subset, str):
                    labels = [(subset,)]
                elif isinstance(subset, tuple):
                    labels = [subset]
                else:
                    labels = [sub if isinstance(sub, tuple) else (sub,) for sub in subset]
                invalids = [label for label in labels if label not in self._internal.column_labels]
                if len(invalids) > 0:
                    raise KeyError(invalids)
            else:
                labels = self._internal.column_labels

            cnt = reduce(
                lambda x, y: x + y,
                [
                    F.when(self._kser_for(label).notna().spark_column, 1).otherwise(0)
                    for label in labels
                ],
                F.lit(0),
            )
            if thresh is not None:
                pred = cnt >= F.lit(int(thresh))
            elif how == "any":
                pred = cnt == F.lit(len(labels))
            elif how == "all":
                pred = cnt > F.lit(0)
            else:
                if how is not None:
                    raise ValueError("invalid how option: {h}".format(h=how))
                else:
                    raise TypeError("must specify how or thresh")

            internal = self._internal.with_filter(pred)
            if inplace:
                self._internal = internal
            else:
                return DataFrame(internal)

        else:
            raise NotImplementedError("dropna currently only works for axis=0 or axis='index'")

    # TODO: add 'limit' when value parameter exists
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

        We can also propagate non-null values forward or backward.

        >>> df.fillna(method='ffill')
             A    B    C  D
        0  NaN  2.0  NaN  0
        1  3.0  4.0  NaN  1
        2  3.0  4.0  NaN  5
        3  3.0  3.0  1.0  4

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
        if value is not None:
            axis = validate_axis(axis)
            inplace = validate_bool_kwarg(inplace, "inplace")
            if axis != 0:
                raise NotImplementedError("fillna currently only works for axis=0 or axis='index'")
            if not isinstance(value, (float, int, str, bool, dict, pd.Series)):
                raise TypeError("Unsupported type %s" % type(value))
            if limit is not None:
                raise ValueError("limit parameter for value is not support now")
            if isinstance(value, pd.Series):
                value = value.to_dict()
            if isinstance(value, dict):
                for v in value.values():
                    if not isinstance(v, (float, int, str, bool)):
                        raise TypeError("Unsupported type %s" % type(v))
                value = {k if isinstance(k, tuple) else (k,): v for k, v in value.items()}

                def op(kser):
                    label = kser._internal.column_labels[0]
                    for k, v in value.items():
                        if k == label[: len(k)]:
                            return kser.fillna(
                                value=value[k], method=method, axis=axis, inplace=False, limit=limit
                            )
                    else:
                        return kser

            else:
                op = lambda kser: kser.fillna(
                    value=value, method=method, axis=axis, inplace=False, limit=limit
                )
        elif method is not None:
            op = lambda kser: kser.fillna(
                value=value, method=method, axis=axis, inplace=False, limit=limit
            )
        else:
            raise ValueError("Must specify a fillna 'value' or 'method' parameter.")

        kdf = self._apply_series_op(op)
        if inplace:
            self._internal = kdf._internal
        else:
            return kdf

    # TODO: add 'downcast' when value parameter exists
    def bfill(self, axis=None, inplace=False, limit=None):
        """
        Synonym for `DataFrame.fillna()` with ``method=`bfill```.

        .. note:: the current implementation of 'bfiff' uses Spark's Window
            without specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
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

        Propagate non-null values backward.

        >>> df.bfill()
             A    B    C  D
        0  3.0  2.0  1.0  0
        1  3.0  4.0  1.0  1
        2  NaN  3.0  1.0  5
        3  NaN  3.0  1.0  4
        """
        return self.fillna(method="bfill", axis=axis, inplace=inplace, limit=limit)

    # TODO: add 'downcast' when value parameter exists
    def ffill(self, axis=None, inplace=False, limit=None):
        """
        Synonym for `DataFrame.fillna()` with ``method=`ffill```.

        .. note:: the current implementation of 'ffiff' uses Spark's Window
            without specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
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

        Propagate non-null values forward.

        >>> df.ffill()
             A    B    C  D
        0  NaN  2.0  NaN  0
        1  3.0  4.0  NaN  1
        2  3.0  4.0  NaN  5
        3  3.0  3.0  1.0  4
        """
        return self.fillna(method="ffill", axis=axis, inplace=inplace, limit=limit)

    def replace(
        self,
        to_replace=None,
        value=None,
        subset=None,
        inplace=False,
        limit=None,
        regex=False,
        method="pad",
    ):
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
        if method != "pad":
            raise NotImplementedError("replace currently works only for method='pad")
        if limit is not None:
            raise NotImplementedError("replace currently works only when limit=None")
        if regex is not False:
            raise NotImplementedError("replace currently doesn't supports regex")
        inplace = validate_bool_kwarg(inplace, "inplace")

        if value is not None and not isinstance(value, (int, float, str, list, dict)):
            raise TypeError("Unsupported type {}".format(type(value)))
        if to_replace is not None and not isinstance(to_replace, (int, float, str, list, dict)):
            raise TypeError("Unsupported type {}".format(type(to_replace)))

        if isinstance(value, list) and isinstance(to_replace, list):
            if len(value) != len(to_replace):
                raise ValueError("Length of to_replace and value must be same")

        # TODO: Do we still need to support this argument?
        if subset is None:
            subset = self._internal.column_labels
        elif isinstance(subset, str):
            subset = [(subset,)]
        elif isinstance(subset, tuple):
            subset = [subset]
        else:
            subset = [sub if isinstance(sub, tuple) else (sub,) for sub in subset]
        subset = [self._internal.spark_column_name_for(label) for label in subset]

        sdf = self._sdf
        if (
            isinstance(to_replace, dict)
            and value is None
            and (not any(isinstance(i, dict) for i in to_replace.values()))
        ):
            sdf = sdf.replace(to_replace, value, subset)
        elif isinstance(to_replace, dict):
            for name, replacement in to_replace.items():
                if isinstance(name, str):
                    name = (name,)
                df_column = self._internal.spark_column_name_for(name)
                if isinstance(replacement, dict):
                    sdf = sdf.replace(replacement, subset=df_column)
                else:
                    sdf = sdf.withColumn(
                        df_column,
                        F.when(scol_for(sdf, df_column) == replacement, value).otherwise(
                            scol_for(sdf, df_column)
                        ),
                    )
        else:
            sdf = sdf.replace(to_replace, value, subset)

        internal = self._internal.with_new_sdf(sdf)
        if inplace:
            self._internal = internal
        else:
            return DataFrame(internal)

    def clip(self, lower: Union[float, int] = None, upper: Union[float, int] = None) -> "DataFrame":
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
            raise ValueError(
                "List-like value are not supported for 'lower' and 'upper' at the " + "moment"
            )

        if lower is None and upper is None:
            return self

        numeric_types = (
            DecimalType,
            DoubleType,
            FloatType,
            ByteType,
            IntegerType,
            LongType,
            ShortType,
        )

        def op(kser):
            if isinstance(kser.spark_type, numeric_types):
                scol = kser.spark_column
                if lower is not None:
                    scol = F.when(scol < lower, lower).otherwise(scol)
                if upper is not None:
                    scol = F.when(scol > upper, upper).otherwise(scol)
                return scol.alias(kser._internal.data_spark_column_names[0])
            else:
                return kser

        return self._apply_series_op(op)

    def head(self, n: int = 5) -> "DataFrame":
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
        if n < 0:
            n = len(self) + n
        if n <= 0:
            return DataFrame(self._internal.with_filter(F.lit(False)))
        else:
            if get_option("compute.ordered_head"):
                sdf = self._sdf.orderBy(NATURAL_ORDER_COLUMN_NAME)
            else:
                sdf = self._sdf
            return DataFrame(self._internal.with_new_sdf(sdf.limit(n)))

    def pivot_table(self, values=None, index=None, columns=None, aggfunc="mean", fill_value=None):
        """
        Create a spreadsheet-style pivot table as a DataFrame. The levels in
        the pivot table will be stored in MultiIndex objects (hierarchical
        indexes) on the index and columns of the result DataFrame.

        Parameters
        ----------
        values : column to aggregate.
            They should be either a list less than three or a string.
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
        ...                        columns='C', aggfunc='sum')
        >>> table.sort_index()  # doctest: +NORMALIZE_WHITESPACE
        C        large  small
        A   B
        bar one    4.0      5
            two    7.0      6
        foo one    4.0      1
            two    NaN      6

        We can also fill missing values using the `fill_value` parameter.

        >>> table = df.pivot_table(values='D', index=['A', 'B'],
        ...                        columns='C', aggfunc='sum', fill_value=0)
        >>> table.sort_index()  # doctest: +NORMALIZE_WHITESPACE
        C        large  small
        A   B
        bar one      4      5
            two      7      6
        foo one      4      1
            two      0      6

        We can also calculate multiple types of aggregations for any given
        value column.

        >>> table = df.pivot_table(values=['D'], index =['C'],
        ...                        columns="A", aggfunc={'D': 'mean'})
        >>> table.sort_index()  # doctest: +NORMALIZE_WHITESPACE
                 D
        A      bar       foo
        C
        large  5.5  2.000000
        small  5.5  2.333333

        The next example aggregates on multiple values.

        >>> table = df.pivot_table(index=['C'], columns="A", values=['D', 'E'],
        ...                         aggfunc={'D': 'mean', 'E': 'sum'})
        >>> table.sort_index() # doctest: +NORMALIZE_WHITESPACE
                 D             E
        A      bar       foo bar foo
        C
        large  5.5  2.000000  15   9
        small  5.5  2.333333  17  13
        """
        if not isinstance(columns, (str, tuple)):
            raise ValueError("columns should be string or tuple.")

        if not isinstance(values, (str, tuple)) and not isinstance(values, list):
            raise ValueError("values should be string or list of one column.")

        if not isinstance(aggfunc, str) and (
            not isinstance(aggfunc, dict)
            or not all(
                isinstance(key, (str, tuple)) and isinstance(value, str)
                for key, value in aggfunc.items()
            )
        ):
            raise ValueError(
                "aggfunc must be a dict mapping from column name (string or tuple) "
                "to aggregate functions (string)."
            )

        if isinstance(aggfunc, dict) and index is None:
            raise NotImplementedError(
                "pivot_table doesn't support aggfunc" " as dict and without index."
            )
        if isinstance(values, list) and index is None:
            raise NotImplementedError("values can't be a list without index.")

        if columns not in self.columns:
            raise ValueError("Wrong columns {}.".format(columns))
        if isinstance(columns, str):
            columns = (columns,)

        if isinstance(values, list):
            values = [col if isinstance(col, tuple) else (col,) for col in values]
            if not all(
                isinstance(self._internal.spark_type_for(col), NumericType) for col in values
            ):
                raise TypeError("values should be a numeric type.")
        else:
            values = values if isinstance(values, tuple) else (values,)
            if not isinstance(self._internal.spark_type_for(values), NumericType):
                raise TypeError("values should be a numeric type.")

        if isinstance(aggfunc, str):
            if isinstance(values, list):
                agg_cols = [
                    F.expr(
                        "{1}(`{0}`) as `{0}`".format(
                            self._internal.spark_column_name_for(value), aggfunc
                        )
                    )
                    for value in values
                ]
            else:
                agg_cols = [
                    F.expr(
                        "{1}(`{0}`) as `{0}`".format(
                            self._internal.spark_column_name_for(values), aggfunc
                        )
                    )
                ]
        elif isinstance(aggfunc, dict):
            aggfunc = {
                key if isinstance(key, tuple) else (key,): value for key, value in aggfunc.items()
            }
            agg_cols = [
                F.expr(
                    "{1}(`{0}`) as `{0}`".format(self._internal.spark_column_name_for(key), value)
                )
                for key, value in aggfunc.items()
            ]
            agg_columns = [key for key, _ in aggfunc.items()]

            if set(agg_columns) != set(values):
                raise ValueError("Columns in aggfunc must be the same as values.")

        if index is None:
            sdf = (
                self._sdf.groupBy()
                .pivot(pivot_col=self._internal.spark_column_name_for(columns))
                .agg(*agg_cols)
            )

        elif isinstance(index, list):
            index = [label if isinstance(label, tuple) else (label,) for label in index]
            sdf = (
                self._sdf.groupBy([self._internal.spark_column_for(label) for label in index])
                .pivot(pivot_col=self._internal.spark_column_name_for(columns))
                .agg(*agg_cols)
            )
        else:
            raise ValueError("index should be a None or a list of columns.")

        if fill_value is not None and isinstance(fill_value, (int, float)):
            sdf = sdf.fillna(fill_value)

        if index is not None:
            if isinstance(values, list):
                index_columns = [self._internal.spark_column_name_for(label) for label in index]
                data_columns = [column for column in sdf.columns if column not in index_columns]

                if len(values) > 1:
                    # If we have two values, Spark will return column's name
                    # in this format: column_values, where column contains
                    # their values in the DataFrame and values is
                    # the column list passed to the pivot_table().
                    # E.g. if column is b and values is ['b','e'],
                    # then ['2_b', '2_e', '3_b', '3_e'].

                    # We sort the columns of Spark DataFrame by values.
                    data_columns.sort(key=lambda x: x.split("_", 1)[1])
                    sdf = sdf.select(index_columns + data_columns)

                    column_name_to_index = dict(
                        zip(self._internal.data_spark_column_names, self._internal.column_labels)
                    )
                    column_labels = [
                        tuple(list(column_name_to_index[name.split("_")[1]]) + [name.split("_")[0]])
                        for name in data_columns
                    ]
                    index_map = OrderedDict(zip(index_columns, index))
                    column_label_names = ([None] * column_labels_level(values)) + [
                        str(columns) if len(columns) > 1 else columns[0]
                    ]
                    internal = _InternalFrame(
                        spark_frame=sdf,
                        index_map=index_map,
                        column_labels=column_labels,
                        data_spark_columns=[scol_for(sdf, col) for col in data_columns],
                        column_label_names=column_label_names,
                    )
                    kdf = DataFrame(internal)
                else:
                    column_labels = [tuple(list(values[0]) + [column]) for column in data_columns]
                    index_map = OrderedDict(zip(index_columns, index))
                    column_label_names = ([None] * len(values[0])) + [
                        str(columns) if len(columns) > 1 else columns[0]
                    ]
                    internal = _InternalFrame(
                        spark_frame=sdf,
                        index_map=index_map,
                        column_labels=column_labels,
                        data_spark_columns=[scol_for(sdf, col) for col in data_columns],
                        column_label_names=column_label_names,
                    )
                    kdf = DataFrame(internal)
                return kdf
            else:
                index_columns = [self._internal.spark_column_name_for(label) for label in index]
                index_map = OrderedDict(zip(index_columns, index))
                column_label_names = [str(columns) if len(columns) > 1 else columns[0]]
                internal = _InternalFrame(
                    spark_frame=sdf, index_map=index_map, column_label_names=column_label_names
                )
                return DataFrame(internal)
        else:
            if isinstance(values, list):
                index_values = values[-1]
            else:
                index_values = values
            index_map = OrderedDict()
            for i, index_value in enumerate(index_values):
                colname = SPARK_INDEX_NAME_FORMAT(i)
                sdf = sdf.withColumn(colname, F.lit(index_value))
                index_map[colname] = None
            column_label_names = [str(columns) if len(columns) > 1 else columns[0]]
            internal = _InternalFrame(
                spark_frame=sdf, index_map=index_map, column_label_names=column_label_names
            )
            return DataFrame(internal)

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
        bar  A  B  C
        foo
        one  1  2  3
        two  4  5  6

        >>> df.pivot(columns='bar', values='baz').sort_index()  # doctest: +NORMALIZE_WHITESPACE
        bar  A    B    C
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
        bar    A    B    C
        foo
        one  1.0  NaN  NaN
        two  NaN  3.0  4.0

        It also support multi-index and multi-index column.
        >>> df.columns = pd.MultiIndex.from_tuples([('a', 'foo'), ('a', 'bar'), ('b', 'baz')])

        >>> df = df.set_index(('a', 'bar'), append=True)
        >>> df  # doctest: +NORMALIZE_WHITESPACE
                      a   b
                    foo baz
          (a, bar)
        0 A         one   1
        1 A         one   2
        2 B         two   3
        3 C         two   4

        >>> df.pivot(columns=('a', 'foo'), values=('b', 'baz')).sort_index()
        ... # doctest: +NORMALIZE_WHITESPACE
        ('a', 'foo')  one  two
          (a, bar)
        0 A           1.0  NaN
        1 A           2.0  NaN
        2 B           NaN  3.0
        3 C           NaN  4.0

        """
        if columns is None:
            raise ValueError("columns should be set.")

        if values is None:
            raise ValueError("values should be set.")

        should_use_existing_index = index is not None
        if should_use_existing_index:
            df = self
            index = [index]
        else:
            # The index after `reset_index()` will never be used, so use "distributed" index
            # as a dummy to avoid overhead.
            with option_context("compute.default_index_type", "distributed"):
                df = self.reset_index()
            index = df._internal.column_labels[: len(self._internal.index_spark_column_names)]

        df = df.pivot_table(index=index, columns=columns, values=values, aggfunc="first")

        if should_use_existing_index:
            return df
        else:
            index_columns = df._internal.index_spark_column_names
            internal = df._internal.copy(
                index_map=OrderedDict(
                    (index_column, name)
                    for index_column, name in zip(index_columns, self._internal.index_names)
                )
            )
            return DataFrame(internal)

    @property
    def columns(self):
        """The column labels of the DataFrame."""
        if self._internal.column_labels_level > 1:
            columns = pd.MultiIndex.from_tuples(self._internal.column_labels)
        else:
            columns = pd.Index([label[0] for label in self._internal.column_labels])
        if self._internal.column_label_names is not None:
            columns.names = self._internal.column_label_names
        return columns

    @columns.setter
    def columns(self, columns):
        if isinstance(columns, pd.MultiIndex):
            column_labels = columns.tolist()
            old_names = self._internal.column_labels
            if len(old_names) != len(column_labels):
                raise ValueError(
                    "Length mismatch: Expected axis has %d elements, new values have %d elements"
                    % (len(old_names), len(column_labels))
                )
            column_label_names = columns.names
            data_columns = [name_like_string(label) for label in column_labels]
            data_spark_columns = [
                self._internal.spark_column_for(label).alias(name)
                for label, name in zip(self._internal.column_labels, data_columns)
            ]
            self._internal = self._internal.with_new_columns(
                data_spark_columns, column_labels=column_labels
            )
            sdf = self._sdf.select(
                self._internal.index_spark_columns
                + [
                    self._internal.spark_column_for(label).alias(name)
                    for label, name in zip(self._internal.column_labels, data_columns)
                ]
                + list(HIDDEN_COLUMNS)
            )
            data_spark_columns = [scol_for(sdf, col) for col in data_columns]
            self._internal = self._internal.copy(
                spark_frame=sdf,
                column_labels=column_labels,
                data_spark_columns=data_spark_columns,
                column_label_names=column_label_names,
            )
        else:
            old_names = self._internal.column_labels
            if len(old_names) != len(columns):
                raise ValueError(
                    "Length mismatch: Expected axis has %d elements, new values have %d elements"
                    % (len(old_names), len(columns))
                )
            column_labels = [col if isinstance(col, tuple) else (col,) for col in columns]
            if isinstance(columns, pd.Index):
                column_label_names = columns.names
            else:
                column_label_names = None
            data_columns = [name_like_string(label) for label in column_labels]
            sdf = self._sdf.select(
                self._internal.index_spark_columns
                + [
                    self._internal.spark_column_for(label).alias(name)
                    for label, name in zip(self._internal.column_labels, data_columns)
                ]
                + list(HIDDEN_COLUMNS)
            )
            data_spark_columns = [scol_for(sdf, col) for col in data_columns]
            self._internal = self._internal.copy(
                spark_frame=sdf,
                column_labels=column_labels,
                data_spark_columns=data_spark_columns,
                column_label_names=column_label_names,
            )

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
        return pd.Series(
            [self._kser_for(label).dtype for label in self._internal.column_labels],
            index=pd.Index(
                [label if len(label) > 1 else label[0] for label in self._internal.column_labels]
            ),
        )

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
            raise ValueError("at least one of include or exclude must be " "nonempty")

        # can't both include AND exclude!
        if set(include).intersection(set(exclude)):
            raise ValueError(
                "include and exclude overlap on {inc_ex}".format(
                    inc_ex=set(include).intersection(set(exclude))
                )
            )

        # Handle Spark types
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

        column_labels = []
        for label in self._internal.column_labels:
            if len(include) > 0:
                should_include = (
                    infer_dtype_from_object(self._kser_for(label).dtype.name) in include_numpy_type
                    or self._internal.spark_type_for(label) in include_spark_type
                )
            else:
                should_include = not (
                    infer_dtype_from_object(self._kser_for(label).dtype.name) in exclude_numpy_type
                    or self._internal.spark_type_for(label) in exclude_spark_type
                )

            if should_include:
                column_labels.append(label)

        data_spark_columns = [self._internal.spark_column_for(label) for label in column_labels]
        return DataFrame(
            self._internal.with_new_columns(data_spark_columns, column_labels=column_labels)
        )

    def count(self, axis=None):
        """
        Count non-NA cells for each column.

        The values `None`, `NaN` are considered NA.

        Parameters
        ----------
        axis : {0 or ‘index’, 1 or ‘columns’}, default 0
            If 0 or ‘index’ counts are generated for each column. If 1 or ‘columns’ counts are
            generated for each row.

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
        Name: 0, dtype: int64

        >>> df.count(axis=1)
        0    3
        1    2
        2    3
        3    3
        4    3
        Name: 0, dtype: int64
        """
        return self._reduce_for_stat_function(
            _Frame._count_expr, name="count", axis=axis, numeric_only=False
        )

    def drop(
        self,
        labels=None,
        axis=1,
        columns: Union[str, Tuple[str, ...], List[str], List[Tuple[str, ...]]] = None,
    ):
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

        Also support for MultiIndex

        >>> df = ks.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6], 'w': [7, 8]},
        ...                   columns=['x', 'y', 'z', 'w'])
        >>> columns = [('a', 'x'), ('a', 'y'), ('b', 'z'), ('b', 'w')]
        >>> df.columns = pd.MultiIndex.from_tuples(columns)
        >>> df  # doctest: +NORMALIZE_WHITESPACE
           a     b
           x  y  z  w
        0  1  3  5  7
        1  2  4  6  8
        >>> df.drop('a')  # doctest: +NORMALIZE_WHITESPACE
           b
           z  w
        0  5  7
        1  6  8

        Notes
        -----
        Currently only axis = 1 is supported in this function,
        axis = 0 is yet to be implemented.
        """
        if labels is not None:
            axis = validate_axis(axis)
            if axis == 1:
                return self.drop(columns=labels)
            raise NotImplementedError("Drop currently only works for axis=1")
        elif columns is not None:
            if isinstance(columns, str):
                columns = [(columns,)]  # type: ignore
            elif isinstance(columns, tuple):
                columns = [columns]
            else:
                columns = [  # type: ignore
                    col if isinstance(col, tuple) else (col,) for col in columns  # type: ignore
                ]
            drop_column_labels = set(
                label
                for label in self._internal.column_labels
                for col in columns
                if label[: len(col)] == col
            )
            if len(drop_column_labels) == 0:
                raise KeyError(columns)
            cols, labels = zip(
                *(
                    (column, label)
                    for column, label in zip(
                        self._internal.data_spark_column_names, self._internal.column_labels
                    )
                    if label not in drop_column_labels
                )
            )
            data_spark_columns = [self._internal.spark_column_for(label) for label in labels]
            internal = self._internal.with_new_columns(
                data_spark_columns, column_labels=list(labels)
            )
            return DataFrame(internal)
        else:
            raise ValueError("Need to specify at least one of 'labels' or 'columns'")

    def _sort(
        self, by: List[Column], ascending: Union[bool, List[bool]], inplace: bool, na_position: str
    ):
        if isinstance(ascending, bool):
            ascending = [ascending] * len(by)
        if len(ascending) != len(by):
            raise ValueError(
                "Length of ascending ({}) != length of by ({})".format(len(ascending), len(by))
            )
        if na_position not in ("first", "last"):
            raise ValueError("invalid na_position: '{}'".format(na_position))

        # Mapper: Get a spark column function for (ascending, na_position) combination
        # Note that 'asc_nulls_first' and friends were added as of Spark 2.4, see SPARK-23847.
        mapper = {
            (True, "first"): lambda x: Column(getattr(x._jc, "asc_nulls_first")()),
            (True, "last"): lambda x: Column(getattr(x._jc, "asc_nulls_last")()),
            (False, "first"): lambda x: Column(getattr(x._jc, "desc_nulls_first")()),
            (False, "last"): lambda x: Column(getattr(x._jc, "desc_nulls_last")()),
        }
        by = [mapper[(asc, na_position)](scol) for scol, asc in zip(by, ascending)]
        sdf = self._sdf.sort(*(by + [NATURAL_ORDER_COLUMN_NAME]))
        kdf = DataFrame(self._internal.with_new_sdf(sdf))  # type: ks.DataFrame
        if inplace:
            self._internal = kdf._internal
            return None
        else:
            return kdf

    def sort_values(
        self,
        by: Union[str, List[str], Tuple[str, ...], List[Tuple[str, ...]]],
        ascending: Union[bool, List[bool]] = True,
        inplace: bool = False,
        na_position: str = "last",
    ) -> Optional["DataFrame"]:
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
        inplace = validate_bool_kwarg(inplace, "inplace")
        if isinstance(by, (str, tuple)):
            by = [by]  # type: ignore
        else:
            by = [b if isinstance(b, tuple) else (b,) for b in by]  # type: ignore

        new_by = []
        for colname in by:
            ser = self[colname]
            if not isinstance(ser, ks.Series):
                raise ValueError(
                    "The column %s is not unique. For a multi-index, the label must be a tuple "
                    "with elements corresponding to each level." % name_like_string(colname)
                )
            new_by.append(ser.spark_column)

        return self._sort(by=new_by, ascending=ascending, inplace=inplace, na_position=na_position)

    def sort_index(
        self,
        axis: int = 0,
        level: Optional[Union[int, List[int]]] = None,
        ascending: bool = True,
        inplace: bool = False,
        kind: str = None,
        na_position: str = "last",
    ) -> Optional["DataFrame"]:
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

        >>> df = ks.DataFrame({'A': range(4), 'B': range(4)[::-1]},
        ...                   index=[['b', 'b', 'a', 'a'], [1, 0, 1, 0]],
        ...                   columns=['A', 'B'])

        >>> df.sort_index()
             A  B
        a 0  3  0
          1  2  1
        b 0  1  2
          1  0  3

        >>> df.sort_index(level=1)  # doctest: +SKIP
             A  B
        a 0  3  0
        b 0  1  2
        a 1  2  1
        b 1  0  3

        >>> df.sort_index(level=[1, 0])
             A  B
        a 0  3  0
        b 0  1  2
        a 1  2  1
        b 1  0  3
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError("No other axis than 0 are supported at the moment")
        if kind is not None:
            raise NotImplementedError(
                "Specifying the sorting algorithm is not supported at the moment."
            )

        if level is None or (is_list_like(level) and len(level) == 0):  # type: ignore
            by = self._internal.index_spark_columns
        elif is_list_like(level):
            by = [self._internal.index_spark_columns[l] for l in level]  # type: ignore
        else:
            by = [self._internal.index_spark_columns[level]]

        return self._sort(by=by, ascending=ascending, inplace=inplace, na_position=na_position)

    # TODO:  add keep = First
    def nlargest(self, n: int, columns: "Any") -> "DataFrame":
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
    def nsmallest(self, n: int, columns: "Any") -> "DataFrame":
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
                % (set(values.keys()).difference(self.columns))
            )

        data_spark_columns = []
        if isinstance(values, dict):
            for i, col in enumerate(self.columns):
                if col in values:
                    data_spark_columns.append(
                        self._internal.spark_column_for(self._internal.column_labels[i])
                        .isin(values[col])
                        .alias(self._internal.data_spark_column_names[i])
                    )
                else:
                    data_spark_columns.append(
                        F.lit(False).alias(self._internal.data_spark_column_names[i])
                    )
        elif is_list_like(values):
            data_spark_columns += [
                self._internal.spark_column_for(label)
                .isin(list(values))
                .alias(self._internal.spark_column_name_for(label))
                for label in self._internal.column_labels
            ]
        else:
            raise TypeError("Values should be iterable, Series, DataFrame or dict.")

        return DataFrame(self._internal.with_new_columns(data_spark_columns))

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

    def merge(
        self,
        right: "DataFrame",
        how: str = "inner",
        on: Optional[Union[str, List[str], Tuple[str, ...], List[Tuple[str, ...]]]] = None,
        left_on: Optional[Union[str, List[str], Tuple[str, ...], List[Tuple[str, ...]]]] = None,
        right_on: Optional[Union[str, List[str], Tuple[str, ...], List[Tuple[str, ...]]]] = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: Tuple[str, str] = ("_x", "_y"),
    ) -> "DataFrame":
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

        See Also
        --------
        DataFrame.join : Join columns of another DataFrame.
        DataFrame.update : Modify in place using non-NA values from another DataFrame.
        DataFrame.hint : Specifies some hint on the current DataFrame.
        broadcast : Marks a DataFrame as small enough for use in broadcast joins.

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
        >>> merged.sort_values(by=['lkey', 'value_x', 'rkey', 'value_y'])  # doctest: +ELLIPSIS
          lkey  value_x rkey  value_y
        ...bar        2  bar        6
        ...baz        3  baz        7
        ...foo        1  foo        5
        ...foo        1  foo        8
        ...foo        5  foo        5
        ...foo        5  foo        8

        >>> left_kdf = ks.DataFrame({'A': [1, 2]})
        >>> right_kdf = ks.DataFrame({'B': ['x', 'y']}, index=[1, 2])

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True).sort_index()
           A  B
        1  2  x

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True, how='left').sort_index()
           A     B
        0  1  None
        1  2     x

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True, how='right').sort_index()
             A  B
        1  2.0  x
        2  NaN  y

        >>> left_kdf.merge(right_kdf, left_index=True, right_index=True, how='outer').sort_index()
             A     B
        0  1.0  None
        1  2.0     x
        2  NaN     y

        Notes
        -----
        As described in #263, joining string columns currently returns None for missing values
            instead of NaN.
        """

        def to_list(
            os: Optional[Union[str, List[str], Tuple[str, ...], List[Tuple[str, ...]]]]
        ) -> List[Tuple[str, ...]]:
            if os is None:
                return []
            elif isinstance(os, tuple):
                return [os]
            elif isinstance(os, str):
                return [(os,)]
            else:
                return [o if isinstance(o, tuple) else (o,) for o in os]  # type: ignore

        if isinstance(right, ks.Series):
            right = right.to_frame()

        if on:
            if left_on or right_on:
                raise ValueError(
                    'Can only pass argument "on" OR "left_on" and "right_on", '
                    "not a combination of both."
                )
            left_key_names = list(map(self._internal.spark_column_name_for, to_list(on)))
            right_key_names = list(map(right._internal.spark_column_name_for, to_list(on)))
        else:
            # TODO: need special handling for multi-index.
            if left_index:
                left_key_names = self._internal.index_spark_column_names
            else:
                left_key_names = list(map(self._internal.spark_column_name_for, to_list(left_on)))
            if right_index:
                right_key_names = right._internal.index_spark_column_names
            else:
                right_key_names = list(
                    map(right._internal.spark_column_name_for, to_list(right_on))
                )

            if left_key_names and not right_key_names:
                raise ValueError("Must pass right_on or right_index=True")
            if right_key_names and not left_key_names:
                raise ValueError("Must pass left_on or left_index=True")
            if not left_key_names and not right_key_names:
                common = list(self.columns.intersection(right.columns))
                if len(common) == 0:
                    raise ValueError(
                        "No common columns to perform merge on. Merge options: "
                        "left_on=None, right_on=None, left_index=False, right_index=False"
                    )
                left_key_names = list(map(self._internal.spark_column_name_for, to_list(common)))
                right_key_names = list(map(right._internal.spark_column_name_for, to_list(common)))
            if len(left_key_names) != len(right_key_names):  # type: ignore
                raise ValueError("len(left_keys) must equal len(right_keys)")

        if how == "full":
            warnings.warn(
                "Warning: While Koalas will accept 'full', you should use 'outer' "
                + "instead to be compatible with the pandas merge API",
                UserWarning,
            )
        if how == "outer":
            # 'outer' in pandas equals 'full' in Spark
            how = "full"
        if how not in ("inner", "left", "right", "full"):
            raise ValueError(
                "The 'how' parameter has to be amongst the following values: ",
                "['inner', 'left', 'right', 'outer']",
            )

        left_table = self._sdf.alias("left_table")
        right_table = right._sdf.alias("right_table")

        left_key_columns = [  # type: ignore
            scol_for(left_table, label) for label in left_key_names
        ]
        right_key_columns = [  # type: ignore
            scol_for(right_table, label) for label in right_key_names
        ]

        join_condition = reduce(
            lambda x, y: x & y,
            [lkey == rkey for lkey, rkey in zip(left_key_columns, right_key_columns)],
        )

        joined_table = left_table.join(right_table, join_condition, how=how)

        # Unpack suffixes tuple for convenience
        left_suffix = suffixes[0]
        right_suffix = suffixes[1]

        # Append suffixes to columns with the same name to avoid conflicts later
        duplicate_columns = set(self._internal.column_labels) & set(right._internal.column_labels)

        exprs = []
        data_columns = []
        column_labels = []

        left_scol_for = lambda label: scol_for(
            left_table, self._internal.spark_column_name_for(label)
        )
        right_scol_for = lambda label: scol_for(
            right_table, right._internal.spark_column_name_for(label)
        )

        for label in self._internal.column_labels:
            col = self._internal.spark_column_name_for(label)
            scol = left_scol_for(label)
            if label in duplicate_columns:
                spark_column_name = self._internal.spark_column_name_for(label)
                if (
                    spark_column_name in left_key_names and spark_column_name in right_key_names
                ):  # type: ignore
                    right_scol = right_scol_for(label)
                    if how == "right":
                        scol = right_scol
                    elif how == "full":
                        scol = F.when(scol.isNotNull(), scol).otherwise(right_scol).alias(col)
                    else:
                        pass
                else:
                    col = col + left_suffix
                    scol = scol.alias(col)
                    label = tuple([label[0] + left_suffix] + list(label[1:]))
            exprs.append(scol)
            data_columns.append(col)
            column_labels.append(label)
        for label in right._internal.column_labels:
            col = right._internal.spark_column_name_for(label)
            scol = right_scol_for(label)
            if label in duplicate_columns:
                spark_column_name = self._internal.spark_column_name_for(label)
                if (
                    spark_column_name in left_key_names and spark_column_name in right_key_names
                ):  # type: ignore
                    continue
                else:
                    col = col + right_suffix
                    scol = scol.alias(col)
                    label = tuple([label[0] + right_suffix] + list(label[1:]))
            exprs.append(scol)
            data_columns.append(col)
            column_labels.append(label)

        left_index_scols = self._internal.index_spark_columns
        right_index_scols = right._internal.index_spark_columns

        # Retain indices if they are used for joining
        if left_index:
            if right_index:
                if how in ("inner", "left"):
                    exprs.extend(left_index_scols)
                    index_map = self._internal.index_map
                elif how == "right":
                    exprs.extend(right_index_scols)
                    index_map = right._internal.index_map
                else:
                    index_map = OrderedDict()
                    for (col, name), left_scol, right_scol in zip(
                        self._internal.index_map.items(), left_index_scols, right_index_scols
                    ):
                        scol = F.when(left_scol.isNotNull(), left_scol).otherwise(right_scol)
                        exprs.append(scol.alias(col))
                        index_map[col] = name
            else:
                exprs.extend(right_index_scols)
                index_map = right._internal.index_map
        elif right_index:
            exprs.extend(left_index_scols)
            index_map = self._internal.index_map
        else:
            index_map = OrderedDict()

        selected_columns = joined_table.select(*exprs)

        internal = _InternalFrame(
            spark_frame=selected_columns,
            index_map=index_map if index_map else None,
            column_labels=column_labels,
            data_spark_columns=[scol_for(selected_columns, col) for col in data_columns],
        )
        return DataFrame(internal)

    def join(
        self,
        right: "DataFrame",
        on: Optional[Union[str, List[str], Tuple[str, ...], List[Tuple[str, ...]]]] = None,
        how: str = "left",
        lsuffix: str = "",
        rsuffix: str = "",
    ) -> "DataFrame":
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
        DataFrame.update : Modify in place using non-NA values from another DataFrame.
        DataFrame.hint : Specifies some hint on the current DataFrame.
        broadcast : Marks a DataFrame as small enough for use in broadcast joins.

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
        >>> join_kdf.index
        Int64Index([0, 1, 2, 3], dtype='int64')
        """
        if isinstance(right, ks.Series):
            common = list(self.columns.intersection([right.name]))
        else:
            common = list(self.columns.intersection(right.columns))
        if len(common) > 0 and not lsuffix and not rsuffix:
            raise ValueError(
                "columns overlap but no suffix specified: " "{rename}".format(rename=common)
            )
        if on:
            self = self.set_index(on)
            join_kdf = self.merge(
                right, left_index=True, right_index=True, how=how, suffixes=(lsuffix, rsuffix)
            ).reset_index()
        else:
            join_kdf = self.merge(
                right, left_index=True, right_index=True, how=how, suffixes=(lsuffix, rsuffix)
            )
        return join_kdf

    def append(
        self,
        other: "DataFrame",
        ignore_index: bool = False,
        verify_integrity: bool = False,
        sort: bool = False,
    ) -> "DataFrame":
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
            raise NotImplementedError("The 'sort' parameter is currently not supported")

        if not ignore_index:
            index_scols = self._internal.index_spark_columns
            if len(index_scols) != len(other._internal.index_spark_columns):
                raise ValueError("Both DataFrames have to have the same number of index levels")

            if verify_integrity and len(index_scols) > 0:
                if (
                    self._sdf.select(index_scols)
                    .intersect(other._sdf.select(other._internal.index_spark_columns))
                    .count()
                ) > 0:
                    raise ValueError("Indices have overlapping values")

        # Lazy import to avoid circular dependency issues
        from databricks.koalas.namespace import concat

        return concat([self, other], ignore_index=ignore_index)

    # TODO: add 'filter_func' and 'errors' parameter
    def update(self, other: "DataFrame", join: str = "left", overwrite: bool = True):
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
        DataFrame.join : Join columns of another DataFrame.
        DataFrame.hint : Specifies some hint on the current DataFrame.
        broadcast : Marks a DataFrame as small enough for use in broadcast joins.

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 3], 'B': [400, 500, 600]}, columns=['A', 'B'])
        >>> new_df = ks.DataFrame({'B': [4, 5, 6], 'C': [7, 8, 9]}, columns=['B', 'C'])
        >>> df.update(new_df)
        >>> df.sort_index()
           A  B
        0  1  4
        1  2  5
        2  3  6

        The DataFrame's length does not increase as a result of the update,
        only values at matching index/column labels are updated.

        >>> df = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']}, columns=['A', 'B'])
        >>> new_df = ks.DataFrame({'B': ['d', 'e', 'f', 'g', 'h', 'i']}, columns=['B'])
        >>> df.update(new_df)
        >>> df.sort_index()
           A  B
        0  a  d
        1  b  e
        2  c  f

        For Series, it's name attribute must be set.

        >>> df = ks.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']}, columns=['A', 'B'])
        >>> new_column = ks.Series(['d', 'e'], name='B', index=[0, 2])
        >>> df.update(new_column)
        >>> df.sort_index()
           A  B
        0  a  d
        1  b  y
        2  c  e

        If `other` contains None the corresponding values are not updated in the original dataframe.

        >>> df = ks.DataFrame({'A': [1, 2, 3], 'B': [400, 500, 600]}, columns=['A', 'B'])
        >>> new_df = ks.DataFrame({'B': [4, None, 6]}, columns=['B'])
        >>> df.update(new_df)
        >>> df.sort_index()
           A      B
        0  1    4.0
        1  2  500.0
        2  3    6.0
        """
        if join != "left":
            raise NotImplementedError("Only left join is supported")

        if isinstance(other, ks.Series):
            other = DataFrame(other)

        update_columns = list(
            set(self._internal.column_labels).intersection(set(other._internal.column_labels))
        )
        update_sdf = self.join(other[update_columns], rsuffix="_new")._sdf

        for column_labels in update_columns:
            column_name = self._internal.spark_column_name_for(column_labels)
            old_col = scol_for(update_sdf, column_name)
            new_col = scol_for(
                update_sdf, other._internal.spark_column_name_for(column_labels) + "_new"
            )
            if overwrite:
                update_sdf = update_sdf.withColumn(
                    column_name, F.when(new_col.isNull(), old_col).otherwise(new_col)
                )
            else:
                update_sdf = update_sdf.withColumn(
                    column_name, F.when(old_col.isNull(), new_col).otherwise(old_col)
                )
        sdf = update_sdf.select(
            [scol_for(update_sdf, col) for col in self._internal.spark_column_names]
            + list(HIDDEN_COLUMNS)
        )
        internal = self._internal.with_new_sdf(sdf)
        self._internal = internal

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        random_state: Optional[int] = None,
    ) -> "DataFrame":
        """
        Return a random sample of items from an axis of object.

        Please call this function using named argument by specifying the ``frac`` argument.

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
            raise NotImplementedError(
                "Function sample currently does not support specifying "
                "exact number of items to return. Use frac instead."
            )

        if frac is None:
            raise ValueError("frac must be specified.")

        sdf = self._sdf.sample(withReplacement=replace, fraction=frac, seed=random_state)
        return DataFrame(self._internal.with_new_sdf(sdf))

    def astype(self, dtype) -> "DataFrame":
        """
        Cast a Koalas object to a specified dtype ``dtype``.

        Parameters
        ----------
        dtype : data type, or dict of column name -> data type
            Use a numpy.dtype or Python type to cast entire Koalas object to
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
        applied = []
        if is_dict_like(dtype):
            for col_name in dtype.keys():
                if col_name not in self.columns:
                    raise KeyError(
                        "Only a column name can be used for the "
                        "key in a dtype mappings argument."
                    )
            for col_name, col in self.items():
                if col_name in dtype:
                    applied.append(col.astype(dtype=dtype[col_name]))
                else:
                    applied.append(col)
        else:
            for col_name, col in self.items():
                applied.append(col.astype(dtype=dtype))
        return DataFrame(self._internal.with_new_columns(applied))

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
        return self._apply_series_op(
            lambda kser: kser.rename(tuple([prefix + i for i in kser._internal.column_labels[0]]))
        )

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
        return self._apply_series_op(
            lambda kser: kser.rename(tuple([i + suffix for i in kser._internal.column_labels[0]]))
        )

    # TODO: include, and exclude should be implemented.
    def describe(self, percentiles: Optional[List[float]] = None) -> "DataFrame":
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
        DataFrame.std: Standard deviation of the observations.

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

        For multi-index columns:

        >>> df.columns = [('num', 'a'), ('num', 'b'), ('obj', 'c')]
        >>> df.describe()  # doctest: +NORMALIZE_WHITESPACE
               num
                 a    b
        count  3.0  3.0
        mean   2.0  5.0
        std    1.0  1.0
        min    1.0  4.0
        25%    1.0  4.0
        50%    2.0  5.0
        75%    3.0  6.0
        max    3.0  6.0

        >>> df[('num', 'b')].describe()
        count    3.0
        mean     5.0
        std      1.0
        min      4.0
        25%      4.0
        50%      5.0
        75%      6.0
        max      6.0
        Name: (num, b), dtype: float64

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
        column_labels = []
        for label in self._internal.column_labels:
            scol = self._internal.spark_column_for(label)
            spark_type = self._internal.spark_type_for(label)
            if isinstance(spark_type, DoubleType) or isinstance(spark_type, FloatType):
                exprs.append(
                    F.nanvl(scol, F.lit(None)).alias(self._internal.spark_column_name_for(label))
                )
                column_labels.append(label)
            elif isinstance(spark_type, NumericType):
                exprs.append(scol)
                column_labels.append(label)

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
        sdf = sdf.replace("stddev", "std", subset="summary")

        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=OrderedDict({"summary": None}),
            column_labels=column_labels,
            data_spark_columns=[
                scol_for(sdf, self._internal.spark_column_name_for(label))
                for label in column_labels
            ],
        )
        return DataFrame(internal).astype("float64")

    def drop_duplicates(self, subset=None, keep="first", inplace=False):
        """
        Return DataFrame with duplicate rows removed, optionally only
        considering certain columns.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', False}, default 'first'
            Determines which duplicates (if any) to keep.
            - ``first`` : Drop duplicates except for the first occurrence.
            - ``last`` : Drop duplicates except for the last occurrence.
            - False : Drop all duplicates.
        inplace : boolean, default False
            Whether to drop duplicates in place or to return a copy.

        Returns
        -------
        DataFrame
            DataFrame with duplicates removed or None if ``inplace=True``.

        >>> df = ks.DataFrame(
        ...     {'a': [1, 2, 2, 2, 3], 'b': ['a', 'a', 'a', 'c', 'd']}, columns = ['a', 'b'])
        >>> df
           a  b
        0  1  a
        1  2  a
        2  2  a
        3  2  c
        4  3  d

        >>> df.drop_duplicates().sort_index()
           a  b
        0  1  a
        1  2  a
        3  2  c
        4  3  d

        >>> df.drop_duplicates('a').sort_index()
           a  b
        0  1  a
        1  2  a
        4  3  d

        >>> df.drop_duplicates(['a', 'b']).sort_index()
           a  b
        0  1  a
        1  2  a
        3  2  c
        4  3  d

        >>> df.drop_duplicates(keep='last').sort_index()
           a  b
        0  1  a
        2  2  a
        3  2  c
        4  3  d

        >>> df.drop_duplicates(keep=False).sort_index()
           a  b
        0  1  a
        3  2  c
        4  3  d
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        sdf, column = self._mark_duplicates(subset, keep)

        sdf = sdf.where(~scol_for(sdf, column)).drop(column)
        internal = self._internal.with_new_sdf(sdf)
        if inplace:
            self._internal = internal
        else:
            return DataFrame(internal)

    def reindex(
        self,
        labels: Optional[Any] = None,
        index: Optional[Any] = None,
        columns: Optional[Any] = None,
        axis: Optional[Union[int, str]] = None,
        copy: Optional[bool] = True,
        fill_value: Optional[Any] = None,
    ) -> "DataFrame":
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
        ...       index=index,
        ...       columns=['http_status', 'response_time'])
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
            axis = validate_axis(axis)
            if axis == 0:
                index = labels
            elif axis == 1:
                columns = labels
            else:
                raise ValueError("No axis named %s for object type %s." % (axis, type(axis)))

        if index is not None and not is_list_like(index):
            raise TypeError(
                "Index must be called with a collection of some kind, "
                "%s was passed" % type(index)
            )

        if columns is not None and not is_list_like(columns):
            raise TypeError(
                "Columns must be called with a collection of some kind, "
                "%s was passed" % type(columns)
            )

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
        assert (
            len(self._internal.index_spark_column_names) <= 1
        ), "Index should be single column or not set."

        index_column = self._internal.index_spark_column_names[0]

        kser = ks.Series(list(index))
        labels = kser._internal._sdf.select(kser.spark_column.alias(index_column))

        joined_df = self._sdf.drop(NATURAL_ORDER_COLUMN_NAME).join(
            labels, on=index_column, how="right"
        )
        internal = self._internal.with_new_sdf(joined_df)

        return internal

    def _reindex_columns(self, columns):
        level = self._internal.column_labels_level
        if level > 1:
            label_columns = list(columns)
            for col in label_columns:
                if not isinstance(col, tuple):
                    raise TypeError("Expected tuple, got {}".format(type(col)))
        else:
            label_columns = [(col,) for col in columns]
        for col in label_columns:
            if len(col) != level:
                raise ValueError(
                    "shape (1,{}) doesn't match the shape (1,{})".format(len(col), level)
                )
        scols, labels = [], []
        for label in label_columns:
            if label in self._internal.column_labels:
                scols.append(self._internal.spark_column_for(label))
            else:
                scols.append(F.lit(np.nan).alias(name_like_string(label)))
            labels.append(label)

        return self._internal.with_new_columns(scols, column_labels=labels)

    def melt(self, id_vars=None, value_vars=None, var_name=None, value_name="value"):
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
            Name to use for the 'variable' column. If None it uses `frame.columns.name` or
            ‘variable’.
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
        ...                    'C': {0: 2, 1: 4, 2: 6}},
        ...                   columns=['A', 'B', 'C'])
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

        >>> df.melt(value_vars='A')
          variable value
        0        A     a
        1        A     b
        2        A     c

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
        column_labels = self._internal.column_labels

        if id_vars is None:
            id_vars = []
        else:
            if isinstance(id_vars, str):
                id_vars = [(id_vars,)]
            elif isinstance(id_vars, tuple):
                if self._internal.column_labels_level == 1:
                    id_vars = [idv if isinstance(idv, tuple) else (idv,) for idv in id_vars]
                else:
                    raise ValueError(
                        "id_vars must be a list of tuples" " when columns are a MultiIndex"
                    )
            else:
                id_vars = [idv if isinstance(idv, tuple) else (idv,) for idv in id_vars]

            non_existence_col = [idv for idv in id_vars if idv not in column_labels]
            if len(non_existence_col) != 0:
                raveled_column_labels = np.ravel(column_labels)
                missing = [
                    nec for nec in np.ravel(non_existence_col) if nec not in raveled_column_labels
                ]
                if len(missing) != 0:
                    raise KeyError(
                        "The following 'id_vars' are not present"
                        " in the DataFrame: {}".format(missing)
                    )
                else:
                    raise KeyError(
                        "None of {} are in the {}".format(non_existence_col, column_labels)
                    )

        if value_vars is None:
            value_vars = []
        else:
            if isinstance(value_vars, str):
                value_vars = [(value_vars,)]
            elif isinstance(value_vars, tuple):
                if self._internal.column_labels_level == 1:
                    value_vars = [
                        valv if isinstance(valv, tuple) else (valv,) for valv in value_vars
                    ]
                else:
                    raise ValueError(
                        "value_vars must be a list of tuples" " when columns are a MultiIndex"
                    )
            else:
                value_vars = [valv if isinstance(valv, tuple) else (valv,) for valv in value_vars]

            non_existence_col = [valv for valv in value_vars if valv not in column_labels]
            if len(non_existence_col) != 0:
                raveled_column_labels = np.ravel(column_labels)
                missing = [
                    nec for nec in np.ravel(non_existence_col) if nec not in raveled_column_labels
                ]
                if len(missing) != 0:
                    raise KeyError(
                        "The following 'value_vars' are not present"
                        " in the DataFrame: {}".format(missing)
                    )
                else:
                    raise KeyError(
                        "None of {} are in the {}".format(non_existence_col, column_labels)
                    )

        if len(value_vars) == 0:
            value_vars = column_labels

        column_labels = [label for label in column_labels if label not in id_vars]

        sdf = self._sdf

        if var_name is None:
            if self._internal.column_label_names is not None:
                var_name = self._internal.column_label_names
            elif self._internal.column_labels_level == 1:
                var_name = ["variable"]
            else:
                var_name = [
                    "variable_{}".format(i) for i in range(self._internal.column_labels_level)
                ]
        elif isinstance(var_name, str):
            var_name = [var_name]

        pairs = F.explode(
            F.array(
                *[
                    F.struct(
                        *(
                            [F.lit(c).alias(name) for c, name in zip(label, var_name)]
                            + [self._internal.spark_column_for(label).alias(value_name)]
                        )
                    )
                    for label in column_labels
                    if label in value_vars
                ]
            )
        )

        columns = (
            [
                self._internal.spark_column_for(label).alias(name_like_string(label))
                for label in id_vars
            ]
            + [F.col("pairs.%s" % name) for name in var_name[: self._internal.column_labels_level]]
            + [F.col("pairs.%s" % value_name)]
        )
        exploded_df = sdf.withColumn("pairs", pairs).select(columns)

        return DataFrame(exploded_df)

    def stack(self):
        """
        Stack the prescribed level(s) from columns to index.

        Return a reshaped DataFrame or Series having a multi-level
        index with one or more new inner-most levels compared to the current
        DataFrame. The new inner-most levels are created by pivoting the
        columns of the current dataframe:

          - if the columns have a single level, the output is a Series;
          - if the columns have multiple levels, the new index
            level(s) is (are) taken from the prescribed level(s) and
            the output is a DataFrame.

        The new index levels are sorted.

        Returns
        -------
        DataFrame or Series
            Stacked dataframe or series.

        See Also
        --------
        DataFrame.unstack : Unstack prescribed level(s) from index axis
            onto column axis.
        DataFrame.pivot : Reshape dataframe from long format to wide
            format.
        DataFrame.pivot_table : Create a spreadsheet-style pivot table
            as a DataFrame.

        Notes
        -----
        The function is named by analogy with a collection of books
        being reorganized from being side by side on a horizontal
        position (the columns of the dataframe) to being stacked
        vertically on top of each other (in the index of the
        dataframe).

        Examples
        --------
        **Single level columns**

        >>> df_single_level_cols = ks.DataFrame([[0, 1], [2, 3]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=['weight', 'height'])

        Stacking a dataframe with a single level column axis returns a Series:

        >>> df_single_level_cols
             weight  height
        cat       0       1
        dog       2       3
        >>> df_single_level_cols.stack().sort_index()
        cat  height    1
             weight    0
        dog  height    3
             weight    2
        Name: 0, dtype: int64

        **Multi level columns: simple case**

        >>> multicol1 = pd.MultiIndex.from_tuples([('weight', 'kg'),
        ...                                        ('weight', 'pounds')])
        >>> df_multi_level_cols1 = ks.DataFrame([[1, 2], [2, 4]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=multicol1)

        Stacking a dataframe with a multi-level column axis:

        >>> df_multi_level_cols1  # doctest: +NORMALIZE_WHITESPACE
            weight
                kg pounds
        cat      1      2
        dog      2      4
        >>> df_multi_level_cols1.stack().sort_index()
                    weight
        cat kg           1
            pounds       2
        dog kg           2
            pounds       4

        **Missing values**

        >>> multicol2 = pd.MultiIndex.from_tuples([('weight', 'kg'),
        ...                                        ('height', 'm')])
        >>> df_multi_level_cols2 = ks.DataFrame([[1.0, 2.0], [3.0, 4.0]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=multicol2)

        It is common to have missing values when stacking a dataframe
        with multi-level columns, as the stacked dataframe typically
        has more values than the original dataframe. Missing values
        are filled with NaNs:

        >>> df_multi_level_cols2
            weight height
                kg      m
        cat    1.0    2.0
        dog    3.0    4.0
        >>> df_multi_level_cols2.stack().sort_index()  # doctest: +SKIP
                height  weight
        cat kg     NaN     1.0
            m      2.0     NaN
        dog kg     NaN     3.0
            m      4.0     NaN
        """
        from databricks.koalas.series import _col

        if len(self._internal.column_labels) == 0:
            return DataFrame(self._internal.with_filter(F.lit(False)))

        column_labels = defaultdict(dict)
        index_values = set()
        should_returns_series = False
        for label in self._internal.column_labels:
            new_label = label[:-1]
            if len(new_label) == 0:
                new_label = ("0",)
                should_returns_series = True
            value = label[-1]

            scol = self._internal.spark_column_for(label)
            column_labels[new_label][value] = scol

            index_values.add(value)

        column_labels = OrderedDict(sorted(column_labels.items(), key=lambda x: x[0]))

        if self._internal.column_label_names is None:
            column_label_names = None
            index_name = None
        else:
            column_label_names = self._internal.column_label_names[:-1]
            if self._internal.column_label_names[-1] is None:
                index_name = None
            else:
                index_name = (self._internal.column_label_names[-1],)

        index_column = SPARK_INDEX_NAME_FORMAT(len(self._internal.index_map))
        index_map = list(self._internal.index_map.items()) + [(index_column, index_name)]
        data_columns = [name_like_string(label) for label in column_labels]

        structs = [
            F.struct(
                [F.lit(value).alias(index_column)]
                + [
                    (
                        column_labels[label][value]
                        if value in column_labels[label]
                        else F.lit(None)
                    ).alias(name)
                    for label, name in zip(column_labels, data_columns)
                ]
            ).alias(value)
            for value in index_values
        ]

        pairs = F.explode(F.array(structs))

        sdf = self._sdf.withColumn("pairs", pairs)
        sdf = sdf.select(
            self._internal.index_spark_columns
            + [sdf["pairs"][index_column].alias(index_column)]
            + [sdf["pairs"][name].alias(name) for name in data_columns]
        )

        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=OrderedDict(index_map),
            column_labels=list(column_labels),
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
            column_label_names=column_label_names,
        )
        kdf = DataFrame(internal)

        if should_returns_series:
            return _col(kdf)
        else:
            return kdf

    def unstack(self):
        """
        Pivot the (necessarily hierarchical) index labels.

        Returns a DataFrame having a new level of column labels whose inner-most level
        consists of the pivoted index labels.

        If the index is not a MultiIndex, the output will be a Series.

        .. note:: If the index is a MultiIndex, the output DataFrame could be very wide, and
            it could cause a serious performance degradation since Spark partitions it row based.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        DataFrame.pivot : Pivot a table based on column values.
        DataFrame.stack : Pivot a level of the column labels (inverse operation from unstack).

        Examples
        --------
        >>> df = ks.DataFrame({"A": {"0": "a", "1": "b", "2": "c"},
        ...                    "B": {"0": "1", "1": "3", "2": "5"},
        ...                    "C": {"0": "2", "1": "4", "2": "6"}},
        ...                   columns=["A", "B", "C"])
        >>> df
           A  B  C
        0  a  1  2
        1  b  3  4
        2  c  5  6

        >>> df.unstack().sort_index()
        A  0    a
           1    b
           2    c
        B  0    1
           1    3
           2    5
        C  0    2
           1    4
           2    6
        Name: 0, dtype: object

        >>> df.columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'), ('Y', 'C')])
        >>> df.unstack().sort_index()
        X  A  0    a
              1    b
              2    c
           B  0    1
              1    3
              2    5
        Y  C  0    2
              1    4
              2    6
        Name: 0, dtype: object

        For MultiIndex case:

        >>> df = ks.DataFrame({"A": ["a", "b", "c"],
        ...                    "B": [1, 3, 5],
        ...                    "C": [2, 4, 6]},
        ...                   columns=["A", "B", "C"])
        >>> df = df.set_index('A', append=True)
        >>> df  # doctest: +NORMALIZE_WHITESPACE
             B  C
          A
        0 a  1  2
        1 b  3  4
        2 c  5  6
        >>> df.unstack().sort_index()  # doctest: +NORMALIZE_WHITESPACE
             B              C
        A    a    b    c    a    b    c
        0  1.0  NaN  NaN  2.0  NaN  NaN
        1  NaN  3.0  NaN  NaN  4.0  NaN
        2  NaN  NaN  5.0  NaN  NaN  6.0
        """
        from databricks.koalas.series import _col

        if len(self._internal.index_spark_column_names) > 1:
            # The index after `reset_index()` will never be used, so use "distributed" index
            # as a dummy to avoid overhead.
            with option_context("compute.default_index_type", "distributed"):
                df = self.reset_index()
            index = df._internal.column_labels[: len(self._internal.index_spark_column_names) - 1]
            columns = df.columns[len(self._internal.index_spark_column_names) - 1]
            df = df.pivot_table(
                index=index, columns=columns, values=self._internal.column_labels, aggfunc="first"
            )
            internal = df._internal.copy(
                index_map=OrderedDict(
                    (index_column, name)
                    for index_column, name in zip(
                        df._internal.index_spark_column_names, self._internal.index_names[:-1]
                    )
                ),
                column_label_names=(
                    df._internal.column_label_names[:-1]
                    + [
                        None
                        if self._internal.index_names[-1] is None
                        else df._internal.column_label_names[-1]
                    ]
                ),
            )
            return DataFrame(internal)

        # TODO: Codes here are similar with melt. Should we deduplicate?
        column_labels = self._internal.column_labels
        ser_name = "0"
        sdf = self._sdf
        new_index_columns = [
            SPARK_INDEX_NAME_FORMAT(i) for i in range(self._internal.column_labels_level)
        ]

        new_index_map = []
        if self._internal.column_label_names is not None:
            new_index_map.extend(zip(new_index_columns, self._internal.column_label_names))
        else:
            new_index_map.extend(zip(new_index_columns, [None] * len(new_index_columns)))

        pairs = F.explode(
            F.array(
                *[
                    F.struct(
                        *(
                            [F.lit(c).alias(name) for c, name in zip(idx, new_index_columns)]
                            + [self._internal.spark_column_for(idx).alias(ser_name)]
                        )
                    )
                    for idx in column_labels
                ]
            )
        )

        columns = [
            F.col("pairs.%s" % name)
            for name in new_index_columns[: self._internal.column_labels_level]
        ] + [F.col("pairs.%s" % ser_name)]

        new_index_len = len(new_index_columns)
        existing_index_columns = []
        for i, index_name in enumerate(self._internal.index_names):
            new_index_map.append((SPARK_INDEX_NAME_FORMAT(i + new_index_len), index_name))
            existing_index_columns.append(
                self._internal.index_spark_columns[i].alias(
                    SPARK_INDEX_NAME_FORMAT(i + new_index_len)
                )
            )

        exploded_df = sdf.withColumn("pairs", pairs).select(existing_index_columns + columns)

        return _col(DataFrame(_InternalFrame(exploded_df, index_map=OrderedDict(new_index_map))))

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
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')

        applied = []
        column_labels = self._internal.column_labels
        for label in column_labels:
            scol = self._internal.spark_column_for(label)
            all_col = F.min(F.coalesce(scol.cast("boolean"), F.lit(True)))
            applied.append(F.when(all_col.isNull(), True).otherwise(all_col))

        # TODO: there is a similar logic to transpose in, for instance,
        #  DataFrame.any, Series.quantile. Maybe we should deduplicate it.
        value_column = "value"
        cols = []
        for label, applied_col in zip(column_labels, applied):
            cols.append(
                F.struct(
                    [F.lit(col).alias(SPARK_INDEX_NAME_FORMAT(i)) for i, col in enumerate(label)]
                    + [applied_col.alias(value_column)]
                )
            )

        sdf = self._sdf.select(F.array(*cols).alias("arrays")).select(F.explode(F.col("arrays")))
        sdf = sdf.selectExpr("col.*")

        index_column_name = lambda i: (
            None
            if self._internal.column_label_names is None
            else (self._internal.column_label_names[i],)
        )
        internal = self._internal.copy(
            spark_frame=sdf,
            index_map=OrderedDict(
                (SPARK_INDEX_NAME_FORMAT(i), index_column_name(i))
                for i in range(self._internal.column_labels_level)
            ),
            column_labels=None,
            data_spark_columns=[scol_for(sdf, value_column)],
            column_label_names=None,
        )

        return DataFrame(internal)[value_column].rename("all")

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
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')

        applied = []
        column_labels = self._internal.column_labels
        for label in column_labels:
            scol = self._internal.spark_column_for(label)
            all_col = F.max(F.coalesce(scol.cast("boolean"), F.lit(False)))
            applied.append(F.when(all_col.isNull(), False).otherwise(all_col))

        # TODO: there is a similar logic to transpose in, for instance,
        #  DataFrame.all, Series.quantile. Maybe we should deduplicate it.
        value_column = "value"
        cols = []
        for label, applied_col in zip(column_labels, applied):
            cols.append(
                F.struct(
                    [F.lit(col).alias(SPARK_INDEX_NAME_FORMAT(i)) for i, col in enumerate(label)]
                    + [applied_col.alias(value_column)]
                )
            )

        sdf = self._sdf.select(F.array(*cols).alias("arrays")).select(F.explode(F.col("arrays")))
        sdf = sdf.selectExpr("col.*")

        index_column_name = lambda i: (
            None
            if self._internal.column_label_names is None
            else (self._internal.column_label_names[i],)
        )
        internal = self._internal.copy(
            spark_frame=sdf,
            index_map=OrderedDict(
                (SPARK_INDEX_NAME_FORMAT(i), index_column_name(i))
                for i in range(self._internal.column_labels_level)
            ),
            column_labels=None,
            data_spark_columns=[scol_for(sdf, value_column)],
            column_label_names=None,
        )

        return DataFrame(internal)[value_column].rename("any")

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
        return self._apply_series_op(lambda kser: kser.rank(method=method, ascending=ascending))

    def filter(self, items=None, like=None, regex=None, axis=None):
        """
        Subset rows or columns of dataframe according to labels in
        the specified index.

        Note that this routine does not filter a dataframe on its
        contents. The filter is applied to the labels of the index.

        Parameters
        ----------
        items : list-like
            Keep labels from axis which are in items.
        like : string
            Keep labels from axis for which "like in label == True".
        regex : string (regular expression)
            Keep labels from axis for which re.search(regex, label) == True.
        axis : int or string axis name
            The axis to filter on.  By default this is the info axis,
            'index' for Series, 'columns' for DataFrame.

        Returns
        -------
        same type as input object

        See Also
        --------
        DataFrame.loc

        Notes
        -----
        The ``items``, ``like``, and ``regex`` parameters are
        enforced to be mutually exclusive.

        ``axis`` defaults to the info axis that is used when indexing
        with ``[]``.

        Examples
        --------
        >>> df = ks.DataFrame(np.array(([1, 2, 3], [4, 5, 6])),
        ...                   index=['mouse', 'rabbit'],
        ...                   columns=['one', 'two', 'three'])

        >>> # select columns by name
        >>> df.filter(items=['one', 'three'])
                one  three
        mouse     1      3
        rabbit    4      6

        >>> # select columns by regular expression
        >>> df.filter(regex='e$', axis=1)
                one  three
        mouse     1      3
        rabbit    4      6

        >>> # select rows containing 'bbi'
        >>> df.filter(like='bbi', axis=0)
                one  two  three
        rabbit    4    5      6
        """

        if sum(x is not None for x in (items, like, regex)) > 1:
            raise TypeError(
                "Keyword arguments `items`, `like`, or `regex` " "are mutually exclusive"
            )

        axis = validate_axis(axis, none_axis=1)

        index_scols = self._internal.index_spark_columns

        if items is not None:
            if is_list_like(items):
                items = list(items)
            else:
                raise ValueError("items should be a list-like object.")
            if axis == 0:
                # TODO: support multi-index here
                if len(index_scols) != 1:
                    raise ValueError("Single index must be specified.")
                col = None
                for item in items:
                    if col is None:
                        col = index_scols[0] == F.lit(item)
                    else:
                        col = col | (index_scols[0] == F.lit(item))
                return DataFrame(self._internal.with_filter(col))
            elif axis == 1:
                return self[items]
        elif like is not None:
            if axis == 0:
                # TODO: support multi-index here
                if len(index_scols) != 1:
                    raise ValueError("Single index must be specified.")
                return DataFrame(self._internal.with_filter(index_scols[0].contains(like)))
            elif axis == 1:
                column_labels = self._internal.column_labels
                output_labels = [label for label in column_labels if any(like in i for i in label)]
                return self[output_labels]
        elif regex is not None:
            if axis == 0:
                # TODO: support multi-index here
                if len(index_scols) != 1:
                    raise ValueError("Single index must be specified.")
                return DataFrame(self._internal.with_filter(index_scols[0].rlike(regex)))
            elif axis == 1:
                column_labels = self._internal.column_labels
                matcher = re.compile(regex)
                output_labels = [
                    label
                    for label in column_labels
                    if any(matcher.search(i) is not None for i in label)
                ]
                return self[output_labels]
        else:
            raise TypeError("Must pass either `items`, `like`, or `regex`")

    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis="index",
        inplace=False,
        level=None,
        errors="ignore",
    ):

        """
        Alter axes labels.
        Function / dict values must be unique (1-to-1). Labels not contained in a dict / Series
        will be left as-is. Extra labels listed don’t throw an error.

        Parameters
        ----------
        mapper : dict-like or function
            Dict-like or functions transformations to apply to that axis’ values.
            Use either `mapper` and `axis` to specify the axis to target with `mapper`, or `index`
            and `columns`.
        index : dict-like or function
            Alternative to specifying axis ("mapper, axis=0" is equivalent to "index=mapper").
        columns : dict-like or function
            Alternative to specifying axis ("mapper, axis=1" is equivalent to "columns=mapper").
        axis : int or str, default 'index'
            Axis to target with mapper. Can be either the axis name ('index', 'columns') or
            number (0, 1).
        inplace : bool, default False
            Whether to return a new DataFrame.
        level : int or level name, default None
            In case of a MultiIndex, only rename labels in the specified level.
        errors : {'ignore', 'raise}, default 'ignore'
            If 'raise', raise a `KeyError` when a dict-like `mapper`, `index`, or `columns`
            contains labels that are not present in the Index being transformed. If 'ignore',
            existing keys will be renamed and extra keys will be ignored.

        Returns
        -------
        DataFrame with the renamed axis labels.

        Raises:
        -------
        `KeyError`
            If any of the labels is not found in the selected axis and "errors='raise'".

        Examples
        --------
        >>> kdf1 = ks.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> kdf1.rename(columns={"A": "a", "B": "c"})  # doctest: +NORMALIZE_WHITESPACE
           a  c
        0  1  4
        1  2  5
        2  3  6

        >>> kdf1.rename(index={1: 10, 2: 20})  # doctest: +NORMALIZE_WHITESPACE
            A  B
        0   1  4
        10  2  5
        20  3  6

        >>> def str_lower(s) -> str:
        ...     return str.lower(s)
        >>> kdf1.rename(str_lower, axis='columns')  # doctest: +NORMALIZE_WHITESPACE
           a  b
        0  1  4
        1  2  5
        2  3  6

        >>> def mul10(x) -> int:
        ...     return x * 10
        >>> kdf1.rename(mul10, axis='index')  # doctest: +NORMALIZE_WHITESPACE
            A  B
        0   1  4
        10  2  5
        20  3  6

        >>> idx = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'), ('Y', 'C'), ('Y', 'D')])
        >>> kdf2 = ks.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=idx)
        >>> kdf2.rename(columns=str_lower, level=0)  # doctest: +NORMALIZE_WHITESPACE
           x     y
           A  B  C  D
        0  1  2  3  4
        1  5  6  7  8

        >>> kdf3 = ks.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=idx, columns=list('ab'))
        >>> kdf3.rename(index=str_lower)  # doctest: +NORMALIZE_WHITESPACE
             a  b
        x a  1  2
          b  3  4
        y c  5  6
          d  7  8
        """

        def gen_mapper_fn(mapper):
            if isinstance(mapper, dict):
                if len(mapper) == 0:
                    if errors == "raise":
                        raise KeyError("Index include label which is not in the `mapper`.")
                    else:
                        return DataFrame(self._internal)

                type_set = set(map(lambda x: type(x), mapper.values()))
                if len(type_set) > 1:
                    raise ValueError("Mapper dict should have the same value type.")
                spark_return_type = as_spark_type(list(type_set)[0])

                def mapper_fn(x):
                    if x in mapper:
                        return mapper[x]
                    else:
                        if errors == "raise":
                            raise KeyError("Index include value which is not in the `mapper`")
                        return x

            elif callable(mapper):
                spark_return_type = infer_return_type(mapper).tpe

                def mapper_fn(x):
                    return mapper(x)

            else:
                raise ValueError(
                    "`mapper` or `index` or `columns` should be "
                    "either dict-like or function type."
                )
            return mapper_fn, spark_return_type

        index_mapper_fn = None
        index_mapper_ret_stype = None
        columns_mapper_fn = None

        inplace = validate_bool_kwarg(inplace, "inplace")
        if mapper:
            axis = validate_axis(axis)
            if axis == 0:
                index_mapper_fn, index_mapper_ret_stype = gen_mapper_fn(mapper)
            elif axis == 1:
                columns_mapper_fn, columns_mapper_ret_stype = gen_mapper_fn(mapper)
            else:
                raise ValueError(
                    "argument axis should be either the axis name "
                    "(‘index’, ‘columns’) or number (0, 1)"
                )
        else:
            if index:
                index_mapper_fn, index_mapper_ret_stype = gen_mapper_fn(index)
            if columns:
                columns_mapper_fn, _ = gen_mapper_fn(columns)

            if not index and not columns:
                raise ValueError("Either `index` or `columns` should be provided.")

        internal = self._internal
        if index_mapper_fn:
            # rename index labels, if `level` is None, rename all index columns, otherwise only
            # rename the corresponding level index.
            # implement this by transform the underlying spark dataframe,
            # Example:
            # suppose the kdf index column in underlying spark dataframe is "index_0", "index_1",
            # if rename level 0 index labels, will do:
            #   ``kdf._sdf.withColumn("index_0", mapper_fn_udf(col("index_0"))``
            # if rename all index labels (`level` is None), then will do:
            #   ```
            #   kdf._sdf.withColumn("index_0", mapper_fn_udf(col("index_0"))
            #           .withColumn("index_1", mapper_fn_udf(col("index_1"))
            #   ```

            index_columns = internal.index_spark_column_names
            num_indices = len(index_columns)
            if level:
                if level < 0 or level >= num_indices:
                    raise ValueError("level should be an integer between [0, num_indices)")

            def gen_new_index_column(level):
                index_col_name = index_columns[level]

                index_mapper_udf = pandas_udf(
                    lambda s: s.map(index_mapper_fn), returnType=index_mapper_ret_stype
                )
                return index_mapper_udf(scol_for(internal.spark_frame, index_col_name))

            sdf = internal.spark_frame
            if level is None:
                for i in range(num_indices):
                    sdf = sdf.withColumn(index_columns[i], gen_new_index_column(i))
            else:
                sdf = sdf.withColumn(index_columns[level], gen_new_index_column(level))
            internal = internal.with_new_sdf(sdf)
        if columns_mapper_fn:
            # rename column name.
            # Will modify the `_internal._column_labels` and transform underlying spark dataframe
            # to the same column name with `_internal._column_labels`.
            if level:
                if level < 0 or level >= internal.column_labels_level:
                    raise ValueError("level should be an integer between [0, column_labels_level)")

            def gen_new_column_labels_entry(column_labels_entry):
                if isinstance(column_labels_entry, tuple):
                    if level is None:
                        # rename all level columns
                        return tuple(map(columns_mapper_fn, column_labels_entry))
                    else:
                        # only rename specified level column
                        entry_list = list(column_labels_entry)
                        entry_list[level] = columns_mapper_fn(entry_list[level])
                        return tuple(entry_list)
                else:
                    return columns_mapper_fn(column_labels_entry)

            new_column_labels = list(map(gen_new_column_labels_entry, internal.column_labels))

            if internal.column_labels_level == 1:
                new_data_columns = [col[0] for col in new_column_labels]
            else:
                new_data_columns = [str(col) for col in new_column_labels]
            new_data_scols = [
                scol_for(internal.spark_frame, old_col_name).alias(new_col_name)
                for old_col_name, new_col_name in zip(
                    internal.data_spark_column_names, new_data_columns
                )
            ]
            internal = internal.with_new_columns(new_data_scols, column_labels=new_column_labels)
        if inplace:
            self._internal = internal
            return self
        else:
            return DataFrame(internal)

    def keys(self):
        """
        Return alias for columns.

        Returns
        -------
        Index
            Columns of the DataFrame.

        Examples
        --------
        >>> df = ks.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=['cobra', 'viper', 'sidewinder'],
        ...                   columns=['max_speed', 'shield'])
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8

        >>> df.keys()
        Index(['max_speed', 'shield'], dtype='object')
        """
        return self.columns

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
        DataFrame

        Examples
        --------
        Percentage change in French franc, Deutsche Mark, and Italian lira
        from 1980-01-01 to 1980-03-01.

        >>> df = ks.DataFrame({
        ...     'FR': [4.0405, 4.0963, 4.3149],
        ...     'GR': [1.7246, 1.7482, 1.8519],
        ...     'IT': [804.74, 810.01, 860.13]},
        ...     index=['1980-01-01', '1980-02-01', '1980-03-01'])
        >>> df
                        FR      GR      IT
        1980-01-01  4.0405  1.7246  804.74
        1980-02-01  4.0963  1.7482  810.01
        1980-03-01  4.3149  1.8519  860.13

        >>> df.pct_change()
                          FR        GR        IT
        1980-01-01       NaN       NaN       NaN
        1980-02-01  0.013810  0.013684  0.006549
        1980-03-01  0.053365  0.059318  0.061876

        You can set periods to shift for forming percent change

        >>> df.pct_change(2)
                          FR        GR       IT
        1980-01-01       NaN       NaN      NaN
        1980-02-01       NaN       NaN      NaN
        1980-03-01  0.067912  0.073814  0.06883
        """
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-periods, -periods)

        def op(kser):
            prev_row = F.lag(kser.spark_column, periods).over(window)
            return ((kser.spark_column - prev_row) / prev_row).alias(
                kser._internal.data_spark_column_names[0]
            )

        return self._apply_series_op(op)

    # TODO: axis = 1
    def idxmax(self, axis=0):
        """
        Return index of first occurrence of maximum over requested axis.
        NA/null values are excluded.

        .. note:: This API collect all rows with maximum value using `to_pandas()`
            because we suppose the number of rows with max values are usually small in general.

        Parameters
        ----------
        axis : 0 or 'index'
            Can only be set to 0 at the moment.

        Returns
        -------
        Series

        See Also
        --------
        Series.idxmax

        Examples
        --------
        >>> kdf = ks.DataFrame({'a': [1, 2, 3, 2],
        ...                     'b': [4.0, 2.0, 3.0, 1.0],
        ...                     'c': [300, 200, 400, 200]})
        >>> kdf
           a    b    c
        0  1  4.0  300
        1  2  2.0  200
        2  3  3.0  400
        3  2  1.0  200

        >>> kdf.idxmax()
        a    2
        b    0
        c    2
        Name: 0, dtype: int64

        For Multi-column Index

        >>> kdf = ks.DataFrame({'a': [1, 2, 3, 2],
        ...                     'b': [4.0, 2.0, 3.0, 1.0],
        ...                     'c': [300, 200, 400, 200]})
        >>> kdf.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> kdf
           a    b    c
           x    y    z
        0  1  4.0  300
        1  2  2.0  200
        2  3  3.0  400
        3  2  1.0  200

        >>> kdf.idxmax().sort_index()
        a  x    2
        b  y    0
        c  z    2
        Name: 0, dtype: int64
        """
        max_cols = map(lambda scol: F.max(scol), self._internal.data_spark_columns)
        sdf_max = self._sdf.select(*max_cols).head()
        # `sdf_max` looks like below
        # +------+------+------+
        # |(a, x)|(b, y)|(c, z)|
        # +------+------+------+
        # |     3|   4.0|   400|
        # +------+------+------+

        conds = (
            scol == max_val for scol, max_val in zip(self._internal.data_spark_columns, sdf_max)
        )
        cond = reduce(lambda x, y: x | y, conds)

        kdf = DataFrame(self._internal.with_filter(cond))
        pdf = kdf.to_pandas()

        return ks.from_pandas(pdf.idxmax())

    # TODO: axis = 1
    def idxmin(self, axis=0):
        """
        Return index of first occurrence of minimum over requested axis.
        NA/null values are excluded.

        .. note:: This API collect all rows with minimum value using `to_pandas()`
            because we suppose the number of rows with min values are usually small in general.

        Parameters
        ----------
        axis : 0 or 'index'
            Can only be set to 0 at the moment.

        Returns
        -------
        Series

        See Also
        --------
        Series.idxmin

        Examples
        --------
        >>> kdf = ks.DataFrame({'a': [1, 2, 3, 2],
        ...                     'b': [4.0, 2.0, 3.0, 1.0],
        ...                     'c': [300, 200, 400, 200]})
        >>> kdf
           a    b    c
        0  1  4.0  300
        1  2  2.0  200
        2  3  3.0  400
        3  2  1.0  200

        >>> kdf.idxmin()
        a    0
        b    3
        c    1
        Name: 0, dtype: int64

        For Multi-column Index

        >>> kdf = ks.DataFrame({'a': [1, 2, 3, 2],
        ...                     'b': [4.0, 2.0, 3.0, 1.0],
        ...                     'c': [300, 200, 400, 200]})
        >>> kdf.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> kdf
           a    b    c
           x    y    z
        0  1  4.0  300
        1  2  2.0  200
        2  3  3.0  400
        3  2  1.0  200

        >>> kdf.idxmin().sort_index()
        a  x    0
        b  y    3
        c  z    1
        Name: 0, dtype: int64
        """
        min_cols = map(lambda scol: F.min(scol), self._internal.data_spark_columns)
        sdf_min = self._sdf.select(*min_cols).head()

        conds = (
            scol == min_val for scol, min_val in zip(self._internal.data_spark_columns, sdf_min)
        )
        cond = reduce(lambda x, y: x | y, conds)

        kdf = DataFrame(self._internal.with_filter(cond))
        pdf = kdf.to_pandas()

        return ks.from_pandas(pdf.idxmin())

    def info(self, verbose=None, buf=None, max_cols=None, null_counts=None):
        """
        Print a concise summary of a DataFrame.

        This method prints information about a DataFrame including
        the index dtype and column dtypes, non-null values and memory usage.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the full summary.
        buf : writable buffer, defaults to sys.stdout
            Where to send the output. By default, the output is printed to
            sys.stdout. Pass a writable buffer if you need to further process
            the output.
        max_cols : int, optional
            When to switch from the verbose to the truncated output. If the
            DataFrame has more than `max_cols` columns, the truncated output
            is used.
        null_counts : bool, optional
            Whether to show the non-null counts.

        Returns
        -------
        None
            This method prints a summary of a DataFrame and returns None.

        See Also
        --------
        DataFrame.describe: Generate descriptive statistics of DataFrame
            columns.

        Examples
        --------
        >>> int_values = [1, 2, 3, 4, 5]
        >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        >>> df = ks.DataFrame(
        ...     {"int_col": int_values, "text_col": text_values, "float_col": float_values},
        ...     columns=['int_col', 'text_col', 'float_col'])
        >>> df
           int_col text_col  float_col
        0        1    alpha       0.00
        1        2     beta       0.25
        2        3    gamma       0.50
        3        4    delta       0.75
        4        5  epsilon       1.00

        Prints information of all columns:

        >>> df.info(verbose=True)  # doctest: +SKIP
        <class 'databricks.koalas.frame.DataFrame'>
        Index: 5 entries, 0 to 4
        Data columns (total 3 columns):
         #   Column     Non-Null Count  Dtype
        ---  ------     --------------  -----
         0   int_col    5 non-null      int64
         1   text_col   5 non-null      object
         2   float_col  5 non-null      float64
        dtypes: float64(1), int64(1), object(1)

        Prints a summary of columns count and its dtypes but not per column
        information:

        >>> df.info(verbose=False)  # doctest: +SKIP
        <class 'databricks.koalas.frame.DataFrame'>
        Index: 5 entries, 0 to 4
        Columns: 3 entries, int_col to float_col
        dtypes: float64(1), int64(1), object(1)

        Pipe output of DataFrame.info to buffer instead of sys.stdout, get
        buffer content and writes to a text file:

        >>> import io
        >>> buffer = io.StringIO()
        >>> df.info(buf=buffer)
        >>> s = buffer.getvalue()
        >>> with open('%s/info.txt' % path, "w",
        ...           encoding="utf-8") as f:
        ...     _ = f.write(s)
        >>> with open('%s/info.txt' % path) as f:
        ...     f.readlines()  # doctest: +SKIP
        ["<class 'databricks.koalas.frame.DataFrame'>\\n",
        'Index: 5 entries, 0 to 4\\n',
        'Data columns (total 3 columns):\\n',
        ' #   Column     Non-Null Count  Dtype  \\n',
        '---  ------     --------------  -----  \\n',
        ' 0   int_col    5 non-null      int64  \\n',
        ' 1   text_col   5 non-null      object \\n',
        ' 2   float_col  5 non-null      float64\\n',
        'dtypes: float64(1), int64(1), object(1)']
        """
        # To avoid pandas' existing config affects Koalas.
        # TODO: should we have corresponding Koalas configs?
        with pd.option_context(
            "display.max_info_columns", sys.maxsize, "display.max_info_rows", sys.maxsize
        ):
            try:
                # hack to use pandas' info as is.
                self._data = self
                count_func = self.count
                self.count = lambda: count_func().to_pandas()
                return pd.DataFrame.info(
                    self,
                    verbose=verbose,
                    buf=buf,
                    max_cols=max_cols,
                    memory_usage=False,
                    null_counts=null_counts,
                )
            finally:
                del self._data
                self.count = count_func

    # TODO: fix parameter 'axis' and 'numeric_only' to work same as pandas'
    def quantile(self, q=0.5, axis=0, numeric_only=True, accuracy=10000):
        """
        Return value at the given quantile.

        .. note:: Unlike pandas', the quantile in Koalas is an approximated quantile based upon
            approximate percentile computation because computing quantile across a large dataset
            is extremely expensive.

        Parameters
        ----------
        q : float or array-like, default 0.5 (50% quantile)
            0 <= q <= 1, the quantile(s) to compute.
        axis : int, default 0 or 'index'
            Can only be set to 0 at the moment.
        numeric_only : bool, default True
            If False, the quantile of datetime and timedelta data will be computed as well.
            Can only be set to True at the moment.
        accuracy : int, optional
            Default accuracy of approximation. Larger value means better accuracy.
            The relative error can be deduced by 1.0 / accuracy.

        Returns
        -------
        Series or DataFrame
            If q is an array, a DataFrame will be returned where the
            index is q, the columns are the columns of self, and the values are the quantiles.
            If q is a float, a Series will be returned where the
            index is the columns of self and the values are the quantiles.

        Examples
        --------
        >>> kdf = ks.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 0]})
        >>> kdf
           a  b
        0  1  6
        1  2  7
        2  3  8
        3  4  9
        4  5  0

        >>> kdf.quantile(.5)
        a    3
        b    7
        Name: 0.5, dtype: int64

        >>> kdf.quantile([.25, .5, .75])
              a  b
        0.25  2  6
        0.5   3  7
        0.75  4  8
        """
        result_as_series = False
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')
        if numeric_only is not True:
            raise NotImplementedError("quantile currently doesn't supports numeric_only")
        if isinstance(q, float):
            result_as_series = True
            key = str(q)
            q = (q,)

        quantiles = q
        # First calculate the percentiles from all columns and map it to each `quantiles`
        # by creating each entry as a struct. So, it becomes an array of structs as below:
        #
        # +-----------------------------------------+
        # |                                   arrays|
        # +-----------------------------------------+
        # |[[0.25, 2, 6], [0.5, 3, 7], [0.75, 4, 8]]|
        # +-----------------------------------------+
        sdf = self._sdf
        args = ", ".join(map(str, quantiles))

        percentile_cols = []
        for column in self._internal.data_spark_column_names:
            percentile_cols.append(
                F.expr("approx_percentile(`%s`, array(%s), %s)" % (column, args, accuracy)).alias(
                    column
                )
            )
        sdf = sdf.select(percentile_cols)
        # Here, after select percntile cols, a spark_frame looks like below:
        # +---------+---------+
        # |        a|        b|
        # +---------+---------+
        # |[2, 3, 4]|[6, 7, 8]|
        # +---------+---------+

        cols_dict = OrderedDict()
        for column in self._internal.data_spark_column_names:
            cols_dict[column] = list()
            for i in range(len(quantiles)):
                cols_dict[column].append(scol_for(sdf, column).getItem(i).alias(column))

        internal_index_column = SPARK_DEFAULT_INDEX_NAME
        cols = []
        for i, col in enumerate(zip(*cols_dict.values())):
            cols.append(F.struct(F.lit("%s" % quantiles[i]).alias(internal_index_column), *col))
        sdf = sdf.select(F.array(*cols).alias("arrays"))

        # And then, explode it and manually set the index.
        # +-----------------+---+---+
        # |__index_level_0__|  a|  b|
        # +-----------------+---+---+
        # |             0.25|  2|  6|
        # |              0.5|  3|  7|
        # |             0.75|  4|  8|
        # +-----------------+---+---+
        sdf = sdf.select(F.explode(F.col("arrays"))).selectExpr("col.*")

        internal = self._internal.copy(
            spark_frame=sdf,
            data_spark_columns=[
                scol_for(sdf, col) for col in self._internal.data_spark_column_names
            ],
            index_map=OrderedDict({internal_index_column: None}),
            column_labels=self._internal.column_labels,
            column_label_names=None,
        )

        return DataFrame(internal) if not result_as_series else DataFrame(internal).T[key]

    def query(self, expr, inplace=False):
        """
        Query the columns of a DataFrame with a boolean expression.

        .. note:: Internal columns that starting with a '__' prefix are able to access, however,
            they are not supposed to be accessed.

        .. note:: This API delegates to Spark SQL so the syntax follows Spark SQL. Therefore, the
            pandas specific syntax such as `@` is not supported. If you want the pandas syntax,
            you can work around with :meth:`DataFrame.apply_batch`, but you should
            be aware that `query_func` will be executed at different nodes in a distributed manner.
            So, for example, to use `@` syntax, make sure the variable is serialized by, for
            example, putting it within the closure as below.

            >>> df = ks.DataFrame({'A': range(2000), 'B': range(2000)})
            >>> def query_func(pdf):
            ...     num = 1995
            ...     return pdf.query('A > @num')
            >>> df.apply_batch(query_func)
                     A     B
            1996  1996  1996
            1997  1997  1997
            1998  1998  1998
            1999  1999  1999

        Parameters
        ----------
        expr : str
            The query string to evaluate.

            You can refer to column names that contain spaces by surrounding
            them in backticks.

            For example, if one of your columns is called ``a a`` and you want
            to sum it with ``b``, your query should be ```a a` + b``.

        inplace : bool
            Whether the query should modify the data in place or return
            a modified copy.

        Returns
        -------
        DataFrame
            DataFrame resulting from the provided query expression.

        Examples
        --------
        >>> df = ks.DataFrame({'A': range(1, 6),
        ...                    'B': range(10, 0, -2),
        ...                    'C C': range(10, 5, -1)})
        >>> df
           A   B  C C
        0  1  10   10
        1  2   8    9
        2  3   6    8
        3  4   4    7
        4  5   2    6

        >>> df.query('A > B')
           A  B  C C
        4  5  2    6

        The previous expression is equivalent to

        >>> df[df.A > df.B]
           A  B  C C
        4  5  2    6

        For columns with spaces in their name, you can use backtick quoting.

        >>> df.query('B == `C C`')
           A   B  C C
        0  1  10   10

        The previous expression is equivalent to

        >>> df[df.B == df['C C']]
           A   B  C C
        0  1  10   10
        """
        if isinstance(self.columns, pd.MultiIndex):
            raise ValueError("Doesn't support for MultiIndex columns")
        if not isinstance(expr, str):
            raise ValueError("expr must be a string to be evaluated, {} given".format(type(expr)))
        inplace = validate_bool_kwarg(inplace, "inplace")

        data_columns = [label[0] for label in self._internal.column_labels]
        sdf = self._sdf.select(
            self._internal.index_spark_columns
            + [
                scol.alias(col)
                for scol, col in zip(self._internal.data_spark_columns, data_columns)
            ]
        ).filter(expr)
        internal = self._internal.with_new_sdf(sdf, data_columns=data_columns)

        if inplace:
            self._internal = internal
        else:
            return DataFrame(internal)

    def explain(self, extended: Optional[bool] = None, mode: Optional[str] = None):
        """
        Prints the underlying (logical and physical) Spark plans to the console for debugging
        purpose.

        Parameters
        ----------
        extended : boolean, default ``False``.
            If ``False``, prints only the physical plan.
        mode : string, default ``None``.
            The expected output format of plans.

        Examples
        --------
        >>> df = ks.DataFrame({'id': range(10)})
        >>> df.explain()  # doctest: +ELLIPSIS
        == Physical Plan ==
        ...

        >>> df.explain(True)  # doctest: +ELLIPSIS
        == Parsed Logical Plan ==
        ...
        == Analyzed Logical Plan ==
        ...
        == Optimized Logical Plan ==
        ...
        == Physical Plan ==
        ...

        >>> df.explain(mode="extended")  # doctest: +ELLIPSIS
        == Parsed Logical Plan ==
        ...
        == Analyzed Logical Plan ==
        ...
        == Optimized Logical Plan ==
        ...
        == Physical Plan ==
        ...
        """
        if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
            if mode is not None:
                if extended is not None:
                    raise Exception("extended and mode can not be specified simultaneously")
                elif mode == "simple":
                    extended = False
                elif mode == "extended":
                    extended = True
                else:
                    raise ValueError(
                        "Unknown explain mode: {}. Accepted explain modes are "
                        "'simple', 'extended'.".format(mode)
                    )
            if extended is None:
                extended = False
            self._internal.to_internal_spark_frame.explain(extended)
        else:
            self._internal.to_internal_spark_frame.explain(extended, mode)

    def take(self, indices, axis=0, **kwargs):
        """
        Return the elements in the given *positional* indices along an axis.

        This means that we are not indexing according to actual values in
        the index attribute of the object. We are indexing according to the
        actual position of the element in the object.

        Parameters
        ----------
        indices : array-like
            An array of ints indicating which positions to take.
        axis : {0 or 'index', 1 or 'columns', None}, default 0
            The axis on which to select elements. ``0`` means that we are
            selecting rows, ``1`` means that we are selecting columns.
        **kwargs
            For compatibility with :meth:`numpy.take`. Has no effect on the
            output.

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
        >>> df = ks.DataFrame([('falcon', 'bird', 389.0),
        ...                    ('parrot', 'bird', 24.0),
        ...                    ('lion', 'mammal', 80.5),
        ...                    ('monkey', 'mammal', np.nan)],
        ...                   columns=['name', 'class', 'max_speed'],
        ...                   index=[0, 2, 3, 1])
        >>> df
             name   class  max_speed
        0  falcon    bird      389.0
        2  parrot    bird       24.0
        3    lion  mammal       80.5
        1  monkey  mammal        NaN

        Take elements at positions 0 and 3 along the axis 0 (default).

        Note how the actual indices selected (0 and 1) do not correspond to
        our selected indices 0 and 3. That's because we are selecting the 0th
        and 3rd rows, not rows whose indices equal 0 and 3.

        >>> df.take([0, 3]).sort_index()
             name   class  max_speed
        0  falcon    bird      389.0
        1  monkey  mammal        NaN

        Take elements at indices 1 and 2 along the axis 1 (column selection).

        >>> df.take([1, 2], axis=1)
            class  max_speed
        0    bird      389.0
        2    bird       24.0
        3  mammal       80.5
        1  mammal        NaN

        We may take elements using negative integers for positive indices,
        starting from the end of the object, just like with Python lists.

        >>> df.take([-1, -2]).sort_index()
             name   class  max_speed
        1  monkey  mammal        NaN
        3    lion  mammal       80.5
        """
        axis = validate_axis(axis)
        if not is_list_like(indices) or isinstance(indices, (dict, set)):
            raise ValueError("`indices` must be a list-like except dict or set")
        if axis == 0:
            return self.iloc[indices, :]
        elif axis == 1:
            return self.iloc[:, indices]

    def eval(self, expr, inplace=False):
        """
        Evaluate a string describing operations on DataFrame columns.

        Operates on columns only, not specific rows or elements. This allows
        `eval` to run arbitrary code, which can make you vulnerable to code
        injection if you pass user input to this function.

        Parameters
        ----------
        expr : str
            The expression string to evaluate.
        inplace : bool, default False
            If the expression contains an assignment, whether to perform the
            operation inplace and mutate the existing DataFrame. Otherwise,
            a new DataFrame is returned.

        Returns
        -------
        The result of the evaluation.

        See Also
        --------
        DataFrame.query : Evaluates a boolean expression to query the columns
            of a frame.
        DataFrame.assign : Can evaluate an expression or function to create new
            values for a column.
        eval : Evaluate a Python expression as a string using various
            backends.

        Examples
        --------
        >>> df = ks.DataFrame({'A': range(1, 6), 'B': range(10, 0, -2)})
        >>> df
           A   B
        0  1  10
        1  2   8
        2  3   6
        3  4   4
        4  5   2
        >>> df.eval('A + B')
        0    11
        1    10
        2     9
        3     8
        4     7
        Name: 0, dtype: int64

        Assignment is allowed though by default the original DataFrame is not
        modified.

        >>> df.eval('C = A + B')
           A   B   C
        0  1  10  11
        1  2   8  10
        2  3   6   9
        3  4   4   8
        4  5   2   7
        >>> df
           A   B
        0  1  10
        1  2   8
        2  3   6
        3  4   4
        4  5   2

        Use ``inplace=True`` to modify the original DataFrame.

        >>> df.eval('C = A + B', inplace=True)
        >>> df
           A   B   C
        0  1  10  11
        1  2   8  10
        2  3   6   9
        3  4   4   8
        4  5   2   7
        """
        from databricks.koalas.series import _col

        if isinstance(self.columns, pd.MultiIndex):
            raise ValueError("`eval` is not supported for multi-index columns")
        inplace = validate_bool_kwarg(inplace, "inplace")
        should_return_series = False
        should_return_scalar = False

        # Since `eva_func` doesn't have a type hint, inferring the schema is always preformed
        # in the `apply_batch`. Hence, the variables `is_seires` and `is_scalar_` can be updated.
        def eval_func(pdf):
            nonlocal should_return_series
            nonlocal should_return_scalar
            result_inner = pdf.eval(expr, inplace=inplace)
            if inplace:
                result_inner = pdf
            if isinstance(result_inner, pd.Series):
                should_return_series = True
                result_inner = result_inner.to_frame()
            elif is_scalar(result_inner):
                should_return_scalar = True
                result_inner = pd.Series(result_inner).to_frame()
            return result_inner

        result = self.apply_batch(eval_func)
        if inplace:
            # Here, the result is always a frame because the error is thrown during schema inference
            # from pandas.
            self._internal = result._internal
        elif should_return_series:
            return _col(result)
        elif should_return_scalar:
            return _col(result)[0]
        else:
            # Returns a frame
            return result

    def _to_internal_pandas(self):
        """
        Return a pandas DataFrame directly from _internal to avoid overhead of copy.

        This method is for internal use only.
        """
        return self._internal.to_pandas_frame

    def __repr__(self):
        max_display_count = get_option("display.max_rows")
        if max_display_count is None:
            return self._to_internal_pandas().to_string()

        pdf = self.head(max_display_count + 1)._to_internal_pandas()
        pdf_length = len(pdf)
        pdf = pdf.iloc[:max_display_count]
        if pdf_length > max_display_count:
            repr_string = pdf.to_string(show_dimensions=True)
            match = REPR_PATTERN.search(repr_string)
            if match is not None:
                nrows = match.group("rows")
                ncols = match.group("columns")
                footer = "\n\n[Showing only the first {nrows} rows x {ncols} columns]".format(
                    nrows=nrows, ncols=ncols
                )
                return REPR_PATTERN.sub(footer, repr_string)
        return pdf.to_string()

    def _repr_html_(self):
        max_display_count = get_option("display.max_rows")
        # pandas 0.25.1 has a regression about HTML representation so 'bold_rows'
        # has to be set as False explicitly. See https://github.com/pandas-dev/pandas/issues/28204
        bold_rows = not (LooseVersion("0.25.1") == LooseVersion(pd.__version__))
        if max_display_count is None:
            return self._to_internal_pandas().to_html(notebook=True, bold_rows=bold_rows)

        pdf = self.head(max_display_count + 1)._to_internal_pandas()
        pdf_length = len(pdf)
        pdf = pdf.iloc[:max_display_count]
        if pdf_length > max_display_count:
            repr_html = pdf.to_html(show_dimensions=True, notebook=True, bold_rows=bold_rows)
            match = REPR_HTML_PATTERN.search(repr_html)
            if match is not None:
                nrows = match.group("rows")
                ncols = match.group("columns")
                by = chr(215)
                footer = (
                    "\n<p>Showing only the first {rows} rows "
                    "{by} {cols} columns</p>\n</div>".format(rows=nrows, by=by, cols=ncols)
                )
                return REPR_HTML_PATTERN.sub(footer, repr_html)
        return pdf.to_html(notebook=True, bold_rows=bold_rows)

    def __getitem__(self, key):
        from databricks.koalas.series import Series

        if key is None:
            raise KeyError("none key")
        if isinstance(key, (str, tuple, list)):
            return self.loc[:, key]
        elif isinstance(key, slice):
            if any(type(n) == int or None for n in [key.start, key.stop]):
                # Seems like pandas Frame always uses int as positional search when slicing
                # with ints.
                return self.iloc[key]
            return self.loc[key]
        elif isinstance(key, Series):
            return self.loc[key.astype(bool)]
        raise NotImplementedError(key)

    def __setitem__(self, key, value):
        from databricks.koalas.series import Series

        if (isinstance(value, Series) and value._kdf is not self) or (
            isinstance(value, DataFrame) and value is not self
        ):
            # Different Series or DataFrames
            key = self._index_normalized_label(key)
            value = self._index_normalized_frame(value)

            def assign_columns(kdf, this_column_labels, that_column_labels):
                assert len(key) == len(that_column_labels)
                # Note that here intentionally uses `zip_longest` that combine
                # that_columns.
                for k, this_label, that_label in zip_longest(
                    key, this_column_labels, that_column_labels
                ):
                    yield (kdf._kser_for(that_label), tuple(["that", *k]))
                    if this_label is not None and this_label[1:] != k:
                        yield (kdf._kser_for(this_label), this_label)

            kdf = align_diff_frames(assign_columns, self, value, fillna=False, how="left")
        elif isinstance(key, list):
            assert isinstance(value, DataFrame)
            # Same DataFrames.
            field_names = value.columns
            kdf = self._assign({k: value[c] for k, c in zip(key, field_names)})
        else:
            # Same Series.
            kdf = self._assign({key: value})

        self._internal = kdf._internal

    def _index_normalized_label(self, labels):
        """
        Returns a label that is normalized against the current column index level.
        For example, the key "abc" can be ("abc", "", "") if the current Frame has
        a multi-index for its column
        """
        level = self._internal.column_labels_level

        if isinstance(labels, str):
            labels = [(labels,)]
        elif isinstance(labels, tuple):
            labels = [labels]
        else:
            labels = [k if isinstance(k, tuple) else (k,) for k in labels]

        if any(len(label) > level for label in labels):
            raise KeyError(
                "Key length ({}) exceeds index depth ({})".format(
                    max(len(label) for label in labels), level
                )
            )
        return [tuple(list(label) + ([""] * (level - len(label)))) for label in labels]

    def _index_normalized_frame(self, kser_or_kdf):
        """
        Returns a frame that is normalized against the current column index level.
        For example, the name in `pd.Series([...], name="abc")` can be can be
        ("abc", "", "") if the current DataFrame has a multi-index for its column
        """

        from databricks.koalas.series import Series

        level = self._internal.column_labels_level
        if isinstance(kser_or_kdf, Series):
            kdf = kser_or_kdf.to_frame()
        else:
            assert isinstance(kser_or_kdf, DataFrame), type(kser_or_kdf)
            kdf = kser_or_kdf.copy()

        kdf.columns = pd.MultiIndex.from_tuples(
            [
                tuple([name_like_string(label)] + ([""] * (level - 1)))
                for label in kdf._internal.column_labels
            ]
        )

        return kdf

    def __getattr__(self, key: str) -> Any:
        if key.startswith("__"):
            raise AttributeError(key)
        if hasattr(_MissingPandasLikeDataFrame, key):
            property_or_func = getattr(_MissingPandasLikeDataFrame, key)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)

        try:
            return self.loc[:, key]
        except KeyError:
            raise AttributeError(
                "'%s' object has no attribute '%s'" % (self.__class__.__name__, key)
            )

    def __len__(self):
        return self._sdf.count()

    def __dir__(self):
        fields = [f for f in self._sdf.schema.fieldNames() if " " not in f]
        return super(DataFrame, self).__dir__() + fields

    def __iter__(self):
        return iter(self.columns)

    # NDArray Compat
    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: Any, **kwargs: Any):
        # TODO: is it possible to deduplicate it with '_map_series_op'?
        if all(isinstance(inp, DataFrame) for inp in inputs) and any(
            inp is not inputs[0] for inp in inputs
        ):
            # binary only
            assert len(inputs) == 2
            this = inputs[0]
            that = inputs[1]
            if this._internal.column_labels_level != that._internal.column_labels_level:
                raise ValueError("cannot join with no overlapping index names")

            # Different DataFrames
            def apply_op(kdf, this_column_labels, that_column_labels):
                for this_label, that_label in zip(this_column_labels, that_column_labels):
                    yield (
                        ufunc(kdf._kser_for(this_label), kdf._kser_for(that_label), **kwargs),
                        this_label,
                    )

            return align_diff_frames(apply_op, this, that, fillna=True, how="full")
        else:
            # DataFrame and Series
            applied = []
            this = inputs[0]
            assert all(inp is this for inp in inputs if isinstance(inp, DataFrame))

            for label in this._internal.column_labels:
                arguments = []
                for inp in inputs:
                    arguments.append(inp[label] if isinstance(inp, DataFrame) else inp)
                # both binary and unary.
                applied.append(ufunc(*arguments, **kwargs))

            internal = this._internal.with_new_columns(applied)
            return DataFrame(internal)

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

    def __init__(self, internal, storage_level=None):
        if storage_level is None:
            self._cached = internal._sdf.cache()
        elif isinstance(storage_level, StorageLevel):
            self._cached = internal._sdf.persist(storage_level)
        else:
            raise TypeError(
                "Only a valid pyspark.StorageLevel type is acceptable for the `storage_level`"
            )
        super(_CachedDataFrame, self).__init__(internal)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.unpersist()

    @property
    def storage_level(self):
        """
        Return the storage level of this cache.

        Examples
        --------
        >>> import pyspark
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df
           dogs  cats
        0   0.2   0.3
        1   0.0   0.6
        2   0.6   0.0
        3   0.2   0.1

        >>> with df.cache() as cached_df:
        ...     print(cached_df.storage_level)
        ...
        Disk Memory Deserialized 1x Replicated

        Set the StorageLevel to `MEMORY_ONLY`.

        >>> with df.persist(pyspark.StorageLevel.MEMORY_ONLY) as cached_df:
        ...     print(cached_df.storage_level)
        ...
        Memory Serialized 1x Replicated
        """
        return self._cached.storageLevel

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
