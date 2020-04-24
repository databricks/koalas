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
A wrapper for GroupedData to behave similar to pandas GroupBy.
"""

import sys
import inspect
from collections import Callable, OrderedDict, namedtuple
from functools import partial
from itertools import product
from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from pandas._libs.parsers import is_datetime64_dtype
from pandas.core.dtypes.common import is_datetime64tz_dtype

from pyspark.sql import Window, functions as F
from pyspark.sql.types import (
    FloatType,
    DoubleType,
    NumericType,
    StructField,
    StructType,
    StringType,
)
from pyspark.sql.functions import PandasUDFType, pandas_udf, Column

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.typedef import infer_return_type
from databricks.koalas.frame import DataFrame
from databricks.koalas.internal import (
    _InternalFrame,
    HIDDEN_COLUMNS,
    NATURAL_ORDER_COLUMN_NAME,
    SPARK_INDEX_NAME_FORMAT,
)
from databricks.koalas.missing.groupby import (
    _MissingPandasLikeDataFrameGroupBy,
    _MissingPandasLikeSeriesGroupBy,
)
from databricks.koalas.series import Series, _col
from databricks.koalas.config import get_option
from databricks.koalas.utils import column_labels_level, scol_for, name_like_string
from databricks.koalas.window import RollingGroupby, ExpandingGroupby

# to keep it the same as pandas
NamedAgg = namedtuple("NamedAgg", ["column", "aggfunc"])


class GroupBy(object):
    """
    :ivar _kdf: The parent dataframe that is used to perform the groupby
    :type _kdf: DataFrame
    :ivar _groupkeys: The list of keys that will be used to perform the grouping
    :type _groupkeys: List[Series]
    """

    # TODO: Series support is not implemented yet.
    # TODO: not all arguments are implemented comparing to Pandas' for now.
    def aggregate(self, func_or_funcs=None, *args, **kwargs):
        """Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func_or_funcs : dict, str or list
             a dict mapping from column name (string) to
             aggregate functions (string or list of strings).

        Returns
        -------
        Series or DataFrame

            The return can be:

            * Series : when DataFrame.agg is called with a single function
            * DataFrame : when DataFrame.agg is called with several functions

            Return Series or DataFrame.

        Notes
        -----
        `agg` is an alias for `aggregate`. Use the alias.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 1, 2, 2],
        ...                    'B': [1, 2, 3, 4],
        ...                    'C': [0.362, 0.227, 1.267, -0.562]},
        ...                   columns=['A', 'B', 'C'])

        >>> df
           A  B      C
        0  1  1  0.362
        1  1  2  0.227
        2  2  3  1.267
        3  2  4 -0.562

        Different aggregations per column

        >>> aggregated = df.groupby('A').agg({'B': 'min', 'C': 'sum'})
        >>> aggregated[['B', 'C']].sort_index()  # doctest: +NORMALIZE_WHITESPACE
           B      C
        A
        1  1  0.589
        2  3  0.705

        >>> aggregated = df.groupby('A').agg({'B': ['min', 'max']})
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             B
           min  max
        A
        1    1    2
        2    3    4

        >>> aggregated = df.groupby('A').agg('min')
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             B      C
        A
        1    1  0.227
        2    3 -0.562

        >>> aggregated = df.groupby('A').agg(['min', 'max'])
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             B           C
           min  max    min    max
        A
        1    1    2  0.227  0.362
        2    3    4 -0.562  1.267

        To control the output names with different aggregations per column, Koalas
        also supports 'named aggregation' or nested renaming in .agg. It can also be
        used when applying multiple aggregation functions to specific columns.

        >>> aggregated = df.groupby('A').agg(b_max=ks.NamedAgg(column='B', aggfunc='max'))
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             b_max
        A
        1        2
        2        4

        >>> aggregated = df.groupby('A').agg(b_max=('B', 'max'), b_min=('B', 'min'))
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             b_max   b_min
        A
        1        2       1
        2        4       3

        >>> aggregated = df.groupby('A').agg(b_max=('B', 'max'), c_min=('C', 'min'))
        >>> aggregated.sort_index()  # doctest: +NORMALIZE_WHITESPACE
             b_max   c_min
        A
        1        2   0.227
        2        4  -0.562
        """
        # I think current implementation of func and arguments in koalas for aggregate is different
        # than pandas, later once arguments are added, this could be removed.
        if func_or_funcs is None and kwargs is None:
            raise ValueError("No aggregation argument or function specified.")

        relabeling = func_or_funcs is None and _is_multi_agg_with_relabel(**kwargs)
        if relabeling:
            func_or_funcs, columns, order = _normalize_keyword_aggregation(kwargs)

        if not isinstance(func_or_funcs, (str, list)):
            if not isinstance(func_or_funcs, dict) or not all(
                isinstance(key, (str, tuple))
                and (
                    isinstance(value, str)
                    or isinstance(value, list)
                    and all(isinstance(v, str) for v in value)
                )
                for key, value in func_or_funcs.items()
            ):
                raise ValueError(
                    "aggs must be a dict mapping from column name (string or tuple) "
                    "to aggregate functions (string or list of strings)."
                )

        else:
            agg_cols = [col.name for col in self._agg_columns]
            func_or_funcs = OrderedDict([(col, func_or_funcs) for col in agg_cols])
        index_map = OrderedDict(
            (SPARK_INDEX_NAME_FORMAT(i), s._internal.column_labels[0])
            for i, s in enumerate(self._groupkeys)
        )
        kdf = DataFrame(
            GroupBy._spark_groupby(self._kdf, func_or_funcs, self._groupkeys_scols, index_map)
        )
        if not self._as_index:
            kdf = kdf.reset_index(drop=self._should_drop_index)

        if relabeling:
            kdf = kdf[order]
            kdf.columns = columns
        return kdf

    agg = aggregate

    @staticmethod
    def _spark_groupby(kdf, func, groupkeys_scols=(), index_map=None):
        assert (len(groupkeys_scols) > 0 and index_map is not None) or (
            len(groupkeys_scols) == 0 and index_map is None
        )

        sdf = kdf._sdf
        groupkey_cols = [s.alias(SPARK_INDEX_NAME_FORMAT(i)) for i, s in enumerate(groupkeys_scols)]
        multi_aggs = any(isinstance(v, list) for v in func.values())
        reordered = []
        data_columns = []
        column_labels = []
        for key, value in func.items():
            label = key if isinstance(key, tuple) else (key,)
            for aggfunc in [value] if isinstance(value, str) else value:
                name = kdf._internal.spark_column_name_for(label)
                data_col = "('{0}', '{1}')".format(name, aggfunc) if multi_aggs else name
                data_columns.append(data_col)
                column_labels.append(tuple(list(label) + [aggfunc]) if multi_aggs else label)
                if aggfunc == "nunique":
                    reordered.append(
                        F.expr("count(DISTINCT `{0}`) as `{1}`".format(name, data_col))
                    )

                # Implement "quartiles" aggregate function for ``describe``.
                elif aggfunc == "quartiles":
                    reordered.append(
                        F.expr(
                            "percentile_approx(`{0}`, array(0.25, 0.5, 0.75)) as `{1}`".format(
                                name, data_col
                            )
                        )
                    )

                else:
                    reordered.append(F.expr("{1}(`{0}`) as `{2}`".format(name, aggfunc, data_col)))
        sdf = sdf.groupby(*groupkey_cols).agg(*reordered)
        return _InternalFrame(
            spark_frame=sdf,
            column_labels=column_labels,
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
            index_map=index_map,
        )

    def count(self):
        """
        Compute count of group, excluding missing values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5],
        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])
        >>> df.groupby('A').count().sort_index()  # doctest: +NORMALIZE_WHITESPACE
            B  C
        A
        1  2  3
        2  2  2
        """
        return self._reduce_for_stat_function(F.count, only_numeric=False)

    # TODO: We should fix See Also when Series implementation is finished.
    def first(self):
        """
        Compute first of group values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.first, only_numeric=False)

    def last(self):
        """
        Compute last of group values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(
            lambda col: F.last(col, ignorenulls=True), only_numeric=False
        )

    def max(self):
        """
        Compute max of group values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.max, only_numeric=False)

    # TODO: examples should be updated.
    def mean(self):
        """
        Compute mean of groups, excluding missing values.

        Returns
        -------
        koalas.Series or koalas.DataFrame

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5],
        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])

        Groupby one column and return the mean of the remaining columns in
        each group.

        >>> df.groupby('A').mean().sort_index()  # doctest: +NORMALIZE_WHITESPACE
             B         C
        A
        1  3.0  1.333333
        2  4.0  1.500000
        """

        return self._reduce_for_stat_function(F.mean, only_numeric=True)

    def min(self):
        """
        Compute min of group values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.min, only_numeric=False)

    # TODO: sync the doc and implement `ddof`.
    def std(self):
        """
        Compute standard deviation of groups, excluding missing values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """

        return self._reduce_for_stat_function(F.stddev, only_numeric=True)

    def sum(self):
        """
        Compute sum of group values

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.sum, only_numeric=True)

    # TODO: sync the doc and implement `ddof`.
    def var(self):
        """
        Compute variance of groups, excluding missing values.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.variance, only_numeric=True)

    # TODO: skipna should be implemented.
    def all(self):
        """
        Returns True if all values in the group are truthful, else False.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ...                    'B': [True, True, True, False, False,
        ...                          False, None, True, None, False]},
        ...                   columns=['A', 'B'])
        >>> df
           A      B
        0  1   True
        1  1   True
        2  2   True
        3  2  False
        4  3  False
        5  3  False
        6  4   None
        7  4   True
        8  5   None
        9  5  False

        >>> df.groupby('A').all().sort_index()  # doctest: +NORMALIZE_WHITESPACE
               B
        A
        1   True
        2  False
        3  False
        4   True
        5  False
        """
        return self._reduce_for_stat_function(
            lambda col: F.min(F.coalesce(col.cast("boolean"), F.lit(True))), only_numeric=False
        )

    # TODO: skipna should be implemented.
    def any(self):
        """
        Returns True if any value in the group is truthful, else False.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
        ...                    'B': [True, True, True, False, False,
        ...                          False, None, True, None, False]},
        ...                   columns=['A', 'B'])
        >>> df
           A      B
        0  1   True
        1  1   True
        2  2   True
        3  2  False
        4  3  False
        5  3  False
        6  4   None
        7  4   True
        8  5   None
        9  5  False

        >>> df.groupby('A').any().sort_index()  # doctest: +NORMALIZE_WHITESPACE
               B
        A
        1   True
        2   True
        3  False
        4   True
        5  False
        """
        return self._reduce_for_stat_function(
            lambda col: F.max(F.coalesce(col.cast("boolean"), F.lit(False))), only_numeric=False
        )

    # TODO: groupby multiply columns should be implemented.
    def size(self):
        """
        Compute group sizes.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 2, 3, 3, 3],
        ...                    'B': [1, 1, 2, 3, 3, 3]},
        ...                   columns=['A', 'B'])
        >>> df
           A  B
        0  1  1
        1  2  1
        2  2  2
        3  3  3
        4  3  3
        5  3  3

        >>> df.groupby('A').size().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        A
        1  1
        2  2
        3  3
        Name: count, dtype: int64

        >>> df.groupby(['A', 'B']).size().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        A  B
        1  1    1
        2  1    1
           2    1
        3  3    3
        Name: count, dtype: int64
        """
        groupkeys = self._groupkeys
        groupkey_cols = [
            s.alias(SPARK_INDEX_NAME_FORMAT(i)) for i, s in enumerate(self._groupkeys_scols)
        ]
        sdf = self._kdf._sdf
        sdf = sdf.groupby(*groupkey_cols).count()
        if (len(self._agg_columns) > 0) and (self._have_agg_columns):
            name = self._agg_columns[0]._internal.data_spark_column_names[0]
            sdf = sdf.withColumnRenamed("count", name)
        else:
            name = "count"
        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=OrderedDict(
                (SPARK_INDEX_NAME_FORMAT(i), s._internal.column_labels[0])
                for i, s in enumerate(groupkeys)
            ),
            data_spark_columns=[scol_for(sdf, name)],
        )
        return _col(DataFrame(internal))

    def diff(self, periods=1):
        """
        First discrete difference of element.

        Calculates the difference of a DataFrame element compared with another element in the
        DataFrame group (default is the element in the same column of the previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference, accepts negative values.

        Returns
        -------
        diffed : DataFrame or Series

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

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

        >>> df.groupby(['b']).diff().sort_index()
             a    c
        0  NaN  NaN
        1  1.0  3.0
        2  NaN  NaN
        3  NaN  NaN
        4  NaN  NaN
        5  NaN  NaN

        Difference with previous column in a group.

        >>> df.groupby(['b'])['a'].diff().sort_index()
        0    NaN
        1    1.0
        2    NaN
        3    NaN
        4    NaN
        5    NaN
        Name: a, dtype: float64
        """
        return self._apply_series_op(
            lambda sg: sg._kser._diff(periods, part_cols=sg._groupkeys_scols)
        )

    def cummax(self):
        """
        Cumulative max for each group.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        Series.cummax
        DataFrame.cummax

        Examples
        --------
        >>> df = ks.DataFrame(
        ...     [[1, None, 4], [1, 0.1, 3], [1, 20.0, 2], [4, 10.0, 1]],
        ...     columns=list('ABC'))
        >>> df
           A     B  C
        0  1   NaN  4
        1  1   0.1  3
        2  1  20.0  2
        3  4  10.0  1

        By default, iterates over rows and finds the sum in each column.

        >>> df.groupby("A").cummax().sort_index()
              B  C
        0   NaN  4
        1   0.1  4
        2  20.0  4
        3  10.0  1

        It works as below in Series.

        >>> df.C.groupby(df.A).cummax().sort_index()
        0    4
        1    4
        2    4
        3    1
        Name: C, dtype: int64

        """
        return self._apply_series_op(
            lambda sg: sg._kser._cum(F.max, True, part_cols=sg._groupkeys_scols)
        )

    def cummin(self):
        """
        Cumulative min for each group.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        Series.cummin
        DataFrame.cummin

        Examples
        --------
        >>> df = ks.DataFrame(
        ...     [[1, None, 4], [1, 0.1, 3], [1, 20.0, 2], [4, 10.0, 1]],
        ...     columns=list('ABC'))
        >>> df
           A     B  C
        0  1   NaN  4
        1  1   0.1  3
        2  1  20.0  2
        3  4  10.0  1

        By default, iterates over rows and finds the sum in each column.

        >>> df.groupby("A").cummin().sort_index()
              B  C
        0   NaN  4
        1   0.1  3
        2   0.1  2
        3  10.0  1

        It works as below in Series.

        >>> df.B.groupby(df.A).cummin().sort_index()
        0     NaN
        1     0.1
        2     0.1
        3    10.0
        Name: B, dtype: float64
        """
        return self._apply_series_op(
            lambda sg: sg._kser._cum(F.min, True, part_cols=sg._groupkeys_scols)
        )

    def cumprod(self):
        """
        Cumulative product for each group.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        Series.cumprod
        DataFrame.cumprod

        Examples
        --------
        >>> df = ks.DataFrame(
        ...     [[1, None, 4], [1, 0.1, 3], [1, 20.0, 2], [4, 10.0, 1]],
        ...     columns=list('ABC'))
        >>> df
           A     B  C
        0  1   NaN  4
        1  1   0.1  3
        2  1  20.0  2
        3  4  10.0  1

        By default, iterates over rows and finds the sum in each column.

        >>> df.groupby("A").cumprod().sort_index()
              B     C
        0   NaN   4.0
        1   0.1  12.0
        2   2.0  24.0
        3  10.0   1.0

        It works as below in Series.

        >>> df.B.groupby(df.A).cumprod().sort_index()
        0     NaN
        1     0.1
        2     2.0
        3    10.0
        Name: B, dtype: float64

        """
        return self._apply_series_op(
            lambda sg: sg._kser._cumprod(True, part_cols=sg._groupkeys_scols)
        )

    def cumsum(self):
        """
        Cumulative sum for each group.

        Returns
        -------
        Series or DataFrame

        See Also
        --------
        Series.cumsum
        DataFrame.cumsum

        Examples
        --------
        >>> df = ks.DataFrame(
        ...     [[1, None, 4], [1, 0.1, 3], [1, 20.0, 2], [4, 10.0, 1]],
        ...     columns=list('ABC'))
        >>> df
           A     B  C
        0  1   NaN  4
        1  1   0.1  3
        2  1  20.0  2
        3  4  10.0  1

        By default, iterates over rows and finds the sum in each column.

        >>> df.groupby("A").cumsum().sort_index()
              B  C
        0   NaN  4
        1   0.1  7
        2  20.1  9
        3  10.0  1

        It works as below in Series.

        >>> df.B.groupby(df.A).cumsum().sort_index()
        0     NaN
        1     0.1
        2    20.1
        3    10.0
        Name: B, dtype: float64

        """
        return self._apply_series_op(
            lambda sg: sg._kser._cum(F.sum, True, part_cols=sg._groupkeys_scols)
        )

    def apply(self, func):
        """
        Apply function `func` group-wise and combine the results together.

        The function passed to `apply` must take a DataFrame as its first
        argument and return a DataFrame. `apply` will
        then take care of combining the results back together into a single
        dataframe. `apply` is therefore a highly flexible
        grouping method.

        While `apply` is a very flexible method, its downside is that
        using it can be quite a bit slower than using more specific methods
        like `agg` or `transform`. Koalas offers a wide range of method that will
        be much faster than using `apply` for their specific purposes, so try to
        use them before reaching for `apply`.

        .. note:: this API executes the function once to infer the type which is
             potentially expensive, for instance, when the dataset is created after
             aggregations or sorting.

             To avoid this, specify return type in ``func``, for instance, as below:

             >>> def pandas_div(x) -> ks.DataFrame[float, float]:
             ...    return x[['B', 'C']] / x[['B', 'C']]

             If the return type is specified, the output column names become
             `c0, c1, c2 ... cn`. These names are positionally mapped to the returned
             DataFrame in ``func``. See examples below.

        .. note:: the dataframe within ``func`` is actually a pandas dataframe. Therefore,
            any pandas APIs within this function is allowed.

        Parameters
        ----------
        func : callable
            A callable that takes a DataFrame as its first argument, and
            returns a dataframe.

        Returns
        -------
        applied : DataFrame or Series

        See Also
        --------
        aggregate : Apply aggregate function to the GroupBy object.
        DataFrame.apply : Apply a function to a DataFrame.
        Series.apply : Apply a function to a Series.

        Examples
        --------
        >>> df = ks.DataFrame({'A': 'a a b'.split(),
        ...                    'B': [1, 2, 3],
        ...                    'C': [4, 6, 5]}, columns=['A', 'B', 'C'])
        >>> g = df.groupby('A')

        Notice that ``g`` has two groups, ``a`` and ``b``.
        Calling `apply` in various ways, we can get different grouping results:

        Below the functions passed to `apply` takes a DataFrame as
        its argument and returns a DataFrame. `apply` combines the result for
        each group together into a new DataFrame:

        >>> def plus_min(x):
        ...    return x + x.min()
        >>> g.apply(plus_min).sort_index()  # doctest: +NORMALIZE_WHITESPACE
            A  B   C
        0  aa  2   8
        1  aa  3  10
        2  bb  6  10

        You can specify the type hint and prevent schema inference for better performance.

        >>> def pandas_div(x) -> ks.DataFrame[float, float]:
        ...    return x[['B', 'C']] / x[['B', 'C']]
        >>> g.apply(pandas_div).sort_index()  # doctest: +NORMALIZE_WHITESPACE
            c0   c1
        0  1.0  1.0
        1  1.0  1.0
        2  1.0  1.0

        >>> def pandas_length(x) -> int:
        ...    return len(x)
        >>> g.apply(pandas_length).sort_index()  # doctest: +NORMALIZE_WHITESPACE
        0    1
        1    2
        Name: 0, dtype: int32

        In case of Series, it works as below.

        >>> def plus_max(x) -> ks.Series[np.int]:
        ...    return x + x.max()
        >>> df.B.groupby(df.A).apply(plus_max).sort_index()
        0    6
        1    3
        2    4
        Name: B, dtype: int32

        >>> def plus_min(x):
        ...    return x + x.min()
        >>> df.B.groupby(df.A).apply(plus_min).sort_index()
        0    2
        1    3
        2    6
        Name: B, dtype: int64

        You can also return a scalar value as a aggregated value of the group:

        >>> def plus_max(x) -> np.int:
        ...    return len(x)
        >>> df.B.groupby(df.A).apply(plus_max).sort_index()
        0    1
        1    2
        Name: B, dtype: int32
        """
        if not isinstance(func, Callable):
            raise TypeError("%s object is not callable" % type(func))

        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        should_infer_schema = return_sig is None
        input_groupnames = [s.name for s in self._groupkeys]

        should_return_series = False
        is_series_groupby = isinstance(self, SeriesGroupBy)
        if is_series_groupby:
            name = self._kser.name

        if should_infer_schema:
            # Here we execute with the first 1000 to get the return type.
            limit = get_option("compute.shortcut_limit")
            pdf = self._kdf.head(limit + 1)._to_internal_pandas()
            if is_series_groupby:
                pser_or_pdf = pdf.groupby(input_groupnames)[name].apply(func)
            else:
                pser_or_pdf = pdf.groupby(input_groupnames).apply(func)
            kser_or_kdf = ks.from_pandas(pser_or_pdf)
            if len(pdf) <= limit:
                return kser_or_kdf

            kdf = kser_or_kdf
            if isinstance(kser_or_kdf, ks.Series):
                should_return_series = True
                kdf = kser_or_kdf.to_frame()

            return_schema = kdf._sdf.drop(*HIDDEN_COLUMNS).schema
        else:
            if not is_series_groupby and getattr(return_sig, "__origin__", None) == ks.Series:
                raise TypeError(
                    "Series as a return type hint at frame groupby is not supported "
                    "currently; however got [%s]. Use DataFrame type hint instead." % return_sig
                )

            return_schema = infer_return_type(func).tpe
            if not isinstance(return_schema, StructType):
                should_return_series = True
                if is_series_groupby:
                    return_schema = StructType([StructField(name, return_schema)])
                else:
                    return_schema = StructType([StructField("0", return_schema)])

        def pandas_groupby_apply(pdf):
            if is_series_groupby:
                pdf_or_ser = pdf.groupby(input_groupnames)[name].apply(func)
            else:
                pdf_or_ser = pdf.groupby(input_groupnames).apply(func)

            if not isinstance(pdf_or_ser, pd.DataFrame):
                return pd.DataFrame(pdf_or_ser)
            else:
                return pdf_or_ser

        sdf = GroupBy._spark_group_map_apply(
            self._kdf,
            pandas_groupby_apply,
            self._groupkeys_scols,
            return_schema,
            retain_index=should_infer_schema,
        )

        if should_infer_schema:
            # If schema is inferred, we can restore indexes too.
            internal = kdf._internal.with_new_sdf(sdf)
        else:
            # Otherwise, it loses index.
            internal = _InternalFrame(spark_frame=sdf, index_map=None)

        if should_return_series:
            return _col(DataFrame(internal))
        else:
            return DataFrame(internal)

    # TODO: implement 'dropna' parameter
    def filter(self, func):
        """
        Return a copy of a DataFrame excluding elements from groups that
        do not satisfy the boolean criterion specified by func.

        Parameters
        ----------
        f : function
            Function to apply to each subframe. Should return True or False.
        dropna : Drop groups that do not pass the filter. True by default;
            if False, groups that evaluate False are filled with NaNs.

        Returns
        -------
        filtered : DataFrame

        Notes
        -----
        Each subframe is endowed the attribute 'name' in case you need to know
        which group you are working on.

        Examples
        --------
        >>> df = ks.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]}, columns=['A', 'B', 'C'])
        >>> grouped = df.groupby('A')
        >>> grouped.filter(lambda x: x['B'].mean() > 3.)
             A  B    C
        1  bar  2  5.0
        3  bar  4  1.0
        5  bar  6  9.0
        """
        if not isinstance(func, Callable):
            raise TypeError("%s object is not callable" % type(func))

        data_schema = self._kdf._sdf.drop(*HIDDEN_COLUMNS).schema
        groupby_names = [s.name for s in self._groupkeys]

        def pandas_filter(pdf):
            return pdf.groupby(groupby_names).filter(func)

        sdf = GroupBy._spark_group_map_apply(
            self._kdf, pandas_filter, self._groupkeys_scols, data_schema, retain_index=True
        )
        return DataFrame(self._kdf._internal.with_new_sdf(sdf))

    @staticmethod
    def _spark_group_map_apply(kdf, func, groupkeys_scols, return_schema, retain_index):
        output_func = GroupBy._make_pandas_df_builder_func(kdf, func, return_schema, retain_index)
        grouped_map_func = pandas_udf(return_schema, PandasUDFType.GROUPED_MAP)(output_func)
        sdf = kdf._sdf.drop(*HIDDEN_COLUMNS)
        input_groupkeys = [s for s in groupkeys_scols]
        sdf = sdf.groupby(*input_groupkeys).apply(grouped_map_func)

        return sdf

    @staticmethod
    def _make_pandas_df_builder_func(kdf, func, return_schema, retain_index):
        """
        Creates a function that can be used inside the pandas UDF. This function can construct
        the same pandas DataFrame as if the Koalas DataFrame is collected to driver side.
        The index, column labels, etc. are re-constructed within the function.
        """
        index_columns = kdf._internal.index_spark_column_names
        index_names = kdf._internal.index_names
        data_columns = kdf._internal.data_spark_column_names
        column_labels = kdf._internal.column_labels

        def rename_output(pdf):
            # TODO: This logic below was borrowed from `DataFrame.to_pandas_frame` to set the index
            #   within each pdf properly. we might have to deduplicate it.
            import pandas as pd

            if len(index_columns) > 0:
                append = False
                for index_field in index_columns:
                    drop = index_field not in data_columns
                    pdf = pdf.set_index(index_field, drop=drop, append=append)
                    append = True
                pdf = pdf[data_columns]

            if column_labels_level(column_labels) > 1:
                pdf.columns = pd.MultiIndex.from_tuples(column_labels)
            else:
                pdf.columns = [None if label is None else label[0] for label in column_labels]

            if len(index_names) > 0:
                pdf.index.names = [
                    name if name is None or len(name) > 1 else name[0] for name in index_names
                ]

            pdf = func(pdf)

            if retain_index:
                # If schema should be inferred, we don't restore index. Pandas seems restoring
                # the index in some cases.
                # When Spark output type is specified, without executing it, we don't know
                # if we should restore the index or not. For instance, see the example in
                # https://github.com/databricks/koalas/issues/628.

                # TODO: deduplicate this logic with _InternalFrame.from_pandas
                new_index_columns = [
                    SPARK_INDEX_NAME_FORMAT(i) for i in range(len(pdf.index.names))
                ]
                new_data_columns = [name_like_string(col) for col in pdf.columns]

                pdf.index.names = new_index_columns
                reset_index = pdf.reset_index()
                reset_index.columns = new_index_columns + new_data_columns
                for name, col in reset_index.iteritems():
                    dt = col.dtype
                    if is_datetime64_dtype(dt) or is_datetime64tz_dtype(dt):
                        continue
                    reset_index[name] = col.replace({np.nan: None})
                pdf = reset_index

            # Just positionally map the column names to given schema's.
            pdf = pdf.rename(columns=dict(zip(pdf.columns, return_schema.fieldNames())))

            return pdf

        return rename_output

    def rank(self, method="average", ascending=True):
        """
        Provide the rank of values within each group.

        Parameters
        ----------
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            * average: average rank of group
            * min: lowest rank in group
            * max: highest rank in group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups
        ascending : boolean, default True
            False for ranks by high (1) to low (N)

        Returns
        -------
        DataFrame with ranking of values within each group

        Examples
        --------

        >>> df = ks.DataFrame({
        ...     'a': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        ...     'b': [1, 2, 2, 2, 3, 3, 3, 4, 4]}, columns=['a', 'b'])
        >>> df
           a  b
        0  1  1
        1  1  2
        2  1  2
        3  2  2
        4  2  3
        5  2  3
        6  3  3
        7  3  4
        8  3  4

        >>> df.groupby("a").rank().sort_index()
             b
        0  1.0
        1  2.5
        2  2.5
        3  1.0
        4  2.5
        5  2.5
        6  1.0
        7  2.5
        8  2.5

        >>> df.b.groupby(df.a).rank(method='max').sort_index()
        0    1.0
        1    3.0
        2    3.0
        3    1.0
        4    3.0
        5    3.0
        6    1.0
        7    3.0
        8    3.0
        Name: b, dtype: float64

        """
        return self._apply_series_op(
            lambda sg: sg._kser._rank(method, ascending, part_cols=sg._groupkeys_scols)
        )

    # TODO: add axis parameter
    def idxmax(self, skipna=True):
        """
        Return index of first occurrence of maximum over requested axis in group.
        NA/null values are excluded.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        See Also
        --------
        Series.idxmax
        DataFrame.idxmax
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 2, 2, 3],
        ...                    'b': [1, 2, 3, 4, 5],
        ...                    'c': [5, 4, 3, 2, 1]}, columns=['a', 'b', 'c'])

        >>> df.groupby(['a'])['b'].idxmax().sort_index() # doctest: +NORMALIZE_WHITESPACE
        a
        1  1
        2  3
        3  4
        Name: b, dtype: int64

        >>> df.groupby(['a']).idxmax().sort_index() # doctest: +NORMALIZE_WHITESPACE
           b  c
        a
        1  1  0
        2  3  2
        3  4  4
        """
        if len(self._kdf._internal.index_names) != 1:
            raise ValueError("idxmax only support one-level index now")
        groupkeys = self._groupkeys
        groupkey_cols = [
            s.alias(SPARK_INDEX_NAME_FORMAT(i)) for i, s in enumerate(self._groupkeys_scols)
        ]
        sdf = self._kdf._sdf
        index = self._kdf._internal.index_spark_column_names[0]

        stat_exprs = []
        for kser, c in zip(self._agg_columns, self._agg_columns_scols):
            name = kser._internal.data_spark_column_names[0]

            if skipna:
                order_column = Column(c._jc.desc_nulls_last())
            else:
                order_column = Column(c._jc.desc_nulls_first())
            window = Window.partitionBy(groupkey_cols).orderBy(
                order_column, NATURAL_ORDER_COLUMN_NAME
            )
            sdf = sdf.withColumn(
                name, F.when(F.row_number().over(window) == 1, scol_for(sdf, index)).otherwise(None)
            )
            stat_exprs.append(F.max(scol_for(sdf, name)).alias(name))
        sdf = sdf.groupby(*groupkey_cols).agg(*stat_exprs)
        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=OrderedDict(
                (SPARK_INDEX_NAME_FORMAT(i), s._internal.column_labels[0])
                for i, s in enumerate(groupkeys)
            ),
            column_labels=[kser._internal.column_labels[0] for kser in self._agg_columns],
            data_spark_columns=[
                scol_for(sdf, kser._internal.data_spark_column_names[0])
                for kser in self._agg_columns
            ],
        )
        return DataFrame(internal)

    # TODO: add axis parameter
    def idxmin(self, skipna=True):
        """
        Return index of first occurrence of minimum over requested axis in group.
        NA/null values are excluded.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        See Also
        --------
        Series.idxmin
        DataFrame.idxmin
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 2, 2, 3],
        ...                    'b': [1, 2, 3, 4, 5],
        ...                    'c': [5, 4, 3, 2, 1]}, columns=['a', 'b', 'c'])

        >>> df.groupby(['a'])['b'].idxmin().sort_index() # doctest: +NORMALIZE_WHITESPACE
        a
        1    0
        2    2
        3    4
        Name: b, dtype: int64

        >>> df.groupby(['a']).idxmin().sort_index() # doctest: +NORMALIZE_WHITESPACE
           b  c
        a
        1  0  1
        2  2  3
        3  4  4
        """
        if len(self._kdf._internal.index_names) != 1:
            raise ValueError("idxmin only support one-level index now")
        groupkeys = self._groupkeys
        groupkey_cols = [
            s.alias(SPARK_INDEX_NAME_FORMAT(i)) for i, s in enumerate(self._groupkeys_scols)
        ]
        sdf = self._kdf._sdf
        index = self._kdf._internal.index_spark_column_names[0]

        stat_exprs = []
        for kser, c in zip(self._agg_columns, self._agg_columns_scols):
            name = kser._internal.data_spark_column_names[0]

            if skipna:
                order_column = Column(c._jc.asc_nulls_last())
            else:
                order_column = Column(c._jc.asc_nulls_first())
            window = Window.partitionBy(groupkey_cols).orderBy(
                order_column, NATURAL_ORDER_COLUMN_NAME
            )
            sdf = sdf.withColumn(
                name, F.when(F.row_number().over(window) == 1, scol_for(sdf, index)).otherwise(None)
            )
            stat_exprs.append(F.max(scol_for(sdf, name)).alias(name))
        sdf = sdf.groupby(*groupkey_cols).agg(*stat_exprs)
        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=OrderedDict(
                (SPARK_INDEX_NAME_FORMAT(i), s._internal.column_labels[0])
                for i, s in enumerate(groupkeys)
            ),
            column_labels=[kser._internal.column_labels[0] for kser in self._agg_columns],
            data_spark_columns=[
                scol_for(sdf, kser._internal.data_spark_column_names[0])
                for kser in self._agg_columns
            ],
        )
        return DataFrame(internal)

    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None):
        """Fill NA/NaN values in group.

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
        ...     'A': [1, 1, 2, 2],
        ...     'B': [2, 4, None, 3],
        ...     'C': [None, None, None, 1],
        ...     'D': [0, 1, 5, 4]
        ...     },
        ...     columns=['A', 'B', 'C', 'D'])
        >>> df
           A    B    C  D
        0  1  2.0  NaN  0
        1  1  4.0  NaN  1
        2  2  NaN  NaN  5
        3  2  3.0  1.0  4

        We can also propagate non-null values forward or backward in group.

        >>> df.groupby(['A'])['B'].fillna(method='ffill').sort_index()
        0    2.0
        1    4.0
        2    NaN
        3    3.0
        Name: B, dtype: float64

        >>> df.groupby(['A']).fillna(method='bfill').sort_index()
             B    C  D
        0  2.0  NaN  0
        1  4.0  NaN  1
        2  3.0  1.0  5
        3  3.0  1.0  4
        """
        return self._fillna(value, method, axis, inplace, limit)

    def bfill(self, limit=None):
        """
        Synonym for `DataFrame.fillna()` with ``method=`bfill```.

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
        ...     'A': [1, 1, 2, 2],
        ...     'B': [2, 4, None, 3],
        ...     'C': [None, None, None, 1],
        ...     'D': [0, 1, 5, 4]
        ...     },
        ...     columns=['A', 'B', 'C', 'D'])
        >>> df
           A    B    C  D
        0  1  2.0  NaN  0
        1  1  4.0  NaN  1
        2  2  NaN  NaN  5
        3  2  3.0  1.0  4

        Propagate non-null values backward.

        >>> df.groupby(['A']).bfill().sort_index()
             B    C  D
        0  2.0  NaN  0
        1  4.0  NaN  1
        2  3.0  1.0  5
        3  3.0  1.0  4
        """
        return self._fillna(method="bfill", limit=limit)

    backfill = bfill

    def ffill(self, limit=None):
        """
        Synonym for `DataFrame.fillna()` with ``method=`ffill```.

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
        ...     'A': [1, 1, 2, 2],
        ...     'B': [2, 4, None, 3],
        ...     'C': [None, None, None, 1],
        ...     'D': [0, 1, 5, 4]
        ...     },
        ...     columns=['A', 'B', 'C', 'D'])
        >>> df
           A    B    C  D
        0  1  2.0  NaN  0
        1  1  4.0  NaN  1
        2  2  NaN  NaN  5
        3  2  3.0  1.0  4

        Propagate non-null values forward.

        >>> df.groupby(['A']).ffill().sort_index()
             B    C  D
        0  2.0  NaN  0
        1  4.0  NaN  1
        2  NaN  NaN  5
        3  3.0  1.0  4
        """
        return self._fillna(method="ffill", limit=limit)

    pad = ffill

    def head(self, n=5):
        """
        Return first n rows of each group.

        Returns
        -------
        DataFrame or Series

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        ...                    'b': [2, 3, 1, 4, 6, 9, 8, 10, 7, 5],
        ...                    'c': [3, 5, 2, 5, 1, 2, 6, 4, 3, 6]},
        ...                   columns=['a', 'b', 'c'],
        ...                   index=[7, 2, 4, 1, 3, 4, 9, 10, 5, 6])
        >>> df
            a   b  c
        7   1   2  3
        2   1   3  5
        4   1   1  2
        1   1   4  5
        3   2   6  1
        4   2   9  2
        9   2   8  6
        10  3  10  4
        5   3   7  3
        6   3   5  6

        >>> df.groupby('a').head(2).sort_index()
            a   b  c
        2   1   3  5
        3   2   6  1
        4   2   9  2
        5   3   7  3
        7   1   2  3
        10  3  10  4

        >>> df.groupby('a')['b'].head(2).sort_index()
        2      3
        3      6
        4      9
        5      7
        7      2
        10    10
        Name: b, dtype: int64
        """
        tmp_col = "__row_number__"
        sdf = self._kdf._sdf
        window = Window.partitionBy(self._groupkeys_scols).orderBy(NATURAL_ORDER_COLUMN_NAME)
        sdf = (
            sdf.withColumn(tmp_col, F.row_number().over(window))
            .filter(F.col(tmp_col) <= n)
            .drop(tmp_col)
        )

        internal = self._kdf._internal.with_new_sdf(sdf)
        return DataFrame(internal)

    def shift(self, periods=1, fill_value=None):
        """
        Shift each group by periods observations.

        Parameters
        ----------
        periods : integer, default 1
            number of periods to shift
        fill_value : optional

        Returns
        -------
        Series or DataFrame
            Object shifted within each group.

        Examples
        --------

        >>> df = ks.DataFrame({
        ...     'a': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        ...     'b': [1, 2, 2, 2, 3, 3, 3, 4, 4]}, columns=['a', 'b'])
        >>> df
           a  b
        0  1  1
        1  1  2
        2  1  2
        3  2  2
        4  2  3
        5  2  3
        6  3  3
        7  3  4
        8  3  4

        >>> df.groupby('a').shift().sort_index()  # doctest: +SKIP
             b
        0  NaN
        1  1.0
        2  2.0
        3  NaN
        4  2.0
        5  3.0
        6  NaN
        7  3.0
        8  4.0

        >>> df.groupby('a').shift(periods=-1, fill_value=0).sort_index()  # doctest: +SKIP
           b
        0  2
        1  2
        2  0
        3  3
        4  3
        5  0
        6  4
        7  4
        8  0
        """
        return self._apply_series_op(
            lambda sg: sg._kser._shift(periods, fill_value, part_cols=sg._groupkeys_scols)
        )

    def transform(self, func):
        """
        Apply function column-by-column to the GroupBy object.

        The function passed to `transform` must take a Series as its first
        argument and return a Series. The given function is executed for
        each series in each grouped data.

        While `transform` is a very flexible method, its downside is that
        using it can be quite a bit slower than using more specific methods
        like `agg` or `transform`. Koalas offers a wide range of method that will
        be much faster than using `transform` for their specific purposes, so try to
        use them before reaching for `transform`.

        .. note:: this API executes the function once to infer the type which is
             potentially expensive, for instance, when the dataset is created after
             aggregations or sorting.

             To avoid this, specify return type in ``func``, for instance, as below:

             >>> def convert_to_string(x) -> ks.Series[str]:
             ...    return x.apply("a string {}".format)

        .. note:: the series within ``func`` is actually a pandas series. Therefore,
            any pandas APIs within this function is allowed.


        Parameters
        ----------
        func : callable
            A callable that takes a Series as its first argument, and
            returns a Series.

        Returns
        -------
        applied : DataFrame

        See Also
        --------
        aggregate : Apply aggregate function to the GroupBy object.
        Series.apply : Apply a function to a Series.

        Examples
        --------

        >>> df = ks.DataFrame({'A': [0, 0, 1],
        ...                    'B': [1, 2, 3],
        ...                    'C': [4, 6, 5]}, columns=['A', 'B', 'C'])

        >>> g = df.groupby('A')

        Notice that ``g`` has two groups, ``0`` and ``1``.
        Calling `transform` in various ways, we can get different grouping results:
        Below the functions passed to `transform` takes a Series as
        its argument and returns a Series. `transform` applies the function on each series
        in each grouped data, and combine them into a new DataFrame:

        >>> def convert_to_string(x) -> ks.Series[str]:
        ...    return x.apply("a string {}".format)
        >>> g.transform(convert_to_string)  # doctest: +NORMALIZE_WHITESPACE
                    B           C
        0  a string 1  a string 4
        1  a string 2  a string 6
        2  a string 3  a string 5

        >>> def plus_max(x) -> ks.Series[np.int]:
        ...    return x + x.max()
        >>> g.transform(plus_max)  # doctest: +NORMALIZE_WHITESPACE
           B   C
        0  3  10
        1  4  12
        2  6  10

        You can omit the type hint and let Koalas infer its type.

        >>> def plus_min(x):
        ...    return x + x.min()
        >>> g.transform(plus_min)  # doctest: +NORMALIZE_WHITESPACE
           B   C
        0  2   8
        1  3  10
        2  6  10

        In case of Series, it works as below.

        >>> df.B.groupby(df.A).transform(plus_max)
        0    3
        1    4
        2    6
        Name: B, dtype: int32

        >>> df.B.groupby(df.A).transform(plus_min)
        0    2
        1    3
        2    6
        Name: B, dtype: int64
        """
        if not isinstance(func, Callable):
            raise TypeError("%s object is not callable" % type(func))

        spec = inspect.getfullargspec(func)
        return_sig = spec.annotations.get("return", None)
        input_groupnames = [s.name for s in self._groupkeys]

        def pandas_transform(pdf):
            # pandas GroupBy.transform drops grouping columns.
            pdf = pdf.drop(columns=input_groupnames)
            return pdf.transform(func)

        should_infer_schema = return_sig is None

        if should_infer_schema:
            # Here we execute with the first 1000 to get the return type.
            # If the records were less than 1000, it uses pandas API directly for a shortcut.
            limit = get_option("compute.shortcut_limit")
            pdf = self._kdf.head(limit + 1)._to_internal_pandas()
            pdf = pdf.groupby(input_groupnames).transform(func)
            kdf = DataFrame(pdf)
            return_schema = kdf._sdf.drop(*HIDDEN_COLUMNS).schema
            if len(pdf) <= limit:
                return kdf

            sdf = GroupBy._spark_group_map_apply(
                self._kdf, pandas_transform, self._groupkeys_scols, return_schema, retain_index=True
            )
            # If schema is inferred, we can restore indexes too.
            internal = kdf._internal.with_new_sdf(sdf)
        else:
            return_type = infer_return_type(func).tpe
            data_columns = self._kdf._internal.data_spark_column_names
            return_schema = StructType(
                [StructField(c, return_type) for c in data_columns if c not in input_groupnames]
            )

            sdf = GroupBy._spark_group_map_apply(
                self._kdf,
                pandas_transform,
                self._groupkeys_scols,
                return_schema,
                retain_index=False,
            )
            # Otherwise, it loses index.
            internal = _InternalFrame(spark_frame=sdf, index_map=None)

        return DataFrame(internal)

    def nunique(self, dropna=True):
        """
        Return DataFrame with number of distinct observations per group for each column.

        Parameters
        ----------
        dropna : boolean, default True
            Don’t include NaN in the counts.

        Returns
        -------
        nunique : DataFrame

        Examples
        --------

        >>> df = ks.DataFrame({'id': ['spam', 'egg', 'egg', 'spam',
        ...                           'ham', 'ham'],
        ...                    'value1': [1, 5, 5, 2, 5, 5],
        ...                    'value2': list('abbaxy')}, columns=['id', 'value1', 'value2'])
        >>> df
             id  value1 value2
        0  spam       1      a
        1   egg       5      b
        2   egg       5      b
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y

        >>> df.groupby('id').nunique().sort_index() # doctest: +NORMALIZE_WHITESPACE
              id  value1  value2
        id
        egg    1       1       1
        ham    1       1       2
        spam   1       2       1

        >>> df.groupby('id')['value1'].nunique().sort_index() # doctest: +NORMALIZE_WHITESPACE
        id
        egg     1
        ham     1
        spam    2
        Name: value1, dtype: int64
        """
        if isinstance(self, DataFrameGroupBy):
            self._agg_columns = self._groupkeys + self._agg_columns
            self._agg_columns_scols = self._groupkeys_scols + self._agg_columns_scols
        if dropna:
            stat_function = lambda col: F.countDistinct(col)
        else:
            stat_function = lambda col: (
                F.countDistinct(col)
                + F.when(F.count(F.when(col.isNull(), 1).otherwise(None)) >= 1, 1).otherwise(0)
            )
        return self._reduce_for_stat_function(stat_function, only_numeric=False)

    def rolling(self, window, min_periods=None):
        """
        Return an rolling grouper, providing rolling
        functionality per group.

        .. note:: 'min_periods' in Koalas works as a fixed window size unlike pandas.
        Unlike pandas, NA is also counted as the period. This might be changed
        in the near future.

        Parameters
        ----------
        window : int, or offset
            Size of the moving window.
            This is the number of observations used for calculating the statistic.
            Each window will be a fixed size.

        min_periods : int, default 1
            Minimum number of observations in window required to have a value
            (otherwise result is NA).

        See Also
        --------
        Series.groupby
        DataFrame.groupby
        """
        return RollingGroupby(self, self._groupkeys, window, min_periods=min_periods)

    def expanding(self, min_periods=1):
        """
        Return an expanding grouper, providing expanding
        functionality per group.

        .. note:: 'min_periods' in Koalas works as a fixed window size unlike pandas.
        Unlike pandas, NA is also counted as the period. This might be changed
        in the near future.

        Parameters
        ----------
        min_periods : int, default 1
            Minimum number of observations in window required to have a value
            (otherwise result is NA).

        See Also
        --------
        Series.groupby
        DataFrame.groupby
        """
        return ExpandingGroupby(self, self._groupkeys, min_periods=min_periods)

    def _reduce_for_stat_function(self, sfun, only_numeric):
        groupkey_cols = [
            s.alias(SPARK_INDEX_NAME_FORMAT(i)) for i, s in enumerate(self._groupkeys_scols)
        ]

        sdf = self._kdf._sdf

        data_columns = []
        column_labels = []
        if len(self._agg_columns) > 0:
            stat_exprs = []
            for kser, c in zip(self._agg_columns, self._agg_columns_scols):
                spark_type = kser.spark_type
                name = kser._internal.data_spark_column_names[0]
                label = kser._internal.column_labels[0]
                # TODO: we should have a function that takes dataframes and converts the numeric
                # types. Converting the NaNs is used in a few places, it should be in utils.
                # Special handle floating point types because Spark's count treats nan as a valid
                # value, whereas Pandas count doesn't include nan.
                if isinstance(spark_type, DoubleType) or isinstance(spark_type, FloatType):
                    stat_exprs.append(sfun(F.nanvl(c, F.lit(None))).alias(name))
                    data_columns.append(name)
                    column_labels.append(label)
                elif isinstance(spark_type, NumericType) or not only_numeric:
                    stat_exprs.append(sfun(c).alias(name))
                    data_columns.append(name)
                    column_labels.append(label)
            sdf = sdf.groupby(*groupkey_cols).agg(*stat_exprs)
        else:
            sdf = sdf.select(*groupkey_cols).distinct()

        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=OrderedDict(
                (SPARK_INDEX_NAME_FORMAT(i), s._internal.column_labels[0])
                for i, s in enumerate(self._groupkeys)
            ),
            column_labels=column_labels,
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
            column_label_names=self._kdf._internal.column_label_names,
        )
        kdf = DataFrame(internal)
        if not self._as_index:
            kdf = kdf.reset_index(drop=self._should_drop_index)
        return kdf


class DataFrameGroupBy(GroupBy):
    def __init__(
        self,
        kdf: DataFrame,
        by: List[Series],
        as_index: bool = True,
        should_drop_index: bool = False,
        agg_columns: List[Union[str, Tuple[str, ...]]] = None,
    ):
        self._kdf = kdf
        self._groupkeys = by
        self._groupkeys_scols = [s.spark_column for s in self._groupkeys]
        self._as_index = as_index
        self._should_drop_index = should_drop_index
        self._have_agg_columns = True

        if agg_columns is None:
            agg_columns = [
                label
                for label in self._kdf._internal.column_labels
                if all(not self._kdf[label]._equals(key) for key in self._groupkeys)
            ]
            self._have_agg_columns = False
        self._agg_columns = [kdf[label] for label in agg_columns]
        self._agg_columns_scols = [s.spark_column for s in self._agg_columns]

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeDataFrameGroupBy, item):
            property_or_func = getattr(_MissingPandasLikeDataFrameGroupBy, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        return self.__getitem__(item)

    def __getitem__(self, item):
        if isinstance(item, str) and self._as_index:
            return SeriesGroupBy(self._kdf[item], self._groupkeys)
        else:
            if isinstance(item, str):
                item = [item]
            item = [i if isinstance(i, tuple) else (i,) for i in item]
            if not self._as_index:
                groupkey_names = set(key.name for key in self._groupkeys)
                for i in item:
                    name = str(i) if len(i) > 1 else i[0]
                    if name in groupkey_names:
                        raise ValueError("cannot insert {}, already exists".format(name))
            return DataFrameGroupBy(
                self._kdf,
                self._groupkeys,
                as_index=self._as_index,
                agg_columns=item,
                should_drop_index=self._should_drop_index,
            )

    def _apply_series_op(self, op):
        applied = []
        for column in self._agg_columns:
            applied.append(op(column.groupby(self._groupkeys)))
        internal = self._kdf._internal.with_new_columns(applied, keep_order=False)
        return DataFrame(internal)

    def _fillna(self, *args, **kwargs):
        applied = []
        kdf = self._kdf

        for label in kdf._internal.column_labels:
            if all(not self._kdf[label]._equals(key) for key in self._groupkeys):
                applied.append(kdf[label].groupby(self._groupkeys)._fillna(*args, **kwargs))

        internal = kdf._internal.with_new_columns(applied, keep_order=False)
        return DataFrame(internal)

    # TODO: Implement 'percentiles', 'include', and 'exclude' arguments.
    # TODO: Add ``DataFrame.select_dtypes`` to See Also when 'include'
    #   and 'exclude' arguments are implemented.
    def describe(self):
        """
        Generate descriptive statistics that summarize the central tendency,
        dispersion and shape of a dataset's distribution, excluding
        ``NaN`` values.

        Analyzes both numeric and object series, as well
        as ``DataFrame`` column sets of mixed data types. The output
        will vary depending on what is provided. Refer to the notes
        below for more detail.

        .. note:: Unlike pandas, the percentiles in Koalas are based upon
            approximate percentile computation because computing percentiles
            across a large dataset is extremely expensive.

        Returns
        -------
        DataFrame
            Summary statistics of the DataFrame provided.

        See Also
        --------
        DataFrame.count
        DataFrame.max
        DataFrame.min
        DataFrame.mean
        DataFrame.std

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
        >>> df
           a  b  c
        0  1  4  7
        1  1  5  8
        2  3  6  9

        Describing a ``DataFrame``. By default only numeric fields
        are returned.

        >>> described = df.groupby('a').describe()
        >>> described.sort_index()  # doctest: +NORMALIZE_WHITESPACE
              b                                        c
          count mean       std min 25% 50% 75% max count mean       std min 25% 50% 75% max
        a
        1   2.0  4.5  0.707107 4.0 4.0 4.0 5.0 5.0   2.0  7.5  0.707107 7.0 7.0 7.0 8.0 8.0
        3   1.0  6.0       NaN 6.0 6.0 6.0 6.0 6.0   1.0  9.0       NaN 9.0 9.0 9.0 9.0 9.0

        """
        for col in self._agg_columns:
            if isinstance(col.spark_type, StringType):
                raise NotImplementedError(
                    "DataFrameGroupBy.describe() doesn't support for string type for now"
                )

        kdf = self.agg(["count", "mean", "std", "min", "quartiles", "max"]).reset_index()
        sdf = kdf._sdf
        agg_cols = [col.name for col in self._agg_columns]
        formatted_percentiles = ["25%", "50%", "75%"]

        # Split "quartiles" columns into first, second, and third quartiles.
        for col in agg_cols:
            quartiles_col = str((col, "quartiles"))
            for i, percentile in enumerate(formatted_percentiles):
                sdf = sdf.withColumn(str((col, percentile)), F.col(quartiles_col)[i])
            sdf = sdf.drop(quartiles_col)

        # Reorder columns lexicographically by agg column followed by stats.
        stats = ["count", "mean", "std", "min"] + formatted_percentiles + ["max"]
        column_labels = list(product(agg_cols, stats))
        data_columns = map(str, column_labels)

        # Reindex the DataFrame to reflect initial grouping and agg columns.
        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=OrderedDict(
                (s._internal.data_spark_column_names[0], s._internal.column_labels[0])
                for s in self._groupkeys
            ),
            column_labels=column_labels,
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
        )

        # Cast columns to ``"float64"`` to match `pandas.DataFrame.groupby`.
        return DataFrame(internal).astype("float64")


class SeriesGroupBy(GroupBy):
    def __init__(self, kser: Series, by: List[Series], as_index: bool = True):
        self._kser = kser
        self._groupkeys = by
        # TODO: this class resolves the groupkeys and agg_columns always by columns names
        #   e.g., F.col("..."). This is because of the limitation of `SeriesGroupBy`
        #   implementation, which reuses the implementation in `GroupBy`.
        #   `SeriesGroupBy` creates another DataFrame and
        #   internal IDs of the columns become different. Maybe we should refactor the whole
        #   class in the future.
        self._groupkeys_scols = [
            F.col(s._internal.data_spark_column_names[0]) for s in self._groupkeys
        ]
        self._agg_columns_scols = [
            F.col(s._internal.data_spark_column_names[0]) for s in self._agg_columns
        ]

        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")
        self._as_index = True
        self._have_agg_columns = True

        # Not used currently. It's a placeholder to match with DataFrameGroupBy.
        self._should_drop_index = False

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeSeriesGroupBy, item):
            property_or_func = getattr(_MissingPandasLikeSeriesGroupBy, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)

    def _apply_series_op(self, op):
        return op(self)

    def _fillna(self, *args, **kwargs):
        return Series._fillna(self._kser, *args, **kwargs, part_cols=self._groupkeys_scols)

    @property
    def _kdf(self) -> DataFrame:
        # TODO: Currently cannot handle the case when the values in current series
        #  and groupkeys series are different but only their names are same.
        series = [self._kser] + [s for s in self._groupkeys if not s._equals(self._kser)]
        return DataFrame(self._kser._kdf._internal.with_new_columns(series))

    @property
    def _agg_columns(self):
        return [self._kser]

    def _reduce_for_stat_function(self, sfun, only_numeric):
        return _col(super(SeriesGroupBy, self)._reduce_for_stat_function(sfun, only_numeric))

    def agg(self, *args, **kwargs):
        return _MissingPandasLikeSeriesGroupBy.agg(self, *args, **kwargs)

    def aggregate(self, *args, **kwargs):
        return _MissingPandasLikeSeriesGroupBy.aggregate(self, *args, **kwargs)

    def transform(self, func):
        return _col(super(SeriesGroupBy, self).transform(func))

    transform.__doc__ = GroupBy.transform.__doc__

    def filter(self, *args, **kwargs):
        return _MissingPandasLikeSeriesGroupBy.filter(self, *args, **kwargs)

    def idxmin(self, skipna=True):
        return _col(super(SeriesGroupBy, self).idxmin(skipna))

    idxmin.__doc__ = GroupBy.idxmin.__doc__

    def idxmax(self, skipna=True):
        return _col(super(SeriesGroupBy, self).idxmax(skipna))

    idxmax.__doc__ = GroupBy.idxmax.__doc__

    def head(self, n=5):
        return _col(super(SeriesGroupBy, self).head(n))

    # TODO: add keep parameter
    def nsmallest(self, n=5):
        """
        Return the first n rows ordered by columns in ascending order in group.

        Return the first n rows with the smallest values in columns, in ascending order.
        The columns that are not specified are returned as well, but not used for ordering.

        Parameters
        ----------
        n : int
            Number of items to retrieve.

        See Also
        --------
        databricks.koalas.Series.nsmallest
        databricks.koalas.DataFrame.nsmallest

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        ...                    'b': [1, 2, 2, 2, 3, 3, 3, 4, 4]}, columns=['a', 'b'])

        >>> df.groupby(['a'])['b'].nsmallest(1).sort_index()  # doctest: +NORMALIZE_WHITESPACE
        a
        1  0    1
        2  3    2
        3  6    3
        Name: b, dtype: int64
        """
        if len(self._kdf._internal.index_names) > 1:
            raise ValueError("nsmallest do not support multi-index now")
        sdf = self._kdf._sdf
        name = self._agg_columns[0]._internal.data_spark_column_names[0]
        window = Window.partitionBy(self._groupkeys_scols).orderBy(
            scol_for(sdf, name), NATURAL_ORDER_COLUMN_NAME
        )
        sdf = sdf.withColumn("rank", F.row_number().over(window)).filter(F.col("rank") <= n)
        internal = _InternalFrame(
            spark_frame=sdf.drop(NATURAL_ORDER_COLUMN_NAME),
            index_map=OrderedDict(
                [
                    (s._internal.data_spark_column_names[0], s._internal.column_labels[0])
                    for s in self._groupkeys
                ]
                + list(self._kdf._internal.index_map.items())
            ),
            data_spark_columns=[scol_for(sdf, name)],
        )
        return _col(DataFrame(internal))

    # TODO: add keep parameter
    def nlargest(self, n=5):
        """
        Return the first n rows ordered by columns in descending order in group.

        Return the first n rows with the smallest values in columns, in descending order.
        The columns that are not specified are returned as well, but not used for ordering.

        Parameters
        ----------
        n : int
            Number of items to retrieve.

        See Also
        --------
        databricks.koalas.Series.nlargest
        databricks.koalas.DataFrame.nlargest

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        ...                    'b': [1, 2, 2, 2, 3, 3, 3, 4, 4]}, columns=['a', 'b'])

        >>> df.groupby(['a'])['b'].nlargest(1).sort_index()  # doctest: +NORMALIZE_WHITESPACE
        a
        1  1    2
        2  4    3
        3  7    4
        Name: b, dtype: int64
        """
        if len(self._kdf._internal.index_names) > 1:
            raise ValueError("nlargest do not support multi-index now")
        sdf = self._kdf._sdf
        name = self._agg_columns[0]._internal.data_spark_column_names[0]
        window = Window.partitionBy(self._groupkeys_scols).orderBy(
            F.col(name).desc(), NATURAL_ORDER_COLUMN_NAME
        )
        sdf = sdf.withColumn("rank", F.row_number().over(window)).filter(F.col("rank") <= n)
        internal = _InternalFrame(
            spark_frame=sdf.drop(NATURAL_ORDER_COLUMN_NAME),
            index_map=OrderedDict(
                [
                    (s._internal.data_spark_column_names[0], s._internal.column_labels[0])
                    for s in self._groupkeys
                ]
                + list(self._kdf._internal.index_map.items())
            ),
            data_spark_columns=[scol_for(sdf, name)],
        )
        return _col(DataFrame(internal))

    # TODO: add bins, normalize parameter
    def value_counts(self, sort=None, ascending=None, dropna=True):
        """
        Compute group sizes.

        Parameters
        ----------
        sort : boolean, default None
            Sort by frequencies.
        ascending : boolean, default False
            Sort in ascending order.
        dropna : boolean, default True
            Don't include counts of NaN.

        See Also
        --------
        databricks.koalas.Series.groupby
        databricks.koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 2, 2, 3, 3, 3],
        ...                    'B': [1, 1, 2, 3, 3, 3]},
        ...                   columns=['A', 'B'])
        >>> df
           A  B
        0  1  1
        1  2  1
        2  2  2
        3  3  3
        4  3  3
        5  3  3

        >>> df.groupby('A')['B'].value_counts().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        A  B
        1  1    1
        2  1    1
           2    1
        3  3    3
        Name: B, dtype: int64
        """
        groupkeys = self._groupkeys + self._agg_columns
        groupkey_cols = [
            s.alias(SPARK_INDEX_NAME_FORMAT(i))
            for i, s in enumerate(self._groupkeys_scols + self._agg_columns_scols)
        ]
        sdf = self._kdf._sdf
        agg_column = self._agg_columns[0]._internal.data_spark_column_names[0]
        sdf = sdf.groupby(*groupkey_cols).count().withColumnRenamed("count", agg_column)

        if sort:
            if ascending:
                sdf = sdf.orderBy(F.col(agg_column).asc())
            else:
                sdf = sdf.orderBy(F.col(agg_column).desc())

        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=OrderedDict(
                (SPARK_INDEX_NAME_FORMAT(i), s._internal.column_labels[0])
                for i, s in enumerate(groupkeys)
            ),
            data_spark_columns=[scol_for(sdf, agg_column)],
        )
        return _col(DataFrame(internal))

    def unique(self):
        """
        Return unique values in group.

        Uniques are returned in order of unknown. It does NOT sort.

        See Also
        --------
        databricks.koalas.Series.unique
        databricks.koalas.Index.unique

        Examples
        --------
        >>> df = ks.DataFrame({'a': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        ...                    'b': [1, 2, 2, 2, 3, 3, 3, 4, 4]}, columns=['a', 'b'])

        >>> df.groupby(['a'])['b'].unique().sort_index()  # doctest: +SKIP
        a
        1    [1, 2]
        2    [2, 3]
        3    [3, 4]
        Name: b, dtype: object
        """
        return self._reduce_for_stat_function(F.collect_set, only_numeric=False)


def _is_multi_agg_with_relabel(**kwargs):
    """
    Check whether the kwargs pass to .agg look like multi-agg with relabling.

    Parameters
    ----------
    **kwargs : dict

    Returns
    -------
    bool

    Examples
    --------
    >>> _is_multi_agg_with_relabel(a='max')
    False
    >>> _is_multi_agg_with_relabel(a_max=('a', 'max'),
    ...                            a_min=('a', 'min'))
    True
    >>> _is_multi_agg_with_relabel()
    False
    """
    if not kwargs:
        return False
    return all(isinstance(v, tuple) and len(v) == 2 for v in kwargs.values())


def _normalize_keyword_aggregation(kwargs):
    """
    Normalize user-provided kwargs.

    Transforms from the new ``Dict[str, NamedAgg]`` style kwargs
    to the old OrderedDict[str, List[scalar]]].

    Parameters
    ----------
    kwargs : dict

    Returns
    -------
    aggspec : dict
        The transformed kwargs.
    columns : List[str]
        The user-provided keys.
    order : List[Tuple[str, str]]
        Pairs of the input and output column names.

    Examples
    --------
    >>> _normalize_keyword_aggregation({'output': ('input', 'sum')})
    (OrderedDict([('input', ['sum'])]), ('output',), [('input', 'sum')])
    """
    # this is due to python version issue, not sure the impact on koalas
    PY36 = sys.version_info >= (3, 6)
    if not PY36:
        kwargs = OrderedDict(sorted(kwargs.items()))

    # TODO(Py35): When we drop python 3.5, change this to defaultdict(list)
    aggspec = OrderedDict()
    order = []
    columns, pairs = list(zip(*kwargs.items()))

    for column, aggfunc in pairs:
        if column in aggspec:
            aggspec[column].append(aggfunc)
        else:
            aggspec[column] = [aggfunc]

        order.append((column, aggfunc))
    return aggspec, columns, order
