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

from functools import partial
from typing import Any, List, Union
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, DoubleType, NumericType

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.dask.compatibility import string_types
from databricks.koalas.generic import _Frame
from databricks.koalas.frame import DataFrame
from databricks.koalas.metadata import Metadata
from databricks.koalas.missing.groupby import _MissingPandasLikeDataFrameGroupBy, \
    _MissingPandasLikeSeriesGroupBy
from databricks.koalas.series import Series, _col


ColumnLike = Union[str, Series]


class GroupBy(object):

    def __new__(cls, obj: _Frame, *args, **kwargs):
        if isinstance(obj, DataFrame):
            return super(GroupBy, cls).__new__(DataFrameGroupBy)
        elif isinstance(obj, Series):
            return super(GroupBy, cls).__new__(SeriesGroupBy)
        else:
            raise TypeError('Constructor expects DataFrame or Series; however, '
                            'got [%s]' % (obj,))

    # TODO: Series support is not implemented yet.
    # TODO: not all arguments are implemented comparing to Pandas' for now.
    def aggregate(self, func_or_funcs, *args, **kwargs):
        """Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        func : dict
             a dict mapping from column name (string) to aggregate functions (string).

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

        Examples
        --------

        >>> df = ks.DataFrame({'A': [1, 1, 2, 2],
        ...                    'B': [1, 2, 3, 4],
        ...                    'C': [0.362, 0.227, 1.267, -0.562]})

        >>> df
           A  B      C
        0  1  1  0.362
        1  1  2  0.227
        2  2  3  1.267
        3  2  4 -0.562

        Different aggregations per column

        >>> df.groupby('A').agg({'B': 'min', 'C': 'sum'})
           B      C
        0  1  0.589
        1  3  0.705

        """
        if not isinstance(func_or_funcs, dict) or \
            not all(isinstance(key, string_types) and isinstance(value, string_types)
                    for key, value in func_or_funcs.items()):
            raise ValueError("aggs must be a dict mapping from column name (string) to aggregate "
                             "functions (string).")

        sdf = self._kdf._sdf
        groupkeys = self._groupkeys
        groupkey_cols = [s._scol.alias('__index_level_{}__'.format(i))
                         for i, s in enumerate(groupkeys)]
        gdf = sdf.groupby(*groupkey_cols).agg(func_or_funcs)
        reordered = ['%s(%s)' % (value, key) for key, value in iter(func_or_funcs.items())]
        kdf = DataFrame(gdf.select(reordered))
        kdf.columns = [key for key in iter(func_or_funcs.keys())]

        return kdf

    agg = aggregate

    def count(self):
        """
        Compute count of group, excluding missing values.

        >>> df = ks.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5],
        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])
        >>> df.groupby('A').count()  # doctest: +NORMALIZE_WHITESPACE
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

        koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.first, only_numeric=False)

    def last(self):
        """
        Compute last of group values.

        See Also
        --------

        koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(lambda col: F.last(col, ignorenulls=True),
                                              only_numeric=False)

    def max(self):
        """
        Compute max of group values.

        See Also
        --------

        koalas.DataFrame.groupby
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

        koalas.DataFrame.groupby

        Examples
        --------
        >>> df = ks.DataFrame({'A': [1, 1, 2, 1, 2],
        ...                    'B': [np.nan, 2, 3, 4, 5],
        ...                    'C': [1, 2, 1, 1, 2]}, columns=['A', 'B', 'C'])

        Groupby one column and return the mean of the remaining columns in
        each group.

        >>> df.groupby('A').mean()  # doctest: +NORMALIZE_WHITESPACE
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

        koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.min, only_numeric=False)

    # TODO: sync the doc and implement `ddof`.
    def std(self):
        """
        Compute standard deviation of groups, excluding missing values.
        """

        return self._reduce_for_stat_function(F.stddev, only_numeric=True)

    def sum(self):
        """
        Compute sum of group values

        See Also
        --------

        koalas.DataFrame.groupby
        """
        return self._reduce_for_stat_function(F.sum, only_numeric=True)

    # TODO: sync the doc and implement `ddof`.
    def var(self):
        """
        Compute variance of groups, excluding missing values.
        """
        return self._reduce_for_stat_function(F.variance, only_numeric=True)

    def _reduce_for_stat_function(self, sfun, only_numeric):
        groupkeys = self._groupkeys
        groupkey_cols = [s._scol.alias('__index_level_{}__'.format(i))
                         for i, s in enumerate(groupkeys)]
        sdf = self._kdf._sdf

        column_fields = []
        if len(self._agg_columns) > 0:
            stat_exprs = []
            for ks in self._agg_columns:
                spark_type = ks.spark_type
                # Special handle floating point types because Spark's count treats nan as a valid
                # value, whereas Pandas count doesn't include nan.
                if isinstance(spark_type, DoubleType) or isinstance(spark_type, FloatType):
                    stat_exprs.append(sfun(F.nanvl(ks._scol, F.lit(None))).alias(ks.name))
                    column_fields.append(ks.name)
                elif isinstance(spark_type, NumericType) or not only_numeric:
                    stat_exprs.append(sfun(ks._scol).alias(ks.name))
                    column_fields.append(ks.name)
            sdf = sdf.groupby(*groupkey_cols).agg(*stat_exprs)
        else:
            sdf = sdf.select(*groupkey_cols).distinct()
        sdf = sdf.sort(*groupkey_cols)
        metadata = Metadata(column_fields=column_fields,
                            index_info=[('__index_level_{}__'.format(i), s.name)
                                        for i, s in enumerate(groupkeys)])
        return DataFrame(sdf, metadata)


class DataFrameGroupBy(GroupBy):

    def __init__(self, kdf: DataFrame, by: List[ColumnLike], agg_columns=None):
        self._kdf = kdf
        self._groupkeys = [_resolve_col(kdf, col) for col in by]

        if agg_columns is None:
            groupkey_names = set(s.name for s in self._groupkeys)
            agg_columns = [col for col in self._kdf._metadata.column_fields
                           if col not in groupkey_names]
        self._agg_columns = [kdf[col] for col in agg_columns]

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeDataFrameGroupBy, item):
            return partial(getattr(_MissingPandasLikeDataFrameGroupBy, item), self)
        return self.__getitem__(item)

    def __getitem__(self, item):
        if isinstance(item, str):
            return SeriesGroupBy(self._kdf[item], self._groupkeys)
        else:
            return DataFrameGroupBy(self._kdf, self._groupkeys, item)


class SeriesGroupBy(GroupBy):

    def __init__(self, ks: Series, by: List[ColumnLike]):
        self._ks = ks
        self._groupkeys = [_resolve_col(ks._kdf, col) for col in by]

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeSeriesGroupBy, item):
            return partial(getattr(_MissingPandasLikeSeriesGroupBy, item), self)
        raise AttributeError(item)

    @property
    def _kdf(self):
        return self._ks._kdf

    @property
    def _agg_columns(self):
        return [self._ks]

    def _reduce_for_stat_function(self, sfun, only_numeric):
        return _col(super(SeriesGroupBy, self)._reduce_for_stat_function(sfun, only_numeric))


def _resolve_col(kdf: DataFrame, col_like: Union[ColumnLike]) -> Series:
    if isinstance(col_like, Series):
        assert kdf == col_like._kdf, \
            "Cannot combine column argument because it comes from a different dataframe"
        return col_like
    elif isinstance(col_like, str):
        return kdf[col_like]
    else:
        raise ValueError(col_like)
