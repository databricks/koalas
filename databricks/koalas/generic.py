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
A base class to be monkey-patched to DataFrame/Column to behave similar to pandas DataFrame/Series.
"""
from collections.abc import Iterable

import pandas as pd

from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, DoubleType, FloatType

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.dask.utils import derived_from

max_display_count = 1000


class _Frame(object):
    """
    The base class for both dataframes and series.
    """

    def to_numpy(self):
        """
        A NumPy ndarray representing the values in this DataFrame
        :return: numpy.ndarray
                 Numpy representation of DataFrame

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.
        """
        return self.toPandas().values

    @derived_from(pd.DataFrame, ua_args=['axis', 'skipna', 'level', 'numeric_only'])
    def mean(self):
        return self._reduce_for_stat_function(F.mean)

    @derived_from(pd.DataFrame, ua_args=['axis', 'skipna', 'level', 'numeric_only', 'min_count'])
    def sum(self):
        return self._reduce_for_stat_function(F.sum)

    @derived_from(pd.DataFrame, ua_args=['axis', 'skipna', 'level', 'numeric_only'])
    def skew(self):
        return self._reduce_for_stat_function(F.skewness)

    @derived_from(pd.DataFrame, ua_args=['axis', 'skipna', 'level', 'numeric_only'])
    def kurtosis(self):
        return self._reduce_for_stat_function(F.kurtosis)

    kurt = kurtosis

    @derived_from(pd.DataFrame, ua_args=['axis', 'skipna', 'level', 'numeric_only'])
    def min(self):
        return self._reduce_for_stat_function(F.min)

    @derived_from(pd.DataFrame, ua_args=['axis', 'skipna', 'level', 'numeric_only'])
    def max(self):
        return self._reduce_for_stat_function(F.max)

    @derived_from(pd.DataFrame, ua_args=['axis', 'skipna', 'level', 'ddof', 'numeric_only'])
    def std(self):
        return self._reduce_for_stat_function(F.stddev)

    @derived_from(pd.DataFrame, ua_args=['axis', 'skipna', 'level', 'ddof', 'numeric_only'])
    def var(self):
        return self._reduce_for_stat_function(F.variance)

    @derived_from(pd.DataFrame)
    def abs(self):
        """
        Return a Series/DataFrame with absolute numeric value of each element.

        :return: :class:`Series` or :class:`DataFrame` with the absolute value of each element.
        """
        return _spark_col_apply(self, F.abs)

    # TODO: by argument only support the grouping name only for now. Documentation should
    # be updated when it's supported.
    def groupby(self, by):
        """
        Group DataFrame or Series using a Series of columns.

        A groupby operation involves some combination of splitting the
        object, applying a function, and combining the results. This can be
        used to group large amounts of data and compute operations on these
        groups.

        Parameters
        ----------
        by : Series, label, or list of labels
            Used to determine the groups for the groupby.
            If Series is passed, the Series or dict VALUES
            will be used to determine the groups. A label or list of
            labels may be passed to group by the columns in ``self``.

        Returns
        -------
        DataFrameGroupBy or SeriesGroupBy
            Depends on the calling object and returns groupby object that
            contains information about the groups.

        See Also
        --------
        koalas.groupby.GroupBy

        Examples
        --------
        >>> df = ks.DataFrame({'Animal': ['Falcon', 'Falcon',
        ...                               'Parrot', 'Parrot'],
        ...                    'Max Speed': [380., 370., 24., 26.]})
        >>> df
           Animal  Max Speed
        0  Falcon      380.0
        1  Falcon      370.0
        2  Parrot       24.0
        3  Parrot       26.0
        >>> df.groupby(['Animal']).mean()  # doctest: +NORMALIZE_WHITESPACE
                Max Speed
        Animal
        Falcon      375.0
        Parrot       25.0
        """
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series
        from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy

        df_or_s = self
        if isinstance(by, str):
            by = [by]
        elif isinstance(by, Series):
            by = [by]
        elif isinstance(by, Iterable):
            by = list(by)
        else:
            raise ValueError('Not a valid index: TODO')
        if not len(by):
            raise ValueError('No group keys passed!')
        if isinstance(df_or_s, DataFrame):
            df = df_or_s  # type: DataFrame
            col_by = [_resolve_col(df, col_or_s) for col_or_s in by]
            return DataFrameGroupBy(df_or_s, col_by)
        if isinstance(df_or_s, Series):
            col = df_or_s  # type: Series
            anchor = df_or_s._kdf
            col_by = [_resolve_col(anchor, col_or_s) for col_or_s in by]
            return SeriesGroupBy(col, col_by)
        raise TypeError('Constructor expects DataFrame or Series; however, '
                        'got [%s]' % (df_or_s,))

    def compute(self):
        """Alias of `to_pandas()` to mimic dask for easily porting tests."""
        return self.toPandas()

    @staticmethod
    def _count_expr(col: spark.Column, spark_type: DataType) -> spark.Column:
        # Special handle floating point types because Spark's count treats nan as a valid value,
        # whereas Pandas count doesn't include nan.
        if isinstance(spark_type, (FloatType, DoubleType)):
            return F.count(F.nanvl(col, F.lit(None)))
        else:
            return F.count(col)


def _resolve_col(kdf, col_like):
    if isinstance(col_like, ks.Series):
        assert kdf == col_like._kdf, \
            "Cannot combine column argument because it comes from a different dataframe"
        return col_like
    elif isinstance(col_like, str):
        return kdf[col_like]
    else:
        raise ValueError(col_like)


def _spark_col_apply(kdf_or_ks, sfun):
    """
    Performs a function to all cells on a dataframe, the function being a known sql function.
    """
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.series import Series
    if isinstance(kdf_or_ks, Series):
        ks = kdf_or_ks
        return Series(sfun(kdf_or_ks._scol), ks._kdf, ks._index_info)
    assert isinstance(kdf_or_ks, DataFrame)
    kdf = kdf_or_ks
    sdf = kdf._sdf
    sdf = sdf.select([sfun(sdf[col]).alias(col) for col in kdf.columns])
    return DataFrame(sdf)
