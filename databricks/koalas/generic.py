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

import numpy as np

from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, DoubleType, FloatType

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.indexing import LocIndexer

max_display_count = 1000


class _Frame(object):
    """
    The base class for both DataFrame and Series.
    """

    def to_numpy(self):
        """
        A NumPy ndarray representing the values in this DataFrame or Series.

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.

        Returns
        -------
        numpy.ndarray
        """
        return self.to_pandas().values

    def mean(self):
        """
        Return the mean of the values.

        Returns
        -------
        mean : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.mean()
        a    2.0
        b    0.2
        dtype: float64

        On a Series:

        >>> df['a'].mean()
        2.0
        """
        return self._reduce_for_stat_function(F.mean)

    def sum(self):
        """
        Return the sum of the values.

        Returns
        -------
        sum : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.sum()
        a    6.0
        b    0.6
        dtype: float64

        On a Series:

        >>> df['a'].sum()
        6.0
        """
        return self._reduce_for_stat_function(F.sum)

    def skew(self):
        """
        Return unbiased skew normalized by N-1.

        Returns
        -------
        skew : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.skew()
        a    0.000000e+00
        b   -3.319678e-16
        dtype: float64

        On a Series:

        >>> df['a'].skew()
        0.0
        """
        return self._reduce_for_stat_function(F.skewness)

    def kurtosis(self):
        """
        Return unbiased kurtosis using Fisher’s definition of kurtosis (kurtosis of normal == 0.0).
        Normalized by N-1.

        Returns
        -------
        kurt : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.kurtosis()
        a   -1.5
        b   -1.5
        dtype: float64

        On a Series:

        >>> df['a'].kurtosis()
        -1.5
        """
        return self._reduce_for_stat_function(F.kurtosis)

    kurt = kurtosis

    def min(self):
        """
        Return the minimum of the values.

        Returns
        -------
        min : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.min()
        a    1.0
        b    0.1
        dtype: float64

        On a Series:

        >>> df['a'].min()
        1.0
        """
        return self._reduce_for_stat_function(F.min)

    def max(self):
        """
        Return the maximum of the values.

        Returns
        -------
        max : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.max()
        a    3.0
        b    0.3
        dtype: float64

        On a Series:

        >>> df['a'].max()
        3.0
        """
        return self._reduce_for_stat_function(F.max)

    def std(self):
        """
        Return sample standard deviation.

        Returns
        -------
        std : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.std()
        a    1.0
        b    0.1
        dtype: float64

        On a Series:

        >>> df['a'].std()
        1.0
        """
        return self._reduce_for_stat_function(F.stddev)

    def var(self):
        """
        Return unbiased variance.

        Returns
        -------
        var : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.var()
        a    1.00
        b    0.01
        dtype: float64

        On a Series:

        >>> df['a'].var()
        1.0
        """
        return self._reduce_for_stat_function(F.variance)

    def abs(self):
        """
        Return a Series/DataFrame with absolute numeric value of each element.

        Returns
        -------
        abs : Series/DataFrame containing the absolute value of each element.

        Examples
        --------
        Absolute numeric values in a Series.

        >>> s = ks.Series([-1.10, 2, -3.33, 4])
        >>> s.abs()
        0    1.10
        1    2.00
        2    3.33
        3    4.00
        Name: abs(0), dtype: float64

        >>> df = ks.DataFrame({
        ...     'a': [4, 5, 6, 7],
        ...     'b': [10, 20, 30, 40],
        ...     'c': [100, 50, -30, -50]
        ...   },
        ...   columns=['a', 'b', 'c'])
        >>> df.abs()
           a   b    c
        0  4  10  100
        1  5  20   50
        2  6  30   30
        3  7  40   50
        """
        # TODO: The first example above should not have "Name: abs(0)".
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
        ...                    'Max Speed': [380., 370., 24., 26.]},
        ...                   columns=['Animal', 'Max Speed'])
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

    @property
    def loc(self):
        return LocIndexer(self)

    loc.__doc__ = LocIndexer.__doc__

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
        return Series(sfun(kdf_or_ks._scol), anchor=ks._kdf, index=ks._index_map)
    assert isinstance(kdf_or_ks, DataFrame)
    kdf = kdf_or_ks
    sdf = kdf._sdf
    sdf = sdf.select([sfun(sdf[col]).alias(col) for col in kdf.columns])
    return DataFrame(sdf)
