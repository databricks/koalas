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
import pandas as pd

from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, DoubleType, FloatType

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

    def groupby(self, by):
        """
        Group DataFrame or Series using a mapper or by a Series of columns.

        A groupby operation involves some combination of splitting the
        object, applying a function, and combining the results. This can be
        used to group large amounts of data and compute operations on these
        groups.

        Parameters
        ----------
        by : mapping, function, label, or list of labels
            Used to determine the groups for the groupby.
            If ``by`` is a function, it's called on each value of the object's
            index. If a dict or Series is passed, the Series or dict VALUES
            will be used to determine the groups (the Series' values are first
            aligned; see ``.align()`` method). If an ndarray is passed, the
            values are used as-is determine the groups. A label or list of
            labels may be passed to group by the columns in ``self``. Notice
            that a tuple is interpreted a (single) key.

        Returns
        -------
        DataFrameGroupBy or SeriesGroupBy
            Depends on the calling object and returns groupby object that
            contains information about the groups.

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

        **Hierarchical Indexes**

        We can groupby different levels of a hierarchical index
        using the `level` parameter:

        >>> arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
        ...           ['Captive', 'Wild', 'Captive', 'Wild']]
        >>> index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))
        >>> df = pd.DataFrame({'Max Speed': [390., 350., 30., 20.]},
        ...                   index=index)
        >>> df  # doctest: +NORMALIZE_WHITESPACE
                        Max Speed
        Animal Type
        Falcon Captive      390.0
               Wild         350.0
        Parrot Captive       30.0
               Wild          20.0
        >>> df.groupby(level=0).mean()  # doctest: +NORMALIZE_WHITESPACE
                Max Speed
        Animal
        Falcon      370.0
        Parrot       25.0
        >>> df.groupby(level=1).mean()  # doctest: +NORMALIZE_WHITESPACE
                 Max Speed
        Type
        Captive      210.0
        Wild         185.0
        """
        from databricks.koalas.groupby import GroupBy
        from databricks.koalas.series import Series
        if isinstance(by, str):
            by = [by]
        elif isinstance(by, Series):
            by = [by]
        else:
            by = list(by)
        if len(by) == 0:
            raise ValueError('No group keys passed!')
        return GroupBy(self, by=by)

    def compute(self):
        """Alias of `toPandas()` to mimic dask for easily porting tests."""
        return self.toPandas()

    @staticmethod
    def _count_expr(col: spark.Column, spark_type: DataType) -> spark.Column:
        # Special handle floating point types because Spark's count treats nan as a valid value,
        # whereas Pandas count doesn't include nan.
        if isinstance(spark_type, (FloatType, DoubleType)):
            return F.count(F.nanvl(col, F.lit(None)))
        else:
            return F.count(col)


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
