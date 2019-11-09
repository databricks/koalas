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
from functools import partial
from typing import Any

from databricks.koalas.internal import _InternalFrame, SPARK_INDEX_NAME_FORMAT
from databricks.koalas.utils import name_like_string
from pyspark.sql import Window
from pyspark.sql import functions as F
from databricks.koalas.missing.window import _MissingPandasLikeRolling, \
    _MissingPandasLikeRollingGroupby, _MissingPandasLikeExpanding, \
    _MissingPandasLikeExpandingGroupby

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.utils import scol_for


class _RollingAndExpanding(object):
    pass


class Rolling(_RollingAndExpanding):
    def __init__(self, kdf_or_kser, window, min_periods=None):
        from databricks.koalas import DataFrame, Series
        from databricks.koalas.groupby import SeriesGroupBy, DataFrameGroupBy

        if window < 0:
            raise ValueError("window must be >= 0")
        if (min_periods is not None) and (min_periods < 0):
            raise ValueError("min_periods must be >= 0")
        self._window_val = window
        if min_periods is not None:
            self._min_periods = min_periods
        else:
            # TODO: 'min_periods' is not equivalent in pandas because it does not count NA as
            #  a value.
            self._min_periods = window

        self.kdf_or_kser = kdf_or_kser
        if not isinstance(kdf_or_kser, (DataFrame, Series, DataFrameGroupBy, SeriesGroupBy)):
            raise TypeError(
                "kdf_or_kser must be a series or dataframe; however, got: %s" % type(kdf_or_kser))
        if isinstance(kdf_or_kser, (DataFrame, Series)):
            self._index_scols = kdf_or_kser._internal.index_scols
            self._window = Window.orderBy(self._index_scols).rowsBetween(
                Window.currentRow - (self._window_val-1), Window.currentRow)

            self._unbounded_window = Window.orderBy(self._index_scols)

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeRolling, item):
            property_or_func = getattr(_MissingPandasLikeRolling, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)

    def _apply_as_series_or_frame(self, func):
        """
        Wraps a function that handles Spark column in order
        to support it in both Koalas Series and DataFrame.
        Note that the given `func` name should be same as the API's method name.
        """
        from databricks.koalas import DataFrame, Series

        if isinstance(self.kdf_or_kser, Series):
            kser = self.kdf_or_kser
            return kser._with_new_scol(func(kser._scol)).rename(kser.name)
        elif isinstance(self.kdf_or_kser, DataFrame):
            kdf = self.kdf_or_kser
            applied = []
            for column in kdf.columns:
                applied.append(
                    getattr(kdf[column].rolling(self._window_val,
                            self._min_periods), func.__name__)())

            sdf = kdf._sdf.select(
                kdf._internal.index_scols + [c._scol for c in applied])
            internal = kdf._internal.copy(
                sdf=sdf,
                column_index=[c._internal.column_index[0] for c in applied],
                column_scols=[scol_for(sdf, c._internal.data_columns[0]) for c in applied])
            return DataFrame(internal)

    def count(self):
        """
        The rolling count of any non-NaN observations inside the window.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.count : Count of the full Series.
        DataFrame.count : Count of the full DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 3, float("nan"), 10])
        >>> s.rolling(1).count()
        0    1.0
        1    1.0
        2    0.0
        3    1.0
        Name: 0, dtype: float64

        >>> s.rolling(3).count()
        0    1.0
        1    2.0
        2    2.0
        3    2.0
        Name: 0, dtype: float64

        >>> s.to_frame().rolling(1).count()
             0
        0  1.0
        1  1.0
        2  0.0
        3  1.0

        >>> s.to_frame().rolling(3).count()
             0
        0  1.0
        1  2.0
        2  2.0
        3  2.0
        """
        def count(scol):
            return F.count(scol).over(self._window)

        return self._apply_as_series_or_frame(count).astype('float64')

    def sum(self):
        """
        Calculate rolling sum of given DataFrame or Series.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Same type as the input, with the same index, containing the
            rolling sum.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.sum : Reducing sum for Series.
        DataFrame.sum : Reducing sum for DataFrame.

        Examples
        --------
        >>> s = ks.Series([4, 3, 5, 2, 6])
        >>> s
        0    4
        1    3
        2    5
        3    2
        4    6
        Name: 0, dtype: int64

        >>> s.rolling(2).sum()
        0    NaN
        1    7.0
        2    8.0
        3    7.0
        4    8.0
        Name: 0, dtype: float64

        >>> s.rolling(3).sum()
        0     NaN
        1     NaN
        2    12.0
        3    10.0
        4    13.0
        Name: 0, dtype: float64

        For DataFrame, each rolling max is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df
           A   B
        0  4  16
        1  3   9
        2  5  25
        3  2   4
        4  6  36

        >>> df.rolling(2).sum()
             A     B
        0  NaN   NaN
        1  7.0  25.0
        2  8.0  34.0
        3  7.0  29.0
        4  8.0  40.0

        >>> df.rolling(3).sum()
              A     B
        0   NaN   NaN
        1   NaN   NaN
        2  12.0  50.0
        3  10.0  38.0
        4  13.0  65.0
        """
        def sum(scol):
            return F.when(
                F.row_number().over(self._unbounded_window) >= self._min_periods,
                F.sum(scol).over(self._window)
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(sum)

    def min(self):
        """
        Calculate the rolling minimum.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the rolling
            calculation.

        See Also
        --------
        Series.rolling : Calling object with a Series.
        DataFrame.rolling : Calling object with a DataFrame.
        Series.min : Similar method for Series.
        DataFrame.min : Similar method for DataFrame.

        Examples
        --------
        >>> s = ks.Series([4, 3, 5, 2, 6])
        >>> s
        0    4
        1    3
        2    5
        3    2
        4    6
        Name: 0, dtype: int64

        >>> s.rolling(2).min()
        0    NaN
        1    3.0
        2    3.0
        3    2.0
        4    2.0
        Name: 0, dtype: float64

        >>> s.rolling(3).min()
        0    NaN
        1    NaN
        2    3.0
        3    2.0
        4    2.0
        Name: 0, dtype: float64

        For DataFrame, each rolling min is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df
           A   B
        0  4  16
        1  3   9
        2  5  25
        3  2   4
        4  6  36

        >>> df.rolling(2).min()
             A    B
        0  NaN  NaN
        1  3.0  9.0
        2  3.0  9.0
        3  2.0  4.0
        4  2.0  4.0

        >>> df.rolling(3).min()
             A    B
        0  NaN  NaN
        1  NaN  NaN
        2  3.0  9.0
        3  2.0  4.0
        4  2.0  4.0
        """
        def min(scol):
            return F.when(
                F.row_number().over(self._unbounded_window) >= self._min_periods,
                F.min(scol).over(self._window)
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(min)

    def max(self):
        """
        Calculate the rolling maximum.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Return type is determined by the caller.

        See Also
        --------
        Series.rolling : Series rolling.
        DataFrame.rolling : DataFrame rolling.
        Series.max : Similar method for Series.
        DataFrame.max : Similar method for DataFrame.

        Examples
        --------
        >>> s = ks.Series([4, 3, 5, 2, 6])
        >>> s
        0    4
        1    3
        2    5
        3    2
        4    6
        Name: 0, dtype: int64

        >>> s.rolling(2).max()
        0    NaN
        1    4.0
        2    5.0
        3    5.0
        4    6.0
        Name: 0, dtype: float64

        >>> s.rolling(3).max()
        0    NaN
        1    NaN
        2    5.0
        3    5.0
        4    6.0
        Name: 0, dtype: float64

        For DataFrame, each rolling max is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df
           A   B
        0  4  16
        1  3   9
        2  5  25
        3  2   4
        4  6  36

        >>> df.rolling(2).max()
             A     B
        0  NaN   NaN
        1  4.0  16.0
        2  5.0  25.0
        3  5.0  25.0
        4  6.0  36.0

        >>> df.rolling(3).max()
             A     B
        0  NaN   NaN
        1  NaN   NaN
        2  5.0  25.0
        3  5.0  25.0
        4  6.0  36.0
        """
        def max(scol):
            return F.when(
                F.row_number().over(self._unbounded_window) >= self._min_periods,
                F.max(scol).over(self._window)
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(max)

    def mean(self):
        """
        Calculate the rolling mean of the values.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the rolling
            calculation.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.mean : Equivalent method for Series.
        DataFrame.mean : Equivalent method for DataFrame.

        Examples
        --------
        >>> s = ks.Series([4, 3, 5, 2, 6])
        >>> s
        0    4
        1    3
        2    5
        3    2
        4    6
        Name: 0, dtype: int64

        >>> s.rolling(2).mean()
        0    NaN
        1    3.5
        2    4.0
        3    3.5
        4    4.0
        Name: 0, dtype: float64

        >>> s.rolling(3).mean()
        0         NaN
        1         NaN
        2    4.000000
        3    3.333333
        4    4.333333
        Name: 0, dtype: float64

        For DataFrame, each rolling max is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df
           A   B
        0  4  16
        1  3   9
        2  5  25
        3  2   4
        4  6  36

        >>> df.rolling(2).mean()
             A     B
        0  NaN   NaN
        1  3.5  12.5
        2  4.0  17.0
        3  3.5  14.5
        4  4.0  20.0

        >>> df.rolling(3).mean()
                  A          B
        0       NaN        NaN
        1       NaN        NaN
        2  4.000000  16.666667
        3  3.333333  12.666667
        4  4.333333  21.666667
        """
        def mean(scol):
            return F.when(
                F.row_number().over(self._unbounded_window) >= self._min_periods,
                F.mean(scol).over(self._window)
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(mean)


class RollingGroupby(Rolling):
    def __init__(self, groupby, groupkeys, window, min_periods=None):
        from databricks.koalas.groupby import SeriesGroupBy
        from databricks.koalas.groupby import DataFrameGroupBy

        if isinstance(groupby, SeriesGroupBy):
            kdf = groupby._ks.to_frame()
        elif isinstance(groupby, DataFrameGroupBy):
            kdf = groupby._kdf
        else:
            raise TypeError(
                "groupby must be a SeriesGroupBy or DataFrameGroupBy; "
                "however, got: %s" % type(groupby))

        super(RollingGroupby, self).__init__(kdf, window, min_periods)
        self._groupby = groupby
        # NOTE THAT this code intentionally uses `F.col` instead of `scol` in
        # given series. This is because, in case of series, we convert it into
        # DataFrame. So, if the given `groupkeys` is a series, they end up with
        # being a different series.
        self._window = self._window.partitionBy(
            *[F.col(name_like_string(ser.name)) for ser in groupkeys])
        self._unbounded_window = self._unbounded_window.partitionBy(
            *[F.col(name_like_string(ser.name)) for ser in groupkeys])
        self._groupkeys = groupkeys
        # Current implementation reuses DataFrameGroupBy implementations for Series as well.
        self.kdf = self.kdf_or_kser

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeRollingGroupby, item):
            property_or_func = getattr(_MissingPandasLikeRollingGroupby, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)

    def _apply_as_series_or_frame(self, func):
        """
        Wraps a function that handles Spark column in order
        to support it in both Koalas Series and DataFrame.
        Note that the given `func` name should be same as the API's method name.
        """
        from databricks.koalas import DataFrame
        from databricks.koalas.series import _col
        from databricks.koalas.groupby import SeriesGroupBy

        kdf = self.kdf
        sdf = self.kdf._sdf

        # Here we need to include grouped key as an index, and shift previous index.
        #   [index_column0, index_column1] -> [grouped key, index_column0, index_column1]
        new_index_scols = []
        new_index_map = []
        for groupkey in self._groupkeys:
            new_index_scols.append(
                # NOTE THAT this code intentionally uses `F.col` instead of `scol` in
                # given series. This is because, in case of series, we convert it into
                # DataFrame. So, if the given `groupkeys` is a series, they end up with
                # being a different series.
                F.col(
                    name_like_string(groupkey.name)
                ).alias(
                    SPARK_INDEX_NAME_FORMAT(len(new_index_scols))
                ))
            new_index_map.append(
                (SPARK_INDEX_NAME_FORMAT(len(new_index_map)),
                 groupkey._internal.column_index[0]))

        for new_index_scol, index_map in zip(kdf._internal.index_scols, kdf._internal.index_map):
            new_index_scols.append(
                new_index_scol.alias(SPARK_INDEX_NAME_FORMAT(len(new_index_scols))))
            _, name = index_map
            new_index_map.append((SPARK_INDEX_NAME_FORMAT(len(new_index_map)), name))

        applied = []
        for column in kdf.columns:
            applied.append(
                kdf[column]._with_new_scol(
                    func(kdf[column]._scol)
                ).rename(kdf[column].name))

        # Seems like pandas filters out when grouped key is NA.
        cond = self._groupkeys[0]._scol.isNotNull()
        for c in self._groupkeys:
            cond = cond | c._scol.isNotNull()
        sdf = sdf.select(new_index_scols + [c._scol for c in applied]).filter(cond)
        internal = kdf._internal.copy(
            sdf=sdf,
            index_map=new_index_map,
            column_index=[c._internal.column_index[0] for c in applied],
            column_scols=[scol_for(sdf, c._internal.data_columns[0]) for c in applied])

        ret = DataFrame(internal)
        if isinstance(self._groupby, SeriesGroupBy):
            return _col(ret)
        else:
            return ret

    def count(self):
        """
        The rolling count of any non-NaN observations inside the window.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the expanding
            calculation.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.count : Count of the full Series.
        DataFrame.count : Count of the full DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5])
        >>> s.groupby(s).rolling(3).count().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        0
        2  0     1.0
           1     2.0
        3  2     1.0
           3     2.0
           4     3.0
        4  5     1.0
           6     2.0
           7     3.0
           8     3.0
        5  9     1.0
           10    2.0
        Name: 0, dtype: float64

        For DataFrame, each rolling count is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).rolling(2).count().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                A    B
        A
        2 0   1.0  1.0
          1   2.0  2.0
        3 2   1.0  1.0
          3   2.0  2.0
          4   2.0  2.0
        4 5   1.0  1.0
          6   2.0  2.0
          7   2.0  2.0
          8   2.0  2.0
        5 9   1.0  1.0
          10  2.0  2.0
        """
        return super(RollingGroupby, self).count()

    def sum(self):
        """
        The rolling sum of any non-NaN observations inside the window.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the rolling
            calculation.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.sum : Sum of the full Series.
        DataFrame.sum : Sum of the full DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5])
        >>> s.groupby(s).rolling(3).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        0
        2  0      NaN
           1      NaN
        3  2      NaN
           3      NaN
           4      9.0
        4  5      NaN
           6      NaN
           7     12.0
           8     12.0
        5  9      NaN
           10     NaN
        Name: 0, dtype: float64

        For DataFrame, each rolling sum is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).rolling(2).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                 A     B
        A
        2 0    NaN   NaN
          1    4.0   8.0
        3 2    NaN   NaN
          3    6.0  18.0
          4    6.0  18.0
        4 5    NaN   NaN
          6    8.0  32.0
          7    8.0  32.0
          8    8.0  32.0
        5 9    NaN   NaN
          10  10.0  50.0
        """
        return super(RollingGroupby, self).sum()

    def min(self):
        """
        The rolling min of any non-NaN observations inside the window.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the rolling
            calculation.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.min : Min of the full Series.
        DataFrame.min : Min of the full DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5])
        >>> s.groupby(s).rolling(3).min().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        0
        2  0     NaN
           1     NaN
        3  2     NaN
           3     NaN
           4     3.0
        4  5     NaN
           6     NaN
           7     4.0
           8     4.0
        5  9     NaN
           10    NaN
        Name: 0, dtype: float64

        For DataFrame, each rolling min is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).rolling(2).min().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                A     B
        A
        2 0   NaN   NaN
          1   2.0   4.0
        3 2   NaN   NaN
          3   3.0   9.0
          4   3.0   9.0
        4 5   NaN   NaN
          6   4.0  16.0
          7   4.0  16.0
          8   4.0  16.0
        5 9   NaN   NaN
          10  5.0  25.0
        """
        return super(RollingGroupby, self).min()

    def max(self):
        """
        The rolling max of any non-NaN observations inside the window.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the rolling
            calculation.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.max : Max of the full Series.
        DataFrame.max : Max of the full DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5])
        >>> s.groupby(s).rolling(3).max().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        0
        2  0     NaN
           1     NaN
        3  2     NaN
           3     NaN
           4     3.0
        4  5     NaN
           6     NaN
           7     4.0
           8     4.0
        5  9     NaN
           10    NaN
        Name: 0, dtype: float64

        For DataFrame, each rolling count is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).rolling(2).max().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                A     B
        A
        2 0   NaN   NaN
          1   2.0   4.0
        3 2   NaN   NaN
          3   3.0   9.0
          4   3.0   9.0
        4 5   NaN   NaN
          6   4.0  16.0
          7   4.0  16.0
          8   4.0  16.0
        5 9   NaN   NaN
          10  5.0  25.0
        """
        return super(RollingGroupby, self).max()

    def mean(self):
        """
        The rolling mean of any non-NaN observations inside the window.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the rolling
            calculation.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.mean : Mean of the full Series.
        DataFrame.mean : Mean of the full DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5])
        >>> s.groupby(s).rolling(3).mean().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        0
        2  0     NaN
           1     NaN
        3  2     NaN
           3     NaN
           4     3.0
        4  5     NaN
           6     NaN
           7     4.0
           8     4.0
        5  9     NaN
           10    NaN
        Name: 0, dtype: float64

        For DataFrame, each rolling mean is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).rolling(2).mean().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                A     B
        A
        2 0   NaN   NaN
          1   2.0   4.0
        3 2   NaN   NaN
          3   3.0   9.0
          4   3.0   9.0
        4 5   NaN   NaN
          6   4.0  16.0
          7   4.0  16.0
          8   4.0  16.0
        5 9   NaN   NaN
          10  5.0  25.0
        """
        return super(RollingGroupby, self).mean()


class Expanding(_RollingAndExpanding):
    def __init__(self, kdf_or_kser, min_periods=1):
        from databricks.koalas import DataFrame, Series

        if min_periods < 0:
            raise ValueError("min_periods must be >= 0")
        self._min_periods = min_periods
        self.kdf_or_kser = kdf_or_kser
        if not isinstance(kdf_or_kser, (DataFrame, Series)):
            raise TypeError(
                "kdf_or_kser must be a series or dataframe; however, got: %s" % type(kdf_or_kser))
        index_scols = kdf_or_kser._internal.index_scols
        self._window = Window.orderBy(index_scols).rowsBetween(
            Window.unboundedPreceding, Window.currentRow)

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeExpanding, item):
            property_or_func = getattr(_MissingPandasLikeExpanding, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)

    # TODO: when add 'center' and 'axis' parameter, should add to here too.
    def __repr__(self):
        return "Expanding [min_periods={}]".format(self._min_periods)

    def _apply_as_series_or_frame(self, func):
        """
        Wraps a function that handles Spark column in order
        to support it in both Koalas Series and DataFrame.

        Note that the given `func` name should be same as the API's method name.
        """
        from databricks.koalas import DataFrame, Series

        if isinstance(self.kdf_or_kser, Series):
            kser = self.kdf_or_kser
            return kser._with_new_scol(func(kser._scol)).rename(kser.name)
        elif isinstance(self.kdf_or_kser, DataFrame):
            kdf = self.kdf_or_kser
            applied = []
            for column in kdf.columns:
                applied.append(
                    getattr(kdf[column].expanding(self._min_periods), func.__name__)())

            sdf = kdf._sdf.select(
                kdf._internal.index_scols + [c._scol for c in applied])
            internal = kdf._internal.copy(
                sdf=sdf,
                column_index=[c._internal.column_index[0] for c in applied],
                column_scols=[scol_for(sdf, c._internal.data_columns[0]) for c in applied])
            return DataFrame(internal)

    def count(self):
        """
        The expanding count of any non-NaN observations inside the window.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the expanding
            calculation.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.count : Count of the full Series.
        DataFrame.count : Count of the full DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 3, float("nan"), 10])
        >>> s.expanding().count()
        0    1.0
        1    2.0
        2    2.0
        3    3.0
        Name: 0, dtype: float64

        >>> s.to_frame().expanding().count()
             0
        0  1.0
        1  2.0
        2  2.0
        3  3.0
        """
        def count(scol):
            # TODO: is this a bug? min_periods is not respected in expanding().count() in pandas.
            # return F.when(
            #     F.row_number().over(self._window) >= self._min_periods,
            #     F.count(scol).over(self._window)
            # ).otherwise(F.lit(None))
            return F.count(scol).over(self._window)

        return self._apply_as_series_or_frame(count).astype('float64')

    def sum(self):
        """
        Calculate expanding sum of given DataFrame or Series.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Same type as the input, with the same index, containing the
            expanding sum.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.sum : Reducing sum for Series.
        DataFrame.sum : Reducing sum for DataFrame.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3, 4, 5])
        >>> s
        0    1
        1    2
        2    3
        3    4
        4    5
        Name: 0, dtype: int64

        >>> s.expanding(3).sum()
        0     NaN
        1     NaN
        2     6.0
        3    10.0
        4    15.0
        Name: 0, dtype: float64

        For DataFrame, each expanding sum is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df
           A   B
        0  1   1
        1  2   4
        2  3   9
        3  4  16
        4  5  25

        >>> df.expanding(3).sum()
              A     B
        0   NaN   NaN
        1   NaN   NaN
        2   6.0  14.0
        3  10.0  30.0
        4  15.0  55.0
        """
        def sum(scol):
            return F.when(
                F.row_number().over(self._window) >= self._min_periods,
                F.sum(scol).over(self._window)
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(sum)

    def min(self):
        """
        Calculate the expanding minimum.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the expanding
            calculation.

        See Also
        --------
        Series.expanding : Calling object with a Series.
        DataFrame.expanding : Calling object with a DataFrame.
        Series.min : Similar method for Series.
        DataFrame.min : Similar method for DataFrame.

        Examples
        --------
        Performing a expanding minimum with a window size of 3.

        >>> s = ks.Series([4, 3, 5, 2, 6])
        >>> s.expanding(3).min()
        0    NaN
        1    NaN
        2    3.0
        3    2.0
        4    2.0
        Name: 0, dtype: float64
        """
        def min(scol):
            return F.when(
                F.row_number().over(self._window) >= self._min_periods,
                F.min(scol).over(self._window)
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(min)

    def max(self):
        """
        Calculate the expanding maximum.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Return type is determined by the caller.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.max : Similar method for Series.
        DataFrame.max : Similar method for DataFrame.
        """
        def max(scol):
            return F.when(
                F.row_number().over(self._window) >= self._min_periods,
                F.max(scol).over(self._window)
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(max)

    def mean(self):
        """
        Calculate the expanding mean of the values.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the expanding
            calculation.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.mean : Equivalent method for Series.
        DataFrame.mean : Equivalent method for DataFrame.

        Examples
        --------
        The below examples will show expanding mean calculations with window sizes of
        two and three, respectively.

        >>> s = ks.Series([1, 2, 3, 4])
        >>> s.expanding(2).mean()
        0    NaN
        1    1.5
        2    2.0
        3    2.5
        Name: 0, dtype: float64

        >>> s.expanding(3).mean()
        0    NaN
        1    NaN
        2    2.0
        3    2.5
        Name: 0, dtype: float64
        """
        def mean(scol):
            return F.when(
                F.row_number().over(self._window) >= self._min_periods,
                F.mean(scol).over(self._window)
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(mean)


class ExpandingGroupby(Expanding):
    def __init__(self, groupby, groupkeys, min_periods=1):
        from databricks.koalas.groupby import SeriesGroupBy
        from databricks.koalas.groupby import DataFrameGroupBy

        if isinstance(groupby, SeriesGroupBy):
            kdf = groupby._ks.to_frame()
        elif isinstance(groupby, DataFrameGroupBy):
            kdf = groupby._kdf
        else:
            raise TypeError(
                "groupby must be a SeriesGroupBy or DataFrameGroupBy; "
                "however, got: %s" % type(groupby))

        super(ExpandingGroupby, self).__init__(kdf, min_periods)
        self._groupby = groupby
        # NOTE THAT this code intentionally uses `F.col` instead of `scol` in
        # given series. This is because, in case of series, we convert it into
        # DataFrame. So, if the given `groupkeys` is a series, they end up with
        # being a different series.
        self._window = self._window.partitionBy(
            *[F.col(name_like_string(ser.name)) for ser in groupkeys])
        self._groupkeys = groupkeys
        # Current implementation reuses DataFrameGroupBy implementations for Series as well.
        self.kdf = self.kdf_or_kser

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeExpandingGroupby, item):
            property_or_func = getattr(_MissingPandasLikeExpandingGroupby, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)

    def _apply_as_series_or_frame(self, func):
        """
        Wraps a function that handles Spark column in order
        to support it in both Koalas Series and DataFrame.

        Note that the given `func` name should be same as the API's method name.
        """
        from databricks.koalas import DataFrame
        from databricks.koalas.series import _col
        from databricks.koalas.groupby import SeriesGroupBy

        kdf = self.kdf
        sdf = self.kdf._sdf

        # Here we need to include grouped key as an index, and shift previous index.
        #   [index_column0, index_column1] -> [grouped key, index_column0, index_column1]
        new_index_scols = []
        new_index_map = []
        for groupkey in self._groupkeys:
            new_index_scols.append(
                # NOTE THAT this code intentionally uses `F.col` instead of `scol` in
                # given series. This is because, in case of series, we convert it into
                # DataFrame. So, if the given `groupkeys` is a series, they end up with
                # being a different series.
                F.col(
                    name_like_string(groupkey.name)
                ).alias(
                    SPARK_INDEX_NAME_FORMAT(len(new_index_scols))
                ))
            new_index_map.append(
                (SPARK_INDEX_NAME_FORMAT(len(new_index_map)),
                 groupkey._internal.column_index[0]))

        for new_index_scol, index_map in zip(kdf._internal.index_scols, kdf._internal.index_map):
            new_index_scols.append(
                new_index_scol.alias(SPARK_INDEX_NAME_FORMAT(len(new_index_scols))))
            _, name = index_map
            new_index_map.append((SPARK_INDEX_NAME_FORMAT(len(new_index_map)), name))

        applied = []
        for column in kdf.columns:
            applied.append(
                kdf[column]._with_new_scol(
                    func(kdf[column]._scol)
                ).rename(kdf[column].name))

        # Seems like pandas filters out when grouped key is NA.
        cond = self._groupkeys[0]._scol.isNotNull()
        for c in self._groupkeys:
            cond = cond | c._scol.isNotNull()
        sdf = sdf.select(new_index_scols + [c._scol for c in applied]).filter(cond)
        internal = kdf._internal.copy(
            sdf=sdf,
            index_map=new_index_map,
            column_index=[c._internal.column_index[0] for c in applied],
            column_scols=[scol_for(sdf, c._internal.data_columns[0]) for c in applied])

        ret = DataFrame(internal)
        if isinstance(self._groupby, SeriesGroupBy):
            return _col(ret)
        else:
            return ret

    def count(self):
        """
        The expanding count of any non-NaN observations inside the window.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the expanding
            calculation.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.count : Count of the full Series.
        DataFrame.count : Count of the full DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5])
        >>> s.groupby(s).expanding(3).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        0
        2  0      NaN
           1      NaN
        3  2      NaN
           3      NaN
           4      9.0
        4  5      NaN
           6      NaN
           7     12.0
           8     16.0
        5  9      NaN
           10     NaN
        Name: 0, dtype: float64

        For DataFrame, each expanding sum is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).expanding(2).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                 A     B
        A
        2 0    NaN   NaN
          1    4.0   8.0
        3 2    NaN   NaN
          3    6.0  18.0
          4    9.0  27.0
        4 5    NaN   NaN
          6    8.0  32.0
          7   12.0  48.0
          8   16.0  64.0
        5 9    NaN   NaN
          10  10.0  50.0
        """
        return super(ExpandingGroupby, self).count()

    def sum(self):
        """
        Calculate expanding sum of given DataFrame or Series.

        Returns
        -------
        Series or DataFrame
            Same type as the input, with the same index, containing the
            expanding sum.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.sum : Reducing sum for Series.
        DataFrame.sum : Reducing sum for DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5])
        >>> s.groupby(s).expanding(3).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        0
        2  0      NaN
           1      NaN
        3  2      NaN
           3      NaN
           4      9.0
        4  5      NaN
           6      NaN
           7     12.0
           8     16.0
        5  9      NaN
           10     NaN
        Name: 0, dtype: float64

        For DataFrame, each expanding sum is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).expanding(2).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                 A     B
        A
        2 0    NaN   NaN
          1    4.0   8.0
        3 2    NaN   NaN
          3    6.0  18.0
          4    9.0  27.0
        4 5    NaN   NaN
          6    8.0  32.0
          7   12.0  48.0
          8   16.0  64.0
        5 9    NaN   NaN
          10  10.0  50.0
        """
        return super(ExpandingGroupby, self).sum()

    def min(self):
        """
        Calculate the expanding minimum.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the expanding
            calculation.

        See Also
        --------
        Series.expanding : Calling object with a Series.
        DataFrame.expanding : Calling object with a DataFrame.
        Series.min : Similar method for Series.
        DataFrame.min : Similar method for DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5])
        >>> s.groupby(s).expanding(3).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        0
        2  0      NaN
           1      NaN
        3  2      NaN
           3      NaN
           4      9.0
        4  5      NaN
           6      NaN
           7     12.0
           8     16.0
        5  9      NaN
           10     NaN
        Name: 0, dtype: float64

        For DataFrame, each expanding sum is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).expanding(2).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                 A     B
        A
        2 0    NaN   NaN
          1    4.0   8.0
        3 2    NaN   NaN
          3    6.0  18.0
          4    9.0  27.0
        4 5    NaN   NaN
          6    8.0  32.0
          7   12.0  48.0
          8   16.0  64.0
        5 9    NaN   NaN
          10  10.0  50.0
        """
        return super(ExpandingGroupby, self).min()

    def max(self):
        """
        Calculate the expanding maximum.

        Returns
        -------
        Series or DataFrame
            Return type is determined by the caller.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.max : Similar method for Series.
        DataFrame.max : Similar method for DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5])
        >>> s.groupby(s).expanding(3).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        0
        2  0      NaN
           1      NaN
        3  2      NaN
           3      NaN
           4      9.0
        4  5      NaN
           6      NaN
           7     12.0
           8     16.0
        5  9      NaN
           10     NaN
        Name: 0, dtype: float64

        For DataFrame, each expanding sum is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).expanding(2).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                 A     B
        A
        2 0    NaN   NaN
          1    4.0   8.0
        3 2    NaN   NaN
          3    6.0  18.0
          4    9.0  27.0
        4 5    NaN   NaN
          6    8.0  32.0
          7   12.0  48.0
          8   16.0  64.0
        5 9    NaN   NaN
          10  10.0  50.0
        """
        return super(ExpandingGroupby, self).max()

    def mean(self):
        """
        Calculate the expanding mean of the values.

        Returns
        -------
        Series or DataFrame
            Returned object type is determined by the caller of the expanding
            calculation.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.mean : Equivalent method for Series.
        DataFrame.mean : Equivalent method for DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5])
        >>> s.groupby(s).expanding(3).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
        0
        2  0      NaN
           1      NaN
        3  2      NaN
           3      NaN
           4      9.0
        4  5      NaN
           6      NaN
           7     12.0
           8     16.0
        5  9      NaN
           10     NaN
        Name: 0, dtype: float64

        For DataFrame, each expanding sum is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).expanding(2).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                 A     B
        A
        2 0    NaN   NaN
          1    4.0   8.0
        3 2    NaN   NaN
          3    6.0  18.0
          4    9.0  27.0
        4 5    NaN   NaN
          6    8.0  32.0
          7   12.0  48.0
          8   16.0  64.0
        5 9    NaN   NaN
          10  10.0  50.0
        """
        return super(ExpandingGroupby, self).mean()
