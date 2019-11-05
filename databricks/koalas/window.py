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

from pyspark.sql import Window
from pyspark.sql import functions as F
from databricks.koalas.missing.window import _MissingPandasLikeRolling, \
    _MissingPandasLikeRollingGroupby, _MissingPandasLikeExpanding, \
    _MissingPandasLikeExpandingGroupby

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.


class _RollingAndExpanding(object):
    pass


class Rolling(_RollingAndExpanding):
    def __init__(self, kdf_or_kser, window, min_periods=None):
        from databricks.koalas import DataFrame, Series
        from databricks.koalas.groupby import SeriesGroupBy, DataFrameGroupBy
        window = window - 1
        min_periods = min_periods if min_periods is not None else 0

        if window < 0:
            raise ValueError("window must be >= 0")
        if min_periods < 0:
            raise ValueError("min_periods must be >= 0")
        self._window_val = window
        self._min_periods = min_periods
        self.kdf_or_kser = kdf_or_kser
        if not isinstance(kdf_or_kser, (DataFrame, Series, DataFrameGroupBy, SeriesGroupBy)):
            raise TypeError(
                "kdf_or_kser must be a series or dataframe; however, got: %s" % type(kdf_or_kser))
        if isinstance(kdf_or_kser, (DataFrame, Series)):
            self._index_scols = kdf_or_kser._internal.index_scols
            self._window = Window.orderBy(self._index_scols).rowsBetween(
                Window.currentRow - window, Window.currentRow)

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
        Decorator that can wraps a function that handles Spark column in order
        to support it in both Koalas Series and DataFrame.
        Note that the given `func` name should be same as the API's method name.
        """
        from databricks.koalas import DataFrame, Series

        if isinstance(self.kdf_or_kser, Series):
            kser = self.kdf_or_kser
            return kser._with_new_scol(
                func(kser._scol)).rename(kser.name)
        elif isinstance(self.kdf_or_kser, DataFrame):
            kdf = self.kdf_or_kser
            applied = []
            for column in kdf.columns:
                applied.append(
                    getattr(kdf[column].rolling(self._window_val + 1,
                            self._min_periods), func.__name__)())

            sdf = kdf._sdf.select(
                kdf._internal.index_scols + [c._scol for c in applied])
            internal = kdf._internal.copy(
                sdf=sdf,
                data_columns=[c._internal.data_columns[0] for c in applied],
                column_index=[c._internal.column_index[0] for c in applied])
            return DataFrame(internal)

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
            window = Window.orderBy(self._index_scols)
            return F.when(
                F.lag(scol, self._window_val).over(window) >= self._min_periods,
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
            window = Window.orderBy(self._index_scols)
            return F.when(
                F.lag(scol, self._window_val).over(window) >= self._min_periods,
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
            window = Window.orderBy(self._index_scols)
            return F.when(
                F.lag(scol, self._window_val).over(window) >= self._min_periods,
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
            window = Window.orderBy(self._index_scols)
            return F.when(
                F.lag(scol, self._window_val).over(window) >= self._min_periods,
                F.mean(scol).over(self._window)
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(mean)


class RollingGroupby(Rolling):
    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeRollingGroupby, item):
            property_or_func = getattr(_MissingPandasLikeRollingGroupby, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)

    def sum(self):
        raise NotImplementedError("groupby.rolling().sum() is currently not implemented yet.")

    def min(self):
        raise NotImplementedError("groupby.rolling().min() is currently not implemented yet.")

    def max(self):
        raise NotImplementedError("groupby.rolling().max() is currently not implemented yet.")

    def mean(self):
        raise NotImplementedError("groupby.rolling().mean() is currently not implemented yet.")


class Expanding(_RollingAndExpanding):
    def __init__(self, kdf_or_kser, min_periods=1):
        from databricks.koalas import DataFrame, Series
        from databricks.koalas.groupby import SeriesGroupBy, DataFrameGroupBy

        if min_periods < 0:
            raise ValueError("min_periods must be >= 0")
        self._min_periods = min_periods
        self.kdf_or_kser = kdf_or_kser
        if not isinstance(kdf_or_kser, (DataFrame, Series, DataFrameGroupBy, SeriesGroupBy)):
            raise TypeError(
                "kdf_or_kser must be a series or dataframe; however, got: %s" % type(kdf_or_kser))
        if isinstance(kdf_or_kser, (DataFrame, Series)):
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
        from databricks.koalas import DataFrame, Series

        if isinstance(self.kdf_or_kser, Series):
            kser = self.kdf_or_kser
            # TODO: is this a bug? min_periods is not respected in expanding().count() in pandas.
            # scol = F.when(
            #     F.row_number().over(self._window) > self._min_periods,
            #     F.count(kser._scol).over(self._window)
            # ).otherwise(F.lit(None))
            scol = F.count(kser._scol).over(self._window)
            return kser._with_new_scol(scol).astype('float64').rename(kser.name)
        elif isinstance(self.kdf_or_kser, DataFrame):
            # TODO: deduplicate with other APIs in expanding.
            kdf = self.kdf_or_kser
            applied = []
            for column in kdf.columns:
                applied.append(kdf[column].expanding(self._min_periods).count())

            sdf = kdf._sdf.select(
                kdf._internal.index_scols + [c._scol for c in applied])
            internal = kdf._internal.copy(
                sdf=sdf,
                data_columns=[c._internal.data_columns[0] for c in applied],
                column_index=[c._internal.column_index[0] for c in applied])
            return DataFrame(internal)


class ExpandingGroupby(Expanding):
    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeExpandingGroupby, item):
            property_or_func = getattr(_MissingPandasLikeExpandingGroupby, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)

    def count(self):
        raise NotImplementedError("groupby.expanding().count() is currently not implemented yet.")
