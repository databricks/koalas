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
from collections import OrderedDict
from functools import partial
from typing import Any, Union, TYPE_CHECKING

from pyspark.sql import Window
from pyspark.sql import functions as F
from databricks.koalas.missing.window import (
    MissingPandasLikeRolling,
    MissingPandasLikeRollingGroupby,
    MissingPandasLikeExpanding,
    MissingPandasLikeExpandingGroupby,
)

# For running doctests and reference resolution in PyCharm.
from databricks import koalas as ks  # noqa: F401

from databricks.koalas.internal import NATURAL_ORDER_COLUMN_NAME, SPARK_INDEX_NAME_FORMAT
from databricks.koalas.utils import scol_for

if TYPE_CHECKING:
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.series import Series


class RollingAndExpanding(object):
    def __init__(self, kdf_or_kser, window, min_periods):
        self._kdf_or_kser = kdf_or_kser
        self._window = window
        # This unbounded Window is later used to handle 'min_periods' for now.
        self._unbounded_window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )
        self._min_periods = min_periods

    def _apply_as_series_or_frame(self, func):
        """
        Wraps a function that handles Spark column in order
        to support it in both Koalas Series and DataFrame.
        Note that the given `func` name should be same as the API's method name.
        """
        raise NotImplementedError(
            "A class that inherits this class should implement this method "
            "to handle the index and columns of output."
        )

    def count(self) -> Union["Series", "DataFrame"]:
        def count(scol):
            return F.count(scol).over(self._window)

        return self._apply_as_series_or_frame(count).astype("float64")

    def sum(self) -> Union["Series", "DataFrame"]:
        def sum(scol):
            return F.when(
                F.row_number().over(self._unbounded_window) >= self._min_periods,
                F.sum(scol).over(self._window),
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(sum)

    def min(self) -> Union["Series", "DataFrame"]:
        def min(scol):
            return F.when(
                F.row_number().over(self._unbounded_window) >= self._min_periods,
                F.min(scol).over(self._window),
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(min)

    def max(self) -> Union["Series", "DataFrame"]:
        def max(scol):
            return F.when(
                F.row_number().over(self._unbounded_window) >= self._min_periods,
                F.max(scol).over(self._window),
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(max)

    def mean(self) -> Union["Series", "DataFrame"]:
        def mean(scol):
            return F.when(
                F.row_number().over(self._unbounded_window) >= self._min_periods,
                F.mean(scol).over(self._window),
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(mean)

    def std(self) -> Union["Series", "DataFrame"]:
        def std(scol):
            return F.when(
                F.row_number().over(self._unbounded_window) >= self._min_periods,
                F.stddev(scol).over(self._window),
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(std)

    def var(self) -> Union["Series", "DataFrame"]:
        def var(scol):
            return F.when(
                F.row_number().over(self._unbounded_window) >= self._min_periods,
                F.variance(scol).over(self._window),
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(var)


class Rolling(RollingAndExpanding):
    def __init__(self, kdf_or_kser, window, min_periods=None):
        from databricks.koalas import DataFrame, Series

        if window < 0:
            raise ValueError("window must be >= 0")
        if (min_periods is not None) and (min_periods < 0):
            raise ValueError("min_periods must be >= 0")
        if min_periods is None:
            # TODO: 'min_periods' is not equivalent in pandas because it does not count NA as
            #  a value.
            min_periods = window

        if not isinstance(kdf_or_kser, (DataFrame, Series)):
            raise TypeError(
                "kdf_or_kser must be a series or dataframe; however, got: %s" % type(kdf_or_kser)
            )

        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(
            Window.currentRow - (window - 1), Window.currentRow
        )

        super().__init__(kdf_or_kser, window, min_periods)

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeRolling, item):
            property_or_func = getattr(MissingPandasLikeRolling, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)

    def _apply_as_series_or_frame(self, func):
        return self._kdf_or_kser._apply_series_op(
            lambda kser: kser._with_new_scol(func(kser.spark.column)), should_resolve=True
        )

    def count(self) -> Union["Series", "DataFrame"]:
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
        dtype: float64

        >>> s.rolling(3).count()
        0    1.0
        1    2.0
        2    2.0
        3    2.0
        dtype: float64

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
        return super().count()

    def sum(self) -> Union["Series", "DataFrame"]:
        """
        Calculate rolling summation of given DataFrame or Series.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Same type as the input, with the same index, containing the
            rolling summation.

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
        dtype: int64

        >>> s.rolling(2).sum()
        0    NaN
        1    7.0
        2    8.0
        3    7.0
        4    8.0
        dtype: float64

        >>> s.rolling(3).sum()
        0     NaN
        1     NaN
        2    12.0
        3    10.0
        4    13.0
        dtype: float64

        For DataFrame, each rolling summation is computed column-wise.

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
        return super().sum()

    def min(self) -> Union["Series", "DataFrame"]:
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
        dtype: int64

        >>> s.rolling(2).min()
        0    NaN
        1    3.0
        2    3.0
        3    2.0
        4    2.0
        dtype: float64

        >>> s.rolling(3).min()
        0    NaN
        1    NaN
        2    3.0
        3    2.0
        4    2.0
        dtype: float64

        For DataFrame, each rolling minimum is computed column-wise.

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
        return super().min()

    def max(self) -> Union["Series", "DataFrame"]:
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
        dtype: int64

        >>> s.rolling(2).max()
        0    NaN
        1    4.0
        2    5.0
        3    5.0
        4    6.0
        dtype: float64

        >>> s.rolling(3).max()
        0    NaN
        1    NaN
        2    5.0
        3    5.0
        4    6.0
        dtype: float64

        For DataFrame, each rolling maximum is computed column-wise.

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
        return super().max()

    def mean(self) -> Union["Series", "DataFrame"]:
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
        dtype: int64

        >>> s.rolling(2).mean()
        0    NaN
        1    3.5
        2    4.0
        3    3.5
        4    4.0
        dtype: float64

        >>> s.rolling(3).mean()
        0         NaN
        1         NaN
        2    4.000000
        3    3.333333
        4    4.333333
        dtype: float64

        For DataFrame, each rolling mean is computed column-wise.

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
        return super().mean()

    def std(self) -> Union["Series", "DataFrame"]:
        """
        Calculate rolling standard deviation.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller of the rolling calculation.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.std : Equivalent method for Series.
        DataFrame.std : Equivalent method for DataFrame.
        numpy.std : Equivalent method for Numpy array.

        Examples
        --------
        >>> s = ks.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).std()
        0         NaN
        1         NaN
        2    0.577350
        3    1.000000
        4    1.000000
        5    1.154701
        6    0.000000
        dtype: float64

        For DataFrame, each rolling standard deviation is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.rolling(2).std()
                  A          B
        0       NaN        NaN
        1  0.000000   0.000000
        2  0.707107   7.778175
        3  0.707107   9.192388
        4  1.414214  16.970563
        5  0.000000   0.000000
        6  0.000000   0.000000
        """
        return super().std()

    def var(self) -> Union["Series", "DataFrame"]:
        """
        Calculate unbiased rolling variance.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller of the rolling calculation.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.var : Equivalent method for Series.
        DataFrame.var : Equivalent method for DataFrame.
        numpy.var : Equivalent method for Numpy array.

        Examples
        --------
        >>> s = ks.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.rolling(3).var()
        0         NaN
        1         NaN
        2    0.333333
        3    1.000000
        4    1.000000
        5    1.333333
        6    0.000000
        dtype: float64

        For DataFrame, each unbiased rolling variance is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.rolling(2).var()
             A      B
        0  NaN    NaN
        1  0.0    0.0
        2  0.5   60.5
        3  0.5   84.5
        4  2.0  288.0
        5  0.0    0.0
        6  0.0    0.0
        """
        return super().var()


class RollingGroupby(Rolling):
    def __init__(self, groupby, window, min_periods=None):
        from databricks.koalas.groupby import SeriesGroupBy
        from databricks.koalas.groupby import DataFrameGroupBy

        if isinstance(groupby, SeriesGroupBy):
            kdf_or_kser = groupby._kser
        elif isinstance(groupby, DataFrameGroupBy):
            kdf_or_kser = groupby._kdf
        else:
            raise TypeError(
                "groupby must be a SeriesGroupBy or DataFrameGroupBy; "
                "however, got: %s" % type(groupby)
            )

        super().__init__(kdf_or_kser, window, min_periods)

        self._groupby = groupby
        self._window = self._window.partitionBy(*[ser.spark.column for ser in groupby._groupkeys])
        self._unbounded_window = self._unbounded_window.partitionBy(
            *[ser.spark.column for ser in groupby._groupkeys]
        )

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeRollingGroupby, item):
            property_or_func = getattr(MissingPandasLikeRollingGroupby, item)
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
        from databricks.koalas.series import first_series
        from databricks.koalas.groupby import SeriesGroupBy

        groupby = self._groupby
        kdf = groupby._kdf

        # Here we need to include grouped key as an index, and shift previous index.
        #   [index_column0, index_column1] -> [grouped key, index_column0, index_column1]
        new_index_scols = []
        new_index_map = OrderedDict()
        for groupkey in groupby._groupkeys:
            index_column_name = SPARK_INDEX_NAME_FORMAT(len(new_index_scols))
            new_index_scols.append(groupkey.spark.column.alias(index_column_name))
            new_index_map[index_column_name] = groupkey._column_label

        for new_index_scol, index_name in zip(
            kdf._internal.index_spark_columns, kdf._internal.index_names
        ):
            index_column_name = SPARK_INDEX_NAME_FORMAT(len(new_index_scols))
            new_index_scols.append(new_index_scol.alias(index_column_name))
            new_index_map[index_column_name] = index_name

        if groupby._agg_columns_selected:
            agg_columns = groupby._agg_columns
        else:
            agg_columns = [
                kdf._kser_for(label)
                for label in kdf._internal.column_labels
                if label not in groupby._column_labels_to_exlcude
            ]

        applied = []
        for agg_column in agg_columns:
            applied.append(agg_column._with_new_scol(func(agg_column.spark.column)))

        # Seems like pandas filters out when grouped key is NA.
        cond = groupby._groupkeys[0].spark.column.isNotNull()
        for c in groupby._groupkeys[1:]:
            cond = cond | c.spark.column.isNotNull()

        sdf = kdf._internal.spark_frame.filter(cond).select(
            new_index_scols + [c.spark.column for c in applied]
        )

        internal = kdf._internal.copy(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in new_index_map.keys()],
            index_names=list(new_index_map.values()),
            column_labels=[c._column_label for c in applied],
            data_spark_columns=[
                scol_for(sdf, c._internal.data_spark_column_names[0]) for c in applied
            ],
        )

        ret = DataFrame(internal)
        if isinstance(groupby, SeriesGroupBy):
            return first_series(ret)
        else:
            return ret

    def count(self) -> Union["Series", "DataFrame"]:
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
        >>> s.groupby(s).rolling(3).count().sort_index()
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
        dtype: float64

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
        return super().count()

    def sum(self) -> Union["Series", "DataFrame"]:
        """
        The rolling summation of any non-NaN observations inside the window.

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
        >>> s.groupby(s).rolling(3).sum().sort_index()
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
        dtype: float64

        For DataFrame, each rolling summation is computed column-wise.

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
        return super().sum()

    def min(self) -> Union["Series", "DataFrame"]:
        """
        The rolling minimum of any non-NaN observations inside the window.

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
        >>> s.groupby(s).rolling(3).min().sort_index()
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
        dtype: float64

        For DataFrame, each rolling minimum is computed column-wise.

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
        return super().min()

    def max(self) -> Union["Series", "DataFrame"]:
        """
        The rolling maximum of any non-NaN observations inside the window.

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
        >>> s.groupby(s).rolling(3).max().sort_index()
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
        dtype: float64

        For DataFrame, each rolling maximum is computed column-wise.

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
        return super().max()

    def mean(self) -> Union["Series", "DataFrame"]:
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
        >>> s.groupby(s).rolling(3).mean().sort_index()
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
        dtype: float64

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
        return super().mean()

    def std(self) -> Union["Series", "DataFrame"]:
        """
        Calculate rolling standard deviation.

        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller of the rolling calculation.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.std : Equivalent method for Series.
        DataFrame.std : Equivalent method for DataFrame.
        numpy.std : Equivalent method for Numpy array.
        """
        return super().std()

    def var(self) -> Union["Series", "DataFrame"]:
        """
        Calculate unbiased rolling variance.

        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller of the rolling calculation.

        See Also
        --------
        Series.rolling : Calling object with Series data.
        DataFrame.rolling : Calling object with DataFrames.
        Series.var : Equivalent method for Series.
        DataFrame.var : Equivalent method for DataFrame.
        numpy.var : Equivalent method for Numpy array.
        """
        return super().var()


class Expanding(RollingAndExpanding):
    def __init__(self, kdf_or_kser, min_periods=1):
        from databricks.koalas import DataFrame, Series

        if min_periods < 0:
            raise ValueError("min_periods must be >= 0")

        if not isinstance(kdf_or_kser, (DataFrame, Series)):
            raise TypeError(
                "kdf_or_kser must be a series or dataframe; however, got: %s" % type(kdf_or_kser)
            )

        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(
            Window.unboundedPreceding, Window.currentRow
        )

        super().__init__(kdf_or_kser, window, min_periods)

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeExpanding, item):
            property_or_func = getattr(MissingPandasLikeExpanding, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)

    # TODO: when add 'center' and 'axis' parameter, should add to here too.
    def __repr__(self):
        return "Expanding [min_periods={}]".format(self._min_periods)

    _apply_as_series_or_frame = Rolling._apply_as_series_or_frame

    def count(self) -> Union["Series", "DataFrame"]:
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
        dtype: float64

        >>> s.to_frame().expanding().count()
             0
        0  1.0
        1  2.0
        2  2.0
        3  3.0
        """

        def count(scol):
            return F.when(
                F.row_number().over(self._unbounded_window) >= self._min_periods,
                F.count(scol).over(self._window),
            ).otherwise(F.lit(None))

        return self._apply_as_series_or_frame(count).astype("float64")  # type: ignore

    def sum(self) -> Union["Series", "DataFrame"]:
        """
        Calculate expanding summation of given DataFrame or Series.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Same type as the input, with the same index, containing the
            expanding summation.

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
        dtype: int64

        >>> s.expanding(3).sum()
        0     NaN
        1     NaN
        2     6.0
        3    10.0
        4    15.0
        dtype: float64

        For DataFrame, each expanding summation is computed column-wise.

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
        return super().sum()

    def min(self) -> Union["Series", "DataFrame"]:
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
        dtype: float64
        """
        return super().min()

    def max(self) -> Union["Series", "DataFrame"]:
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

        Examples
        --------
        Performing a expanding minimum with a window size of 3.

        >>> s = ks.Series([4, 3, 5, 2, 6])
        >>> s.expanding(3).max()
        0    NaN
        1    NaN
        2    5.0
        3    5.0
        4    6.0
        dtype: float64
        """
        return super().max()

    def mean(self) -> Union["Series", "DataFrame"]:
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
        dtype: float64

        >>> s.expanding(3).mean()
        0    NaN
        1    NaN
        2    2.0
        3    2.5
        dtype: float64
        """
        return super().mean()

    def std(self) -> Union["Series", "DataFrame"]:
        """
        Calculate expanding standard deviation.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller of the expanding calculation.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.std : Equivalent method for Series.
        DataFrame.std : Equivalent method for DataFrame.
        numpy.std : Equivalent method for Numpy array.

        Examples
        --------
        >>> s = ks.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.expanding(3).std()
        0         NaN
        1         NaN
        2    0.577350
        3    0.957427
        4    0.894427
        5    0.836660
        6    0.786796
        dtype: float64

        For DataFrame, each expanding standard deviation variance is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.expanding(2).std()
                  A          B
        0       NaN        NaN
        1  0.000000   0.000000
        2  0.577350   6.350853
        3  0.957427  11.412712
        4  0.894427  10.630146
        5  0.836660   9.928075
        6  0.786796   9.327379
        """
        return super().std()

    def var(self) -> Union["Series", "DataFrame"]:
        """
        Calculate unbiased expanding variance.

        .. note:: the current implementation of this API uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller of the expanding calculation.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.var : Equivalent method for Series.
        DataFrame.var : Equivalent method for DataFrame.
        numpy.var : Equivalent method for Numpy array.

        Examples
        --------
        >>> s = ks.Series([5, 5, 6, 7, 5, 5, 5])
        >>> s.expanding(3).var()
        0         NaN
        1         NaN
        2    0.333333
        3    0.916667
        4    0.800000
        5    0.700000
        6    0.619048
        dtype: float64

        For DataFrame, each unbiased expanding variance is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.expanding(2).var()
                  A           B
        0       NaN         NaN
        1  0.000000    0.000000
        2  0.333333   40.333333
        3  0.916667  130.250000
        4  0.800000  113.000000
        5  0.700000   98.566667
        6  0.619048   87.000000
        """
        return super().var()


class ExpandingGroupby(Expanding):
    def __init__(self, groupby, min_periods=1):
        from databricks.koalas.groupby import SeriesGroupBy
        from databricks.koalas.groupby import DataFrameGroupBy

        if isinstance(groupby, SeriesGroupBy):
            kdf_or_kser = groupby._kser
        elif isinstance(groupby, DataFrameGroupBy):
            kdf_or_kser = groupby._kdf
        else:
            raise TypeError(
                "groupby must be a SeriesGroupBy or DataFrameGroupBy; "
                "however, got: %s" % type(groupby)
            )

        super().__init__(kdf_or_kser, min_periods)

        self._groupby = groupby
        self._window = self._window.partitionBy(*[ser.spark.column for ser in groupby._groupkeys])
        self._unbounded_window = self._window.partitionBy(
            *[ser.spark.column for ser in groupby._groupkeys]
        )

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeExpandingGroupby, item):
            property_or_func = getattr(MissingPandasLikeExpandingGroupby, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)

    _apply_as_series_or_frame = RollingGroupby._apply_as_series_or_frame  # type: ignore

    def count(self) -> Union["Series", "DataFrame"]:
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
        >>> s.groupby(s).expanding(3).count().sort_index()
        2  0     NaN
           1     NaN
        3  2     NaN
           3     NaN
           4     3.0
        4  5     NaN
           6     NaN
           7     3.0
           8     4.0
        5  9     NaN
           10    NaN
        dtype: float64

        For DataFrame, each expanding count is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).expanding(2).count().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                A    B
        A
        2 0   NaN  NaN
          1   2.0  2.0
        3 2   NaN  NaN
          3   2.0  2.0
          4   3.0  3.0
        4 5   NaN  NaN
          6   2.0  2.0
          7   3.0  3.0
          8   4.0  4.0
        5 9   NaN  NaN
          10  2.0  2.0
        """
        return super().count()

    def sum(self) -> Union["Series", "DataFrame"]:
        """
        Calculate expanding summation of given DataFrame or Series.

        Returns
        -------
        Series or DataFrame
            Same type as the input, with the same index, containing the
            expanding summation.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.sum : Reducing sum for Series.
        DataFrame.sum : Reducing sum for DataFrame.

        Examples
        --------
        >>> s = ks.Series([2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5])
        >>> s.groupby(s).expanding(3).sum().sort_index()
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
        dtype: float64

        For DataFrame, each expanding summation is computed column-wise.

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
        return super().sum()

    def min(self) -> Union["Series", "DataFrame"]:
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
        >>> s.groupby(s).expanding(3).min().sort_index()
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
        dtype: float64

        For DataFrame, each expanding minimum is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).expanding(2).min().sort_index()  # doctest: +NORMALIZE_WHITESPACE
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
        return super().min()

    def max(self) -> Union["Series", "DataFrame"]:
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
        >>> s.groupby(s).expanding(3).max().sort_index()
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
        dtype: float64

        For DataFrame, each expanding maximum is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).expanding(2).max().sort_index()  # doctest: +NORMALIZE_WHITESPACE
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
        return super().max()

    def mean(self) -> Union["Series", "DataFrame"]:
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
        >>> s.groupby(s).expanding(3).mean().sort_index()
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
        dtype: float64

        For DataFrame, each expanding mean is computed column-wise.

        >>> df = ks.DataFrame({"A": s.to_numpy(), "B": s.to_numpy() ** 2})
        >>> df.groupby(df.A).expanding(2).mean().sort_index()  # doctest: +NORMALIZE_WHITESPACE
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
        return super().mean()

    def std(self) -> Union["Series", "DataFrame"]:
        """
        Calculate expanding standard deviation.


        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller of the expanding calculation.

        See Also
        --------
        Series.expanding: Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.std : Equivalent method for Series.
        DataFrame.std : Equivalent method for DataFrame.
        numpy.std : Equivalent method for Numpy array.
        """
        return super().std()

    def var(self) -> Union["Series", "DataFrame"]:
        """
        Calculate unbiased expanding variance.

        Returns
        -------
        Series or DataFrame
            Returns the same object type as the caller of the expanding calculation.

        See Also
        --------
        Series.expanding : Calling object with Series data.
        DataFrame.expanding : Calling object with DataFrames.
        Series.var : Equivalent method for Series.
        DataFrame.var : Equivalent method for DataFrame.
        numpy.var : Equivalent method for Numpy array.
        """
        return super().var()
