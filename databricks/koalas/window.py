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
    def __init__(self, obj):
        self.obj = obj

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeRolling, item):
            property_or_func = getattr(_MissingPandasLikeRolling, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)


class RollingGroupby(Rolling):
    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeRollingGroupby, item):
            property_or_func = getattr(_MissingPandasLikeRollingGroupby, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError(item)


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

    # TODO: when add 'center' and 'axis' parameter, should add to here too.
    def __repr__(self):
        return "Expanding [min_periods={}]".format(self._min_periods)

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
