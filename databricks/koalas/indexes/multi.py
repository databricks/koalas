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

from distutils.version import LooseVersion
from functools import partial
from typing import Any, Optional, Tuple, Union, cast
import warnings

import pandas as pd
from pandas.api.types import is_list_like
from pandas.api.types import is_hashable

import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F, Window

# For running doctests and reference resolution in PyCharm.
from databricks import koalas as ks  # noqa: F401
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeMultiIndex
from databricks.koalas.series import Series, first_series
from databricks.koalas.utils import (
    compare_disallow_null,
    default_session,
    is_name_like_tuple,
    name_like_string,
    scol_for,
    verify_temp_column_name,
)
from databricks.koalas.internal import (
    InternalFrame,
    NATURAL_ORDER_COLUMN_NAME,
    SPARK_INDEX_NAME_FORMAT,
)
from databricks.koalas.typedef import Scalar


class MultiIndex(Index):
    """
    Koalas MultiIndex that corresponds to pandas MultiIndex logically. This might hold Spark Column
    internally.

    Parameters
    ----------
    levels : sequence of arrays
        The unique labels for each level.
    codes : sequence of arrays
        Integers for each level designating which label at each location.
    sortorder : optional int
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : optional sequence of objects
        Names for each of the index levels. (name is accepted for compat).
    copy : bool, default False
        Copy the meta-data.
    verify_integrity : bool, default True
        Check that the levels/codes are consistent and valid.

    See Also
    --------
    MultiIndex.from_arrays  : Convert list of arrays to MultiIndex.
    MultiIndex.from_product : Create a MultiIndex from the cartesian product
                              of iterables.
    MultiIndex.from_tuples  : Convert list of tuples to a MultiIndex.
    MultiIndex.from_frame   : Make a MultiIndex from a DataFrame.
    Index : A single-level Index.

    Examples
    --------
    >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[[1, 2, 3], [4, 5, 6]]).index  # doctest: +SKIP
    MultiIndex([(1, 4),
                (2, 5),
                (3, 6)],
               )

    >>> ks.DataFrame({'a': [1, 2, 3]}, index=[list('abc'), list('def')]).index  # doctest: +SKIP
    MultiIndex([('a', 'd'),
                ('b', 'e'),
                ('c', 'f')],
               )
    """

    def __new__(
        cls,
        levels=None,
        codes=None,
        sortorder=None,
        names=None,
        dtype=None,
        copy=False,
        name=None,
        verify_integrity: bool = True,
    ):
        if LooseVersion(pd.__version__) < LooseVersion("0.24"):
            if levels is None or codes is None:
                raise TypeError("Must pass both levels and codes")

            pidx = pd.MultiIndex(
                levels=levels,
                labels=codes,
                sortorder=sortorder,
                names=names,
                dtype=dtype,
                copy=copy,
                name=name,
                verify_integrity=verify_integrity,
            )
        else:
            pidx = pd.MultiIndex(
                levels=levels,
                codes=codes,
                sortorder=sortorder,
                names=names,
                dtype=dtype,
                copy=copy,
                name=name,
                verify_integrity=verify_integrity,
            )
        return ks.from_pandas(pidx)

    @property
    def _internal(self):
        internal = self._kdf._internal
        scol = F.struct(internal.index_spark_columns)
        return internal.copy(
            column_labels=[None],
            data_spark_columns=[scol],
            data_dtypes=[None],
            column_label_names=None,
        )

    @property
    def _column_label(self):
        return None

    def __abs__(self):
        raise TypeError("TypeError: cannot perform __abs__ with this index type: MultiIndex")

    def _with_new_scol(self, scol: spark.Column, *, dtype=None):
        raise NotImplementedError("Not supported for type MultiIndex")

    def _align_and_column_op(self, f, *args) -> Index:
        raise NotImplementedError("Not supported for type MultiIndex")

    def any(self, *args, **kwargs) -> None:
        raise TypeError("cannot perform any with this index type: MultiIndex")

    def all(self, *args, **kwargs) -> None:
        raise TypeError("cannot perform all with this index type: MultiIndex")

    @staticmethod
    def from_tuples(tuples, sortorder=None, names=None) -> "MultiIndex":
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index : MultiIndex

        Examples
        --------

        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> ks.MultiIndex.from_tuples(tuples, names=('number', 'color'))  # doctest: +SKIP
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        return cast(
            MultiIndex,
            ks.from_pandas(
                pd.MultiIndex.from_tuples(tuples=tuples, sortorder=sortorder, names=names)
            ),
        )

    @staticmethod
    def from_arrays(arrays, sortorder=None, names=None) -> "MultiIndex":
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays: list / sequence of array-likes
            Each array-like gives one level’s value for each data point. len(arrays)
            is the number of levels.
        sortorder: int or None
            Level of sortedness (must be lexicographically sorted by that level).
        names: list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index: MultiIndex

        Examples
        --------

        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        >>> ks.MultiIndex.from_arrays(arrays, names=('number', 'color'))  # doctest: +SKIP
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        return cast(
            MultiIndex,
            ks.from_pandas(
                pd.MultiIndex.from_arrays(arrays=arrays, sortorder=sortorder, names=names)
            ),
        )

    @staticmethod
    def from_product(iterables, sortorder=None, names=None) -> "MultiIndex":
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index : MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ['green', 'purple']
        >>> ks.MultiIndex.from_product([numbers, colors],
        ...                            names=['number', 'color'])  # doctest: +SKIP
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """
        return cast(
            MultiIndex,
            ks.from_pandas(
                pd.MultiIndex.from_product(iterables=iterables, sortorder=sortorder, names=names)
            ),
        )

    @staticmethod
    def from_frame(df, names=None) -> "MultiIndex":
        """
        Make a MultiIndex from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            DataFrame to be converted to MultiIndex.
        names : list-like, optional
            If no names are provided, use the column names, or tuple of column
            names if the columns is a MultiIndex. If a sequence, overwrite
            names with the given sequence.

        Returns
        -------
        MultiIndex
            The MultiIndex representation of the given DataFrame.

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.
        MultiIndex.from_product : Make a MultiIndex from cartesian product
                                  of iterables.

        Examples
        --------
        >>> df = ks.DataFrame([['HI', 'Temp'], ['HI', 'Precip'],
        ...                    ['NJ', 'Temp'], ['NJ', 'Precip']],
        ...                   columns=['a', 'b'])
        >>> df  # doctest: +SKIP
              a       b
        0    HI    Temp
        1    HI  Precip
        2    NJ    Temp
        3    NJ  Precip

        >>> ks.MultiIndex.from_frame(df)  # doctest: +SKIP
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['a', 'b'])

        Using explicit names, instead of the column names

        >>> ks.MultiIndex.from_frame(df, names=['state', 'observation'])  # doctest: +SKIP
        MultiIndex([('HI',   'Temp'),
                    ('HI', 'Precip'),
                    ('NJ',   'Temp'),
                    ('NJ', 'Precip')],
                   names=['state', 'observation'])
        """
        if not isinstance(df, DataFrame):
            raise TypeError("Input must be a DataFrame")
        sdf = df.to_spark()

        if names is None:
            names = df._internal.column_labels
        elif not is_list_like(names):
            raise ValueError("Names should be list-like for a MultiIndex")
        else:
            names = [name if is_name_like_tuple(name) else (name,) for name in names]

        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in sdf.columns],
            index_names=names,
        )
        return cast(MultiIndex, DataFrame(internal).index)

    @property
    def name(self) -> str:
        raise PandasNotImplementedError(class_name="pd.MultiIndex", property_name="name")

    @name.setter
    def name(self, name: str) -> None:
        raise PandasNotImplementedError(class_name="pd.MultiIndex", property_name="name")

    def _verify_for_rename(self, name):
        if is_list_like(name):
            if self._internal.index_level != len(name):
                raise ValueError(
                    "Length of new names must be {}, got {}".format(
                        self._internal.index_level, len(name)
                    )
                )
            if any(not is_hashable(n) for n in name):
                raise TypeError("MultiIndex.name must be a hashable type")
            return [n if is_name_like_tuple(n) else (n,) for n in name]
        else:
            raise TypeError("Must pass list-like as `names`.")

    def swaplevel(self, i=-2, j=-1) -> "MultiIndex":
        """
        Swap level i with level j.
        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int, str, default -2
            First level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.
        j : int, str, default -1
            Second level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.

        Returns
        -------
        MultiIndex
            A new MultiIndex.

        Examples
        --------
        >>> midx = ks.MultiIndex.from_arrays([['a', 'b'], [1, 2]], names = ['word', 'number'])
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 1),
                    ('b', 2)],
                   names=['word', 'number'])

        >>> midx.swaplevel(0, 1)  # doctest: +SKIP
        MultiIndex([(1, 'a'),
                    (2, 'b')],
                   names=['number', 'word'])

        >>> midx.swaplevel('number', 'word')  # doctest: +SKIP
        MultiIndex([(1, 'a'),
                    (2, 'b')],
                   names=['number', 'word'])
        """
        for index in (i, j):
            if not isinstance(index, int) and index not in self.names:
                raise KeyError("Level %s not found" % index)

        i = i if isinstance(i, int) else self.names.index(i)
        j = j if isinstance(j, int) else self.names.index(j)

        for index in (i, j):
            if index >= len(self.names) or index < -len(self.names):
                raise IndexError(
                    "Too many levels: Index has only %s levels, "
                    "%s is not a valid level number" % (len(self.names), index)
                )

        index_map = list(
            zip(
                self._internal.index_spark_columns,
                self._internal.index_names,
                self._internal.index_dtypes,
            )
        )
        index_map[i], index_map[j], = index_map[j], index_map[i]
        index_spark_columns, index_names, index_dtypes = zip(*index_map)
        internal = self._internal.copy(
            index_spark_columns=list(index_spark_columns),
            index_names=list(index_names),
            index_dtypes=list(index_dtypes),
            column_labels=[],
            data_spark_columns=[],
            data_dtypes=[],
        )
        return cast(MultiIndex, DataFrame(internal).index)

    @property
    def levshape(self) -> Tuple[int, ...]:
        """
        A tuple with the length of each level.

        Examples
        --------
        >>> midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )

        >>> midx.levshape
        (3, 3)
        """
        result = self._internal.spark_frame.agg(
            *(F.countDistinct(c) for c in self._internal.index_spark_columns)
        ).collect()[0]
        return tuple(result)

    @staticmethod
    def _comparator_for_monotonic_increasing(data_type):
        return compare_disallow_null

    def _is_monotonic(self, order):
        if order == "increasing":
            return self._is_monotonic_increasing().all()
        else:
            return self._is_monotonic_decreasing().all()

    def _is_monotonic_increasing(self):
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-1, -1)

        cond = F.lit(True)
        has_not_null = F.lit(True)
        for scol in self._internal.index_spark_columns[::-1]:
            data_type = self._internal.spark_type_for(scol)
            prev = F.lag(scol, 1).over(window)
            compare = MultiIndex._comparator_for_monotonic_increasing(data_type)
            # Since pandas 1.1.4, null value is not allowed at any levels of MultiIndex.
            # Therefore, we should check `has_not_null` over the all levels.
            has_not_null = has_not_null & scol.isNotNull()
            cond = F.when(scol.eqNullSafe(prev), cond).otherwise(
                compare(scol, prev, spark.Column.__gt__)
            )

        cond = has_not_null & (prev.isNull() | cond)

        cond_name = verify_temp_column_name(
            self._internal.spark_frame.select(self._internal.index_spark_columns),
            "__is_monotonic_increasing_cond__",
        )

        sdf = self._internal.spark_frame.select(
            self._internal.index_spark_columns + [cond.alias(cond_name)]
        )

        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )

        return first_series(DataFrame(internal))

    @staticmethod
    def _comparator_for_monotonic_decreasing(data_type):
        return compare_disallow_null

    def _is_monotonic_decreasing(self):
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-1, -1)

        cond = F.lit(True)
        has_not_null = F.lit(True)
        for scol in self._internal.index_spark_columns[::-1]:
            data_type = self._internal.spark_type_for(scol)
            prev = F.lag(scol, 1).over(window)
            compare = MultiIndex._comparator_for_monotonic_increasing(data_type)
            # Since pandas 1.1.4, null value is not allowed at any levels of MultiIndex.
            # Therefore, we should check `has_not_null` over the all levels.
            has_not_null = has_not_null & scol.isNotNull()
            cond = F.when(scol.eqNullSafe(prev), cond).otherwise(
                compare(scol, prev, spark.Column.__lt__)
            )

        cond = has_not_null & (prev.isNull() | cond)

        cond_name = verify_temp_column_name(
            self._internal.spark_frame.select(self._internal.index_spark_columns),
            "__is_monotonic_decreasing_cond__",
        )

        sdf = self._internal.spark_frame.select(
            self._internal.index_spark_columns + [cond.alias(cond_name)]
        )

        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
            index_dtypes=self._internal.index_dtypes,
        )

        return first_series(DataFrame(internal))

    def to_frame(self, index=True, name=None) -> DataFrame:
        """
        Create a DataFrame with the levels of the MultiIndex as columns.
        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        Parameters
        ----------
        index : boolean, default True
            Set the index of the returned DataFrame as the original MultiIndex.
        name : list / sequence of strings, optional
            The passed names should substitute index level names.

        Returns
        -------
        DataFrame : a DataFrame containing the original MultiIndex data.

        See Also
        --------
        DataFrame

        Examples
        --------
        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> idx = ks.MultiIndex.from_tuples(tuples, names=('number', 'color'))
        >>> idx  # doctest: +SKIP
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        >>> idx.to_frame()  # doctest: +NORMALIZE_WHITESPACE
                      number color
        number color
        1      red         1   red
               blue        1  blue
        2      red         2   red
               blue        2  blue

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
           number color
        0       1   red
        1       1  blue
        2       2   red
        3       2  blue

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(name=['n', 'c'])  # doctest: +NORMALIZE_WHITESPACE
                      n     c
        number color
        1      red    1   red
               blue   1  blue
        2      red    2   red
               blue   2  blue
        """
        if name is None:
            name = [
                name if name is not None else (i,)
                for i, name in enumerate(self._internal.index_names)
            ]
        elif is_list_like(name):
            if len(name) != self._internal.index_level:
                raise ValueError("'name' should have same length as number of levels on index.")
            name = [n if is_name_like_tuple(n) else (n,) for n in name]
        else:
            raise TypeError("'name' must be a list / sequence of column names.")

        return self._to_frame(index=index, names=name)

    def to_pandas(self) -> pd.MultiIndex:
        """
        Return a pandas MultiIndex.

        .. note:: This method should only be used if the resulting pandas object is expected
                  to be small, as all the data is loaded into the driver's memory.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=[list('abcd'), list('efgh')])
        >>> df['dogs'].index.to_pandas()  # doctest: +SKIP
        MultiIndex([('a', 'e'),
                    ('b', 'f'),
                    ('c', 'g'),
                    ('d', 'h')],
                   )
        """
        # TODO: We might need to handle internal state change.
        # So far, we don't have any functions to change the internal state of MultiIndex except for
        # series-like operations. In that case, it creates new Index object instead of MultiIndex.
        return super().to_pandas()

    def toPandas(self) -> pd.MultiIndex:
        warnings.warn(
            "MultiIndex.toPandas is deprecated as of MultiIndex.to_pandas. "
            "Please use the API instead.",
            FutureWarning,
        )
        return self.to_pandas()

    toPandas.__doc__ = to_pandas.__doc__

    def nunique(self, dropna=True) -> None:  # type: ignore
        raise NotImplementedError("nunique is not defined for MultiIndex")

    # TODO: add 'name' parameter after pd.MultiIndex.name is implemented
    def copy(self, deep=None) -> "MultiIndex":  # type: ignore
        """
        Make a copy of this object.

        Parameters
        ----------
        deep : None
            this parameter is not supported but just dummy parameter to match pandas.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=[list('abcd'), list('efgh')])
        >>> df['dogs'].index  # doctest: +SKIP
        MultiIndex([('a', 'e'),
                    ('b', 'f'),
                    ('c', 'g'),
                    ('d', 'h')],
                   )

        Copy index

        >>> df.index.copy()  # doctest: +SKIP
        MultiIndex([('a', 'e'),
                    ('b', 'f'),
                    ('c', 'g'),
                    ('d', 'h')],
                   )
        """
        return super().copy(deep=deep)  # type: ignore

    def symmetric_difference(self, other, result_name=None, sort=None) -> "MultiIndex":
        """
        Compute the symmetric difference of two MultiIndex objects.

        Parameters
        ----------
        other : Index or array-like
        result_name : list
        sort : True or None, default None
            Whether to sort the resulting index.
            * True : Attempt to sort the result.
            * None : Do not sort the result.

        Returns
        -------
        symmetric_difference : MiltiIndex

        Notes
        -----
        ``symmetric_difference`` contains elements that appear in either
        ``idx1`` or ``idx2`` but not both. Equivalent to the Index created by
        ``idx1.difference(idx2) | idx2.difference(idx1)`` with duplicates
        dropped.

        Examples
        --------
        >>> midx1 = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                        ['speed', 'weight', 'length']],
        ...                       [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                        [0, 0, 0, 0, 1, 2, 0, 1, 2]])
        >>> midx2 = pd.MultiIndex([['koalas', 'cow', 'falcon'],
        ...                        ['speed', 'weight', 'length']],
        ...                       [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                        [0, 0, 0, 0, 1, 2, 0, 1, 2]])
        >>> s1 = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...                index=midx1)
        >>> s2 = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...              index=midx2)

        >>> s1.index.symmetric_difference(s2.index)  # doctest: +SKIP
        MultiIndex([('koalas', 'speed'),
                    (  'lama', 'speed')],
                   )

        You can set names of result Index.

        >>> s1.index.symmetric_difference(s2.index, result_name=['a', 'b'])  # doctest: +SKIP
        MultiIndex([('koalas', 'speed'),
                    (  'lama', 'speed')],
                   names=['a', 'b'])

        You can set sort to `True`, if you want to sort the resulting index.

        >>> s1.index.symmetric_difference(s2.index, sort=True)  # doctest: +SKIP
        MultiIndex([('koalas', 'speed'),
                    (  'lama', 'speed')],
                   )

        You can also use the ``^`` operator:

        >>> s1.index ^ s2.index  # doctest: +SKIP
        MultiIndex([('koalas', 'speed'),
                    (  'lama', 'speed')],
                   )
        """
        if type(self) != type(other):
            raise NotImplementedError(
                "Doesn't support symmetric_difference between Index & MultiIndex for now"
            )

        sdf_self = self._kdf._internal.spark_frame.select(self._internal.index_spark_columns)
        sdf_other = other._kdf._internal.spark_frame.select(other._internal.index_spark_columns)

        sdf_symdiff = sdf_self.union(sdf_other).subtract(sdf_self.intersect(sdf_other))

        if sort:
            sdf_symdiff = sdf_symdiff.sort(self._internal.index_spark_columns)

        internal = InternalFrame(  # TODO: dtypes?
            spark_frame=sdf_symdiff,
            index_spark_columns=[
                scol_for(sdf_symdiff, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
        )
        result = cast(MultiIndex, DataFrame(internal).index)

        if result_name:
            result.names = result_name

        return result

    # TODO: ADD error parameter
    def drop(self, codes, level=None) -> "MultiIndex":
        """
        Make new MultiIndex with passed list of labels deleted

        Parameters
        ----------
        codes : array-like
            Must be a list of tuples
        level : int or level name, default None

        Returns
        -------
        dropped : MultiIndex

        Examples
        --------
        >>> index = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> index # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )

        >>> index.drop(['a']) # doctest: +SKIP
        MultiIndex([('b', 'y'),
                    ('c', 'z')],
                   )

        >>> index.drop(['x', 'y'], level=1) # doctest: +SKIP
        MultiIndex([('c', 'z')],
                   )
        """
        internal = self._internal.resolved_copy
        sdf = internal.spark_frame
        index_scols = internal.index_spark_columns
        if level is None:
            scol = index_scols[0]
        elif isinstance(level, int):
            scol = index_scols[level]
        else:
            scol = None
            for index_spark_column, index_name in zip(
                internal.index_spark_columns, internal.index_names
            ):
                if not isinstance(level, tuple):
                    level = (level,)
                if level == index_name:
                    if scol is not None:
                        raise ValueError(
                            "The name {} occurs multiple times, use a level number".format(
                                name_like_string(level)
                            )
                        )
                    scol = index_spark_column
            if scol is None:
                raise KeyError("Level {} not found".format(name_like_string(level)))
        sdf = sdf[~scol.isin(codes)]

        internal = InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in internal.index_spark_column_names],
            index_names=internal.index_names,
            index_dtypes=internal.index_dtypes,
            column_labels=[],
            data_spark_columns=[],
            data_dtypes=[],
        )
        return cast(MultiIndex, DataFrame(internal).index)

    def value_counts(
        self, normalize=False, sort=True, ascending=False, bins=None, dropna=True
    ) -> Series:
        if (
            LooseVersion(pyspark.__version__) < LooseVersion("2.4")
            and default_session().conf.get("spark.sql.execution.arrow.enabled") == "true"
            and isinstance(self, MultiIndex)
        ):
            raise RuntimeError(
                "if you're using pyspark < 2.4, set conf "
                "'spark.sql.execution.arrow.enabled' to 'false' "
                "for using this function with MultiIndex"
            )
        return super().value_counts(
            normalize=normalize, sort=sort, ascending=ascending, bins=bins, dropna=dropna
        )

    value_counts.__doc__ = IndexOpsMixin.value_counts.__doc__

    def argmax(self) -> None:
        raise TypeError("reduction operation 'argmax' not allowed for this dtype")

    def argmin(self) -> None:
        raise TypeError("reduction operation 'argmin' not allowed for this dtype")

    def asof(self, label) -> None:
        raise NotImplementedError(
            "only the default get_loc method is currently supported for MultiIndex"
        )

    @property
    def is_all_dates(self) -> bool:
        """
        is_all_dates always returns False for MultiIndex

        Examples
        --------
        >>> from datetime import datetime

        >>> idx = ks.MultiIndex.from_tuples(
        ...     [(datetime(2019, 1, 1, 0, 0, 0), datetime(2019, 1, 1, 0, 0, 0)),
        ...      (datetime(2019, 1, 1, 0, 0, 0), datetime(2019, 1, 1, 0, 0, 0))])
        >>> idx  # doctest: +SKIP
        MultiIndex([('2019-01-01', '2019-01-01'),
                    ('2019-01-01', '2019-01-01')],
                   )

        >>> idx.is_all_dates
        False
        """
        return False

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeMultiIndex, item):
            property_or_func = getattr(MissingPandasLikeMultiIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError("'MultiIndex' object has no attribute '{}'".format(item))

    def _get_level_number(self, level) -> Optional[int]:
        """
        Return the level number if a valid level is given.
        """
        count = self.names.count(level)
        if (count > 1) and not isinstance(level, int):
            raise ValueError("The name %s occurs multiple times, use a level number" % level)
        if level in self.names:
            level = self.names.index(level)
        elif isinstance(level, int):
            nlevels = self.nlevels
            if level >= nlevels:
                raise IndexError(
                    "Too many levels: Index has only %d "
                    "levels, %d is not a valid level number" % (nlevels, level)
                )
            if level < 0:
                if (level + nlevels) < 0:
                    raise IndexError(
                        "Too many levels: Index has only %d levels, "
                        "not %d" % (nlevels, level + 1)
                    )
                level = level + nlevels
        else:
            raise KeyError("Level %s not found" % str(level))
            return None

        return level

    def get_level_values(self, level) -> Index:
        """
        Return vector of label values for requested level,
        equal to the length of the index.

        Parameters
        ----------
        level : int or str
            ``level`` is either the integer position of the level in the
            MultiIndex, or the name of the level.

        Returns
        -------
        values : Index
            Values is a level of this MultiIndex converted to
            a single :class:`Index` (or subclass thereof).

        Examples
        --------

        Create a MultiIndex:

        >>> mi = ks.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'a')])
        >>> mi.names = ['level_1', 'level_2']

        Get level values by supplying level as either integer or name:

        >>> mi.get_level_values(0)
        Index(['x', 'x', 'y'], dtype='object', name='level_1')

        >>> mi.get_level_values('level_2')
        Index(['a', 'b', 'a'], dtype='object', name='level_2')
        """
        level = self._get_level_number(level)
        index_scol = self._internal.index_spark_columns[level]
        index_name = self._internal.index_names[level]
        index_dtype = self._internal.index_dtypes[level]
        internal = self._internal.copy(
            index_spark_columns=[index_scol],
            index_names=[index_name],
            index_dtypes=[index_dtype],
            column_labels=[],
            data_spark_columns=[],
            data_dtypes=[],
        )
        return DataFrame(internal).index

    def insert(self, loc: int, item) -> Index:
        """
        Make new MultiIndex inserting new item at location.

        Follows Python list.append semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        new_index : MultiIndex

        Examples
        --------
        >>> kmidx = ks.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
        >>> kmidx.insert(3, ("h", "j"))  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z'),
                    ('h', 'j')],
                   )

        For negative values

        >>> kmidx.insert(-2, ("h", "j"))  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('h', 'j'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )
        """
        length = len(self)
        if loc < 0:
            loc = loc + length
            if loc < 0:
                raise IndexError(
                    "index {} is out of bounds for axis 0 with size {}".format(
                        (loc - length), length
                    )
                )
        else:
            if loc > length:
                raise IndexError(
                    "index {} is out of bounds for axis 0 with size {}".format(loc, length)
                )

        index_name = self._internal.index_spark_column_names
        sdf_before = self.to_frame(name=index_name)[:loc].to_spark()
        sdf_middle = Index([item]).to_frame(name=index_name).to_spark()
        sdf_after = self.to_frame(name=index_name)[loc:].to_spark()
        sdf = sdf_before.union(sdf_middle).union(sdf_after)

        internal = InternalFrame(  # TODO: dtypes?
            spark_frame=sdf,
            index_spark_columns=[
                scol_for(sdf, col) for col in self._internal.index_spark_column_names
            ],
            index_names=self._internal.index_names,
        )
        return DataFrame(internal).index

    def item(self) -> Tuple[Scalar, ...]:
        """
        Return the first element of the underlying data as a python tuple.

        Returns
        -------
        tuple
            The first element of MultiIndex.

        Raises
        ------
        ValueError
            If the data is not length-1.

        Examples
        --------
        >>> kmidx = ks.MultiIndex.from_tuples([('a', 'x')])
        >>> kmidx.item()
        ('a', 'x')
        """
        return self._kdf.head(2)._to_internal_pandas().index.item()

    def intersection(self, other) -> "MultiIndex":
        """
        Form the intersection of two Index objects.

        This returns a new Index with elements common to the index and `other`.

        Parameters
        ----------
        other : Index or array-like

        Returns
        -------
        intersection : MultiIndex

        Examples
        --------
        >>> midx1 = ks.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
        >>> midx2 = ks.MultiIndex.from_tuples([("c", "z"), ("d", "w")])
        >>> midx1.intersection(midx2).sort_values()  # doctest: +SKIP
        MultiIndex([('c', 'z')],
                   )
        """
        if isinstance(other, Series) or not is_list_like(other):
            raise TypeError("other must be a MultiIndex or a list of tuples")
        elif isinstance(other, DataFrame):
            raise ValueError("Index data must be 1-dimensional")
        elif isinstance(other, MultiIndex):
            spark_frame_other = other.to_frame().to_spark()
            keep_name = self.names == other.names
        elif isinstance(other, Index):
            # Always returns an empty MultiIndex if `other` is Index.
            return self.to_frame().head(0).index  # type: ignore
        elif not all(isinstance(item, tuple) for item in other):
            raise TypeError("other must be a MultiIndex or a list of tuples")
        else:
            other = MultiIndex.from_tuples(list(other))
            spark_frame_other = other.to_frame().to_spark()
            keep_name = True

        default_name = [SPARK_INDEX_NAME_FORMAT(i) for i in range(self.nlevels)]
        spark_frame_self = self.to_frame(name=default_name).to_spark()
        spark_frame_intersected = spark_frame_self.intersect(spark_frame_other)
        if keep_name:
            index_names = self._internal.index_names
        else:
            index_names = None
        internal = InternalFrame(  # TODO: dtypes?
            spark_frame=spark_frame_intersected,
            index_spark_columns=[scol_for(spark_frame_intersected, col) for col in default_name],
            index_names=index_names,
        )
        return cast(MultiIndex, DataFrame(internal).index)

    @property
    def hasnans(self):
        raise NotImplementedError("hasnans is not defined for MultiIndex")

    @property
    def inferred_type(self) -> str:
        """
        Return a string of the type inferred from the values.
        """
        # Always returns "mixed" for MultiIndex
        return "mixed"

    @property
    def asi8(self) -> None:
        """
        Integer representation of the values.
        """
        # Always returns None for MultiIndex
        return None

    def factorize(
        self, sort: bool = True, na_sentinel: Optional[int] = -1
    ) -> Tuple[Union["Series", "Index"], pd.Index]:
        return MissingPandasLikeMultiIndex.factorize(self, sort=sort, na_sentinel=na_sentinel)

    def __iter__(self):
        return MissingPandasLikeMultiIndex.__iter__(self)
