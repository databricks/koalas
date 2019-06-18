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
Wrappers for Indexes to behave similar to pandas Index, MultiIndex.
"""

from functools import partial
from typing import Any, List, Optional

import pandas as pd
from pandas.api.types import is_list_like
from pyspark import sql as spark
from pyspark.sql import functions as F

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import DataFrame
from databricks.koalas.generic import max_display_count
from databricks.koalas.missing.indexes import _MissingPandasLikeIndex, _MissingPandasLikeMultiIndex
from databricks.koalas.series import Series


class Index(IndexOpsMixin):
    """
    Koala Index that corresponds to Pandas Index logically. This might hold Spark Column
    internally.

    :ivar _kdf: The parent dataframe
    :type _kdf: DataFrame
    :ivar _scol: Spark Column instance
    :type _scol: pyspark.Column

    See Also
    --------
    MultiIndex : A multi-level, or hierarchical, Index.

    Examples
    --------
    >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[1, 2, 3]).index
    Int64Index([1, 2, 3], dtype='int64')

    >>> ks.DataFrame({'a': [1, 2, 3]}, index=list('abc')).index
    Index(['a', 'b', 'c'], dtype='object')
    """

    def __init__(self, kdf: DataFrame, scol: Optional[spark.Column] = None) -> None:
        assert len(kdf._internal._index_map) == 1
        if scol is None:
            IndexOpsMixin.__init__(
                self, kdf._internal.copy(scol=kdf._sdf[kdf._internal.index_columns[0]]), kdf)
        else:
            IndexOpsMixin.__init__(self, kdf._internal.copy(scol=scol), kdf)

    def _with_new_scol(self, scol: spark.Column) -> 'Index':
        """
        Copy Koalas Index with the new Spark Column.

        :param scol: the new Spark Column
        :return: the copied Index
        """
        return Index(self._kdf, scol)

    @property
    def _columns(self) -> List[spark.Column]:
        """ Returns spark Columns corresponding to index columns. """
        kdf = self._kdf
        return [kdf._sdf[field] for field in kdf._internal.index_columns]

    def to_pandas(self) -> pd.Index:
        """
        Return a pandas Series.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory. If the input
                  is large, set max_rows parameter.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=list('abcd'))
        >>> df['dogs'].index.to_pandas()
        Index(['a', 'b', 'c', 'd'], dtype='object')
        """
        return self._kdf[[]].to_pandas().index

    toPandas = to_pandas

    @property
    def spark_type(self):
        """ Returns the data type as defined by Spark, as a Spark DataType object."""
        return self.to_series().dataType

    @property
    def name(self) -> str:
        """Return name of the Index."""
        return self.names[0]

    @name.setter
    def name(self, name: str) -> None:
        self.names = [name]

    @property
    def names(self) -> List[str]:
        """Return names of the Index."""
        return self._kdf._internal.index_names.copy()

    @names.setter
    def names(self, names: List[str]) -> None:
        if not is_list_like(names):
            raise ValueError('Names must be a list-like')
        internal = self._kdf._internal
        if len(internal.index_map) != len(names):
            raise ValueError('Length of new names must be {}, got {}'
                             .format(len(internal.index_map), len(names)))
        self._kdf._internal = internal.copy(index_map=list(zip(internal.index_columns, names)))

    def to_series(self, name: str = None) -> Series:
        """
        Create a Series with both index and values equal to the index keys
        useful with map for returning an indexer based on an index.

        Parameters
        ----------
        name : string, optional
            name of resulting Series. If None, defaults to name of original
            index

        Returns
        -------
        Series : dtype will be based on the type of the Index values.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=list('abcd'))
        >>> df['dogs'].index.to_series()
        a    a
        b    b
        c    c
        d    d
        Name: __index_level_0__, dtype: object
        """
        kdf = self._kdf
        scol = self._scol
        return Series(kdf._internal.copy(scol=scol if name is None else scol.alias(name)),
                      anchor=kdf)

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeIndex, item):
            property_or_func = getattr(_MissingPandasLikeIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError("'Index' object has no attribute '{}'".format(item))

    def __repr__(self):
        pser = self._kdf.head(max_display_count + 1).index.to_pandas()
        pser_length = len(pser)
        repr_string = repr(pser[:max_display_count])
        if pser_length > max_display_count:
            footer = '\nShowing only the first {}'.format(max_display_count)
            return repr_string + footer
        return repr_string


class MultiIndex(Index):

    def __init__(self, kdf: DataFrame, scol: Optional[spark.Column] = None):
        assert len(kdf._internal._index_map) > 1
        self._kdf = kdf
        if scol is None:
            IndexOpsMixin.__init__(self, kdf._internal.copy(scol=F.struct(self._columns)), kdf)
        else:
            IndexOpsMixin.__init__(self, kdf._internal.copy(scol=scol), kdf)

    def _with_new_scol(self, scol: spark.Column) -> 'MultiIndex':
        """
        Copy Koalas MultiIndex with the new Spark Column.

        :param scol: the new Spark Column
        :return: the copied MultiIndex
        """
        return MultiIndex(self._kdf, scol)

    def any(self, *args, **kwargs):
        raise TypeError("cannot perform any with this index type: MultiIndex")

    def all(self, *args, **kwargs):
        raise TypeError("cannot perform all with this index type: MultiIndex")

    @property
    def name(self) -> str:
        raise PandasNotImplementedError(class_name='pd.MultiIndex', property_name='name')

    @name.setter
    def name(self, name: str) -> None:
        raise PandasNotImplementedError(class_name='pd.MultiIndex', property_name='name')

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeMultiIndex, item):
            property_or_func = getattr(_MissingPandasLikeMultiIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError("'MultiIndex' object has no attribute '{}'".format(item))
