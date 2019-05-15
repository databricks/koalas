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
from typing import Any, List

import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
from pyspark import sql as spark
from pyspark.sql import functions as F

from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.frame import DataFrame
from databricks.koalas.generic import max_display_count
from databricks.koalas.missing.indexes import _MissingPandasLikeIndex, _MissingPandasLikeMultiIndex
from databricks.koalas.series import Series


class Index(object):
    """
    :ivar _kdf: The parent dataframe that is used to perform the groupby
    :type _kdf: DataFrame
    """

    def __init__(self, kdf: DataFrame) -> None:
        assert len(kdf._metadata._index_map) == 1
        self._kdf = kdf

    @property
    def _columns(self) -> List[spark.Column]:
        kdf = self._kdf
        return [kdf._sdf[field] for field in kdf._metadata.index_columns]

    def to_pandas(self) -> pd.Index:
        return self._kdf[[]].to_pandas().index

    toPandas = to_pandas

    @property
    def dtype(self) -> np.dtype:
        return self.to_series().dtype

    @property
    def name(self) -> str:
        return self.names[0]

    @name.setter
    def name(self, name: str) -> None:
        self.names = [name]

    @property
    def names(self) -> List[str]:
        return self._kdf._metadata.index_names

    @names.setter
    def names(self, names: List[str]) -> None:
        if not is_list_like(names):
            raise ValueError('Names must be a list-like')
        metadata = self._kdf._metadata
        if len(metadata.index_map) != len(names):
            raise ValueError('Length of new names must be {}, got {}'
                             .format(len(metadata.index_map), len(names)))
        self._kdf._metadata = metadata.copy(index_map=list(zip(metadata.index_columns, names)))

    def to_series(self, name: str = None) -> Series:
        kdf = self._kdf
        scol = self._columns[0]
        return Series(scol if name is None else scol.alias(name),
                      anchor=kdf,
                      index=kdf._metadata.index_map)

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

    def __init__(self, kdf: DataFrame):
        assert len(kdf._metadata._index_map) > 1
        self._kdf = kdf

    @property
    def name(self) -> str:
        raise PandasNotImplementedError(class_name='pd.MultiIndex', property_name='name')

    @name.setter
    def name(self, name: str) -> None:
        raise PandasNotImplementedError(class_name='pd.MultiIndex', property_name='name')

    def to_series(self, name: str = None) -> Series:
        kdf = self._kdf
        scol = F.struct(self._columns)
        return Series(scol if name is None else scol.alias(name),
                      anchor=kdf,
                      index=kdf._metadata.index_map)

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeMultiIndex, item):
            property_or_func = getattr(_MissingPandasLikeMultiIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError("'MultiIndex' object has no attribute '{}'".format(item))
