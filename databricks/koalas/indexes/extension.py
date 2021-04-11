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
from datetime import datetime
from typing import Any, Callable, Union

import pandas as pd
import numpy as np

from pyspark.sql.functions import pandas_udf, PandasUDFType

import databricks.koalas as ks
from databricks.koalas.indexes.base import Index
from databricks.koalas.internal import SPARK_DEFAULT_INDEX_NAME
from databricks.koalas.typedef.typehints import as_spark_type, Dtype, Scalar


def getOrElse(input: pd.Series, pos, return_type: Union[Scalar, Dtype], default_value=None):
    try:
        return input.loc[pos]
    except:
        if default_value is None:
            return return_type(pos)  # type: ignore
        else:
            return return_type(default_value)  # type: ignore


# TODO: Implement na_action similar functionality to pandas
# NB: Passing return_type into class cause Serialisation errors; instead pass at method level
class MapExtension:
    def __init__(self, index, na_action: Any):
        self._index = index
        if na_action is not None:
            raise NotImplementedError("Currently do not support na_action functionality")
        else:
            self._na_action = na_action

    def map(self, mapper: Union[dict, Callable[[Any], Any], dict, pd.Series]) -> Index:
        """
        Single callable/entry point to map Index values

        Parameters
        ----------
        mapper: dict, function or pd.Series

        Returns
        -------
        Index

        """
        if isinstance(mapper, dict):
            idx = self._map_dict(mapper)
        elif isinstance(mapper, pd.Series):
            idx = self._map_series(mapper)
        elif isinstance(mapper, ks.Series):
            raise NotImplementedError("Currently do not support input of ks.Series in Index.map")
        else:
            idx = self._map_lambda(mapper)
        return idx

    def _map_dict(self, mapper: dict) -> Index:
        """
        Helper method that has been isolated to merely help map Index values
        when argument in dict type.

        Parameters
        ----------
        mapper: dict
            Key-value pairs that are used to instruct mapping from index value
            to new value

        Returns
        -------
        Index

        .. note:: Default return value for missing elements is the index's original value

        """
        return_type = self._mapper_return_type(mapper)

        @pandas_udf(as_spark_type(return_type), PandasUDFType.SCALAR)
        def pyspark_mapper(col):
            return col.apply(lambda i: mapper.get(i, return_type(i)))  # type: ignore

        return self._index._with_new_scol(pyspark_mapper(SPARK_DEFAULT_INDEX_NAME))

    def _map_series(self, mapper: pd.Series) -> Index:
        """
        Helper method that has been isolated to merely help map an Index values
        when argument in pd.Series type.

        Parameters
        ----------
        mapper: pandas.Series
            Series of (index, value) that is used to instruct mapping from index value
            to new value

        Returns
        -------
        Index

        .. note:: Default return value for missing elements is the index's original value

        """
        return_type = self._mapper_return_type(mapper)

        @pandas_udf(as_spark_type(return_type), PandasUDFType.SCALAR)
        def pyspark_mapper(col):
            return col.apply(lambda i: getOrElse(mapper, i, return_type))

        return self._index._with_new_scol(pyspark_mapper(SPARK_DEFAULT_INDEX_NAME))

    def _map_lambda(self, mapper: Callable[[Any], Any]) -> Index:
        """
        Helper method that has been isolated to merely help map Index values when the argument is a
        generic lambda function.

        Parameters
        ----------
        mapper: function
            Generic lambda function to apply to index

        Returns
        -------
        Index

        """
        return_type = self._mapper_return_type(mapper)

        @pandas_udf(as_spark_type(return_type), PandasUDFType.SCALAR)
        def pyspark_mapper(col):
            return col.apply(mapper)

        return self._index._with_new_scol(scol=pyspark_mapper(SPARK_DEFAULT_INDEX_NAME))

    def _mapper_return_type(
        self, mapper: Union[dict, Callable[[Any], Any], pd.Series]
    ) -> Union[Scalar, Dtype]:
        """
        Helper method to get the mapper's return type. The return type is required for
        the pandas_udf

        Parameters
        ----------
        mapper: dict, function or pd.Series

        Returns
        -------
        Scalar or Dtype

        """

        if isinstance(mapper, dict):
            types = list(set(type(k) for k in mapper.values()))
            return_type = types[0] if len(types) == 1 else str
        elif isinstance(mapper, pd.Series):
            # Pandas dtype('O') means pandas str
            return_type = str if mapper.dtype == np.dtype("object") else mapper.dtype
        else:
            if isinstance(self._index, ks.CategoricalIndex):
                return_type = self._index.categories.dtype
            else:
                return_type = type(mapper(self._index.min()))

        # Handle pandas Timestamp - map to basic datetime
        if return_type == pd._libs.tslibs.timestamps.Timestamp:
            return_type = datetime
        return return_type
