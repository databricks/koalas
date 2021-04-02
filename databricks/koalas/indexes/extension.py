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

from typing import Any, Union, Callable

import pandas as pd

from pyspark.sql.functions import pandas_udf, PandasUDFType

import databricks.koalas as ks
from databricks.koalas.indexes.base import Index
from databricks.koalas.internal import SPARK_DEFAULT_INDEX_NAME

# TODO: User to supply ReturnType (default to StringType) & ability to supply value for missing
class MapExtension:
    def __init__(self, index):
        self._index = index

    def map(self, mapper: Union[dict, Callable[[Any], Any], pd.Series]):
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
        # Default missing values to None
        @pandas_udf("string", PandasUDFType.SCALAR)
        def pyspark_mapper(col):
            return col.apply(lambda i: mapper.get(i, None))

        return self._index._with_new_scol(pyspark_mapper(SPARK_DEFAULT_INDEX_NAME))

    def _map_series(self, mapper: pd.Series):
        # TODO: clean up, maybe move somewhere else
        def getOrElse(i):
            try:
                return mapper.loc[i]
            except:
                return f"{i}"

        @pandas_udf("string", PandasUDFType.SCALAR)
        def pyspark_mapper(col):
            return col.apply(lambda i: getOrElse(i))

        return self._index._with_new_scol(pyspark_mapper(SPARK_DEFAULT_INDEX_NAME))

    def _map_lambda(self, mapper: Callable[[Any], Any]):
        result = mapper(self._index)

        # Try to use this result if we can
        if isinstance(result, str):
            result = self._lambda_str(mapper)

        if not isinstance(result, Index):
            raise TypeError("The map function must return an Index object")
        return result

    def _lambda_str(self, mapper):
        """
        Mapping helper when lambda returns str

        Parameters
        ----------
        mapper
            A lambda function that does something

        Returns
        -------
        Index

        """

        @pandas_udf("string", PandasUDFType.SCALAR)
        def pyspark_mapper(col):
            return col.apply(mapper)

        return self._index._with_new_scol(pyspark_mapper(SPARK_DEFAULT_INDEX_NAME))
