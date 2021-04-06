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

from typing import Any, Callable, Union

import pandas as pd

from pyspark.sql.functions import pandas_udf, PandasUDFType

import databricks.koalas as ks
from databricks.koalas.indexes.base import Index
from databricks.koalas.internal import SPARK_DEFAULT_INDEX_NAME
from databricks.koalas.typedef.typehints import Dtype, as_spark_type


# TODO: Implement na_action similar functionality to pandas
# NB: Passing return_type into class cause Serialisation errors; instead pass at method level
class MapExtension:
    def __init__(self, index, na_action: Any):
        self._index = index
        if na_action is not None:
            raise NotImplementedError("Currently do not support na_action functionality")
        else:
            self._na_action = na_action

    def map(
        self, mapper: Union[dict, Callable[[Any], Any], pd.Series], return_type: Dtype
    ) -> Index:
        """
        Single callable/entry point to map Index values

        Parameters
        ----------
        mapper: dict, function or pd.Series
        return_type: Dtype

        Returns
        -------
        ks.Index

        """
        if isinstance(mapper, dict):
            idx = self._map_dict(mapper, return_type)
        elif isinstance(mapper, pd.Series):
            idx = self._map_series(mapper, return_type)
        elif isinstance(mapper, ks.Series):
            raise NotImplementedError("Currently do not support input of ks.Series in Index.map")
        else:
            idx = self._map_lambda(mapper, return_type)
        return idx

    def _map_dict(self, mapper: dict, return_type: Dtype) -> Index:
        """
        Helper method that has been isolated to merely help map an Index when argument in dict type.

        Parameters
        ----------
        mapper: dict
            Key-value pairs that are used to instruct mapping from index value to new value
        return_type: Dtype
            Data type of returned value

        Returns
        -------
        ks.Index

        .. note:: Default return value for missing elements is the index's original value

        """

        @pandas_udf(as_spark_type(return_type), PandasUDFType.SCALAR)
        def pyspark_mapper(col):
            return col.apply(lambda i: mapper.get(i, return_type(i)))

        return self._index._with_new_scol(pyspark_mapper(SPARK_DEFAULT_INDEX_NAME))

    def _map_series(self, mapper: pd.Series, return_type: Dtype) -> Index:
        """
        Helper method that has been isolated to merely help map an Index
        when argument in pandas.Series type.

        Parameters
        ----------
        mapper: pandas.Series
            Series of (index, value) that is used to instruct mapping from index value to new value
        return_type: Dtype
            Data type of returned value

        Returns
        -------
        ks.Index

        .. note:: Default return value for missing elements is the index's original value

        """
        # TODO: clean up, maybe move somewhere else
        def getOrElse(i):
            try:
                return mapper.loc[i]
            except:
                return return_type(i)

        @pandas_udf(as_spark_type(return_type), PandasUDFType.SCALAR)
        def pyspark_mapper(col):
            return col.apply(lambda i: getOrElse(i))

        return self._index._with_new_scol(pyspark_mapper(SPARK_DEFAULT_INDEX_NAME))

    def _map_lambda(self, mapper: Callable[[Any], Any], return_type: Dtype) -> Index:
        """
        Helper method that has been isolated to merely help map Index when the argument is a
        generic lambda function.

        Parameters
        ----------
        mapper: Callable[[Any], Any]
            Generic lambda function that is applied to index
        return_type: Dtype
            Data type of returned value

        Returns
        -------
        ks.Index

        """

        @pandas_udf(as_spark_type(return_type), PandasUDFType.SCALAR)
        def pyspark_mapper(col):
            return col.apply(mapper)

        return self._index._with_new_scol(pyspark_mapper(SPARK_DEFAULT_INDEX_NAME))
