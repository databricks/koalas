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
from typing import Any, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np
from pandas.api.types import (
    is_list_like,
    is_interval_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from pandas.core.accessor import CachedAccessor
from pandas.io.formats.printing import pprint_thing
from pandas.api.types import CategoricalDtype, is_hashable
from pandas._libs import lib

from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, FractionalType, IntegralType, TimestampType

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.config import get_option, option_context
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import DataFrame
from databricks.koalas.missing.indexes import MissingPandasLikeIndex
from databricks.koalas.series import Series, first_series
from databricks.koalas.spark.accessors import SparkIndexMethods
from databricks.koalas.utils import (
    is_name_like_tuple,
    is_name_like_value,
    name_like_string,
    same_anchor,
    scol_for,
    verify_temp_column_name,
    validate_bool_kwarg,
    ERROR_MESSAGE_CANNOT_COMBINE,
)
from databricks.koalas.internal import (
    InternalFrame,
    DEFAULT_SERIES_NAME,
    SPARK_DEFAULT_INDEX_NAME,
    SPARK_INDEX_NAME_FORMAT,
)
from databricks.koalas.typedef import Scalar
from databricks.koalas.indexes.base import Index

class MapExtension():
    def __init__(self, index):
        self._index = index

    def map(self, mapper: Union[dict, callable]):
        if isinstance(mapper, dict):
            idx = self._map_dict(mapper)
        else:
            idx = self._map_lambda(mapper)
        return idx

    def _map_dict(self, mapper: dict) -> Index:
        # Default missing values to None
        vfunc = np.vectorize(lambda i: mapper.get(i, None))
        return Index(vfunc(self._index.values))

    def _map_lambda(self, mapper):
        try:
            result = mapper(self._index)

            # Try to use this result if we can
            if isinstance(result, str):
                result = self._lambda_str(mapper)

            if not isinstance(result, Index):
                raise TypeError("The map function must return an Index object")
            return result
        except Exception:
            return self.astype(object).map(mapper)

    def _lambda_str(self, mapper):
        """
        Mapping helper when lambda returns str

        Parameters
        ----------
        mapper
            A lamdba function that does something

        Returns
        -------
        Index

        """
        vfunc = np.vectorize(mapper)
        return Index(vfunc(self._index.values))

