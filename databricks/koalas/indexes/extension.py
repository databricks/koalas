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

from typing import Union

import numpy as np

from databricks.koalas.indexes.base import Index


class MapExtension:
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
