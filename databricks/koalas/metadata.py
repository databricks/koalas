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
A metadata to manage indexes.
"""
import pandas as pd

from databricks.koalas.dask.compatibility import string_types


class Metadata(object):
    """
    Manages column names and index information
    """

    def __init__(self, column_fields, index_info=None):
        """ Create a new metadata to manage column fields and index fields and names.

        :param column_fields: list of string
                              Field names to appear as columns.
        :param index_info: list of string pair
                           Each pair holds the index field name which exists in Spark fields,
                           and the index name.
        """
        assert all(isinstance(col, string_types) for col in column_fields)
        assert index_info is None \
            or all(isinstance(index_field, string_types)
                   and (index_name is None or isinstance(index_name, string_types))
                   for index_field, index_name in index_info)
        self._column_fields = column_fields
        self._index_info = index_info or []

    @property
    def column_fields(self):
        return self._column_fields

    @property
    def index_info(self):
        return self._index_info

    @property
    def index_fields(self):
        return [index_field for index_field, _ in self._index_info]

    @property
    def index_names(self):
        return [name for _, name in self._index_info]

    @property
    def all_fields(self):
        index_fields = self.index_fields
        return index_fields + [field for field in self._column_fields
                               if field not in index_fields]

    def copy(self, column_fields=None, index_info=None):
        if column_fields is None:
            column_fields = self._column_fields
        if index_info is None:
            index_info = self._index_info
        return Metadata(column_fields=column_fields.copy(), index_info=index_info.copy())

    @staticmethod
    def from_pandas(pdf):
        column_fields = [str(col) for col in pdf.columns]
        index = pdf.index
        if isinstance(index, pd.MultiIndex):
            if index.names is None:
                index_info = [('__index_level_{}__'.format(i), None)
                              for i in range(len(index.levels))]
            else:
                index_info = [('__index_level_{}__'.format(i) if name is None else name, name)
                              for i, name in enumerate(index.names)]
        else:
            index_info = [(index.name
                          if index.name is not None else '__index_level_0__', index.name)]

        return Metadata(column_fields=column_fields, index_info=index_info)
