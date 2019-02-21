"""
A metadata to manage indexes.
"""
import sys

import pandas as pd

from ._dask_stubs.compatibility import string_types


class Metadata(object):
    """
    Manages column names and index information
    """

    def __init__(self, column_fields, index_info=None):
        assert all(isinstance(col, string_types) for col in column_fields)
        assert index_info is None \
            or all(isinstance(index_column_name, string_types)
                   and (index_name is None or isinstance(index_name, string_types))
                   for index_column_name, index_name in index_info)
        self._column_fields = column_fields
        self._index_info = index_info or []

    @property
    def column_fields(self):
        return self._column_fields

    @property
    def index_info(self):
        return self._index_info

    @property
    def index_columns(self):
        return [column for column, _ in self._index_info]

    @property
    def index_names(self):
        return [name for _, name in self._index_info]

    @property
    def all_columns(self):
        index_columns = self.index_columns
        return index_columns + [column for column in self._column_fields
                                if column not in index_columns]

    def copy(self, column_fields=None, index_info=None):
        if column_fields is None:
            column_fields = self._column_fields
        if index_info is None:
            index_info = self._index_info
        return Metadata(column_fields=column_fields.copy(), index_info=index_info.copy())

    @staticmethod
    def from_pandas(pdf):
        columns = [str(col) for col in pdf.columns]
        index = pdf.index
        if isinstance(index, pd.MultiIndex):
            if index.names is None:
                index_info = [('__index_level_{}__'.format(i), None)
                              for i in range(len(index.levels))]
            else:
                index_info = [('__index_level_{}__'.format(i) if name is None else name, name)
                              for name, i in enumerate(index.names)]
        else:
            index_info = [(index.name
                          if index.name is not None else '__index_level_0__', index.name)]

        return Metadata(column_fields=columns, index_info=index_info)
