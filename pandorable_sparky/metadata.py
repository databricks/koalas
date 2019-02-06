"""
A metadata to manage indexes.
"""
import sys

import pandas as pd

if sys.version > '3':
    basestring = unicode = str


class Metadata(object):
    """
    Manages column names and index information
    """

    def __init__(self, columns, index_info=[]):
        assert all(isinstance(col, basestring) for col in columns)
        assert index_info is not None
        assert all(isinstance(info, tuple) and len(info) == 2
                   and isinstance(info[0], basestring)
                   and (info[1] is None or isinstance(info[1], basestring))
                   for info in index_info)
        self._columns = columns
        self._index_info = index_info

    @property
    def columns(self):
        return self._columns

    @property
    def index_info(self):
        return self._index_info

    @property
    def _index_columns(self):
        return [column for column, _ in self._index_info]

    @property
    def _index_names(self):
        return [name for _, name in self._index_info]

    @property
    def all_columns(self):
        index_columns = self._index_columns
        return index_columns + [column for column in self._columns if column not in index_columns]

    def copy(self, columns=None, index_info=None):
        if columns is None:
            columns = self._columns
        if index_info is None:
            index_info = self._index_info
        return Metadata(columns=columns.copy(), index_info=index_info.copy())

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

        return Metadata(columns=columns, index_info=index_info)
