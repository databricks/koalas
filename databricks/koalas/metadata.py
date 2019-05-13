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

from typing import List, Optional, Tuple

import pandas as pd

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.


IndexMap = Tuple[str, Optional[str]]


class Metadata(object):
    """
    Manages column names and index information.

    :ivar _data_columns: list of the Spark field names to be seen as columns in Koalas DataFrame.
    :ivar _index_map: list of pair holding the Spark field names for indexes,
                       and the index name to be seen in Koalas DataFrame.

    .. note:: this is an internal class. It is not supposed to be exposed to users and users
        should not directly access to it.

    Metadata represents the index information for a DataFrame it belongs to. For instance,
    if we have a Koalas DataFrame as below, Pandas DataFrame does not store the index as columns.

    >>> kdf = ks.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': [5, 6, 7, 8],
    ...     'C': [9, 10, 11, 12],
    ...     'D': [13, 14, 15, 16],
    ...     'E': [17, 18, 19, 20]}, columns = ['A', 'B', 'C', 'D', 'E'])
    >>> kdf  # doctest: +NORMALIZE_WHITESPACE
       A  B   C   D   E
    0  1  5   9  13  17
    1  2  6  10  14  18
    2  3  7  11  15  19
    3  4  8  12  16  20

    However, all columns including index column are also stored in Spark DataFrame internally
    as below.

    >>> kdf.to_spark().show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+

    In order to fill this gap, the current metadata is used by mapping Spark's internal column
    to Koalas' index. See the method below:

    * `data_columns` represents non-indexing columns

    * `index_columns` represents internal index columns

    * `columns` represents all columns

    * `index_names` represents the external index name

    * `index_map` is zipped pairs of `index_columns` and `index_names`

    >>> metadata = kdf._metadata
    >>> metadata.data_columns
    ['A', 'B', 'C', 'D', 'E']
    >>> metadata.index_columns
    ['__index_level_0__']
    >>> metadata.columns
    ['__index_level_0__', 'A', 'B', 'C', 'D', 'E']
    >>> metadata.index_names
    [None]
    >>> metadata.index_map
    [('__index_level_0__', None)]

    In case that index is set to one of the existing column as below:

    >>> kdf1 = kdf.set_index("A")
    >>> kdf1  # doctest: +NORMALIZE_WHITESPACE
       B   C   D   E
    A
    1  5   9  13  17
    2  6  10  14  18
    3  7  11  15  19
    4  8  12  16  20

    >>> kdf1.to_spark().show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+

    >>> metadata = kdf1._metadata
    >>> metadata.data_columns
    ['B', 'C', 'D', 'E']
    >>> metadata.index_columns
    ['A']
    >>> metadata.columns
    ['A', 'B', 'C', 'D', 'E']
    >>> metadata.index_names
    ['A']
    >>> metadata.index_map
    [('A', 'A')]

    In case that index becomes a multi index as below:

    >>> kdf2 = kdf.set_index("A", append=True)
    >>> kdf2  # doctest: +NORMALIZE_WHITESPACE
         B   C   D   E
      A
    0 1  5   9  13  17
    1 2  6  10  14  18
    2 3  7  11  15  19
    3 4  8  12  16  20

    >>> kdf2.to_spark().show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+

    >>> metadata = kdf2._metadata
    >>> metadata.data_columns
    ['B', 'C', 'D', 'E']
    >>> metadata.index_columns
    ['__index_level_0__', 'A']
    >>> metadata.columns
    ['__index_level_0__', 'A', 'B', 'C', 'D', 'E']
    >>> metadata.index_names
    [None, 'A']
    >>> metadata.index_map
    [('__index_level_0__', None), ('A', 'A')]
    """

    def __init__(self, data_columns: List[str],
                 index_map: Optional[List[IndexMap]] = None) -> None:
        """ Create a new metadata to manage column fields and index fields and names.

        :param data_columns: list of string
                              Field names to appear as columns.
        :param index_map: list of string pair
                           Each pair holds the index field name which exists in Spark fields,
                           and the index name.
        """
        assert all(isinstance(col, str) for col in data_columns)
        assert index_map is None \
            or all(isinstance(index_field, str)
                   and (index_name is None or isinstance(index_name, str))
                   for index_field, index_name in index_map)
        self._data_columns = data_columns  # type: List[str]
        self._index_map = index_map or []  # type: List[IndexMap]

    @property
    def data_columns(self) -> List[str]:
        """ Returns the managed column field names. """
        return self._data_columns

    @property
    def index_columns(self) -> List[str]:
        """ Returns the managed index field names. """
        return [index_column for index_column, _ in self._index_map]

    @property
    def columns(self) -> List[str]:
        """ Return all the field names including index field names. """
        index_columns = self.index_columns
        return index_columns + [column for column in self._data_columns
                                if column not in index_columns]

    @property
    def index_map(self) -> List[IndexMap]:
        """ Return the managed index information. """
        return self._index_map

    @property
    def index_names(self) -> List[Optional[str]]:
        """ Return the managed index names. """
        return [index_name for _, index_name in self._index_map]

    def copy(self, data_columns: Optional[List[str]] = None,
             index_map: Optional[List[IndexMap]] = None) -> 'Metadata':
        """ Copy the metadata.

        :param data_columns: the new column field names. If None, then the original ones are used.
        :param index_map: the new index information. If None, then the original one is used.
        :return: the copied metadata.
        """
        if data_columns is None:
            data_columns = self._data_columns
        if index_map is None:
            index_map = self._index_map
        return Metadata(data_columns=data_columns.copy(), index_map=index_map.copy())

    @staticmethod
    def from_pandas(pdf: pd.DataFrame) -> 'Metadata':
        """ Create a metadata from pandas DataFrame.

        :param pdf: :class:`pd.DataFrame`
        :return: the created metadata
        """
        data_columns = [str(col) for col in pdf.columns]
        index = pdf.index

        index_map = []  # type: List[IndexMap]
        if isinstance(index, pd.MultiIndex):
            if index.names is None:
                index_map = [('__index_level_{}__'.format(i), None)
                             for i in range(len(index.levels))]
            else:
                index_map = [('__index_level_{}__'.format(i) if name is None else name, name)
                             for i, name in enumerate(index.names)]
        else:
            index_map = [(index.name
                          if index.name is not None else '__index_level_0__', index.name)]

        return Metadata(data_columns=data_columns, index_map=index_map)
