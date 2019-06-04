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
An internal immutable DataFrame with some metadata to manage indexes.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_datetime64tz_dtype
from pyspark import sql as spark
from pyspark.sql.types import StructField, StructType, to_arrow_type

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.typedef import infer_pd_series_spark_type
from databricks.koalas.utils import default_session, lazy_property


IndexMap = Tuple[str, Optional[str]]


class _InternalFrame(object):
    """
    The internal immutable DataFrame which manages Spark DataFrame and column names and index
    information.

    :ivar _sdf: Spark DataFrame
    :ivar _index_map: list of pair holding the Spark field names for indexes,
                       and the index name to be seen in Koalas DataFrame.
    :ivar _scol: Spark Column
    :ivar _data_columns: list of the Spark field names to be seen as columns in Koalas DataFrame.

    .. note:: this is an internal class. It is not supposed to be exposed to users and users
        should not directly access to it.

    The internal immutable DataFrame represents the index information for a DataFrame it belongs to.
    For instance, if we have a Koalas DataFrame as below, Pandas DataFrame does not store the index
    as columns.

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

    * `sdf` represents the internal Spark DataFrame

    * `data_columns` represents non-indexing columns

    * `index_columns` represents internal index columns

    * `columns` represents all columns

    * `index_names` represents the external index name

    * `index_map` is zipped pairs of `index_columns` and `index_names`

    * `spark_df` represents Spark DataFrame derived by the metadata

    * `pandas_df` represents pandas DataFrame derived by the metadata

    >>> internal = kdf._internal
    >>> internal.sdf.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+
    >>> internal.data_columns
    ['A', 'B', 'C', 'D', 'E']
    >>> internal.index_columns
    ['__index_level_0__']
    >>> internal.columns
    ['__index_level_0__', 'A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [None]
    >>> internal.index_map
    [('__index_level_0__', None)]
    >>> internal.spark_df.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+
    >>> internal.pandas_df
       A  B   C   D   E
    0  1  5   9  13  17
    1  2  6  10  14  18
    2  3  7  11  15  19
    3  4  8  12  16  20

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

    >>> internal = kdf1._internal
    >>> internal.sdf.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+
    >>> internal.data_columns
    ['B', 'C', 'D', 'E']
    >>> internal.index_columns
    ['A']
    >>> internal.columns
    ['A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    ['A']
    >>> internal.index_map
    [('A', 'A')]
    >>> internal.spark_df.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+
    >>> internal.pandas_df  # doctest: +NORMALIZE_WHITESPACE
       B   C   D   E
    A
    1  5   9  13  17
    2  6  10  14  18
    3  7  11  15  19
    4  8  12  16  20

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

    >>> internal = kdf2._internal
    >>> internal.sdf.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+
    >>> internal.data_columns
    ['B', 'C', 'D', 'E']
    >>> internal.index_columns
    ['__index_level_0__', 'A']
    >>> internal.columns
    ['__index_level_0__', 'A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [None, 'A']
    >>> internal.index_map
    [('__index_level_0__', None), ('A', 'A')]
    >>> internal.spark_df.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+
    >>> internal.pandas_df  # doctest: +NORMALIZE_WHITESPACE
         B   C   D   E
      A
    0 1  5   9  13  17
    1 2  6  10  14  18
    2 3  7  11  15  19
    3 4  8  12  16  20

    For series, it also holds scol to represent the column.

    >>> kseries = kdf1.B
    >>> kseries
    A
    1    5
    2    6
    3    7
    4    8
    Name: B, dtype: int64

    >>> internal = kseries._internal
    >>> internal.sdf.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+
    >>> internal.scol
    Column<b'B'>
    >>> internal.data_columns
    ['B']
    >>> internal.index_columns
    ['A']
    >>> internal.columns
    ['A', 'B']
    >>> internal.index_names
    ['A']
    >>> internal.index_map
    [('A', 'A')]
    >>> internal.spark_df.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+
    |  A|  B|
    +---+---+
    |  1|  5|
    |  2|  6|
    |  3|  7|
    |  4|  8|
    +---+---+
    >>> internal.pandas_df  # doctest: +NORMALIZE_WHITESPACE
       B
    A
    1  5
    2  6
    3  7
    4  8
    """

    def __init__(self, sdf: spark.DataFrame,
                 index_map: Optional[List[IndexMap]] = None,
                 scol: Optional[spark.Column] = None,
                 data_columns: Optional[List[str]] = None) -> None:
        """
        Create a new internal immutable DataFrame to manage Spark DataFrame, column fields and
        index fields and names.

        :param sdf: Spark DataFrame to be managed.
        :param index_map: list of string pair
                           Each pair holds the index field name which exists in Spark fields,
                           and the index name.
        :param scol: Spark Column to be managed.
        :param data_columns: list of string
                              Field names to appear as columns. If scol is not None, this
                              argument is ignored, otherwise if this is None, calculated from sdf.
        """
        assert isinstance(sdf, spark.DataFrame)
        assert index_map is None \
            or all(isinstance(index_field, str)
                   and (index_name is None or isinstance(index_name, str))
                   for index_field, index_name in index_map)
        assert scol is None or isinstance(scol, spark.Column)
        assert data_columns is None or all(isinstance(col, str) for col in data_columns)

        self._sdf = sdf  # type: spark.DataFrame
        self._index_map = (index_map if index_map is not None else [])  # type: List[IndexMap]
        self._scol = scol  # type: Optional[spark.Column]
        if scol is not None:
            self._data_columns = sdf.select(scol).columns
        elif data_columns is None:
            index_columns = set(index_column for index_column, _ in self._index_map)
            self._data_columns = [column for column in sdf.columns if column not in index_columns]
        else:
            self._data_columns = data_columns

    @property
    def sdf(self) -> spark.DataFrame:
        """ Return the managed Spark DataFrame. """
        return self._sdf

    @property
    def data_columns(self) -> List[str]:
        """ Return the managed column field names. """
        return self._data_columns

    @lazy_property
    def index_columns(self) -> List[str]:
        """ Return the managed index field names. """
        return [index_column for index_column, _ in self._index_map]

    @lazy_property
    def columns(self) -> List[str]:
        """ Return all the field names including index field names. """
        index_columns = set(self.index_columns)
        return self.index_columns + [column for column in self._data_columns
                                     if column not in index_columns]

    @property
    def index_map(self) -> List[IndexMap]:
        """ Return the managed index information. """
        return self._index_map

    @lazy_property
    def index_names(self) -> List[Optional[str]]:
        """ Return the managed index names. """
        return [index_name for _, index_name in self.index_map]

    @property
    def scol(self) -> Optional[spark.Column]:
        """ Return the managed Spark Column. """
        return self._scol

    @lazy_property
    def spark_df(self) -> spark.DataFrame:
        """ Return as Spark DataFrame. """
        if self._scol is None:
            sdf = self._sdf
        else:
            sdf = self._sdf.select(self.index_columns + [self._scol])
        return sdf.select(['`{}`'.format(name) for name in self.columns])

    @lazy_property
    def pandas_df(self):
        """ Return as pandas DataFrame. """
        sdf = self.spark_df
        pdf = sdf.toPandas()
        if len(pdf) == 0 and len(sdf.schema) > 0:
            pdf = pdf.astype({field.name: to_arrow_type(field.dataType).to_pandas_dtype()
                              for field in sdf.schema})

        index_columns = self.index_columns
        if len(index_columns) > 0:
            append = False
            for index_field in index_columns:
                drop = index_field not in self.data_columns
                pdf = pdf.set_index(index_field, drop=drop, append=append)
                append = True
            pdf = pdf[self.data_columns]

        index_names = self.index_names
        if len(index_names) > 0:
            if isinstance(pdf.index, pd.MultiIndex):
                pdf.index.names = index_names
            else:
                pdf.index.name = index_names[0]
        return pdf

    def copy(self, sdf: Optional[spark.DataFrame] = None,
             index_map: Optional[List[IndexMap]] = None,
             scol: Optional[spark.Column] = None,
             data_columns: Optional[List[str]] = None) -> '_InternalFrame':
        """ Copy the immutable DataFrame.

        :param sdf: the new Spark DataFrame. If None, then the original one is used.
        :param index_map: the new index information. If None, then the original one is used.
        :param scol: the new Spark Column. If None, then the original one is used.
        :param data_columns: the new column field names. If None, then the original ones are used.
        :return: the copied immutable DataFrame.
        """
        if sdf is None:
            sdf = self._sdf
        if index_map is None:
            index_map = self._index_map
        if scol is None:
            scol = self._scol
        if data_columns is None:
            data_columns = self._data_columns
        return _InternalFrame(sdf, index_map=index_map.copy(), scol=scol, data_columns=data_columns)

    @staticmethod
    def from_pandas(pdf: pd.DataFrame) -> '_InternalFrame':
        """ Create an immutable DataFrame from pandas DataFrame.

        :param pdf: :class:`pd.DataFrame`
        :return: the created immutable DataFrame
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

        index_columns = [index_column for index_column, _ in index_map]

        reset_index = pdf.reset_index()
        reset_index.columns = index_columns + data_columns
        schema = StructType([StructField(name, infer_pd_series_spark_type(col),
                                         nullable=bool(col.isnull().any()))
                             for name, col in reset_index.iteritems()])
        for name, col in reset_index.iteritems():
            dt = col.dtype
            if is_datetime64_dtype(dt) or is_datetime64tz_dtype(dt):
                continue
            reset_index[name] = col.replace({np.nan: None})
        sdf = default_session().createDataFrame(reset_index, schema=schema)
        return _InternalFrame(sdf=sdf, index_map=index_map, data_columns=data_columns)
