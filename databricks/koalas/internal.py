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
import re
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from itertools import accumulate

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_datetime64tz_dtype, is_list_like
from pyspark import sql as spark
from pyspark._globals import _NoValue, _NoValueType
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.types import BooleanType, DataType, StructField, StructType, LongType
try:
    from pyspark.sql.types import to_arrow_type
except ImportError:
    from pyspark.sql.pandas.types import to_arrow_type

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
if TYPE_CHECKING:
    # This is required in old Python 3.5 to prevent circular reference.
    from databricks.koalas.series import Series
from databricks.koalas.config import get_option
from databricks.koalas.typedef import infer_pd_series_spark_type, spark_type_to_pandas_dtype
from databricks.koalas.utils import (column_index_level, default_session, lazy_property,
                                     name_like_string, scol_for)


# A function to turn given numbers to Spark columns that represent Koalas index.
SPARK_INDEX_NAME_FORMAT = "__index_level_{}__".format
# A pattern to check if the name of a Spark column is a Koalas index name or not.
SPARK_INDEX_NAME_PATTERN = re.compile(r"__index_level_[0-9]+__")

NATURAL_ORDER_COLUMN_NAME = '__natural_order__'

HIDDEN_COLUMNS = set([NATURAL_ORDER_COLUMN_NAME])

IndexMap = Tuple[str, Optional[Tuple[str, ...]]]


class _InternalFrame(object):
    """
    The internal immutable DataFrame which manages Spark DataFrame and column names and index
    information.

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

    >>> kdf._internal.spark_internal_df.show()  # doctest: +NORMALIZE_WHITESPACE
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
    >>> internal.sdf.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|...|
    |                1|  2|  6| 10| 14| 18|...|
    |                2|  3|  7| 11| 15| 19|...|
    |                3|  4|  8| 12| 16| 20|...|
    +-----------------+---+---+---+---+---+-----------------+
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
    >>> internal.spark_internal_df.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+
    >>> internal.spark_df.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+
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

    >>> kdf1._internal.spark_internal_df.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+

    >>> internal = kdf1._internal
    >>> internal.sdf.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|...|
    |                1|  2|  6| 10| 14| 18|...|
    |                2|  3|  7| 11| 15| 19|...|
    |                3|  4|  8| 12| 16| 20|...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_columns
    ['B', 'C', 'D', 'E']
    >>> internal.index_columns
    ['A']
    >>> internal.columns
    ['A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [('A',)]
    >>> internal.index_map
    [('A', ('A',))]
    >>> internal.spark_internal_df.show()  # doctest: +NORMALIZE_WHITESPACE
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

    The `spark_df` will drop the index columns:

    >>> internal.spark_df.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+
    |  B|  C|  D|  E|
    +---+---+---+---+
    |  5|  9| 13| 17|
    |  6| 10| 14| 18|
    |  7| 11| 15| 19|
    |  8| 12| 16| 20|
    +---+---+---+---+

    but if `drop=False`, the columns will still remain in `spark_df`:

    >>> kdf.set_index("A", drop=False)._internal.spark_df.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+

    In case that index becomes a multi index as below:

    >>> kdf2 = kdf.set_index("A", append=True)
    >>> kdf2  # doctest: +NORMALIZE_WHITESPACE
         B   C   D   E
      A
    0 1  5   9  13  17
    1 2  6  10  14  18
    2 3  7  11  15  19
    3 4  8  12  16  20

    >>> kdf2._internal.spark_internal_df.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+

    >>> internal = kdf2._internal
    >>> internal.sdf.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|...|
    |                1|  2|  6| 10| 14| 18|...|
    |                2|  3|  7| 11| 15| 19|...|
    |                3|  4|  8| 12| 16| 20|...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_columns
    ['B', 'C', 'D', 'E']
    >>> internal.index_columns
    ['__index_level_0__', 'A']
    >>> internal.columns
    ['__index_level_0__', 'A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [None, ('A',)]
    >>> internal.index_map
    [('__index_level_0__', None), ('A', ('A',))]
    >>> internal.spark_internal_df.show()  # doctest: +NORMALIZE_WHITESPACE
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

    For multi-level columns, it also holds column_index

    >>> columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'),
    ...                                      ('Y', 'C'), ('Y', 'D')])
    >>> kdf3 = ks.DataFrame([
    ...     [1, 2, 3, 4],
    ...     [5, 6, 7, 8],
    ...     [9, 10, 11, 12],
    ...     [13, 14, 15, 16],
    ...     [17, 18, 19, 20]], columns = columns)
    >>> kdf3  # doctest: +NORMALIZE_WHITESPACE
        X       Y
        A   B   C   D
    0   1   2   3   4
    1   5   6   7   8
    2   9  10  11  12
    3  13  14  15  16
    4  17  18  19  20

    >>> internal = kdf3._internal
    >>> internal.sdf.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+------+------+------+------+-----------------+
    |__index_level_0__|(X, A)|(X, B)|(Y, C)|(Y, D)|__natural_order__|
    +-----------------+------+------+------+------+-----------------+
    |                0|     1|     2|     3|     4|...|
    |                1|     5|     6|     7|     8|...|
    |                2|     9|    10|    11|    12|...|
    |                3|    13|    14|    15|    16|...|
    |                4|    17|    18|    19|    20|...|
    +-----------------+------+------+------+------+-----------------+
    >>> internal.data_columns
    ['(X, A)', '(X, B)', '(Y, C)', '(Y, D)']
    >>> internal.column_index
    [('X', 'A'), ('X', 'B'), ('Y', 'C'), ('Y', 'D')]

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
    >>> internal.sdf.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|...|
    |                1|  2|  6| 10| 14| 18|...|
    |                2|  3|  7| 11| 15| 19|...|
    |                3|  4|  8| 12| 16| 20|...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.scol
    Column<b'B'>
    >>> internal.data_columns
    ['B']
    >>> internal.index_columns
    ['A']
    >>> internal.columns
    ['A', 'B']
    >>> internal.index_names
    [('A',)]
    >>> internal.index_map
    [('A', ('A',))]
    >>> internal.spark_internal_df.show()  # doctest: +NORMALIZE_WHITESPACE
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
                 index_map: Optional[List[IndexMap]],
                 column_index: Optional[List[Tuple[str, ...]]] = None,
                 column_scols: Optional[List[spark.Column]] = None,
                 column_index_names: Optional[List[str]] = None,
                 scol: Optional[spark.Column] = None) -> None:
        """
        Create a new internal immutable DataFrame to manage Spark DataFrame, column fields and
        index fields and names.

        :param sdf: Spark DataFrame to be managed.
        :param index_map: list of string pair
                           Each pair holds the index field name which exists in Spark fields,
                           and the index name.
        :param column_index: list of tuples with the same length
                              The multi-level values in the tuples.
        :param column_scols: list of Spark Column
                              Spark Columns to appear as columns. If scol is not None, this
                              argument is ignored, otherwise if this is None, calculated from sdf.
        :param column_index_names: Names for each of the index levels.
        :param scol: Spark Column to be managed.

        See the examples below to refer what each parameter means.

        >>> column_index = pd.MultiIndex.from_tuples(
        ...     [('a', 'x'), ('a', 'y'), ('b', 'z')], names=["column_index_a", "column_index_b"])
        >>> row_index = pd.MultiIndex.from_tuples(
        ...     [('foo', 'bar'), ('foo', 'bar'), ('zoo', 'bar')],
        ...     names=["row_index_a", "row_index_b"])
        >>> kdf = ks.DataFrame(
        ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=row_index, columns=column_index)
        >>> kdf.set_index(('a', 'x'), append=True, inplace=True)
        >>> kdf  # doctest: +NORMALIZE_WHITESPACE
        column_index_a                  a  b
        column_index_b                  y  z
        row_index_a row_index_b (a, x)
        foo         bar         1       2  3
                                4       5  6
        zoo         bar         7       8  9

        >>> internal = kdf[('a', 'y')]._internal

        >>> internal._sdf.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        +-----------+-----------+------+------+------+...
        |row_index_a|row_index_b|(a, x)|(a, y)|(b, z)|...
        +-----------+-----------+------+------+------+...
        |        foo|        bar|     1|     2|     3|...
        |        foo|        bar|     4|     5|     6|...
        |        zoo|        bar|     7|     8|     9|...
        +-----------+-----------+------+------+------+...

        >>> internal._index_map  # doctest: +NORMALIZE_WHITESPACE
        [('row_index_a', ('row_index_a',)), ('row_index_b', ('row_index_b',)),
         ('(a, x)', ('a', 'x'))]

        >>> internal._column_index
        [('a', 'y')]

        >>> internal._column_scols
        [Column<b'(a, y)'>]

        >>> list(internal._column_index_names)
        ['column_index_a', 'column_index_b']

        >>> internal._scol
        Column<b'(a, y)'>
        """
        assert isinstance(sdf, spark.DataFrame)

        if index_map is None:
            # Here is when Koalas DataFrame is created directly from Spark DataFrame.
            assert not any(SPARK_INDEX_NAME_PATTERN.match(name) for name in sdf.schema.names), \
                "Index columns should not appear in columns of the Spark DataFrame. Avoid " \
                "index colum names [%s]." % SPARK_INDEX_NAME_PATTERN

            # Create default index.
            index_map = [(SPARK_INDEX_NAME_FORMAT(0), None)]
            sdf = _InternalFrame.attach_default_index(sdf)

        if NATURAL_ORDER_COLUMN_NAME not in sdf.columns:
            sdf = sdf.withColumn(NATURAL_ORDER_COLUMN_NAME, F.monotonically_increasing_id())

        assert all(isinstance(index_field, str)
                   and (index_name is None or (isinstance(index_name, tuple)
                                               and all(isinstance(name, str)
                                                       for name in index_name)))
                   for index_field, index_name in index_map), index_map
        assert scol is None or isinstance(scol, spark.Column)
        assert column_scols is None or all(isinstance(scol, spark.Column) for scol in column_scols)

        self._sdf = sdf  # type: spark.DataFrame
        self._index_map = index_map  # type: List[IndexMap]
        self._scol = scol  # type: Optional[spark.Column]
        if scol is not None:
            self._column_scols = [scol]
        elif column_scols is None:
            index_columns = set(index_column for index_column, _ in self._index_map)
            self._column_scols = [scol_for(sdf, col) for col in sdf.columns
                                  if col not in index_columns and col not in HIDDEN_COLUMNS]
        else:
            self._column_scols = column_scols

        if scol is not None:
            assert column_index is not None and len(column_index) == 1, column_index
            assert all(idx is None or (isinstance(idx, tuple) and len(idx) > 0)
                       for idx in column_index), column_index
            self._column_index = column_index
        elif column_index is None:
            self._column_index = [(sdf.select(scol).columns[0],) for scol in self._column_scols]
        else:
            assert len(column_index) == len(self._column_scols), \
                (len(column_index), len(self._column_scols))
            assert all(isinstance(i, tuple) for i in column_index), column_index
            assert len(set(len(i) for i in column_index)) <= 1, column_index
            self._column_index = column_index

        if column_index_names is not None and not is_list_like(column_index_names):
            raise ValueError('Column_index_names should be list-like or None for a MultiIndex')

        if isinstance(column_index_names, list):
            if all(name is None for name in column_index_names):
                self._column_index_names = None
            else:
                self._column_index_names = column_index_names
        else:
            self._column_index_names = column_index_names

    @staticmethod
    def attach_default_index(sdf, default_index_type=None):
        """
        This method attaches a default index to Spark DataFrame. Spark does not have the index
        notion so corresponding column should be generated.
        There are several types of default index can be configured by `compute.default_index_type`.
        """
        if default_index_type is None:
            default_index_type = get_option("compute.default_index_type")
        scols = [scol_for(sdf, column) for column in sdf.columns]
        if default_index_type == "sequence":
            sequential_index = F.row_number().over(
                Window.orderBy(F.monotonically_increasing_id())) - 1
            return sdf.select(sequential_index.alias(SPARK_INDEX_NAME_FORMAT(0)), *scols)
        elif default_index_type == "distributed-sequence":
            # 1. Calculates counts per each partition ID. `counts` here is, for instance,
            #     {
            #         1: 83,
            #         6: 83,
            #         3: 83,
            #         ...
            #     }
            sdf = sdf.withColumn("__spark_partition_id", F.spark_partition_id())
            counts = map(lambda x: (x["key"], x["count"]),
                         sdf.groupby(sdf['__spark_partition_id'].alias("key")).count().collect())

            # 2. Calculates cumulative sum in an order of partition id.
            #     Note that it does not matter if partition id guarantees its order or not.
            #     We just need a one-by-one sequential id.

            # sort by partition key.
            sorted_counts = sorted(counts, key=lambda x: x[0])
            # get cumulative sum in an order of partition key.
            cumulative_counts = [0] + list(accumulate(map(lambda count: count[1], sorted_counts)))
            # zip it with partition key.
            sums = dict(zip(map(lambda count: count[0], sorted_counts), cumulative_counts))

            # 3. Attach offset for each partition.
            @pandas_udf(LongType(), PandasUDFType.SCALAR)
            def offset(id):
                current_partition_offset = sums[id.iloc[0]]
                return pd.Series(current_partition_offset).repeat(len(id))

            sdf = sdf.withColumn('__offset__', offset('__spark_partition_id'))

            # 4. Calculate row_number in each partition.
            w = Window.partitionBy('__spark_partition_id').orderBy(F.monotonically_increasing_id())
            row_number = F.row_number().over(w)
            sdf = sdf.withColumn('__row_number__', row_number)

            # 5. Calcuate the index.
            return sdf.select(
                F.expr('__offset__ + __row_number__ - 1').alias(SPARK_INDEX_NAME_FORMAT(0)), *scols)
        elif default_index_type == "distributed":
            return sdf.select(
                F.monotonically_increasing_id().alias(SPARK_INDEX_NAME_FORMAT(0)), *scols)
        else:
            raise ValueError("'compute.default_index_type' should be one of 'sequence',"
                             " 'distributed-sequence' and 'distributed'")

    @lazy_property
    def _column_index_to_name(self) -> Dict[Tuple[str, ...], str]:
        return dict(zip(self.column_index, self.data_columns))

    def column_name_for(self, column_name_or_index: Union[str, Tuple[str, ...]]) -> str:
        """ Return the actual Spark column name for the given column name or index. """
        if column_name_or_index in self._column_index_to_name:
            return self._column_index_to_name[column_name_or_index]
        else:
            if not isinstance(column_name_or_index, str):
                raise KeyError(name_like_string(column_name_or_index))
            return column_name_or_index

    @lazy_property
    def _column_index_to_scol(self) -> Dict[Tuple[str, ...], spark.Column]:
        return dict(zip(self.column_index, self.column_scols))

    def scol_for(self, column_name_or_index: Union[str, Tuple[str, ...]]):
        """ Return Spark Column for the given column name or index. """
        if column_name_or_index in self._column_index_to_scol:
            return self._column_index_to_scol[column_name_or_index]
        else:
            return scol_for(self._sdf, self.column_name_for(column_name_or_index))

    def spark_type_for(self, column_name_or_index: Union[str, Tuple[str, ...]]) -> DataType:
        """ Return DataType for the given column name or index. """
        return self._sdf.select(self.scol_for(column_name_or_index)).schema[0].dataType

    @property
    def sdf(self) -> spark.DataFrame:
        """ Return the managed Spark DataFrame. """
        return self._sdf

    @lazy_property
    def data_columns(self) -> List[str]:
        """ Return the managed column field names. """
        return self.sdf.select(self.column_scols).columns

    @property
    def column_scols(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed data columns. """
        return self._column_scols

    @lazy_property
    def index_columns(self) -> List[str]:
        """ Return the managed index field names. """
        return [index_column for index_column, _ in self._index_map]

    @lazy_property
    def index_scols(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed index columns. """
        return [self.scol_for(column) for column in self.index_columns]

    @lazy_property
    def columns(self) -> List[str]:
        """ Return all the field names including index field names. """
        index_columns = set(self.index_columns)
        return self.index_columns + [column for column in self.data_columns
                                     if column not in index_columns]

    @lazy_property
    def scols(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed columns including index columns. """
        return [self.scol_for(column) for column in self.columns]

    @property
    def index_map(self) -> List[IndexMap]:
        """ Return the managed index information. """
        assert len(self._index_map) > 0
        return self._index_map

    @lazy_property
    def index_names(self) -> List[Optional[Tuple[str, ...]]]:
        """ Return the managed index names. """
        return [index_name for _, index_name in self.index_map]

    @property
    def scol(self) -> Optional[spark.Column]:
        """ Return the managed Spark Column. """
        return self._scol

    @property
    def column_index(self) -> List[Tuple[str, ...]]:
        """ Return the managed column index. """
        return self._column_index

    @lazy_property
    def column_index_level(self) -> int:
        """ Return the level of the column index. """
        return column_index_level(self._column_index)

    @property
    def column_index_names(self) -> Optional[List[str]]:
        """ Return names of the index levels. """
        return self._column_index_names

    @lazy_property
    def spark_internal_df(self) -> spark.DataFrame:
        """
        Return as Spark DataFrame. This contains index columns as well
        and should be only used for internal purposes.
        """
        index_columns = set(self.index_columns)
        data_columns = []
        for i, (column, idx) in enumerate(zip(self.data_columns, self.column_index)):
            if column not in index_columns:
                scol = self.scol_for(idx)
                name = str(i) if idx is None else name_like_string(idx)
                if column != name:
                    scol = scol.alias(name)
                data_columns.append(scol)
        return self._sdf.select(self.index_scols + data_columns)

    @lazy_property
    def spark_df(self) -> spark.DataFrame:
        """ Return as Spark DataFrame. """
        data_columns = []
        for i, (column, idx) in enumerate(zip(self.data_columns, self.column_index)):
            scol = self.scol_for(idx)
            name = str(i) if idx is None else name_like_string(idx)
            if column != name:
                scol = scol.alias(name)
            data_columns.append(scol)
        return self._sdf.select(data_columns)

    @lazy_property
    def pandas_df(self):
        """ Return as pandas DataFrame. """
        sdf = self.spark_internal_df
        pdf = sdf.toPandas()
        if len(pdf) == 0 and len(sdf.schema) > 0:
            pdf = pdf.astype({field.name: spark_type_to_pandas_dtype(field.dataType)
                              for field in sdf.schema})

        index_columns = self.index_columns
        if len(index_columns) > 0:
            append = False
            for index_field in index_columns:
                drop = index_field not in self.data_columns
                pdf = pdf.set_index(index_field, drop=drop, append=append)
                append = True
            pdf = pdf[[col if col in index_columns
                       else str(i) if idx is None else name_like_string(idx)
                       for i, (col, idx) in enumerate(zip(self.data_columns, self.column_index))]]

        if self.column_index_level > 1:
            pdf.columns = pd.MultiIndex.from_tuples(self._column_index)
        else:
            pdf.columns = [None if idx is None else idx[0] for idx in self._column_index]
        if self._column_index_names is not None:
            pdf.columns.names = self._column_index_names

        index_names = self.index_names
        if len(index_names) > 0:
            pdf.index.names = [name if name is None or len(name) > 1 else name[0]
                               for name in index_names]
        return pdf

    def with_new_columns(self, scols_or_ksers: List[Union[spark.Column, 'Series']],
                         column_index: Optional[List[Tuple[str, ...]]] = None,
                         keep_order: bool = True) -> '_InternalFrame':
        """ Copy the immutable DataFrame with the updates by the specified Spark Columns or Series.

        :param scols_or_ksers: the new Spark Columns or Series.
        :param column_index: the new column index.
            If None, the its column_index is used when the corresponding `scols_or_ksers` is Series,
            otherwise the original one is used.
        :return: the copied immutable DataFrame.
        """
        from databricks.koalas.series import Series

        if column_index is None:
            if all(isinstance(scol_or_kser, Series) for scol_or_kser in scols_or_ksers):
                column_index = [kser._internal.column_index[0] for kser in scols_or_ksers]
            else:
                assert len(scols_or_ksers) == len(self.column_index), \
                    (len(scols_or_ksers), len(self.column_index))
                column_index = []
                for scol_or_kser, idx in zip(scols_or_ksers, self.column_index):
                    if isinstance(scol_or_kser, Series):
                        column_index.append(scol_or_kser._internal.column_index[0])
                    else:
                        column_index.append(idx)
        else:
            assert len(scols_or_ksers) == len(column_index), \
                (len(scols_or_ksers), len(column_index))

        column_scols = []
        for scol_or_kser, idx in zip(scols_or_ksers, column_index):
            if isinstance(scol_or_kser, Series):
                scol = scol_or_kser._internal.scol
            else:
                scol = scol_or_kser
            column_scols.append(scol.alias(name_like_string(idx)))  # type: ignore

        hidden_columns = []
        if keep_order:
            hidden_columns.append(NATURAL_ORDER_COLUMN_NAME)

        sdf = self._sdf.select(self.index_scols + column_scols + hidden_columns)

        return self.copy(
            sdf=sdf,
            column_index=column_index,
            column_scols=[scol_for(sdf, name_like_string(idx)) for idx in column_index],
            scol=None)

    def with_filter(self, pred: Union[spark.Column, 'Series']):
        """ Copy the immutable DataFrame with the updates by the predicate.

        :param pred: the predicate to filter.
        :return: the copied immutable DataFrame.
        """
        from databricks.koalas.series import Series
        if isinstance(pred, Series):
            assert isinstance(pred.spark_type, BooleanType), pred.spark_type
            pred = pred._scol
        else:
            spark_type = self._sdf.select(pred).schema[0].dataType
            assert isinstance(spark_type, BooleanType), spark_type

        return self.copy(sdf=self._sdf.drop(NATURAL_ORDER_COLUMN_NAME).filter(pred))

    def copy(self, sdf: Union[spark.DataFrame, _NoValueType] = _NoValue,
             index_map: Union[List[IndexMap], _NoValueType] = _NoValue,
             column_index: Union[List[Tuple[str, ...]], _NoValueType] = _NoValue,
             column_scols: Union[List[spark.Column], _NoValueType] = _NoValue,
             column_index_names: Optional[Union[List[str], _NoValueType]] = _NoValue,
             scol: Union[spark.Column, _NoValueType] = _NoValue) -> '_InternalFrame':
        """ Copy the immutable DataFrame.

        :param sdf: the new Spark DataFrame. If None, then the original one is used.
        :param index_map: the new index information. If None, then the original one is used.
        :param column_index: the new column index.
        :param column_scols: the new Spark Columns. If None, then the original ones are used.
        :param column_index_names: the new names of the index levels.
        :param scol: the new Spark Column. If None, then the original one is used.
        :return: the copied immutable DataFrame.
        """
        if sdf is _NoValue:
            sdf = self._sdf
        if index_map is _NoValue:
            index_map = self._index_map
        if column_index is _NoValue:
            column_index = self._column_index
        if column_scols is _NoValue:
            column_scols = self._column_scols
        if column_index_names is _NoValue:
            column_index_names = self._column_index_names
        if scol is _NoValue:
            scol = self._scol
        return _InternalFrame(sdf, index_map=index_map, column_index=column_index,
                              column_scols=column_scols, column_index_names=column_index_names,
                              scol=scol)

    @staticmethod
    def from_pandas(pdf: pd.DataFrame) -> '_InternalFrame':
        """ Create an immutable DataFrame from pandas DataFrame.

        :param pdf: :class:`pd.DataFrame`
        :return: the created immutable DataFrame
        """
        columns = pdf.columns
        data_columns = [name_like_string(col) for col in columns]
        if isinstance(columns, pd.MultiIndex):
            column_index = columns.tolist()
        else:
            column_index = None
        column_index_names = columns.names

        index = pdf.index

        index_map = []  # type: List[IndexMap]
        if isinstance(index, pd.MultiIndex):
            if index.names is None:
                index_map = [(SPARK_INDEX_NAME_FORMAT(i), None)
                             for i in range(len(index.levels))]
            else:
                index_map = [
                    (SPARK_INDEX_NAME_FORMAT(i) if name is None else name_like_string(name),
                     name if name is None or isinstance(name, tuple) else (name,))
                    for i, name in enumerate(index.names)]
        else:
            name = index.name
            index_map = [(name_like_string(name)
                          if name is not None else SPARK_INDEX_NAME_FORMAT(0),
                          name if name is None or isinstance(name, tuple) else (name,))]

        index_columns = [index_column for index_column, _ in index_map]

        reset_index = pdf.reset_index()
        reset_index.columns = index_columns + data_columns
        schema = StructType([StructField(name_like_string(name), infer_pd_series_spark_type(col),
                                         nullable=bool(col.isnull().any()))
                             for name, col in reset_index.iteritems()])
        for name, col in reset_index.iteritems():
            dt = col.dtype
            if is_datetime64_dtype(dt) or is_datetime64tz_dtype(dt):
                continue
            reset_index[name] = col.replace({np.nan: None})
        sdf = default_session().createDataFrame(reset_index, schema=schema)
        return _InternalFrame(sdf=sdf,
                              index_map=index_map,
                              column_index=column_index,
                              column_scols=[scol_for(sdf, col) for col in data_columns],
                              column_index_names=column_index_names)
