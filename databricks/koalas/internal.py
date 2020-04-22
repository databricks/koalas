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
from collections import OrderedDict

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
from databricks.koalas.utils import (
    column_labels_level,
    default_session,
    lazy_property,
    name_like_string,
    scol_for,
    verify_temp_column_name,
)


# A function to turn given numbers to Spark columns that represent Koalas index.
SPARK_INDEX_NAME_FORMAT = "__index_level_{}__".format
SPARK_DEFAULT_INDEX_NAME = SPARK_INDEX_NAME_FORMAT(0)
# A pattern to check if the name of a Spark column is a Koalas index name or not.
SPARK_INDEX_NAME_PATTERN = re.compile(r"__index_level_[0-9]+__")

NATURAL_ORDER_COLUMN_NAME = "__natural_order__"

HIDDEN_COLUMNS = {NATURAL_ORDER_COLUMN_NAME}


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

    >>> kdf._internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
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

    * `spark_frame` represents the internal Spark DataFrame

    * `data_spark_column_names` represents non-indexing Spark column names

    * `data_spark_columns` represents non-indexing Spark columns

    * `index_spark_column_names` represents internal index Spark column names

    * `index_spark_columns` represents internal index Spark columns

    * `spark_column_names` represents all columns

    * `index_names` represents the external index name as a label

    * `index_map` is zipped pairs of `index_spark_column_names` and `index_names`

    * `to_external_spark_frame` represents Spark DataFrame derived by the metadata.

    * `to_internal_spark_frame` represents Spark DataFrame derived by the metadata. Includes index.

    * `to_pandas_frame` represents pandas DataFrame derived by the metadata

    >>> internal = kdf._internal
    >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|...|
    |                1|  2|  6| 10| 14| 18|...|
    |                2|  3|  7| 11| 15| 19|...|
    |                3|  4|  8| 12| 16| 20|...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_spark_column_names
    ['A', 'B', 'C', 'D', 'E']
    >>> internal.index_spark_column_names
    ['__index_level_0__']
    >>> internal.spark_column_names
    ['__index_level_0__', 'A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [None]
    >>> internal.index_map
    OrderedDict([('__index_level_0__', None)])
    >>> internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+
    >>> internal.to_external_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+
    >>> internal.to_pandas_frame
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

    >>> kdf1._internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+

    >>> internal = kdf1._internal
    >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|...|
    |                1|  2|  6| 10| 14| 18|...|
    |                2|  3|  7| 11| 15| 19|...|
    |                3|  4|  8| 12| 16| 20|...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_spark_column_names
    ['B', 'C', 'D', 'E']
    >>> internal.index_spark_column_names
    ['A']
    >>> internal.spark_column_names
    ['A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [('A',)]
    >>> internal.index_map
    OrderedDict([('A', ('A',))])
    >>> internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+---+
    |  A|  B|  C|  D|  E|
    +---+---+---+---+---+
    |  1|  5|  9| 13| 17|
    |  2|  6| 10| 14| 18|
    |  3|  7| 11| 15| 19|
    |  4|  8| 12| 16| 20|
    +---+---+---+---+---+
    >>> internal.to_pandas_frame  # doctest: +NORMALIZE_WHITESPACE
       B   C   D   E
    A
    1  5   9  13  17
    2  6  10  14  18
    3  7  11  15  19
    4  8  12  16  20

    The `to_external_spark_frame` will drop the index columns:

    >>> internal.to_external_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+---+---+
    |  B|  C|  D|  E|
    +---+---+---+---+
    |  5|  9| 13| 17|
    |  6| 10| 14| 18|
    |  7| 11| 15| 19|
    |  8| 12| 16| 20|
    +---+---+---+---+

    but if `drop=False`, the columns will still remain in `to_external_spark_frame`:

    >>> kdf.set_index("A", drop=False)._internal.to_external_spark_frame.show()
    ... # doctest: +NORMALIZE_WHITESPACE
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

    >>> kdf2._internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+

    >>> internal = kdf2._internal
    >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|...|
    |                1|  2|  6| 10| 14| 18|...|
    |                2|  3|  7| 11| 15| 19|...|
    |                3|  4|  8| 12| 16| 20|...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_spark_column_names
    ['B', 'C', 'D', 'E']
    >>> internal.index_spark_column_names
    ['__index_level_0__', 'A']
    >>> internal.spark_column_names
    ['__index_level_0__', 'A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [None, ('A',)]
    >>> internal.index_map
    OrderedDict([('__index_level_0__', None), ('A', ('A',))])
    >>> internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+
    >>> internal.to_pandas_frame  # doctest: +NORMALIZE_WHITESPACE
         B   C   D   E
      A
    0 1  5   9  13  17
    1 2  6  10  14  18
    2 3  7  11  15  19
    3 4  8  12  16  20

    For multi-level columns, it also holds column_labels

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
    >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+------+------+------+------+-----------------+
    |__index_level_0__|(X, A)|(X, B)|(Y, C)|(Y, D)|__natural_order__|
    +-----------------+------+------+------+------+-----------------+
    |                0|     1|     2|     3|     4|...|
    |                1|     5|     6|     7|     8|...|
    |                2|     9|    10|    11|    12|...|
    |                3|    13|    14|    15|    16|...|
    |                4|    17|    18|    19|    20|...|
    +-----------------+------+------+------+------+-----------------+
    >>> internal.data_spark_column_names
    ['(X, A)', '(X, B)', '(Y, C)', '(Y, D)']
    >>> internal.column_labels
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
    >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|...|
    |                1|  2|  6| 10| 14| 18|...|
    |                2|  3|  7| 11| 15| 19|...|
    |                3|  4|  8| 12| 16| 20|...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.spark_column
    Column<b'B'>
    >>> internal.data_spark_column_names
    ['B']
    >>> internal.index_spark_column_names
    ['A']
    >>> internal.spark_column_names
    ['A', 'B']
    >>> internal.index_names
    [('A',)]
    >>> internal.index_map
    OrderedDict([('A', ('A',))])
    >>> internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +---+---+
    |  A|  B|
    +---+---+
    |  1|  5|
    |  2|  6|
    |  3|  7|
    |  4|  8|
    +---+---+
    >>> internal.to_pandas_frame  # doctest: +NORMALIZE_WHITESPACE
       B
    A
    1  5
    2  6
    3  7
    4  8
    """

    def __init__(
        self,
        spark_frame: spark.DataFrame,
        index_map: Optional[Dict[str, Optional[Tuple[str, ...]]]],
        column_labels: Optional[List[Tuple[str, ...]]] = None,
        data_spark_columns: Optional[List[spark.Column]] = None,
        column_label_names: Optional[List[str]] = None,
        spark_column: Optional[spark.Column] = None,
    ) -> None:
        """
        Create a new internal immutable DataFrame to manage Spark DataFrame, column fields and
        index fields and names.

        :param spark_frame: Spark DataFrame to be managed.
        :param index_map: dictionary of string pairs
                           Each pair holds the index field name which exists in Spark fields,
                           and the index name.
        :param column_labels: list of tuples with the same length
                              The multi-level values in the tuples.
        :param data_spark_columns: list of Spark Column
                                   Spark Columns to appear as columns. If spark_column is not None,
                                   this argument is ignored, otherwise if this is None, calculated
                                   from spark_frame.
        :param column_label_names: Names for each of the index levels.
        :param spark_column: Spark Column to be managed.

        See the examples below to refer what each parameter means.

        >>> column_labels = pd.MultiIndex.from_tuples(
        ...     [('a', 'x'), ('a', 'y'), ('b', 'z')], names=["column_labels_a", "column_labels_b"])
        >>> row_index = pd.MultiIndex.from_tuples(
        ...     [('foo', 'bar'), ('foo', 'bar'), ('zoo', 'bar')],
        ...     names=["row_index_a", "row_index_b"])
        >>> kdf = ks.DataFrame(
        ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=row_index, columns=column_labels)
        >>> kdf.set_index(('a', 'x'), append=True, inplace=True)
        >>> kdf  # doctest: +NORMALIZE_WHITESPACE
        column_labels_a                  a  b
        column_labels_b                  y  z
        row_index_a row_index_b (a, x)
        foo         bar         1       2  3
                                4       5  6
        zoo         bar         7       8  9

        >>> internal = kdf[('a', 'y')]._internal

        >>> internal._sdf.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        +-----------------+-----------------+------+------+------+...
        |__index_level_0__|__index_level_1__|(a, x)|(a, y)|(b, z)|...
        +-----------------+-----------------+------+------+------+...
        |              foo|              bar|     1|     2|     3|...
        |              foo|              bar|     4|     5|     6|...
        |              zoo|              bar|     7|     8|     9|...
        +-----------------+-----------------+------+------+------+...

        >>> internal._index_map  # doctest: +NORMALIZE_WHITESPACE
        OrderedDict([('__index_level_0__', ('row_index_a',)),
         ('__index_level_1__', ('row_index_b',)),
         ('(a, x)', ('a', 'x'))])

        >>> internal._column_labels
        [('a', 'y')]

        >>> internal._data_spark_columns
        [Column<b'(a, y)'>]

        >>> list(internal._column_label_names)
        ['column_labels_a', 'column_labels_b']

        >>> internal.spark_column
        Column<b'(a, y)'>
        """

        assert isinstance(spark_frame, spark.DataFrame)
        assert not spark_frame.isStreaming, "Koalas does not support Structured Streaming."

        if index_map is None:
            assert not any(SPARK_INDEX_NAME_PATTERN.match(name) for name in spark_frame.columns), (
                "Index columns should not appear in columns of the Spark DataFrame. Avoid "
                "index column names [%s]." % SPARK_INDEX_NAME_PATTERN
            )

            # Create default index.
            spark_frame = _InternalFrame.attach_default_index(spark_frame)
            index_map = OrderedDict({SPARK_DEFAULT_INDEX_NAME: None})

        if NATURAL_ORDER_COLUMN_NAME not in spark_frame.columns:
            spark_frame = spark_frame.withColumn(
                NATURAL_ORDER_COLUMN_NAME, F.monotonically_increasing_id()
            )

        assert isinstance(index_map, OrderedDict), index_map
        assert all(
            isinstance(index_field, str)
            and (
                index_name is None
                or (
                    isinstance(index_name, tuple)
                    and all(isinstance(name, str) for name in index_name)
                )
            )
            for index_field, index_name in index_map.items()
        ), index_map
        assert spark_column is None or isinstance(spark_column, spark.Column)
        assert data_spark_columns is None or all(
            isinstance(scol, spark.Column) for scol in data_spark_columns
        )

        self._sdf = spark_frame  # type: spark.DataFrame
        self._index_map = index_map  # type: Dict[str, Optional[Tuple[str, ...]]]
        self._spark_column = spark_column  # type: Optional[spark.Column]
        if spark_column is not None:
            self._data_spark_columns = [spark_column]
        elif data_spark_columns is None:
            index_columns = set(index_column for index_column in self._index_map)
            self._data_spark_columns = [
                scol_for(spark_frame, col)
                for col in spark_frame.columns
                if col not in index_columns and col not in HIDDEN_COLUMNS
            ]
        else:
            self._data_spark_columns = data_spark_columns

        if spark_column is not None:
            assert column_labels is not None and len(column_labels) == 1, column_labels
            assert all(
                label is None or (isinstance(label, tuple) and len(label) > 0)
                for label in column_labels
            ), column_labels
            self._column_labels = column_labels
        elif column_labels is None:
            self._column_labels = [
                (spark_frame.select(scol).columns[0],) for scol in self._data_spark_columns
            ]
        else:
            assert len(column_labels) == len(self._data_spark_columns), (
                len(column_labels),
                len(self._data_spark_columns),
            )
            assert all(isinstance(i, tuple) for i in column_labels), column_labels
            assert len(set(len(i) for i in column_labels)) <= 1, column_labels
            self._column_labels = column_labels

        if column_label_names is not None and not is_list_like(column_label_names):
            raise ValueError("Column_index_names should be list-like or None for a MultiIndex")

        if isinstance(column_label_names, list):
            if all(name is None for name in column_label_names):
                self._column_label_names = None
            else:
                self._column_label_names = column_label_names
        else:
            self._column_label_names = column_label_names

    @staticmethod
    def attach_default_index(sdf, default_index_type=None):
        """
        This method attaches a default index to Spark DataFrame. Spark does not have the index
        notion so corresponding column should be generated.
        There are several types of default index can be configured by `compute.default_index_type`.

        >>> spark_frame = ks.range(10).to_spark()
        >>> spark_frame
        DataFrame[id: bigint]

        It adds the default index column '__index_level_0__'.

        >>> spark_frame = _InternalFrame.attach_default_index(spark_frame)
        >>> spark_frame
        DataFrame[__index_level_0__: int, id: bigint]

        It throws an exception if the given column name already exists.

        >>> _InternalFrame.attach_default_index(spark_frame)
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
          ...
        AssertionError: '__index_level_0__' already exists...
        """
        index_column = SPARK_DEFAULT_INDEX_NAME
        assert (
            index_column not in sdf.columns
        ), "'%s' already exists in the Spark column names '%s'" % (index_column, sdf.columns)

        if default_index_type is None:
            default_index_type = get_option("compute.default_index_type")

        scols = [scol_for(sdf, column) for column in sdf.columns]
        if default_index_type == "sequence":
            sequential_index = (
                F.row_number().over(Window.orderBy(F.monotonically_increasing_id())) - 1
            )
            return sdf.select(sequential_index.alias(index_column), *scols)
        elif default_index_type == "distributed-sequence":
            return _InternalFrame.attach_distributed_sequence_column(sdf, column_name=index_column)
        elif default_index_type == "distributed":
            return _InternalFrame.attach_distributed_column(sdf, column_name=index_column)
        else:
            raise ValueError(
                "'compute.default_index_type' should be one of 'sequence',"
                " 'distributed-sequence' and 'distributed'"
            )

    @staticmethod
    def attach_distributed_column(sdf, column_name):
        scols = [scol_for(sdf, column) for column in sdf.columns]
        return sdf.select(F.monotonically_increasing_id().alias(column_name), *scols)

    @staticmethod
    def attach_distributed_sequence_column(sdf, column_name):
        """
        This method attaches a Spark column that has a sequence in a distributed manner.
        This is equivalent to the column assigned when default index type 'distributed-sequence'.

        >>> sdf = ks.DataFrame(['a', 'b', 'c']).to_spark()
        >>> sdf = _InternalFrame.attach_distributed_sequence_column(sdf, column_name="sequence")
        >>> sdf.sort("sequence").show()  # doctest: +NORMALIZE_WHITESPACE
        +--------+---+
        |sequence|  0|
        +--------+---+
        |       0|  a|
        |       1|  b|
        |       2|  c|
        +--------+---+
        """

        scols = [scol_for(sdf, column) for column in sdf.columns]

        spark_partition_column = verify_temp_column_name(sdf, "__spark_partition_id__")
        offset_column = verify_temp_column_name(sdf, "__offset__")
        row_number_column = verify_temp_column_name(sdf, "__row_number__")

        # 1. Calculates counts per each partition ID. `counts` here is, for instance,
        #     {
        #         1: 83,
        #         6: 83,
        #         3: 83,
        #         ...
        #     }
        sdf = sdf.withColumn(spark_partition_column, F.spark_partition_id())
        counts = map(
            lambda x: (x["key"], x["count"]),
            sdf.groupby(sdf[spark_partition_column].alias("key")).count().collect(),
        )

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

        sdf = sdf.withColumn(offset_column, offset(spark_partition_column))

        # 4. Calculate row_number in each partition.
        w = Window.partitionBy(spark_partition_column).orderBy(F.monotonically_increasing_id())
        row_number = F.row_number().over(w)
        sdf = sdf.withColumn(row_number_column, row_number)

        # 5. Calculate the index.
        return sdf.select(
            (sdf[offset_column] + sdf[row_number_column] - 1).alias(column_name), *scols
        )

    def spark_column_name_for(self, labels: Tuple[str, ...]) -> str:
        """ Return the actual Spark column name for the given column name. """
        return self._sdf.select(self.spark_column_for(labels)).columns[0]

    def spark_column_for(self, labels: Tuple[str, ...]):
        """ Return Spark Column for the given column name. """
        column_labels_to_scol = dict(zip(self.column_labels, self.data_spark_columns))
        if labels in column_labels_to_scol:
            return column_labels_to_scol[labels]  # type: ignore
        else:
            raise KeyError(name_like_string(labels))

    def spark_type_for(self, labels: Tuple[str, ...]) -> DataType:
        """ Return DataType for the given column name. """
        return self._sdf.select(self.spark_column_for(labels)).schema[0].dataType

    @property
    def spark_frame(self) -> spark.DataFrame:
        """ Return the managed Spark DataFrame. """
        return self._sdf

    @property
    def spark_column(self) -> Optional[spark.Column]:
        """ Return the managed Spark Column. """
        return self._spark_column

    @lazy_property
    def data_spark_column_names(self) -> List[str]:
        """ Return the managed column field names. """
        return self.spark_frame.select(self.data_spark_columns).columns

    @property
    def data_spark_columns(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed data columns. """
        return self._data_spark_columns

    @lazy_property
    def index_spark_column_names(self) -> List[str]:
        """ Return the managed index field names. """
        return list(self.index_map.keys())

    @lazy_property
    def index_spark_columns(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed index columns. """
        return [scol_for(self._sdf, column) for column in self.index_spark_column_names]

    @lazy_property
    def spark_column_names(self) -> List[str]:
        """ Return all the field names including index field names. """
        index_columns = set(self.index_spark_column_names)
        return self.index_spark_column_names + [
            column for column in self.data_spark_column_names if column not in index_columns
        ]

    @lazy_property
    def spark_columns(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed columns including index columns. """
        index_columns = set(self.index_spark_column_names)
        return self.index_spark_columns + [
            self.spark_column_for(label)
            for label in self.column_labels
            if self.spark_column_name_for(label) not in index_columns
        ]

    @property
    def index_map(self) -> Dict[str, Optional[Tuple[str, ...]]]:
        """ Return the managed index information. """
        assert len(self._index_map) > 0
        return self._index_map

    @lazy_property
    def index_names(self) -> List[Optional[Tuple[str, ...]]]:
        """ Return the managed index names. """
        return list(self.index_map.values())

    @property
    def column_labels(self) -> List[Tuple[str, ...]]:
        """ Return the managed column index. """
        return self._column_labels

    @lazy_property
    def column_labels_level(self) -> int:
        """ Return the level of the column index. """
        return column_labels_level(self._column_labels)

    @property
    def column_label_names(self) -> Optional[List[str]]:
        """ Return names of the index levels. """
        return self._column_label_names

    @lazy_property
    def to_internal_spark_frame(self) -> spark.DataFrame:
        """
        Return as Spark DataFrame. This contains index columns as well
        and should be only used for internal purposes.
        """
        index_columns = set(self.index_spark_column_names)
        data_columns = []
        for i, (column, label) in enumerate(zip(self.data_spark_column_names, self.column_labels)):
            if column not in index_columns:
                scol = self.spark_column_for(label)
                name = str(i) if label is None else name_like_string(label)
                if column != name:
                    scol = scol.alias(name)
                data_columns.append(scol)
        return self._sdf.select(self.index_spark_columns + data_columns)

    @lazy_property
    def to_external_spark_frame(self) -> spark.DataFrame:
        """ Return as new Spark DataFrame. """
        data_columns = []
        for i, (column, label) in enumerate(zip(self.data_spark_column_names, self.column_labels)):
            scol = self.spark_column_for(label)
            name = str(i) if label is None else name_like_string(label)
            if column != name:
                scol = scol.alias(name)
            data_columns.append(scol)
        return self._sdf.select(data_columns)

    @lazy_property
    def to_pandas_frame(self) -> pd.DataFrame:
        """ Return as pandas DataFrame. """
        sdf = self.to_internal_spark_frame
        pdf = sdf.toPandas()
        if len(pdf) == 0 and len(sdf.schema) > 0:
            pdf = pdf.astype(
                {field.name: spark_type_to_pandas_dtype(field.dataType) for field in sdf.schema}
            )

        index_columns = self.index_spark_column_names
        if len(index_columns) > 0:
            append = False
            for index_field in index_columns:
                drop = index_field not in self.data_spark_column_names
                pdf = pdf.set_index(index_field, drop=drop, append=append)
                append = True
            pdf = pdf[
                [
                    col
                    if col in index_columns
                    else str(i)
                    if label is None
                    else name_like_string(label)
                    for i, (col, label) in enumerate(
                        zip(self.data_spark_column_names, self.column_labels)
                    )
                ]
            ]

        if self.column_labels_level > 1:
            pdf.columns = pd.MultiIndex.from_tuples(self._column_labels)
        else:
            pdf.columns = [None if label is None else label[0] for label in self._column_labels]
        if self._column_label_names is not None:
            pdf.columns.names = self._column_label_names

        index_names = self.index_names
        if len(index_names) > 0:
            pdf.index.names = [
                name if name is None or len(name) > 1 else name[0] for name in index_names
            ]
        return pdf

    def with_new_sdf(
        self, sdf: spark.DataFrame, data_columns: Optional[List[str]] = None
    ) -> "_InternalFrame":
        """ Copy the immutable _InternalFrame with the updates by the specified Spark DataFrame.

        :param sdf: the new Spark DataFrame
        :param data_columns: the new column names.
            If None, the original one is used.
        :return: the copied _InternalFrame.
        """
        assert self.spark_column is None

        if data_columns is None:
            data_columns = self.data_spark_column_names
        else:
            assert len(data_columns) == len(self.column_labels), (
                len(data_columns),
                len(self.column_labels),
            )
        sdf = sdf.drop(NATURAL_ORDER_COLUMN_NAME)
        return self.copy(
            spark_frame=sdf, data_spark_columns=[scol_for(sdf, col) for col in data_columns]
        )

    def with_new_columns(
        self,
        scols_or_ksers: List[Union[spark.Column, "Series"]],
        column_labels: Optional[List[Tuple[str, ...]]] = None,
        keep_order: bool = True,
    ) -> "_InternalFrame":
        """
        Copy the immutable _InternalFrame with the updates by the specified Spark Columns or Series.

        :param scols_or_ksers: the new Spark Columns or Series.
        :param column_labels: the new column index.
            If None, the its column_labels is used when the corresponding `scols_or_ksers` is
            Series, otherwise the original one is used.
        :return: the copied _InternalFrame.
        """
        from databricks.koalas.series import Series

        if column_labels is None:
            if all(isinstance(scol_or_kser, Series) for scol_or_kser in scols_or_ksers):
                column_labels = [kser._internal.column_labels[0] for kser in scols_or_ksers]
            else:
                assert len(scols_or_ksers) == len(self.column_labels), (
                    len(scols_or_ksers),
                    len(self.column_labels),
                )
                column_labels = []
                for scol_or_kser, label in zip(scols_or_ksers, self.column_labels):
                    if isinstance(scol_or_kser, Series):
                        column_labels.append(scol_or_kser._internal.column_labels[0])
                    else:
                        column_labels.append(label)
        else:
            assert len(scols_or_ksers) == len(column_labels), (
                len(scols_or_ksers),
                len(column_labels),
            )

        data_spark_columns = []
        for scol_or_kser, label in zip(scols_or_ksers, column_labels):
            if isinstance(scol_or_kser, Series):
                scol = scol_or_kser._internal.spark_column
            else:
                scol = scol_or_kser
            data_spark_columns.append(scol)

        hidden_columns = []
        if keep_order:
            hidden_columns.append(NATURAL_ORDER_COLUMN_NAME)

        sdf = self._sdf.select(self.index_spark_columns + data_spark_columns + hidden_columns)

        return self.copy(
            spark_frame=sdf,
            column_labels=column_labels,
            data_spark_columns=[
                scol_for(sdf, col) for col in self._sdf.select(data_spark_columns).columns
            ],
            spark_column=None,
        )

    def with_filter(self, pred: Union[spark.Column, "Series"]):
        """ Copy the immutable _InternalFrame with the updates by the predicate.

        :param pred: the predicate to filter.
        :return: the copied _InternalFrame.
        """
        from databricks.koalas.series import Series

        if isinstance(pred, Series):
            assert isinstance(pred.spark_type, BooleanType), pred.spark_type
            pred = pred.spark_column
        else:
            spark_type = self._sdf.select(pred).schema[0].dataType
            assert isinstance(spark_type, BooleanType), spark_type

        return self.copy(spark_frame=self._sdf.drop(NATURAL_ORDER_COLUMN_NAME).filter(pred))

    def copy(
        self,
        spark_frame: Union[spark.DataFrame, _NoValueType] = _NoValue,
        index_map: Optional[Union[Dict[str, Optional[Tuple[str, ...]]], _NoValueType]] = _NoValue,
        column_labels: Optional[Union[List[Tuple[str, ...]], _NoValueType]] = _NoValue,
        data_spark_columns: Optional[Union[List[spark.Column], _NoValueType]] = _NoValue,
        column_label_names: Optional[Union[List[str], _NoValueType]] = _NoValue,
        spark_column: Optional[Union[spark.Column, _NoValueType]] = _NoValue,
    ) -> "_InternalFrame":
        """ Copy the immutable DataFrame.

        :param spark_frame: the new Spark DataFrame. If None, then the original one is used.
        :param index_map: the new index information. If None, then the original one is used.
        :param column_labels: the new column index.
        :param data_spark_columns: the new Spark Columns. If None, then the original ones are used.
        :param column_label_names: the new names of the index levels.
        :param spark_column: the new Spark Column. If None, then the original one is used.
        :return: the copied immutable DataFrame.
        """
        if spark_frame is _NoValue:
            spark_frame = self._sdf
        if index_map is _NoValue:
            index_map = self._index_map
        if column_labels is _NoValue:
            column_labels = self._column_labels
        if data_spark_columns is _NoValue:
            data_spark_columns = self._data_spark_columns
        if column_label_names is _NoValue:
            column_label_names = self._column_label_names
        if spark_column is _NoValue:
            spark_column = self.spark_column
        return _InternalFrame(
            spark_frame,
            index_map=index_map,
            column_labels=column_labels,
            data_spark_columns=data_spark_columns,
            column_label_names=column_label_names,
            spark_column=spark_column,
        )

    @staticmethod
    def from_pandas(pdf: pd.DataFrame) -> "_InternalFrame":
        """ Create an immutable DataFrame from pandas DataFrame.

        :param pdf: :class:`pd.DataFrame`
        :return: the created immutable DataFrame
        """
        columns = pdf.columns
        data_columns = [name_like_string(col) for col in columns]
        if isinstance(columns, pd.MultiIndex):
            column_labels = columns.tolist()
        else:
            column_labels = None
        column_label_names = columns.names

        index_names = [
            name if name is None or isinstance(name, tuple) else (name,) for name in pdf.index.names
        ]
        index_columns = [SPARK_INDEX_NAME_FORMAT(i) for i in range(len(index_names))]

        pdf = pdf.copy()
        pdf.index.names = index_columns
        reset_index = pdf.reset_index()
        reset_index.columns = index_columns + data_columns
        schema = StructType(
            [
                StructField(
                    name, infer_pd_series_spark_type(col), nullable=bool(col.isnull().any()),
                )
                for name, col in reset_index.iteritems()
            ]
        )
        for name, col in reset_index.iteritems():
            dt = col.dtype
            if is_datetime64_dtype(dt) or is_datetime64tz_dtype(dt):
                continue
            reset_index[name] = col.replace({np.nan: None})
        sdf = default_session().createDataFrame(reset_index, schema=schema)
        return _InternalFrame(
            spark_frame=sdf,
            index_map=OrderedDict(zip(index_columns, index_names)),
            column_labels=column_labels,
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
            column_label_names=column_label_names,
        )
