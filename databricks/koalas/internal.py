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
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
from itertools import accumulate
import py4j

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_datetime64tz_dtype
from pyspark import sql as spark
from pyspark._globals import _NoValue, _NoValueType
from pyspark.sql import functions as F, Window
from pyspark.sql.functions import PandasUDFType, pandas_udf
from pyspark.sql.types import BooleanType, DataType, StructField, StructType, LongType

try:
    from pyspark.sql.types import to_arrow_type
except ImportError:
    from pyspark.sql.pandas.types import to_arrow_type  # noqa: F401

# For running doctests and reference resolution in PyCharm.
from databricks import koalas as ks  # noqa: F401

if TYPE_CHECKING:
    # This is required in old Python 3.5 to prevent circular reference.
    from databricks.koalas.series import Series
from databricks.koalas.config import get_option
from databricks.koalas.typedef import (
    infer_pd_series_spark_type,
    spark_type_to_pandas_dtype,
)
from databricks.koalas.utils import (
    column_labels_level,
    default_session,
    is_name_like_tuple,
    is_testing,
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

DEFAULT_SERIES_NAME = 0
SPARK_DEFAULT_SERIES_NAME = str(DEFAULT_SERIES_NAME)


class InternalFrame(object):
    """
    The internal immutable DataFrame which manages Spark DataFrame and column names and index
    information.

    .. note:: this is an internal class. It is not supposed to be exposed to users and users
        should not directly access to it.

    The internal immutable DataFrame represents the index information for a DataFrame it belongs to.
    For instance, if we have a Koalas DataFrame as below, pandas DataFrame does not store the index
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

    * `to_internal_spark_frame` represents Spark DataFrame derived by the metadata. Includes index.

    * `to_pandas_frame` represents pandas DataFrame derived by the metadata

    >>> internal = kdf._internal
    >>> internal.spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    +-----------------+---+---+---+---+---+-----------------+
    |__index_level_0__|  A|  B|  C|  D|  E|__natural_order__|
    +-----------------+---+---+---+---+---+-----------------+
    |                0|  1|  5|  9| 13| 17|              ...|
    |                1|  2|  6| 10| 14| 18|              ...|
    |                2|  3|  7| 11| 15| 19|              ...|
    |                3|  4|  8| 12| 16| 20|              ...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_spark_column_names
    ['A', 'B', 'C', 'D', 'E']
    >>> internal.index_spark_column_names
    ['__index_level_0__']
    >>> internal.spark_column_names
    ['__index_level_0__', 'A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [None]
    >>> internal.to_internal_spark_frame.show()  # doctest: +NORMALIZE_WHITESPACE
    +-----------------+---+---+---+---+---+
    |__index_level_0__|  A|  B|  C|  D|  E|
    +-----------------+---+---+---+---+---+
    |                0|  1|  5|  9| 13| 17|
    |                1|  2|  6| 10| 14| 18|
    |                2|  3|  7| 11| 15| 19|
    |                3|  4|  8| 12| 16| 20|
    +-----------------+---+---+---+---+---+
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
    |                0|  1|  5|  9| 13| 17|              ...|
    |                1|  2|  6| 10| 14| 18|              ...|
    |                2|  3|  7| 11| 15| 19|              ...|
    |                3|  4|  8| 12| 16| 20|              ...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_spark_column_names
    ['B', 'C', 'D', 'E']
    >>> internal.index_spark_column_names
    ['A']
    >>> internal.spark_column_names
    ['A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [('A',)]
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
    |                0|  1|  5|  9| 13| 17|              ...|
    |                1|  2|  6| 10| 14| 18|              ...|
    |                2|  3|  7| 11| 15| 19|              ...|
    |                3|  4|  8| 12| 16| 20|              ...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_spark_column_names
    ['B', 'C', 'D', 'E']
    >>> internal.index_spark_column_names
    ['__index_level_0__', 'A']
    >>> internal.spark_column_names
    ['__index_level_0__', 'A', 'B', 'C', 'D', 'E']
    >>> internal.index_names
    [None, ('A',)]
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
    |                0|     1|     2|     3|     4|              ...|
    |                1|     5|     6|     7|     8|              ...|
    |                2|     9|    10|    11|    12|              ...|
    |                3|    13|    14|    15|    16|              ...|
    |                4|    17|    18|    19|    20|              ...|
    +-----------------+------+------+------+------+-----------------+
    >>> internal.data_spark_column_names
    ['(X, A)', '(X, B)', '(Y, C)', '(Y, D)']
    >>> internal.column_labels
    [('X', 'A'), ('X', 'B'), ('Y', 'C'), ('Y', 'D')]

    For Series, it also holds scol to represent the column.

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
    |                0|  1|  5|  9| 13| 17|              ...|
    |                1|  2|  6| 10| 14| 18|              ...|
    |                2|  3|  7| 11| 15| 19|              ...|
    |                3|  4|  8| 12| 16| 20|              ...|
    +-----------------+---+---+---+---+---+-----------------+
    >>> internal.data_spark_column_names
    ['B']
    >>> internal.index_spark_column_names
    ['A']
    >>> internal.spark_column_names
    ['A', 'B']
    >>> internal.index_names
    [('A',)]
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
        index_spark_columns: Optional[List[spark.Column]],
        index_names: Optional[List[Optional[Tuple]]] = None,
        column_labels: Optional[List[Tuple]] = None,
        data_spark_columns: Optional[List[spark.Column]] = None,
        column_label_names: Optional[List[Optional[Tuple]]] = None,
    ) -> None:
        """
        Create a new internal immutable DataFrame to manage Spark DataFrame, column fields and
        index fields and names.

        :param spark_frame: Spark DataFrame to be managed.
        :param index_spark_columns: list of Spark Column
                                    Spark Columns for the index.
        :param index_names: list of tuples
                            the index names.
        :param column_labels: list of tuples with the same length
                              The multi-level values in the tuples.
        :param data_spark_columns: list of Spark Column
                                   Spark Columns to appear as columns. If this is None, calculated
                                   from spark_frame.
        :param column_label_names: Names for each of the column index levels.

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

        >>> internal = kdf._internal

        >>> internal._sdf.show()  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        +-----------------+-----------------+------+------+------+...
        |__index_level_0__|__index_level_1__|(a, x)|(a, y)|(b, z)|...
        +-----------------+-----------------+------+------+------+...
        |              foo|              bar|     1|     2|     3|...
        |              foo|              bar|     4|     5|     6|...
        |              zoo|              bar|     7|     8|     9|...
        +-----------------+-----------------+------+------+------+...

        >>> internal._index_spark_columns
        [Column<b'__index_level_0__'>, Column<b'__index_level_1__'>, Column<b'(a, x)'>]

        >>> internal._index_names
        [('row_index_a',), ('row_index_b',), ('a', 'x')]

        >>> internal._column_labels
        [('a', 'y'), ('b', 'z')]

        >>> internal._data_spark_columns
        [Column<b'(a, y)'>, Column<b'(b, z)'>]

        >>> internal._column_label_names
        [('column_labels_a',), ('column_labels_b',)]
        """

        assert isinstance(spark_frame, spark.DataFrame)
        assert not spark_frame.isStreaming, "Koalas does not support Structured Streaming."

        if not index_spark_columns:
            if data_spark_columns is not None:
                if column_labels is not None:
                    data_spark_columns = [
                        scol.alias(name_like_string(label))
                        for scol, label in zip(data_spark_columns, column_labels)
                    ]
                spark_frame = spark_frame.select(data_spark_columns)

            assert not any(SPARK_INDEX_NAME_PATTERN.match(name) for name in spark_frame.columns), (
                "Index columns should not appear in columns of the Spark DataFrame. Avoid "
                "index column names [%s]." % SPARK_INDEX_NAME_PATTERN
            )

            # Create default index.
            spark_frame = InternalFrame.attach_default_index(spark_frame)
            index_spark_columns = [scol_for(spark_frame, SPARK_DEFAULT_INDEX_NAME)]

            if data_spark_columns is not None:
                data_spark_columns = [
                    scol_for(spark_frame, col)
                    for col in spark_frame.columns
                    if col != SPARK_DEFAULT_INDEX_NAME
                ]

        if NATURAL_ORDER_COLUMN_NAME not in spark_frame.columns:
            spark_frame = spark_frame.withColumn(
                NATURAL_ORDER_COLUMN_NAME, F.monotonically_increasing_id()
            )

        if not index_names:
            index_names = [None] * len(index_spark_columns)

        assert len(index_spark_columns) == len(index_names), (
            len(index_spark_columns),
            len(index_names),
        )
        assert all(
            isinstance(index_scol, spark.Column) for index_scol in index_spark_columns
        ), index_spark_columns
        assert all(
            is_name_like_tuple(index_name, check_type=True) for index_name in index_names
        ), index_names
        assert data_spark_columns is None or all(
            isinstance(scol, spark.Column) for scol in data_spark_columns
        )

        self._sdf = spark_frame  # type: spark.DataFrame
        self._index_spark_columns = index_spark_columns  # type: List[spark.Column]
        self._index_names = index_names  # type: List[Optional[Tuple]]

        if data_spark_columns is None:
            self._data_spark_columns = [
                scol_for(spark_frame, col)
                for col in spark_frame.columns
                if all(
                    not scol_for(spark_frame, col)._jc.equals(index_scol._jc)
                    for index_scol in index_spark_columns
                )
                and col not in HIDDEN_COLUMNS
            ]
        else:
            self._data_spark_columns = data_spark_columns

        if column_labels is None:
            self._column_labels = [
                (col,) for col in spark_frame.select(self._data_spark_columns).columns
            ]  # type: List[Tuple]
        else:
            assert len(column_labels) == len(self._data_spark_columns), (
                len(column_labels),
                len(self._data_spark_columns),
            )
            if len(column_labels) == 1:
                column_label = column_labels[0]
                assert is_name_like_tuple(column_label, check_type=True), column_label
            else:
                assert all(
                    is_name_like_tuple(column_label, check_type=True)
                    for column_label in column_labels
                ), column_labels
                assert len(set(len(label) for label in column_labels)) <= 1, column_labels
            self._column_labels = column_labels

        if column_label_names is None:
            self._column_label_names = [None] * column_labels_level(
                self._column_labels
            )  # type: List[Optional[Tuple]]
        else:
            if len(self._column_labels) > 0:
                assert len(column_label_names) == column_labels_level(self._column_labels), (
                    len(column_label_names),
                    column_labels_level(self._column_labels),
                )
            else:
                assert len(column_label_names) > 0, len(column_label_names)
            assert all(
                is_name_like_tuple(column_label_name, check_type=True)
                for column_label_name in column_label_names
            ), column_label_names
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

        >>> spark_frame = InternalFrame.attach_default_index(spark_frame)
        >>> spark_frame
        DataFrame[__index_level_0__: bigint, id: bigint]

        It throws an exception if the given column name already exists.

        >>> InternalFrame.attach_default_index(spark_frame)
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

        if default_index_type == "sequence":
            return InternalFrame.attach_sequence_column(sdf, column_name=index_column)
        elif default_index_type == "distributed-sequence":
            return InternalFrame.attach_distributed_sequence_column(sdf, column_name=index_column)
        elif default_index_type == "distributed":
            return InternalFrame.attach_distributed_column(sdf, column_name=index_column)
        else:
            raise ValueError(
                "'compute.default_index_type' should be one of 'sequence',"
                " 'distributed-sequence' and 'distributed'"
            )

    @staticmethod
    def attach_sequence_column(sdf, column_name):
        scols = [scol_for(sdf, column) for column in sdf.columns]
        sequential_index = (
            F.row_number().over(Window.orderBy(F.monotonically_increasing_id())).cast("long") - 1
        )
        return sdf.select(sequential_index.alias(column_name), *scols)

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
        >>> sdf = InternalFrame.attach_distributed_sequence_column(sdf, column_name="sequence")
        >>> sdf.show()  # doctest: +NORMALIZE_WHITESPACE
        +--------+---+
        |sequence|  0|
        +--------+---+
        |       0|  a|
        |       1|  b|
        |       2|  c|
        +--------+---+
        """
        if len(sdf.columns) > 0:
            try:
                jdf = sdf._jdf.toDF()

                sql_ctx = sdf.sql_ctx
                encoders = sql_ctx._jvm.org.apache.spark.sql.Encoders
                encoder = encoders.tuple(jdf.exprEnc(), encoders.scalaLong())

                jrdd = jdf.localCheckpoint(False).rdd().zipWithIndex()

                df = spark.DataFrame(
                    sql_ctx.sparkSession._jsparkSession.createDataset(jrdd, encoder).toDF(), sql_ctx
                )
                columns = df.columns
                return df.selectExpr(
                    "`{}` as `{}`".format(columns[1], column_name), "`{}`.*".format(columns[0])
                )
            except py4j.protocol.Py4JError:
                if is_testing():
                    raise
                return InternalFrame._attach_distributed_sequence_column(sdf, column_name)
        else:
            cnt = sdf.count()
            if cnt > 0:
                return default_session().range(cnt).toDF(column_name)
            else:
                return default_session().createDataFrame(
                    [], schema=StructType().add(column_name, data_type=LongType(), nullable=False)
                )

    @staticmethod
    def _attach_distributed_sequence_column(sdf, column_name):
        """
        >>> sdf = ks.DataFrame(['a', 'b', 'c']).to_spark()
        >>> sdf = InternalFrame._attach_distributed_sequence_column(sdf, column_name="sequence")
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

        # Checkpoint the DataFrame to fix the partition ID.
        sdf = sdf.localCheckpoint(eager=False)

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

    def spark_column_name_for(self, label: Tuple) -> str:
        """ Return the actual Spark column name for the given column label. """
        return self.spark_frame.select(self.spark_column_for(label)).columns[0]

    def spark_column_for(self, label: Tuple):
        """ Return Spark Column for the given column label. """
        column_labels_to_scol = dict(zip(self.column_labels, self.data_spark_columns))
        if label in column_labels_to_scol:
            return column_labels_to_scol[label]  # type: ignore
        else:
            raise KeyError(name_like_string(label))

    def spark_type_for(self, label: Tuple) -> DataType:
        """ Return DataType for the given column label. """
        return self.spark_frame.select(self.spark_column_for(label)).schema[0].dataType

    def spark_column_nullable_for(self, label: Tuple) -> bool:
        """ Return nullability for the given column label. """
        return self.spark_frame.select(self.spark_column_for(label)).schema[0].nullable

    @property
    def spark_frame(self) -> spark.DataFrame:
        """ Return the managed Spark DataFrame. """
        return self._sdf

    @lazy_property
    def data_spark_column_names(self) -> List[str]:
        """ Return the managed column field names. """
        return self.spark_frame.select(self.data_spark_columns).columns

    @property
    def data_spark_columns(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed data columns. """
        return self._data_spark_columns

    @property
    def index_spark_column_names(self) -> List[str]:
        """ Return the managed index field names. """
        return self.spark_frame.select(self.index_spark_columns).columns

    @property
    def index_spark_columns(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed index columns. """
        return self._index_spark_columns

    @lazy_property
    def spark_column_names(self) -> List[str]:
        """ Return all the field names including index field names. """
        return self.spark_frame.select(self.spark_columns).columns

    @lazy_property
    def spark_columns(self) -> List[spark.Column]:
        """ Return Spark Columns for the managed columns including index columns. """
        index_spark_columns = self.index_spark_columns
        return index_spark_columns + [
            spark_column
            for spark_column in self.data_spark_columns
            if all(not spark_column._jc.equals(scol._jc) for scol in index_spark_columns)
        ]

    @property
    def index_names(self) -> List[Optional[Tuple]]:
        """ Return the managed index names. """
        return self._index_names

    @lazy_property
    def index_level(self) -> int:
        """ Return the level of the index. """
        return len(self._index_names)

    @property
    def column_labels(self) -> List[Tuple]:
        """ Return the managed column index. """
        return self._column_labels

    @lazy_property
    def column_labels_level(self) -> int:
        """ Return the level of the column index. """
        return len(self._column_label_names)

    @property
    def column_label_names(self) -> List[Optional[Tuple]]:
        """ Return names of the index levels. """
        return self._column_label_names

    @lazy_property
    def to_internal_spark_frame(self) -> spark.DataFrame:
        """
        Return as Spark DataFrame. This contains index columns as well
        and should be only used for internal purposes.
        """
        index_spark_columns = self.index_spark_columns
        data_columns = []
        for i, (label, spark_column, column_name) in enumerate(
            zip(self.column_labels, self.data_spark_columns, self.data_spark_column_names)
        ):
            if all(not spark_column._jc.equals(scol._jc) for scol in index_spark_columns):
                name = str(i) if label is None else name_like_string(label)
                if column_name != name:
                    spark_column = spark_column.alias(name)
                data_columns.append(spark_column)
        return self.spark_frame.select(index_spark_columns + data_columns)

    @lazy_property
    def to_pandas_frame(self) -> pd.DataFrame:
        """ Return as pandas DataFrame. """
        sdf = self.to_internal_spark_frame
        pdf = sdf.toPandas()
        if len(pdf) == 0 and len(sdf.schema) > 0:
            pdf = pdf.astype(
                {field.name: spark_type_to_pandas_dtype(field.dataType) for field in sdf.schema}
            )

        column_names = []
        for i, (label, spark_column, column_name) in enumerate(
            zip(self.column_labels, self.data_spark_columns, self.data_spark_column_names)
        ):
            for index_spark_column_name, index_spark_column in zip(
                self.index_spark_column_names, self.index_spark_columns
            ):
                if spark_column._jc.equals(index_spark_column._jc):
                    column_names.append(index_spark_column_name)
                    break
            else:
                name = str(i) if label is None else name_like_string(label)
                if column_name != name:
                    column_name = name
                column_names.append(column_name)

        append = False
        for index_field in self.index_spark_column_names:
            drop = index_field not in column_names
            pdf = pdf.set_index(index_field, drop=drop, append=append)
            append = True
        pdf = pdf[column_names]

        names = [
            name if name is None or len(name) > 1 else name[0] for name in self._column_label_names
        ]
        if self.column_labels_level > 1:
            pdf.columns = pd.MultiIndex.from_tuples(self._column_labels, names=names)
        else:
            pdf.columns = pd.Index(
                [None if label is None else label[0] for label in self._column_labels],
                name=names[0],
            )

        pdf.index.names = [
            name if name is None or len(name) > 1 else name[0] for name in self.index_names
        ]

        return pdf

    @lazy_property
    def resolved_copy(self) -> "InternalFrame":
        """ Copy the immutable InternalFrame with the updates resolved. """
        sdf = self.spark_frame.select(self.spark_columns + list(HIDDEN_COLUMNS))
        return self.copy(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self.index_spark_column_names],
            data_spark_columns=[scol_for(sdf, col) for col in self.data_spark_column_names],
        )

    def with_new_sdf(
        self, spark_frame: spark.DataFrame, data_columns: Optional[List[str]] = None
    ) -> "InternalFrame":
        """ Copy the immutable InternalFrame with the updates by the specified Spark DataFrame.

        :param spark_frame: the new Spark DataFrame
        :param data_columns: the new column names.
            If None, the original one is used.
        :return: the copied InternalFrame.
        """
        if data_columns is None:
            data_columns = self.data_spark_column_names
        else:
            assert len(data_columns) == len(self.column_labels), (
                len(data_columns),
                len(self.column_labels),
            )
        sdf = spark_frame.drop(NATURAL_ORDER_COLUMN_NAME)
        return self.copy(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in self.index_spark_column_names],
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
        )

    def with_new_columns(
        self,
        scols_or_ksers: List[Union[spark.Column, "Series"]],
        column_labels: Optional[List[Tuple]] = None,
        column_label_names: Optional[Union[List[Optional[Tuple]], _NoValueType]] = _NoValue,
        keep_order: bool = True,
    ) -> "InternalFrame":
        """
        Copy the immutable InternalFrame with the updates by the specified Spark Columns or Series.

        :param scols_or_ksers: the new Spark Columns or Series.
        :param column_labels: the new column index.
            If None, the its column_labels is used when the corresponding `scols_or_ksers` is
            Series, otherwise the original one is used.
        :param column_label_names: the new names of the column index levels.
        :return: the copied InternalFrame.
        """
        from databricks.koalas.series import Series

        if column_labels is None:
            if all(isinstance(scol_or_kser, Series) for scol_or_kser in scols_or_ksers):
                column_labels = [kser._column_label for kser in scols_or_ksers]
            else:
                assert len(scols_or_ksers) == len(self.column_labels), (
                    len(scols_or_ksers),
                    len(self.column_labels),
                )
                column_labels = []
                for scol_or_kser, label in zip(scols_or_ksers, self.column_labels):
                    if isinstance(scol_or_kser, Series):
                        column_labels.append(scol_or_kser._column_label)
                    else:
                        column_labels.append(label)
        else:
            assert len(scols_or_ksers) == len(column_labels), (
                len(scols_or_ksers),
                len(column_labels),
            )

        data_spark_columns = []
        for scol_or_kser in scols_or_ksers:
            if isinstance(scol_or_kser, Series):
                scol = scol_or_kser.spark.column
            else:
                scol = scol_or_kser
            data_spark_columns.append(scol)

        sdf = self.spark_frame
        if not keep_order:
            sdf = self.spark_frame.select(self.index_spark_columns + data_spark_columns)
            index_spark_columns = [scol_for(sdf, col) for col in self.index_spark_column_names]
            data_spark_columns = [
                scol_for(sdf, col) for col in self.spark_frame.select(data_spark_columns).columns
            ]
        else:
            index_spark_columns = self.index_spark_columns

        if column_label_names is _NoValue:
            column_label_names = self._column_label_names

        return self.copy(
            spark_frame=sdf,
            index_spark_columns=index_spark_columns,
            column_labels=column_labels,
            data_spark_columns=data_spark_columns,
            column_label_names=column_label_names,
        )

    def with_filter(self, pred: Union[spark.Column, "Series"]) -> "InternalFrame":
        """ Copy the immutable InternalFrame with the updates by the predicate.

        :param pred: the predicate to filter.
        :return: the copied InternalFrame.
        """
        from databricks.koalas.series import Series

        if isinstance(pred, Series):
            assert isinstance(pred.spark.data_type, BooleanType), pred.spark.data_type
            pred = pred.spark.column
        else:
            spark_type = self.spark_frame.select(pred).schema[0].dataType
            assert isinstance(spark_type, BooleanType), spark_type

        return self.with_new_sdf(self.spark_frame.filter(pred).select(self.spark_columns))

    def with_new_spark_column(
        self, column_label: Tuple, scol: spark.Column, keep_order: bool = True
    ) -> "InternalFrame":
        """
        Copy the immutable InternalFrame with the updates by the specified Spark Column.

        :param column_label: the column label to be updated.
        :param scol: the new Spark Column
        :return: the copied InternalFrame.
        """
        assert column_label in self.column_labels, column_label

        idx = self.column_labels.index(column_label)
        data_spark_columns = self.data_spark_columns.copy()
        data_spark_columns[idx] = scol
        return self.with_new_columns(data_spark_columns, keep_order=keep_order)

    def select_column(self, column_label: Tuple) -> "InternalFrame":
        """
        Copy the immutable InternalFrame with the specified column.

        :param column_label: the column label to use.
        :return: the copied InternalFrame.
        """
        assert column_label in self.column_labels, column_label

        return self.copy(
            column_labels=[column_label],
            data_spark_columns=[self.spark_column_for(column_label)],
            column_label_names=None,
        )

    def copy(
        self,
        spark_frame: Union[spark.DataFrame, _NoValueType] = _NoValue,
        index_spark_columns: Union[List[spark.Column], _NoValueType] = _NoValue,
        index_names: Union[List[Optional[Tuple]], _NoValueType] = _NoValue,
        column_labels: Optional[Union[List[Tuple], _NoValueType]] = _NoValue,
        data_spark_columns: Optional[Union[List[spark.Column], _NoValueType]] = _NoValue,
        column_label_names: Optional[Union[List[Optional[Tuple]], _NoValueType]] = _NoValue,
    ) -> "InternalFrame":
        """ Copy the immutable InternalFrame.

        :param spark_frame: the new Spark DataFrame. If not specified, the original one is used.
        :param index_spark_columns: the list of Spark Column.
                                    If not specified, the original ones are used.
        :param index_names: the index names. If not specified, the original ones are used.
        :param column_labels: the new column labels. If not specified, the original ones are used.
        :param data_spark_columns: the new Spark Columns.
                                   If not specified, the original ones are used.
        :param column_label_names: the new names of the column index levels.
                                   If not specified, the original ones are used.
        :return: the copied immutable InternalFrame.
        """
        if spark_frame is _NoValue:
            spark_frame = self.spark_frame
        if index_spark_columns is _NoValue:
            index_spark_columns = self.index_spark_columns
        if index_names is _NoValue:
            index_names = self.index_names
        if column_labels is _NoValue:
            column_labels = self.column_labels
        if data_spark_columns is _NoValue:
            data_spark_columns = self.data_spark_columns
        if column_label_names is _NoValue:
            column_label_names = self.column_label_names
        return InternalFrame(
            spark_frame=spark_frame,
            index_spark_columns=index_spark_columns,
            index_names=index_names,
            column_labels=column_labels,
            data_spark_columns=data_spark_columns,
            column_label_names=column_label_names,
        )

    @staticmethod
    def from_pandas(pdf: pd.DataFrame) -> "InternalFrame":
        """ Create an immutable DataFrame from pandas DataFrame.

        :param pdf: :class:`pd.DataFrame`
        :return: the created immutable DataFrame
        """
        columns = pdf.columns
        data_columns = [name_like_string(col) for col in columns]
        if isinstance(columns, pd.MultiIndex):
            column_labels = columns.tolist()
        else:
            column_labels = [(col,) for col in columns]
        column_label_names = [
            name if name is None or isinstance(name, tuple) else (name,) for name in columns.names
        ]

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
        return InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in index_columns],
            index_names=index_names,
            column_labels=column_labels,
            data_spark_columns=[scol_for(sdf, col) for col in data_columns],
            column_label_names=column_label_names,
        )
