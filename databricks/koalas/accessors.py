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
Koalas specific features.
"""
from collections import OrderedDict
from typing import Tuple, Union, TYPE_CHECKING

from databricks.koalas.internal import InternalFrame, SPARK_INDEX_NAME_FORMAT
from databricks.koalas.utils import name_like_string, scol_for

if TYPE_CHECKING:
    from databricks.koalas.frame import DataFrame


class KoalasFrameMethods(object):
    """ Koalas specific features for DataFrame. """

    def __init__(self, frame: "DataFrame"):
        self._kdf = frame

    def attach_id_column(self, id_type: str, column: Union[str, Tuple[str, ...]]) -> "DataFrame":
        """
        Attach a column to be used as identifier of rows similar to the default index.

        See also `Default Index type
        <https://koalas.readthedocs.io/en/latest/user_guide/options.html#default-index-type>`_.

        Parameters
        ----------
        id_type : string
            The id type.

            - 'sequence' : a sequence that increases one by one.

              .. note:: this uses Spark's Window without specifying partition specification.
                  This leads to move all data into single partition in single machine and
                  could cause serious performance degradation.
                  Avoid this method against very large dataset.

            - 'distributed-sequence' : a sequence that increases one by one,
              by group-by and group-map approach in a distributed manner.
            - 'distributed' : a monotonically increasing sequence simply by using PySpark’s
              monotonically_increasing_id function in a fully distributed manner.

        column : string or tuple of string
            The column name.

        Returns
        -------
        DataFrame
            The DataFrame attached the column.

        Examples
        --------
        >>> df = ks.DataFrame({"x": ['a', 'b', 'c']})
        >>> df.koalas.attach_id_column(id_type="sequence", column="id")
           x  id
        0  a   0
        1  b   1
        2  c   2

        >>> df.koalas.attach_id_column(id_type="distributed-sequence", column="id").sort_index()
           x  id
        0  a   0
        1  b   1
        2  c   2

        >>> df.koalas.attach_id_column(id_type="distributed", column="id")
        ... # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
           x   id
        0  a  ...
        1  b  ...
        2  c  ...

        For multi-index columns:

        >>> df = ks.DataFrame({("x", "y"): ['a', 'b', 'c']})
        >>> df.koalas.attach_id_column(id_type="sequence", column=("id-x", "id-y"))
           x id-x
           y id-y
        0  a    0
        1  b    1
        2  c    2
        """
        from databricks.koalas.frame import DataFrame

        if id_type == "sequence":
            attach_func = InternalFrame.attach_sequence_column
        elif id_type == "distributed-sequence":
            attach_func = InternalFrame.attach_distributed_sequence_column
        elif id_type == "distributed":
            attach_func = InternalFrame.attach_distributed_column
        else:
            raise ValueError(
                "id_type should be one of 'sequence', 'distributed-sequence' and 'distributed'"
            )

        if isinstance(column, str):
            column = (column,)
        else:
            assert isinstance(column, tuple), type(column)

        internal = self._kdf._internal

        if len(column) != internal.column_labels_level:
            raise ValueError(
                "The given column `{}` must be the same length as the existing columns.".format(
                    column
                )
            )
        elif column in internal.column_labels:
            raise ValueError(
                "The given column `{}` already exists.".format(name_like_string(column))
            )

        # Make sure the underlying Spark column names are the form of
        # `name_like_string(column_label)`.
        sdf = internal.spark_frame.select(
            [
                scol.alias(SPARK_INDEX_NAME_FORMAT(i))
                for i, scol in enumerate(internal.index_spark_columns)
            ]
            + [
                scol.alias(name_like_string(label))
                for scol, label in zip(internal.data_spark_columns, internal.column_labels)
            ]
        )
        sdf = attach_func(sdf, name_like_string(column))

        return DataFrame(
            InternalFrame(
                spark_frame=sdf,
                index_map=OrderedDict(
                    [
                        (SPARK_INDEX_NAME_FORMAT(i), name)
                        for i, name in enumerate(internal.index_names)
                    ]
                ),
                column_labels=internal.column_labels + [column],
                data_spark_columns=(
                    [scol_for(sdf, name_like_string(label)) for label in internal.column_labels]
                    + [scol_for(sdf, name_like_string(column))]
                ),
                column_label_names=internal.column_label_names,
            ).resolved_copy
        )
