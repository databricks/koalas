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
Utility features. The features here are specific to Koalas.
"""
from collections import OrderedDict
from typing import Tuple, Union, TYPE_CHECKING

from databricks.koalas.internal import InternalFrame, SPARK_INDEX_NAME_FORMAT
from databricks.koalas.utils import name_like_string, scol_for

if TYPE_CHECKING:
    from databricks.koalas.frame import DataFrame


class UtilsFrameMethods(object):
    """ Utility features for DataFrame. Usually, The features here are specific to Koalas. """

    def __init__(self, frame: "DataFrame"):
        self._kdf = frame

    def attach_sequence_column(self, column: Union[str, Tuple[str, ...]]) -> "DataFrame":
        """
        Attach a column of a sequence that increases one by one.

        .. note:: this uses Spark's Window without specifying partition specification.
            This leads to move all data into single partition in single machine and
            could cause serious performance degradation.
            Avoid this method against very large dataset.

        Parameters
        ----------
        column : string or list of string
            The column name to be attached.

        Returns
        -------
        DataFrame
            The DataFrame attached the column.

        Examples
        --------
        >>> df = ks.DataFrame({"x": ['a', 'b', 'c']})
        >>> df.utils.attach_sequence_column("sequence")
           x  sequence
        0  a         0
        1  b         1
        2  c         2
        """
        return self._attach_default_index_like_column(column, InternalFrame.attach_sequence_column)

    def attach_distributed_sequence_column(
        self, column: Union[str, Tuple[str, ...]]
    ) -> "DataFrame":
        """
        Attach a column of a sequence that increases one by one, by group-by and group-map approach
        in a distributed manner.

        Parameters
        ----------
        column : string or list of string
            The column name to be attached.

        Returns
        -------
        DataFrame
            The DataFrame attached the column.

        Examples
        --------
        >>> df = ks.DataFrame({"x": ['a', 'b', 'c']})
        >>> df.utils.attach_distributed_sequence_column("distributed_sequence").sort_index()
           x  distributed_sequence
        0  a                     0
        1  b                     1
        2  c                     2
        """
        return self._attach_default_index_like_column(
            column, InternalFrame.attach_distributed_sequence_column
        )

    def attach_distributed_column(self, column: Union[str, Tuple[str, ...]]) -> "DataFrame":
        """
        Attach a column of a monotonically increasing sequence simply by using PySpark’s
        monotonically_increasing_id function in a fully distributed manner.

        Parameters
        ----------
        column : string or list of string
            The column name to be attached.

        Returns
        -------
        DataFrame
            The DataFrame attached the column.

        Examples
        --------
        >>> df = ks.DataFrame({"x": ['a', 'b', 'c']})
        >>> df.utils.attach_distributed_column("distributed")
        ... # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
           x  distributed
        0  a          ...
        1  b          ...
        2  c          ...
        """
        return self._attach_default_index_like_column(
            column, InternalFrame.attach_distributed_column
        )

    def _attach_default_index_like_column(
        self, column_label: Union[str, Tuple[str, ...]], attach_func
    ) -> "DataFrame":
        from databricks.koalas.frame import DataFrame

        if isinstance(column_label, str):
            column_label = (column_label,)
        else:
            assert isinstance(column_label, tuple), type(column_label)

        internal = self._kdf._internal

        if len(column_label) != internal.column_labels_level:
            raise ValueError(
                "The given column `{}` must be the same length as the existing columns.".format(
                    column_label
                )
            )
        elif column_label in internal.column_labels:
            raise ValueError(
                "The given column `{}` already exists.".format(name_like_string(column_label))
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
        sdf = attach_func(sdf, name_like_string(column_label))

        return DataFrame(
            InternalFrame(
                spark_frame=sdf,
                index_map=OrderedDict(
                    [
                        (SPARK_INDEX_NAME_FORMAT(i), name)
                        for i, name in enumerate(internal.index_names)
                    ]
                ),
                column_labels=internal.column_labels + [column_label],
                data_spark_columns=(
                    [scol_for(sdf, name_like_string(label)) for label in internal.column_labels]
                    + [scol_for(sdf, name_like_string(column_label))]
                ),
                column_label_names=internal.column_label_names,
            ).resolved_copy
        )
