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
A loc indexer for Koalas DataFrame/Series.
"""
from functools import reduce

from pandas.api.types import is_list_like
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, LongType
from pyspark.sql.utils import AnalysisException

from databricks.koalas.internal import _InternalFrame, NATURAL_ORDER_COLUMN_NAME
from databricks.koalas.exceptions import SparkPandasIndexingError, SparkPandasNotImplementedError
from databricks.koalas.utils import (
    column_labels_level,
    lazy_property,
    name_like_string,
    verify_temp_column_name,
)


class _IndexerLike(object):
    def __init__(self, kdf_or_kser):
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series

        assert isinstance(kdf_or_kser, (DataFrame, Series)), "unexpected argument type: {}".format(
            type(kdf_or_kser)
        )
        self._kdf_or_kser = kdf_or_kser

    @property
    def _is_df(self):
        from databricks.koalas.frame import DataFrame

        return isinstance(self._kdf_or_kser, DataFrame)

    @property
    def _is_series(self):
        from databricks.koalas.series import Series

        return isinstance(self._kdf_or_kser, Series)

    @property
    def _internal(self):
        return self._kdf_or_kser._internal


class AtIndexer(_IndexerLike):
    """
    Access a single value for a row/column label pair.
    If the index is not unique, all matching pairs are returned as an array.
    Similar to ``loc``, in that both provide label-based lookups. Use ``at`` if you only need to
    get a single value in a DataFrame or Series.

    .. note:: Unlike pandas, Koalas only allows using ``at`` to get values but not to set them.

    .. note:: Warning: If ``row_index`` matches a lot of rows, large amounts of data will be
        fetched, potentially causing your machine to run out of memory.

    Raises
    ------
    KeyError
        When label does not exist in DataFrame

    Examples
    --------
    >>> kdf = ks.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],
    ...                    index=[4, 5, 5], columns=['A', 'B', 'C'])
    >>> kdf
        A   B   C
    4   0   2   3
    5   0   4   1
    5  10  20  30

    Get value at specified row/column pair

    >>> kdf.at[4, 'B']
    2

    Get array if an index occurs multiple times

    >>> kdf.at[5, 'B']
    array([ 4, 20])
    """

    def __getitem__(self, key):
        if self._is_df:
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError("Use DataFrame.at like .at[row_index, column_name]")
            row_sel, col_sel = key
        else:
            assert self._is_series, type(self._kdf_or_kser)
            if isinstance(key, tuple) and len(key) != 1:
                raise TypeError("Use Series.at like .at[row_index]")
            row_sel = key
            col_sel = self._internal.column_labels[0]

        if len(self._internal.index_map) == 1:
            if is_list_like(row_sel):
                raise ValueError("At based indexing on a single index can only have a single value")
            row_sel = (row_sel,)
        elif not isinstance(row_sel, tuple):
            raise ValueError("At based indexing on multi-index can only have tuple values")
        if not (
            isinstance(col_sel, str)
            or (isinstance(col_sel, tuple) and all(isinstance(col, str) for col in col_sel))
        ):
            raise ValueError("At based indexing on multi-index can only have tuple values")
        if isinstance(col_sel, str):
            col_sel = (col_sel,)

        cond = reduce(
            lambda x, y: x & y,
            [scol == row for scol, row in zip(self._internal.index_scols, row_sel)],
        )
        pdf = (
            self._internal.sdf.drop(NATURAL_ORDER_COLUMN_NAME)
            .filter(cond)
            .select(self._internal.scol_for(col_sel))
            .toPandas()
        )

        if len(pdf) < 1:
            raise KeyError(name_like_string(row_sel))

        values = pdf.iloc[:, 0].values
        return (
            values
            if (len(row_sel) < len(self._internal.index_map) or len(values) > 1)
            else values[0]
        )


class iAtIndexer(_IndexerLike):
    """
    Access a single value for a row/column pair by integer position.

    Similar to ``iloc``, in that both provide integer-based lookups. Use
    ``iat`` if you only need to get or set a single value in a DataFrame
    or Series.

    Raises
    ------
    KeyError
        When label does not exist in DataFrame

    Examples
    --------
    >>> df = ks.DataFrame([[0, 2, 3], [0, 4, 1], [10, 20, 30]],
    ...                   columns=['A', 'B', 'C'])
    >>> df
        A   B   C
    0   0   2   3
    1   0   4   1
    2  10  20  30

    Get value at specified row/column pair

    >>> df.iat[1, 2]
    1

    Get value within a series

    >>> kser = ks.Series([1, 2, 3], index=[10, 20, 30])
    >>> kser
    10    1
    20    2
    30    3
    Name: 0, dtype: int64

    >>> kser.iat[1]
    2
    """

    def __getitem__(self, key):
        if self._is_df:
            if not isinstance(key, tuple) or len(key) != 2:
                raise TypeError(
                    "Use DataFrame.iat like .iat[row_integer_position, column_integer_position]"
                )
            row_sel, col_sel = key
            if not isinstance(row_sel, int) or not isinstance(col_sel, int):
                raise ValueError("iAt based indexing can only have integer indexers")
            return self._kdf_or_kser.iloc[row_sel, col_sel]
        else:
            assert self._is_series, type(self._kdf_or_kser)
            if not isinstance(key, int) and len(key) != 1:
                raise TypeError("Use Series.iat like .iat[row_integer_position]")
            if not isinstance(key, int):
                raise ValueError("iAt based indexing can only have integer indexers")
            return self._kdf_or_kser.iloc[key]


class _LocIndexerLike(_IndexerLike):
    def __getitem__(self, key):
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series

        if self._is_series:
            if isinstance(key, Series) and key._kdf is not self._kdf_or_kser._kdf:
                kdf = self._kdf_or_kser.to_frame()
                kdf["__temp_col__"] = key
                return type(self)(kdf[self._kdf_or_kser.name])[kdf["__temp_col__"]]

            cond, limit, remaining_index = self._select_rows(key)
            if cond is None and limit is None:
                return self._kdf_or_kser

            column_labels = self._internal.column_labels
            column_scols = self._internal.column_scols
            returns_series = True
        else:
            assert self._is_df
            if isinstance(key, tuple):
                if len(key) != 2:
                    raise SparkPandasIndexingError("Only accepts pairs of candidates")
                rows_sel, cols_sel = key
            else:
                rows_sel = key
                cols_sel = None

            if isinstance(rows_sel, Series) and rows_sel._kdf is not self._kdf_or_kser:
                kdf = self._kdf_or_kser.copy()
                kdf["__temp_col__"] = rows_sel
                return type(self)(kdf)[kdf["__temp_col__"], cols_sel][
                    list(self._kdf_or_kser.columns)
                ]

            cond, limit, remaining_index = self._select_rows(rows_sel)
            column_labels, column_scols, returns_series = self._select_cols(cols_sel)

            if cond is None and limit is None and returns_series:
                return self._kdf_or_kser._kser_for(column_labels[0])

        if remaining_index is not None:
            index_scols = self._internal.index_scols[-remaining_index:]
            index_map = self._internal.index_map[-remaining_index:]
        else:
            index_scols = self._internal.index_scols
            index_map = self._internal.index_map

        if self._internal.column_label_names is None:
            column_label_names = None
        else:
            # Manage column index names
            level = column_labels_level(column_labels)
            column_label_names = self._internal.column_label_names[-level:]

        try:
            sdf = self._internal._sdf
            if cond is not None:
                sdf = sdf.drop(NATURAL_ORDER_COLUMN_NAME).filter(cond)
            if limit is not None:
                if limit >= 0:
                    sdf = sdf.limit(limit)
                else:
                    sdf = sdf.limit(sdf.count() + limit)

            sdf = sdf.select(index_scols + column_scols)
        except AnalysisException:
            raise KeyError(
                "[{}] don't exist in columns".format([col._jc.toString() for col in column_scols])
            )

        internal = _InternalFrame(
            sdf=sdf,
            index_map=index_map,
            column_labels=column_labels,
            column_label_names=column_label_names,
        )
        kdf = DataFrame(internal)

        if returns_series:
            kdf_or_kser = Series(kdf._internal.copy(scol=kdf._internal.column_scols[0]), anchor=kdf)
        else:
            kdf_or_kser = kdf

        if remaining_index is not None and remaining_index == 0:
            pdf_or_pser = kdf_or_kser.head(2).to_pandas()
            length = len(pdf_or_pser)
            if length == 0:
                raise KeyError(name_like_string(key))
            elif length == 1:
                return pdf_or_pser.iloc[0]
            else:
                return kdf_or_kser
        else:
            return kdf_or_kser


class LocIndexer(_LocIndexerLike):
    """
    Access a group of rows and columns by label(s) or a boolean Series.

    ``.loc[]`` is primarily label based, but may also be used with a
    conditional boolean Series derived from the DataFrame or Series.

    Allowed inputs are:

    - A single label, e.g. ``5`` or ``'a'``, (note that ``5`` is
      interpreted as a *label* of the index, and **never** as an
      integer position along the index) for column selection.

    - A list or array of labels, e.g. ``['a', 'b', 'c']``.

    - A slice object with labels, e.g. ``'a':'f'``.

    - A conditional boolean Series derived from the DataFrame or Series

    Not allowed inputs which pandas allows are:

    - A boolean array of the same length as the axis being sliced,
      e.g. ``[True, False, True]``.
    - A ``callable`` function with one argument (the calling Series, DataFrame
      or Panel) and that returns valid output for indexing (one of the above)

    .. note:: MultiIndex is not supported yet.

    .. note:: Note that contrary to usual python slices, **both** the
        start and the stop are included, and the step of the slice is not allowed.

    .. note:: With a list or array of labels for row selection,
        Koalas behaves as a filter without reordering by the labels.

    See Also
    --------
    Series.loc : Access group of values using labels.

    Examples
    --------
    **Getting values**

    >>> df = ks.DataFrame([[1, 2], [4, 5], [7, 8]],
    ...                   index=['cobra', 'viper', 'sidewinder'],
    ...                   columns=['max_speed', 'shield'])
    >>> df
                max_speed  shield
    cobra               1       2
    viper               4       5
    sidewinder          7       8

    Single label. Note this returns the row as a Series.

    >>> df.loc['viper']
    max_speed    4
    shield       5
    Name: viper, dtype: int64

    List of labels. Note using ``[[]]`` returns a DataFrame.
    Also note that Koalas behaves just a filter without reordering by the labels.

    >>> df.loc[['viper', 'sidewinder']]
                max_speed  shield
    viper               4       5
    sidewinder          7       8

    >>> df.loc[['sidewinder', 'viper']]
                max_speed  shield
    viper               4       5
    sidewinder          7       8

    Single label for column.

    >>> df.loc['cobra', 'shield']
    2

    List of labels for row.

    >>> df.loc[['cobra'], 'shield']
    cobra    2
    Name: shield, dtype: int64

    List of labels for column.

    >>> df.loc['cobra', ['shield']]
    shield    2
    Name: cobra, dtype: int64

    List of labels for both row and column.

    >>> df.loc[['cobra'], ['shield']]
           shield
    cobra       2

    Slice with labels for row and single label for column. As mentioned
    above, note that both the start and stop of the slice are included.

    >>> df.loc['cobra':'viper', 'max_speed']
    cobra    1
    viper    4
    Name: max_speed, dtype: int64

    Conditional that returns a boolean Series

    >>> df.loc[df['shield'] > 6]
                max_speed  shield
    sidewinder          7       8

    Conditional that returns a boolean Series with column labels specified

    >>> df.loc[df['shield'] > 6, ['max_speed']]
                max_speed
    sidewinder          7

    **Setting values**

    Setting value for all items matching the list of labels.

    >>> df.loc[['viper', 'sidewinder'], ['shield']] = 50
    >>> df
                max_speed  shield
    cobra               1       2
    viper               4      50
    sidewinder          7      50

    Setting value for an entire row is not allowed

    >>> df.loc['cobra'] = 10
    Traceback (most recent call last):
     ...
    databricks.koalas.exceptions.SparkPandasNotImplementedError: ...

    Set value for an entire column

    >>> df.loc[:, 'max_speed'] = 30
    >>> df
                max_speed  shield
    cobra              30       2
    viper              30      50
    sidewinder         30      50

    Set value for an entire list of columns

    >>> df.loc[:, ['max_speed', 'shield']] = 100
    >>> df
                max_speed  shield
    cobra             100     100
    viper             100     100
    sidewinder        100     100

    Set value with Series

    >>> df.loc[:, 'shield'] = df['shield'] * 2
    >>> df
                max_speed  shield
    cobra             100     200
    viper             100     200
    sidewinder        100     200

    **Getting values on a DataFrame with an index that has integer labels**

    Another example using integers for the index

    >>> df = ks.DataFrame([[1, 2], [4, 5], [7, 8]],
    ...                   index=[7, 8, 9],
    ...                   columns=['max_speed', 'shield'])
    >>> df
       max_speed  shield
    7          1       2
    8          4       5
    9          7       8

    Slice with integer labels for rows. As mentioned above, note that both
    the start and stop of the slice are included.

    >>> df.loc[7:9]
       max_speed  shield
    7          1       2
    8          4       5
    9          7       8
    """

    @staticmethod
    def _raiseNotImplemented(description):
        raise SparkPandasNotImplementedError(
            description=description,
            pandas_function=".loc[..., ...]",
            spark_target_function="select, where",
        )

    def _select_rows(self, rows_sel):
        from databricks.koalas.series import Series

        if isinstance(rows_sel, Series):
            assert isinstance(rows_sel.spark_type, BooleanType), rows_sel.spark_type
            return rows_sel._scol, None, None
        elif isinstance(rows_sel, slice):
            assert len(self._internal.index_columns) > 0
            if rows_sel.step is not None:
                LocIndexer._raiseNotImplemented("Cannot use step with Spark.")
            if rows_sel == slice(None):
                # If slice is None - select everything, so nothing to do
                return None, None, None
            elif len(self._internal.index_columns) == 1:
                sdf = self._internal.sdf
                index = self._kdf_or_kser.index
                index_column = index.to_series()
                index_data_type = index_column.spark_type
                start = rows_sel.start
                stop = rows_sel.stop

                # get natural order from '__natural_order__' from start to stop
                # to keep natural order.
                start_and_stop = (
                    sdf.select(index_column._scol, NATURAL_ORDER_COLUMN_NAME)
                    .where(
                        (index_column._scol == F.lit(start).cast(index_data_type))
                        | (index_column._scol == F.lit(stop).cast(index_data_type))
                    )
                    .collect()
                )

                start = [row[1] for row in start_and_stop if row[0] == start]
                start = start[0] if len(start) > 0 else None

                stop = [row[1] for row in start_and_stop if row[0] == stop]
                stop = stop[-1] if len(stop) > 0 else None

                cond = []
                if start is not None:
                    cond.append(F.col(NATURAL_ORDER_COLUMN_NAME) >= F.lit(start).cast(LongType()))
                if stop is not None:
                    cond.append(F.col(NATURAL_ORDER_COLUMN_NAME) <= F.lit(stop).cast(LongType()))

                # if index order is not monotonic increasing or decreasing
                # and specified values don't exist in index, raise KeyError
                if (start is None and rows_sel.start is not None) or (
                    stop is None and rows_sel.stop is not None
                ):
                    inc, dec = (
                        sdf.select(
                            index_column._is_monotonic()._scol.alias("__increasing__"),
                            index_column._is_monotonic_decreasing()._scol.alias("__decreasing__"),
                        )
                        .select(
                            F.min(F.coalesce("__increasing__", F.lit(True))),
                            F.min(F.coalesce("__decreasing__", F.lit(True))),
                        )
                        .first()
                    )
                    if start is None and rows_sel.start is not None:
                        start = rows_sel.start
                        if inc is not False:
                            cond.append(index_column._scol >= F.lit(start).cast(index_data_type))
                        elif dec is not False:
                            cond.append(index_column._scol <= F.lit(start).cast(index_data_type))
                        else:
                            raise KeyError(rows_sel.start)
                    if stop is None and rows_sel.stop is not None:
                        stop = rows_sel.stop
                        if inc is not False:
                            cond.append(index_column._scol <= F.lit(stop).cast(index_data_type))
                        elif dec is not False:
                            cond.append(index_column._scol >= F.lit(stop).cast(index_data_type))
                        else:
                            raise KeyError(rows_sel.stop)

                if len(cond) > 0:
                    return reduce(lambda x, y: x & y, cond), None, None
            else:
                LocIndexer._raiseNotImplemented("Cannot use slice for MultiIndex with Spark.")
        elif is_list_like(rows_sel) and not isinstance(rows_sel, tuple):
            rows_sel = list(rows_sel)
            if len(rows_sel) == 0:
                return F.lit(False), None, None
            elif len(self._internal.index_columns) == 1:
                index_column = self._kdf_or_kser.index.to_series()
                index_data_type = index_column.spark_type
                if len(rows_sel) == 1:
                    return (
                        index_column._scol == F.lit(rows_sel[0]).cast(index_data_type),
                        None,
                        None,
                    )
                else:
                    return (
                        index_column._scol.isin([F.lit(r).cast(index_data_type) for r in rows_sel]),
                        None,
                        None,
                    )
            else:
                LocIndexer._raiseNotImplemented("Cannot select with MultiIndex with Spark.")
        else:
            if not isinstance(rows_sel, tuple):
                rows_sel = (rows_sel,)
            if len(rows_sel) > len(self._internal.index_map):
                raise SparkPandasIndexingError("Too many indexers")

            rows = [scol == value for scol, value in zip(self._internal.index_scols, rows_sel)]
            return (
                reduce(lambda x, y: x & y, rows),
                None,
                len(self._internal.index_map) - len(rows_sel),
            )

    def _get_from_multiindex_column(self, key, labels=None):
        """ Select columns from multi-index columns.

        :param key: the multi-index column keys represented by tuple
        :return: DataFrame or Series
        """
        assert isinstance(key, tuple)
        if labels is None:
            labels = [(label, label) for label in self._internal.column_labels]
        for k in key:
            labels = [(label, lbl[1:]) for label, lbl in labels if lbl[0] == k]
            if len(labels) == 0:
                raise KeyError(k)

        if all(len(lbl) > 0 and lbl[0] == "" for _, lbl in labels):
            # If the head is '', drill down recursively.
            labels = [(label, tuple([str(key), *lbl[1:]])) for i, (label, lbl) in enumerate(labels)]
            return self._get_from_multiindex_column((str(key),), labels)
        else:
            returns_series = all(len(lbl) == 0 for _, lbl in labels)
            if returns_series:
                labels = set(label for label, _ in labels)
                assert len(labels) == 1
                label = list(labels)[0]
                column_labels = [label]
                column_scols = [self._internal.scol_for(label)]
            else:
                column_labels = [lbl for _, lbl in labels]
                column_scols = [self._internal.scol_for(label) for label, _ in labels]

        return column_labels, column_scols, returns_series

    def _select_cols(self, cols_sel):
        from databricks.koalas.series import Series

        returns_series = False

        if isinstance(cols_sel, slice):
            if cols_sel == slice(None):
                cols_sel = None
            else:
                raise LocIndexer._raiseNotImplemented(
                    "Can only select columns either by name or reference or all"
                )
        elif isinstance(cols_sel, (Series, spark.Column)):
            returns_series = True
            cols_sel = [cols_sel]

        if cols_sel is None:
            column_labels = self._internal.column_labels
            column_scols = self._internal.column_scols
        elif isinstance(cols_sel, (str, tuple)):
            if isinstance(cols_sel, str):
                cols_sel = (cols_sel,)
            return self._get_from_multiindex_column(cols_sel)
        elif all(isinstance(key, Series) for key in cols_sel):
            column_labels = [key._internal.column_labels[0] for key in cols_sel]
            column_scols = [key._scol for key in cols_sel]
        elif all(isinstance(key, spark.Column) for key in cols_sel):
            column_labels = [(self._internal.sdf.select(col).columns[0],) for col in cols_sel]
            column_scols = cols_sel
        elif any(isinstance(key, str) for key in cols_sel) and any(
            isinstance(key, tuple) for key in cols_sel
        ):
            raise TypeError("Expected tuple, got str")
        else:
            if all(isinstance(key, tuple) for key in cols_sel):
                level = self._internal.column_labels_level
                if any(len(key) != level for key in cols_sel):
                    raise ValueError("All the key level should be the same as column index level.")

            column_labels = []
            column_scols = []
            for key in cols_sel:
                found = False
                for label in self._internal.column_labels:
                    if label == key or label[0] == key:
                        column_labels.append(label)
                        column_scols.append(self._internal.scol_for(label))
                        found = True
                if not found:
                    raise KeyError("['{}'] not in index".format(name_like_string(key)))

        return column_labels, column_scols, returns_series

    def __setitem__(self, key, value):
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series, _col

        if self._is_series:
            raise SparkPandasNotImplementedError(
                description="Can only assign value to dataframes",
                pandas_function=".loc[..., ...] = ...",
                spark_target_function="withColumn, select",
            )

        if (not isinstance(key, tuple)) or (len(key) != 2):
            raise SparkPandasNotImplementedError(
                description="Only accepts pairs of candidates",
                pandas_function=".loc[..., ...] = ...",
                spark_target_function="withColumn, select",
            )

        rows_sel, cols_sel = key

        if (not isinstance(rows_sel, slice)) or (rows_sel != slice(None)):
            if isinstance(rows_sel, list):
                if isinstance(cols_sel, str):
                    cols_sel = [cols_sel]
                kdf = self._kdf_or_kser.copy()
                for col_sel in cols_sel:
                    # Uses `kdf` to allow operations on different DataFrames.
                    # TODO: avoid temp column name or declare `__` prefix is
                    #  reserved for Koalas' internal columns.
                    kdf["__indexing_temp_col__"] = value
                    new_col = kdf["__indexing_temp_col__"]._scol
                    kdf[col_sel] = Series(
                        kdf[col_sel]._internal.copy(
                            scol=F.when(
                                kdf._internal.index_scols[0].isin(rows_sel), new_col
                            ).otherwise(kdf[col_sel]._scol)
                        ),
                        anchor=kdf,
                    )
                    kdf = kdf.drop(labels=["__indexing_temp_col__"])

                self._kdf_or_kser._internal = kdf._internal.copy()
            else:
                raise SparkPandasNotImplementedError(
                    description="""Can only assign value to the whole dataframe, the row index
                    has to be `slice(None)` or `:`""",
                    pandas_function=".loc[..., ...] = ...",
                    spark_target_function="withColumn, select",
                )

        if not isinstance(cols_sel, (str, list)):
            raise ValueError("""only column names or list of column names can be assigned""")

        if isinstance(value, DataFrame):
            if len(value.columns) == 1:
                self._kdf_or_kser[cols_sel] = _col(value)
            else:
                raise ValueError("Only a dataframe with one column can be assigned")
        else:
            if isinstance(cols_sel, str):
                cols_sel = [cols_sel]
            if (not isinstance(rows_sel, list)) and (isinstance(cols_sel, list)):
                for col_sel in cols_sel:
                    self._kdf_or_kser[col_sel] = value


class iLocIndexer(_LocIndexerLike):
    """
    Purely integer-location based indexing for selection by position.

    ``.iloc[]`` is primarily integer position based (from ``0`` to
    ``length-1`` of the axis), but may also be used with a conditional boolean Series.

    Allowed inputs are:

    - An integer for column selection, e.g. ``5``.
    - A list or array of integers for column selection, e.g. ``[4, 3, 0]``.
    - A boolean array for column selection.
    - A slice object with ints for column selection, e.g. ``1:7``.
    - A slice object with ints without start and step for row selection, e.g. ``:7``.
    - A conditional boolean Index for row selection.

    Not allowed inputs which pandas allows are:

    - An integer for row selection, e.g. ``5``.
    - A list or array of integers for row selection, e.g. ``[4, 3, 0]``.
    - A boolean array for row selection.
    - A ``callable`` function with one argument (the calling Series, DataFrame
      or Panel) and that returns valid output for indexing (one of the above).
      This is useful in method chains, when you don't have a reference to the
      calling object, but would like to base your selection on some value.

    ``.iloc`` will raise ``IndexError`` if a requested indexer is
    out-of-bounds, except *slice* indexers which allow out-of-bounds
    indexing (this conforms with python/numpy *slice* semantics).

    See Also
    --------
    DataFrame.loc : Purely label-location based indexer for selection by label.
    Series.iloc : Purely integer-location based indexing for
                   selection by position.

    Examples
    --------

    >>> mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
    ...           {'a': 100, 'b': 200, 'c': 300, 'd': 400},
    ...           {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
    >>> df = ks.DataFrame(mydict, columns=['a', 'b', 'c', 'd'])
    >>> df
          a     b     c     d
    0     1     2     3     4
    1   100   200   300   400
    2  1000  2000  3000  4000

    **Indexing just the rows**

    A scalar integer for row selection.

    >>> df.iloc[0]
    a    1
    b    2
    c    3
    d    4
    Name: 0, dtype: int64

    A list of integers for row selection is not allowed.

    >>> df.iloc[[0]]
    Traceback (most recent call last):
     ...
    databricks.koalas.exceptions.SparkPandasNotImplementedError: ...

    With a `slice` object.

    >>> df.iloc[:3]
          a     b     c     d
    0     1     2     3     4
    1   100   200   300   400
    2  1000  2000  3000  4000

    Conditional that returns a boolean Series

    >>> df.iloc[df.index % 2 == 0]
          a     b     c     d
    0     1     2     3     4
    2  1000  2000  3000  4000

    **Indexing both axes**

    You can mix the indexer types for the index and columns. Use ``:`` to
    select the entire axis.

    With scalar integers.

    >>> df.iloc[:1, 1]
    0    2
    Name: b, dtype: int64

    With lists of integers.

    >>> df.iloc[:2, [1, 3]]
         b    d
    0    2    4
    1  200  400

    With `slice` objects.

    >>> df.iloc[:2, 0:3]
         a    b    c
    0    1    2    3
    1  100  200  300

    With a boolean array whose length matches the columns.

    >>> df.iloc[:, [True, False, True, False]]
          a     c
    0     1     3
    1   100   300
    2  1000  3000
    """

    @staticmethod
    def _raiseNotImplemented(description):
        raise SparkPandasNotImplementedError(
            description=description,
            pandas_function=".iloc[..., ...]",
            spark_target_function="select, where",
        )

    @lazy_property
    def _internal(self):
        internal = super(iLocIndexer, self)._internal
        sdf = _InternalFrame.attach_distributed_sequence_column(
            internal.sdf, column_name=self._sequence_col
        )
        return internal.with_new_sdf(sdf.orderBy(NATURAL_ORDER_COLUMN_NAME))

    @lazy_property
    def _sequence_col(self):
        internal = super(iLocIndexer, self)._internal
        return verify_temp_column_name(internal.sdf, "__distributed_sequence_column__")

    def _select_rows(self, rows_sel):
        from databricks.koalas.indexes import Index

        if isinstance(rows_sel, tuple) and len(rows_sel) > 1:
            raise SparkPandasIndexingError("Too many indexers")
        elif isinstance(rows_sel, Index):
            assert isinstance(rows_sel.spark_type, BooleanType), rows_sel.spark_type
            return rows_sel._scol, None, None
        elif isinstance(rows_sel, slice):
            if rows_sel == slice(None):
                # If slice is None - select everything, so nothing to do
                return None, None, None
            elif (rows_sel.start is not None) or (rows_sel.step is not None):
                iLocIndexer._raiseNotImplemented("Cannot use start or step with Spark.")
            elif not isinstance(rows_sel.stop, int):
                raise TypeError(
                    "cannot do slice indexing with these indexers [{}] of {}".format(
                        rows_sel.stop, type(rows_sel.stop)
                    )
                )
            else:
                return None, rows_sel.stop, None
        elif isinstance(rows_sel, int):
            sdf = self._internal.sdf
            return (sdf[self._sequence_col] == rows_sel), None, 0
        else:
            iLocIndexer._raiseNotImplemented(
                ".iloc requires numeric slice or conditional "
                "boolean Index, got {}".format(type(rows_sel))
            )

    def _select_cols(self, cols_sel):
        from databricks.koalas.series import Series

        returns_series = cols_sel is not None and isinstance(cols_sel, (Series, int))

        # make cols_sel a 1-tuple of string if a single string
        if isinstance(cols_sel, Series) and cols_sel._equals(self._kdf_or_kser):
            column_labels = cols_sel._internal.column_labels
            column_scols = cols_sel._internal.column_scols
        elif isinstance(cols_sel, int):
            if cols_sel > len(self._internal.column_labels):
                raise KeyError(cols_sel)
            column_labels = [self._internal.column_labels[cols_sel]]
            column_scols = [self._internal.column_scols[cols_sel]]
        elif cols_sel is None or cols_sel == slice(None):
            column_labels = self._internal.column_labels
            column_scols = self._internal.column_scols
        elif isinstance(cols_sel, slice):
            if all(
                s is None or isinstance(s, int)
                for s in (cols_sel.start, cols_sel.stop, cols_sel.step)
            ):
                column_labels = self._internal.column_labels[cols_sel]
                column_scols = self._internal.column_scols[cols_sel]
            else:
                not_none = (
                    cols_sel.start
                    if cols_sel.start is not None
                    else cols_sel.stop
                    if cols_sel.stop is not None
                    else cols_sel.step
                )
                raise TypeError(
                    "cannot do slice indexing with these indexers {} of {}".format(
                        not_none, type(not_none)
                    )
                )
        elif is_list_like(cols_sel):
            if all(isinstance(s, bool) for s in cols_sel):
                cols_sel = [i for i, s in enumerate(cols_sel) if s]
            if all(isinstance(s, int) for s in cols_sel):
                column_labels = [self._internal.column_labels[s] for s in cols_sel]
                column_scols = [self._internal.column_scols[s] for s in cols_sel]
            else:
                raise TypeError("cannot perform reduce with flexible type")
        else:
            raise ValueError(
                "Location based indexing can only have [integer, integer slice, "
                "listlike of integers, boolean array] types, got {}".format(cols_sel)
            )

        return column_labels, column_scols, returns_series
