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
from pyspark.sql.types import BooleanType
from pyspark.sql.utils import AnalysisException

from databricks.koalas.exceptions import SparkPandasIndexingError, SparkPandasNotImplementedError


def _make_col(c):
    from databricks.koalas.series import Series
    if isinstance(c, Series):
        return c._scol
    elif isinstance(c, str):
        return F.col(c)
    else:
        raise SparkPandasNotImplementedError(
            description="Can only convert a string to a column type.")


def _unfold(key, kseries):
    """ Return row selection and column selection pair.

    If kseries parameter is not None, the key should be row selection and the column selection will
    be the kseries parameter.

    >>> s = ks.Series([1, 2, 3], name='a')
    >>> _unfold(slice(1, 2), s)
    (slice(1, 2, None), 0    1
    1    2
    2    3
    Name: a, dtype: int64)

    >>> _unfold((slice(1, 2), slice(None)), None)
    (slice(1, 2, None), slice(None, None, None))

    >>> _unfold((slice(1, 2), s), None)
    (slice(1, 2, None), 0    1
    1    2
    2    3
    Name: a, dtype: int64)

    >>> _unfold((slice(1, 2), 'col'), None)
    (slice(1, 2, None), 'col')
    """
    if kseries is not None:
        if isinstance(key, tuple):
            if len(key) > 1:
                raise SparkPandasIndexingError('Too many indexers')
            key = key[0]
        rows_sel = key
        cols_sel = kseries
    elif isinstance(key, tuple):
        if len(key) != 2:
            raise SparkPandasIndexingError("Only accepts pairs of candidates")
        rows_sel, cols_sel = key
    else:
        rows_sel = key
        cols_sel = None

    return rows_sel, cols_sel


class AtIndexer(object):
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
    def __init__(self, df_or_s):
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series
        assert isinstance(df_or_s, (DataFrame, Series)), \
            'unexpected argument type: {}'.format(type(df_or_s))
        if isinstance(df_or_s, DataFrame):
            self._kdf = df_or_s
            self._ks = None
        else:
            # If df_or_col is Column, store both the DataFrame anchored to the Column and
            # the Column itself.
            self._kdf = df_or_s._kdf
            self._ks = df_or_s

    def __getitem__(self, key):
        if self._ks is None and (not isinstance(key, tuple) or len(key) != 2):
            raise TypeError("Use DataFrame.at like .at[row_index, column_name]")
        if self._ks is not None and not isinstance(key, str) and len(key) != 1:
            raise TypeError("Use Series.at like .at[row_index]")

        # TODO Maybe extend to multilevel indices in the future
        if len(self._kdf._internal.index_columns) != 1:
            raise ValueError("'.at' only supports indices with level 1 right now")

        column = key[1] if self._ks is None else self._ks.name
        if column is not None and column not in self._kdf._internal.data_columns:
            raise KeyError("%s" % column)
        series = self._ks if self._ks is not None else self._kdf[column]

        row = key[0] if self._ks is None else key
        pdf = (series._kdf._sdf
               .where(F.col(self._kdf._internal.index_columns[0]) == row)
               .select(column)
               .toPandas())
        if len(pdf) < 1:
            raise KeyError("%s" % row)

        values = pdf.iloc[:, 0].values
        return values[0] if len(values) == 1 else values


class LocIndexer(object):
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
        In addition, with a slice, Koalas works as a filter between the range.

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

    A single label for row selection is not allowed.

    >>> df.loc['viper']
    Traceback (most recent call last):
     ...
    databricks.koalas.exceptions.SparkPandasNotImplementedError: ...

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

    Single label for column

    >>> df.loc[['cobra'], 'shield']
    cobra    2
    Name: shield, dtype: int64

    List of labels for column. Note using list returns a DataFrame.

    >>> df.loc[['cobra'], ['shield']]
           shield
    cobra       2

    Slice with labels for row and single label for column. As mentioned
    above, note that both the start and stop of the slice are included.

    Also note that the row for 'sidewinder' is included since 'sidewinder'
    is between 'cobra' and 'viper'.

    >>> df.loc['cobra':'viper', 'max_speed']
    cobra         1
    viper         4
    sidewinder    7
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

    Setting value for all items matching the list of labels is not allowed

    >>> df.loc[['viper', 'sidewinder'], ['shield']] = 50
    Traceback (most recent call last):
     ...
    databricks.koalas.exceptions.SparkPandasNotImplementedError: ...

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
    viper              30       5
    sidewinder         30       8

    Set value with Series

    >>> df.loc[:, 'shield'] = df['shield'] * 2
    >>> df
                max_speed  shield
    cobra              30       4
    viper              30      10
    sidewinder         30      16

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

    def __init__(self, df_or_s):
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series
        assert isinstance(df_or_s, (DataFrame, Series)), \
            'unexpected argument type: {}'.format(type(df_or_s))
        if isinstance(df_or_s, DataFrame):
            self._kdf = df_or_s
            self._ks = None
        else:
            # If df_or_col is Column, store both the DataFrame anchored to the Column and
            # the Column itself.
            self._kdf = df_or_s._kdf
            self._ks = df_or_s

    def __getitem__(self, key):
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series

        def raiseNotImplemented(description):
            raise SparkPandasNotImplementedError(
                description=description,
                pandas_function=".loc[..., ...]",
                spark_target_function="select, where")

        rows_sel, cols_sel = _unfold(key, self._ks)

        sdf = self._kdf._sdf
        if isinstance(rows_sel, Series):
            sdf_for_check_schema = sdf.select(rows_sel._scol)
            assert isinstance(sdf_for_check_schema.schema.fields[0].dataType, BooleanType), \
                (str(sdf_for_check_schema), sdf_for_check_schema.schema.fields[0].dataType)
            sdf = sdf.where(rows_sel._scol)
        elif isinstance(rows_sel, slice):
            if rows_sel.step is not None:
                raiseNotImplemented("Cannot use step with Spark.")
            if rows_sel == slice(None):
                # If slice is None - select everything, so nothing to do
                pass
            elif len(self._kdf._internal.index_columns) == 0:
                raiseNotImplemented("Cannot use slice for Spark if no index provided.")
            elif len(self._kdf._internal.index_columns) == 1:
                start = rows_sel.start
                stop = rows_sel.stop

                index_column = self._kdf.index.to_series()
                index_data_type = index_column.schema[0].dataType
                cond = []
                if start is not None:
                    cond.append(index_column._scol >= F.lit(start).cast(index_data_type))
                if stop is not None:
                    cond.append(index_column._scol <= F.lit(stop).cast(index_data_type))

                if len(cond) > 0:
                    sdf = sdf.where(reduce(lambda x, y: x & y, cond))
            else:
                raiseNotImplemented("Cannot use slice for MultiIndex with Spark.")
        elif isinstance(rows_sel, str):
            raiseNotImplemented("Cannot use a scalar value for row selection with Spark.")
        else:
            try:
                rows_sel = list(rows_sel)
            except TypeError:
                raiseNotImplemented("Cannot use a scalar value for row selection with Spark.")
            if len(rows_sel) == 0:
                sdf = sdf.where(F.lit(False))
            elif len(self._kdf._internal.index_columns) == 1:
                index_column = self._kdf.index.to_series()
                index_data_type = index_column.schema[0].dataType
                if len(rows_sel) == 1:
                    sdf = sdf.where(
                        index_column._scol == F.lit(rows_sel[0]).cast(index_data_type))
                else:
                    sdf = sdf.where(index_column._scol.isin(
                        [F.lit(r).cast(index_data_type) for r in rows_sel]))
            else:
                raiseNotImplemented("Cannot select with MultiIndex with Spark.")

        # make cols_sel a 1-tuple of string if a single string
        if isinstance(cols_sel, (str, Series)):
            cols_sel = _make_col(cols_sel)
        elif isinstance(cols_sel, slice) and cols_sel != slice(None):
            raise raiseNotImplemented("Can only select columns either by name or reference or all")
        elif isinstance(cols_sel, slice) and cols_sel == slice(None):
            cols_sel = None

        if cols_sel is None:
            columns = [_make_col(c) for c in self._kdf._internal.data_columns]
        elif isinstance(cols_sel, spark.Column):
            columns = [cols_sel]
        else:
            columns = [_make_col(c) for c in cols_sel]
        try:
            kdf = DataFrame(sdf.select(self._kdf._internal.index_columns + columns))
        except AnalysisException:
            raise KeyError('[{}] don\'t exist in columns'
                           .format([col._jc.toString() for col in columns]))
        kdf._internal = kdf._internal.copy(
            data_columns=kdf._internal.data_columns[-len(columns):],
            index_map=self._kdf._internal.index_map)
        if cols_sel is not None and isinstance(cols_sel, spark.Column):
            from databricks.koalas.series import _col
            return _col(kdf)
        else:
            return kdf

    def __setitem__(self, key, value):
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series, _col

        if (not isinstance(key, tuple)) or (len(key) != 2):
            raise SparkPandasNotImplementedError(
                description="Only accepts pairs of candidates",
                pandas_function=".loc[..., ...] = ...",
                spark_target_function="withColumn, select")

        rows_sel, cols_sel = key

        if (not isinstance(rows_sel, slice)) or (rows_sel != slice(None)):
            raise SparkPandasNotImplementedError(
                description="""Can only assign value to the whole dataframe, the row index
                has to be `slice(None)` or `:`""",
                pandas_function=".loc[..., ...] = ...",
                spark_target_function="withColumn, select")

        if not isinstance(cols_sel, str):
            raise ValueError("""only column names can be assigned""")

        if isinstance(value, DataFrame):
            if len(value.columns) == 1:
                self._kdf[cols_sel] = _col(value)
            else:
                raise ValueError("Only a dataframe with one column can be assigned")
        else:
            self._kdf[cols_sel] = value


class ILocIndexer(object):
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

    A scalar integer for row selection is not allowed.

    >>> df.iloc[0]
    Traceback (most recent call last):
     ...
    databricks.koalas.exceptions.SparkPandasNotImplementedError: ...

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

    def __init__(self, df_or_s):
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series
        assert isinstance(df_or_s, (DataFrame, Series)), \
            'unexpected argument type: {}'.format(type(df_or_s))
        if isinstance(df_or_s, DataFrame):
            self._kdf = df_or_s
            self._ks = None
        else:
            # If df_or_col is Column, store both the DataFrame anchored to the Column and
            # the Column itself.
            self._kdf = df_or_s._kdf
            self._ks = df_or_s

    def __getitem__(self, key):
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.indexes import Index
        from databricks.koalas.series import Series

        def raiseNotImplemented(description):
            raise SparkPandasNotImplementedError(
                description=description,
                pandas_function=".iloc[..., ...]",
                spark_target_function="select, where")

        rows_sel, cols_sel = _unfold(key, self._ks)

        sdf = self._kdf._sdf
        if isinstance(rows_sel, Index):
            sdf_for_check_schema = sdf.select(rows_sel._scol)
            assert isinstance(sdf_for_check_schema.schema.fields[0].dataType, BooleanType), \
                (str(sdf_for_check_schema), sdf_for_check_schema.schema.fields[0].dataType)
            sdf = sdf.where(rows_sel._scol)
        elif isinstance(rows_sel, slice):
            if rows_sel == slice(None):
                # If slice is None - select everything, so nothing to do
                pass
            elif (rows_sel.start is not None) or (rows_sel.step is not None):
                raiseNotImplemented("Cannot use start or step with Spark.")
            elif not isinstance(rows_sel.stop, int):
                raise TypeError("cannot do slice indexing with these indexers [{}] of {}"
                                .format(rows_sel.stop, type(rows_sel.stop)))
            elif rows_sel.stop >= 0:
                sdf = sdf.limit(rows_sel.stop)
            else:
                sdf = sdf.limit(sdf.count() + rows_sel.stop)
        else:
            raiseNotImplemented(".iloc requires numeric slice or conditional boolean Index, "
                                "got {}".format(rows_sel))

        # make cols_sel a 1-tuple of string if a single string
        if isinstance(cols_sel, Series):
            columns = [_make_col(cols_sel)]
        elif isinstance(cols_sel, int):
            columns = [_make_col(self._kdf.columns[cols_sel])]
        elif cols_sel is None or cols_sel == slice(None):
            columns = [_make_col(col) for col in self._kdf.columns]
        elif isinstance(cols_sel, slice):
            if all(s is None or isinstance(s, int)
                   for s in (cols_sel.start, cols_sel.stop, cols_sel.step)):
                columns = [_make_col(col) for col in self._kdf.columns[cols_sel]]
            else:
                not_none = cols_sel.start if cols_sel.start is not None \
                    else cols_sel.stop if cols_sel.stop is not None else cols_sel.step
                raise TypeError('cannot do slice indexing with these indexers {} of {}'
                                .format(not_none, type(not_none)))
        elif is_list_like(cols_sel):
            if all(isinstance(s, int) for s in cols_sel):
                columns = [_make_col(col) for col in self._kdf.columns[cols_sel]]
            else:
                raise TypeError('cannot perform reduce with flexible type')
        else:
            raise ValueError("Location based indexing can only have [integer, integer slice, "
                             "listlike of integers, boolean array] types, got {}".format(cols_sel))

        try:
            kdf = DataFrame(sdf.select(self._kdf._internal.index_columns + columns))
        except AnalysisException:
            raise KeyError('[{}] don\'t exist in columns'
                           .format([col._jc.toString() for col in columns]))
        kdf._internal = kdf._internal.copy(
            data_columns=kdf._internal.data_columns[-len(columns):],
            index_map=self._kdf._internal.index_map)
        if cols_sel is not None and isinstance(cols_sel, (Series, int)):
            from databricks.koalas.series import _col
            return _col(kdf)
        else:
            return kdf
