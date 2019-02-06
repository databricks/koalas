"""
A locator for PandasLikeDataFrame.
"""
import sys
from functools import reduce

from pyspark.sql import Column, DataFrame
from pyspark.sql.types import BooleanType

from .exceptions import SparkPandasIndexingError, SparkPandasNotImplementedError

if sys.version > '3':
    basestring = unicode = str


def _make_col(c):
    if isinstance(c, Column):
        return c
    elif isinstance(c, str):
        from pyspark.sql.functions import _spark_col
        return _spark_col(c)
    else:
        raise SparkPandasNotImplementedError(
            description="Can only convert a string to a column type")


def _unfold(key, col):
    about_cols = """Can only select columns either by name or reference or all"""

    if col is not None:
        if isinstance(key, tuple):
            if len(key) > 1:
                raise SparkPandasIndexingError('Too many indexers')
            key = key[0]
        rows = key
        cols = col
    elif isinstance(key, tuple):
        if len(key) != 2:
            raise SparkPandasIndexingError("Only accepts pairs of candidates")
        rows, cols = key

        # make cols a 1-tuple of string if a single string
        if isinstance(cols, (str, Column)):
            cols = _make_col(cols)
        elif isinstance(cols, slice) and cols != slice(None):
            raise SparkPandasNotImplementedError(
                description=about_cols,
                pandas_function="loc",
                spark_target_function="select, where, withColumn")
        elif isinstance(cols, slice) and cols == slice(None):
            cols = None
    else:
        rows = key
        cols = None

    return rows, cols


class SparkDataFrameLocator(object):
    """
    A locator to slice a group of rows and columns by conditional and label(s).

    Allowed inputs are a slice with all indices or conditional for rows, and string(s) or
    :class:`Column`(s) for cols.
    """

    def __init__(self, df_or_col):
        assert isinstance(df_or_col, (DataFrame, Column)), \
            'unexpected argument: {}'.format(df_or_col)
        if isinstance(df_or_col, DataFrame):
            self.df = df_or_col
            self.col = None
        else:
            self.df = df_or_col._pandas_anchor
            self.col = df_or_col

    def __getitem__(self, key):

        def raiseNotImplemented():
            about_rows = """Can only slice with all indices or a column that evaluates to Boolean"""
            raise SparkPandasNotImplementedError(
                description=about_rows,
                pandas_function=".loc[..., ...]",
                spark_target_function="select, where")

        rows, cols = _unfold(key, self.col)

        df = self.df
        if isinstance(rows, Column):
            assert isinstance(self.df._spark_select(rows).schema.fields[0].dataType, BooleanType)
            df = df._spark_where(rows)
        elif isinstance(rows, basestring):
            raiseNotImplemented()
        elif isinstance(rows, slice):
            if rows.step is not None:
                raiseNotImplemented()
            if len(self.df._index_columns) == 1:
                start = rows.start
                stop = rows.stop

                index_column = self.df._index_columns[0]
                cond = []
                if start is not None:
                    cond.append(index_column >= start)
                if stop is not None:
                    cond.append(index_column <= stop)

                if len(cond) > 0:
                    df = df._spark_where(reduce(lambda x, y: x & y, cond))
            else:
                raiseNotImplemented()
        else:
            try:
                rows = list(rows)
                if len(rows) == 0:
                    from pyspark.sql.functions import _spark_lit
                    df = df._spark_where(_spark_lit(False))
                if len(self.df._index_columns) == 1:
                    index_column = self.df._index_columns[0]
                    if len(rows) == 1:
                        df = df._spark_where(index_column == rows[0])
                    else:
                        df = df._spark_where(index_column.isin(rows))
                else:
                    raiseNotImplemented()
            except Exception:
                raiseNotImplemented()
        if cols is None:
            columns = [_make_col(c) for c in self.df._metadata.columns]
        elif isinstance(cols, Column):
            columns = [cols]
        else:
            columns = [_make_col(c) for c in cols]
        df = df._spark_select(self.df._metadata._index_columns + columns)
        df._metadata = self.df._metadata.copy(columns=[col.name for col in columns])
        if cols is not None and isinstance(cols, Column):
            from .structures import _col
            return _col(df)
        else:
            return df

    def __setitem__(self, key, value):

        if (not isinstance(key, tuple)) or (len(key) != 2):
            raise NotImplementedError("Only accepts pairs of candidates")

        rows, cols = key

        if (not isinstance(rows, slice)) or (rows != slice(None)):
            raise SparkPandasNotImplementedError(
                description="""Can only assign value to the whole dataframe, the row index
                has to be `slice(None)` or `:`""",
                pandas_function=".loc[..., ...] = ...",
                spark_target_function="withColumn, select")

        if not isinstance(cols, str):
            raise ValueError("""only column names can be assigned""")

        if isinstance(value, Column):
            self.df[cols] = value
        elif isinstance(value, DataFrame) and len(value.columns) == 1:
            from pyspark.sql.functions import _spark_col
            self.df[cols] = _spark_col(value.columns[0])
        elif isinstance(value, DataFrame) and len(value.columns) != 1:
            raise ValueError("Only a dataframe with one column can be assigned")
        else:
            raise ValueError("Only a column or dataframe with single column can be assigned")
