"""
A locator for PandasLikeDataFrame.
"""
from functools import reduce

from pyspark.sql import Column, DataFrame
from pyspark.sql.types import BooleanType
from pyspark.sql.utils import AnalysisException

from ._dask_stubs.compatibility import string_types
from .exceptions import SparkPandasIndexingError, SparkPandasNotImplementedError


def _make_col(c):
    if isinstance(c, Column):
        return c
    elif isinstance(c, str):
        from pyspark.sql.functions import _spark_col
        return _spark_col(c)
    else:
        raise SparkPandasNotImplementedError(
            description="Can only convert a string to a column type.")


def _unfold(key, col):
    if col is not None:
        if isinstance(key, tuple):
            if len(key) > 1:
                raise SparkPandasIndexingError('Too many indexers')
            key = key[0]
        rows_sel = key
        cols_sel = col
    elif isinstance(key, tuple):
        if len(key) != 2:
            raise SparkPandasIndexingError("Only accepts pairs of candidates")
        rows_sel, cols_sel = key

        # make cols_sel a 1-tuple of string if a single string
        if isinstance(cols_sel, (str, Column)):
            cols_sel = _make_col(cols_sel)
        elif isinstance(cols_sel, slice) and cols_sel != slice(None):
            raise SparkPandasNotImplementedError(
                description="Can only select columns either by name or reference or all",
                pandas_function="loc",
                spark_target_function="select, where, withColumn")
        elif isinstance(cols_sel, slice) and cols_sel == slice(None):
            cols_sel = None
    else:
        rows_sel = key
        cols_sel = None

    return rows_sel, cols_sel


class SparkDataFrameLocator(object):
    """
    A locator to slice a group of rows and columns by conditional and label(s).

    Allowed inputs are a slice with all indices or conditional for rows, and string(s) or
    :class:`Column`(s) for cols.
    """

    def __init__(self, df_or_col):
        assert isinstance(df_or_col, (DataFrame, Column)), \
            'unexpected argument type: {}'.format(type(df_or_col))
        if isinstance(df_or_col, DataFrame):
            self.df = df_or_col
            self.col = None
        else:
            self.df = df_or_col._pandas_anchor
            self.col = df_or_col

    def __getitem__(self, key):
        from pyspark.sql.functions import _spark_lit

        def raiseNotImplemented(description):
            raise SparkPandasNotImplementedError(
                description=description,
                pandas_function=".loc[..., ...]",
                spark_target_function="select, where")

        rows_sel, cols_sel = _unfold(key, self.col)

        df = self.df
        if isinstance(rows_sel, Column):
            df_for_check_schema = self.df._spark_select(rows_sel)
            assert isinstance(df_for_check_schema.schema.fields[0].dataType, BooleanType), \
                (str(df_for_check_schema), df_for_check_schema.schema.fields[0].dataType)
            df = df._spark_where(rows_sel)
        elif isinstance(rows_sel, slice):
            if rows_sel.step is not None:
                raiseNotImplemented("Cannot use step with Spark.")
            if len(self.df._index_columns) == 1:
                start = rows_sel.start
                stop = rows_sel.stop

                index_column = self.df._index_columns[0]
                index_data_type = index_column.schema[0].dataType
                cond = []
                if start is not None:
                    cond.append(index_column >=
                                _spark_lit(start)._spark_cast(index_data_type))
                if stop is not None:
                    cond.append(index_column <=
                                _spark_lit(stop)._spark_cast(index_data_type))

                if len(cond) > 0:
                    df = df._spark_where(reduce(lambda x, y: x & y, cond))
            else:
                raiseNotImplemented("Cannot use slice for MultiIndex with Spark.")
        elif isinstance(rows_sel, string_types):
            raiseNotImplemented("Cannot use a scalar value for row selection with Spark.")
        else:
            try:
                rows_sel = list(rows_sel)
            except TypeError:
                raiseNotImplemented("Cannot use a scalar value for row selection with Spark.")
            if len(rows_sel) == 0:
                df = df._spark_where(_spark_lit(False))
            if len(self.df._index_columns) == 1:
                index_column = self.df._index_columns[0]
                index_data_type = index_column.schema[0].dataType
                if len(rows_sel) == 1:
                    df = df._spark_where(
                        index_column == _spark_lit(rows_sel[0])._spark_cast(index_data_type))
                else:
                    df = df._spark_where(index_column._spark_isin(
                        [_spark_lit(r)._spark_cast(index_data_type) for r in rows_sel]))
            else:
                raiseNotImplemented("Cannot select with MultiIndex with Spark.")
        if cols_sel is None:
            columns = [_make_col(c) for c in self.df._metadata._column_fields]
        elif isinstance(cols_sel, Column):
            columns = [cols_sel]
        else:
            columns = [_make_col(c) for c in cols_sel]
        try:
            df = df._spark_select(self.df._metadata.index_columns + columns)
        except AnalysisException:
            raise KeyError('[{}] don\'t exist in columns'
                           .format([col.name for col in columns]))
        df._metadata = self.df._metadata.copy(column_fields=[col.name for col in columns])
        if cols_sel is not None and isinstance(cols_sel, Column):
            from .structures import _col
            return _col(df)
        else:
            return df

    def __setitem__(self, key, value):

        if (not isinstance(key, tuple)) or (len(key) != 2):
            raise NotImplementedError("Only accepts pairs of candidates")

        rows_sel, cols_sel = key

        if (not isinstance(rows_sel, slice)) or (rows_sel != slice(None)):
            raise SparkPandasNotImplementedError(
                description="""Can only assign value to the whole dataframe, the row index
                has to be `slice(None)` or `:`""",
                pandas_function=".loc[..., ...] = ...",
                spark_target_function="withColumn, select")

        if not isinstance(cols_sel, str):
            raise ValueError("""only column names can be assigned""")

        if isinstance(value, Column):
            self.df[cols_sel] = value
        elif isinstance(value, DataFrame) and len(value.columns) == 1:
            from pyspark.sql.functions import _spark_col
            self.df[cols_sel] = _spark_col(value.columns[0])
        elif isinstance(value, DataFrame) and len(value.columns) != 1:
            raise ValueError("Only a dataframe with one column can be assigned")
        else:
            raise ValueError("Only a column or dataframe with single column can be assigned")
