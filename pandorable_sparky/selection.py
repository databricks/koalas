"""
A locator for PandasLikeDataFrame.
"""
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import BooleanType

from .exceptions import PandorableSparkyNotImplementedError


def _make_col(c):
    if isinstance(c, Column):
        return c
    elif isinstance(c, str):
        return col(c)
    else:
        raise PandorableSparkyNotImplementedError(
            description="Can only convert a string to a column type")


def _unfold(key):
    about_cols = """Can only select columns either by name or reference or all"""

    if (not isinstance(key, tuple)) or (len(key) != 2):
        raise NotImplementedError("Only accepts pairs of candidates")

    rows, cols = key
    # make cols a 1-tuple of string if a single string
    if isinstance(cols, (str, Column)):
        cols = (cols,)
    elif isinstance(cols, slice) and cols != slice(None):
        raise PandorableSparkyNotImplementedError(
            description=about_cols,
            pandas_source="loc",
            spark_target_function="select, where, withColumn")
    elif isinstance(cols, slice) and cols == slice(None):
        cols = ("*",)

    return rows, cols


class SparkDataFrameLocator(object):
    """
    A locator to slice a group of rows and columns by conditional and label(s).

    Allowed inputs are a slice with all indices or conditional for rows, and string(s) or
    :class:`Column`(s) for cols.
    """

    def __init__(self, df):
        self.df = df

    def __getitem__(self, key):

        about_rows = """Can only slice with all indices or a column that evaluates to Boolean"""

        rows, cols = _unfold(key)

        if isinstance(rows, slice) and rows != slice(None):
            raise PandorableSparkyNotImplementedError(
                description=about_rows,
                pandas_source=".loc[..., ...]",
                spark_target_function="select, where")
        elif isinstance(rows, slice) and rows == slice(None):
            df = self.df
        else:   # not isinstance(rows, slice):
            try:
                assert isinstance(self.df._spark_select(rows).schema.fields[0].dataType,
                                  BooleanType)
                df = self.df._spark_where(rows)
            except Exception as e:
                raise PandorableSparkyNotImplementedError(
                    description=about_rows,
                    pandas_source=".loc[..., ...]",
                    spark_target_function="select, where")
        return df._spark_select([_make_col(c) for c in cols])

    def __setitem__(self, key, value):

        if (not isinstance(key, tuple)) or (len(key) != 2):
            raise NotImplementedError("Only accepts pairs of candidates")

        rows, cols = key

        if (not isinstance(rows, slice)) or (rows != slice(None)):
            raise PandorableSparkyNotImplementedError(
                description="""Can only assign value to the whole dataframe, the row index
                has to be `slice(None)` or `:`""",
                pandas_source=".loc[..., ...] = ...",
                spark_target_function="withColumn, select")

        if not isinstance(cols, str):
            raise ValueError("""only column names can be assigned""")

        if isinstance(value, Column):
            df = self.df._spark_withColumn(cols, value)
        elif isinstance(value, DataFrame) and len(value.columns) == 1:
            df = self.df._spark_withColumn(cols, col(value.columns[0]))
        elif isinstance(value, DataFrame) and len(value.columns) != 1:
            raise ValueError("Only a dataframe with one column can be assigned")
        else:
            raise ValueError("Only a column or dataframe with single column can be assigned")

        from .structures import _reassign_jdf
        _reassign_jdf(self.df, df)
