"""
Wrappers around spark that correspond to common pandas functions.
"""
import pyspark.sql
import numpy as np
import pandas as pd
from .typing import Col, pandas_wrap
from pyspark.sql import Column
from pyspark.sql.types import StructType


def default_session():
    return pyspark.sql.SparkSession.builder.getOrCreate()


def read_csv(path, header='infer'):
    b = default_session().read.format("csv").option("inferSchema", "true")
    if header == 'infer':
        b = b.option("header", "true")
    elif header == 0:
        pass
    else:
        raise ValueError("Unknown header argument {}".format(header))
    return b.load(path)


def read_parquet(path, columns=None):
    if columns is None or (isinstance(columns, (list, tuple)) and len(columns) > 0):
        df = default_session().read.parquet(path)
        if columns is not None:
            fields = [field.name for field in df.schema]
            cols = [col for col in columns if col in fields]
            if len(cols) > 0:
                df = df.select(cols)
            else:
                df = default_session().createDataFrame([], schema=StructType())
    else:
        df = default_session().createDataFrame([], schema=StructType())
    return df


def to_datetime(arg, errors='raise', format=None):
    if isinstance(arg, Column):
        return _to_datetime1(arg, errors=errors, format=format)
    if isinstance(arg, dict):
        return _to_datetime2(
            arg_year=arg['year'],
            arg_month=arg['month'],
            arg_day=arg['day'],
            errors=errors,
            format=format
        )


# @pandas_wrap(return_col=np.datetime64)
@pandas_wrap
def _to_datetime1(arg, errors, format) -> Col[np.datetime64]:
    return pd.to_datetime(arg, errors=errors, format=format).astype(np.datetime64)


# @pandas_wrap(return_col=np.datetime64)
@pandas_wrap
def _to_datetime2(arg_year=None, arg_month=None, arg_day=None,
                  errors=None, format=None) -> Col[np.datetime64]:
    arg = dict(year=arg_year, month=arg_month, day=arg_day)
    for key in arg:
        if arg[key] is None:
            del arg[key]
    return pd.to_datetime(arg, errors=errors, format=format).astype(np.datetime64)
