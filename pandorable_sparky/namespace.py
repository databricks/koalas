"""
Wrappers around spark that correspond to common pandas functions.
"""
import sys

import pyspark.sql
import numpy as np
import pandas as pd
from .typing import Col, pandas_wrap
from pyspark.sql import Column
from pyspark.sql.types import StructType

if sys.version > '3':
    basestring = unicode = str


def default_session():
    return pyspark.sql.SparkSession.builder.getOrCreate()


def read_csv(path, header='infer', names=None, usecols=None,
             mangle_dupe_cols=True, parse_dates=False, comment=None):
    if usecols is not None and not callable(usecols):
        usecols = list(usecols)
    if usecols is None or callable(usecols) or len(usecols) > 0:
        reader = default_session().read.option("inferSchema", "true")

        if header == 'infer':
            header = 0 if names is None else None
        if header == 0:
            reader.option("header", True)
        elif header is None:
            reader.option("header", False)
        else:
            raise ValueError("Unknown header argument {}".format(header))

        if comment is not None:
            if not isinstance(comment, basestring) or len(comment) != 1:
                raise ValueError("Only length-1 comment characters supported")
            reader.option("comment", comment)

        df = reader.csv(path)

        if header is None:
            df = df.selectExpr(*["`%s` as `%s`" % (field.name, i)
                                 for i, field in enumerate(df.schema)])
        if names is not None:
            names = list(names)
            if len(set(names)) != len(names):
                raise ValueError('Found non-unique column index')
            if len(names) != len(df.schema):
                raise ValueError('Names do not match the number of columns: %d' % len(names))
            df = df.selectExpr(*["`%s` as `%s`" % (field.name, name)
                                 for field, name in zip(df.schema, names)])

        if usecols is not None:
            if callable(usecols):
                cols = [field.name for field in df.schema if usecols(field.name)]
                missing = []
            elif all(isinstance(col, int) for col in usecols):
                cols = [field.name for i, field in enumerate(df.schema) if i in usecols]
                missing = [col for col in usecols
                           if col >= len(df.schema) or df.schema[col].name not in cols]
            elif all(isinstance(col, basestring) for col in usecols):
                cols = [field.name for field in df.schema if field.name in usecols]
                missing = [col for col in usecols if col not in cols]
            else:
                raise ValueError("'usecols' must either be list-like of all strings, "
                                 "all unicode, all integers or a callable.")
            if len(missing) > 0:
                raise ValueError('Usecols do not match columns, columns expected but not found: %s'
                                 % missing)

            if len(cols) > 0:
                df = df.select(cols)
            else:
                df = default_session().createDataFrame([], schema=StructType())
    else:
        df = default_session().createDataFrame([], schema=StructType())
    return df


def read_parquet(path, columns=None):
    if columns is not None:
        columns = list(columns)
    if columns is None or len(columns) > 0:
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
