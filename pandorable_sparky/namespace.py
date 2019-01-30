"""
Wrappers around spark that correspond to common pandas functions.
"""
import pyspark
import numpy as np
import pandas as pd
from .typing import Col, pandas_wrap
from pyspark.sql import Column


def default_session():
    return pyspark.sql.SparkSession.builder.getOrCreate()


def read_csv(path, header='infer', names=None, usecols=None,
             mangle_dupe_cols=True, parse_dates=False, comment=None):
    """Read CSV (comma-separated) file into DataFrame using the default session.

    See :meth:`pyspark.sql.session.SparkSession.read_csv`.
    """
    return default_session().read_csv(path=path, header=header, names=names, usecols=usecols,
                                      mangle_dupe_cols=mangle_dupe_cols, parse_dates=parse_dates,
                                      comment=comment)


def read_parquet(path, columns=None):
    """Read Parquet file into DataFrame using the default session.

    See :meth:`pyspark.sql.session.SparkSession.read_parquet`.
    """
    return default_session().read_parquet(path=path, columns=columns)


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
