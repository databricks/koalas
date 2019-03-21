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
Wrappers around spark that correspond to common pandas functions.
"""
import pyspark
import numpy as np
import pandas as pd
from .typing import Col, pandas_wrap
from pyspark.sql import Column, DataFrame


def default_session():
    return pyspark.sql.SparkSession.builder.getOrCreate()


def from_pandas(pdf):
    """Create DataFrame from pandas DataFrame.

    This is similar to `DataFrame.createDataFrame()` with pandas DataFrame, but this also picks
    the index in the given pandas DataFrame.

    :param pdf: :class:`pandas.DataFrame`
    """
    return default_session().from_pandas(pdf)


def read_csv(path, header='infer', names=None, usecols=None,
             mangle_dupe_cols=True, parse_dates=False, comment=None):
    """Read CSV (comma-separated) file into DataFrame.

    :param path: The path string storing the CSV file to be read.
    :param header: Whether to to use as the column names, and the start of the data.
                   Default behavior is to infer the column names: if no names are passed
                   the behavior is identical to `header=0` and column names are inferred from
                   the first line of the file, if column names are passed explicitly then
                   the behavior is identical to `header=None`. Explicitly pass `header=0` to be
                   able to replace existing names
    :param names: List of column names to use. If file contains no header row, then you should
                  explicitly pass `header=None`. Duplicates in this list will cause an error to be
                  issued.
    :param usecols: Return a subset of the columns. If list-like, all elements must either be
                    positional (i.e. integer indices into the document columns) or strings that
                    correspond to column names provided either by the user in names or inferred
                    from the document header row(s).
                    If callable, the callable function will be evaluated against the column names,
                    returning names where the callable function evaluates to `True`.
    :param mangle_dupe_cols: Duplicate columns will be specified as 'X0', 'X1', ... 'XN', rather
                             than 'X' ... 'X'. Passing in False will cause data to be overwritten if
                             there are duplicate names in the columns.
                             Currently only `True` is allowed.
    :param parse_dates: boolean or list of ints or names or list of lists or dict, default `False`.
                        Currently only `False` is allowed.
    :param comment: Indicates the line should not be parsed.
    :return: :class:`DataFrame`
    """
    return default_session().read_csv(path=path, header=header, names=names, usecols=usecols,
                                      mangle_dupe_cols=mangle_dupe_cols, parse_dates=parse_dates,
                                      comment=comment)


def read_parquet(path, columns=None):
    """Load a parquet object from the file path, returning a DataFrame.

    :param path: File path
    :param columns: If not None, only these columns will be read from the file.
    :return: :class:`DataFrame`
    """
    return default_session().read_parquet(path=path, columns=columns)


def to_datetime(arg, errors='raise', format=None, infer_datetime_format=False):
    if isinstance(arg, Column):
        return _to_datetime1(
            arg,
            errors=errors,
            format=format,
            infer_datetime_format=infer_datetime_format)
    if isinstance(arg, (dict, DataFrame)):
        return _to_datetime2(
            arg_year=arg['year'],
            arg_month=arg['month'],
            arg_day=arg['day'],
            errors=errors,
            format=format,
            infer_datetime_format=infer_datetime_format)


# @pandas_wrap(return_col=np.datetime64)
@pandas_wrap
def _to_datetime1(arg, errors, format, infer_datetime_format) -> Col[np.datetime64]:
    return pd.to_datetime(
        arg,
        errors=errors,
        format=format,
        infer_datetime_format=infer_datetime_format)


# @pandas_wrap(return_col=np.datetime64)
@pandas_wrap
def _to_datetime2(arg_year, arg_month, arg_day,
                  errors, format, infer_datetime_format) -> Col[np.datetime64]:
    arg = dict(year=arg_year, month=arg_month, day=arg_day)
    for key in arg:
        if arg[key] is None:
            del arg[key]
    return pd.to_datetime(
        arg,
        errors=errors,
        format=format,
        infer_datetime_format=infer_datetime_format)
