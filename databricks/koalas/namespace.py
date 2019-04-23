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
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.types import *

from databricks.koalas.dask.compatibility import string_types
from databricks.koalas.dask.utils import derived_from
from databricks.koalas.utils import default_session
from databricks.koalas.frame import DataFrame, _reduce_spark_multi
from databricks.koalas.typing import Col, pandas_wrap
from databricks.koalas.series import Series


def from_pandas(pdf):
    """Create DataFrame from pandas DataFrame.

    This is similar to `DataFrame.createDataFrame()` with pandas DataFrame, but this also picks
    the index in the given pandas DataFrame.

    :param pdf: :class:`pandas.DataFrame`
    """
    if isinstance(pdf, pd.Series):
        return Series(pdf)
    elif isinstance(pdf, pd.DataFrame):
        return DataFrame(pdf)
    else:
        raise ValueError("Unknown data type: {}".format(type(pdf)))


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
    if mangle_dupe_cols is not True:
        raise ValueError("mangle_dupe_cols can only be `True`: %s" % mangle_dupe_cols)
    if parse_dates is not False:
        raise ValueError("parse_dates can only be `False`: %s" % parse_dates)

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
            if not isinstance(comment, string_types) or len(comment) != 1:
                raise ValueError("Only length-1 comment characters supported")
            reader.option("comment", comment)

        sdf = reader.csv(path)

        if header is None:
            sdf = sdf.selectExpr(*["`%s` as `%s`" % (field.name, i)
                                   for i, field in enumerate(sdf.schema)])
        if names is not None:
            names = list(names)
            if len(set(names)) != len(names):
                raise ValueError('Found non-unique column index')
            if len(names) != len(sdf.schema):
                raise ValueError('Names do not match the number of columns: %d' % len(names))
            sdf = sdf.selectExpr(*["`%s` as `%s`" % (field.name, name)
                                   for field, name in zip(sdf.schema, names)])

        if usecols is not None:
            if callable(usecols):
                cols = [field.name for field in sdf.schema if usecols(field.name)]
                missing = []
            elif all(isinstance(col, int) for col in usecols):
                cols = [field.name for i, field in enumerate(sdf.schema) if i in usecols]
                missing = [col for col in usecols
                           if col >= len(sdf.schema) or sdf.schema[col].name not in cols]
            elif all(isinstance(col, string_types) for col in usecols):
                cols = [field.name for field in sdf.schema if field.name in usecols]
                missing = [col for col in usecols if col not in cols]
            else:
                raise ValueError("'usecols' must either be list-like of all strings, "
                                 "all unicode, all integers or a callable.")
            if len(missing) > 0:
                raise ValueError('Usecols do not match columns, columns expected but not '
                                 'found: %s' % missing)

            if len(cols) > 0:
                sdf = sdf.select(cols)
            else:
                sdf = default_session().createDataFrame([], schema=StructType())
    else:
        sdf = default_session().createDataFrame([], schema=StructType())
    return DataFrame(sdf)


def read_parquet(path, columns=None):
    """Load a parquet object from the file path, returning a DataFrame.

    :param path: File path
    :param columns: If not None, only these columns will be read from the file.
    :return: :class:`DataFrame`
    """
    if columns is not None:
        columns = list(columns)
    if columns is None or len(columns) > 0:
        sdf = default_session().read.parquet(path)
        if columns is not None:
            fields = [field.name for field in sdf.schema]
            cols = [col for col in columns if col in fields]
            if len(cols) > 0:
                sdf = sdf.select(cols)
            else:
                sdf = default_session().createDataFrame([], schema=StructType())
    else:
        sdf = default_session().createDataFrame([], schema=StructType())
    return DataFrame(sdf)


def to_datetime(arg, errors='raise', format=None, infer_datetime_format=False):
    if isinstance(arg, Series):
        return Series(_to_datetime1(
            arg._scol,
            errors=errors,
            format=format,
            infer_datetime_format=infer_datetime_format), arg._kdf, arg._index_info)
    if isinstance(arg, DataFrame):
        return Series(_to_datetime2(
            arg_year=arg['year']._scol,
            arg_month=arg['month']._scol,
            arg_day=arg['day']._scol,
            errors=errors,
            format=format,
            infer_datetime_format=infer_datetime_format), arg, arg._metadata.index_info)
    if isinstance(arg, dict):
        return _to_datetime2(
            arg_year=arg['year'],
            arg_month=arg['month'],
            arg_day=arg['day'],
            errors=errors,
            format=format,
            infer_datetime_format=infer_datetime_format)


@derived_from(pd)
def get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False,
                drop_first=False, dtype=None):
    if sparse is not False:
        raise NotImplementedError("get_dummies currently does not support sparse")

    if isinstance(columns, string_types):
        columns = [columns]
    if dtype is None:
        dtype = 'byte'

    if isinstance(data, Series):
        if prefix is not None:
            prefix = [str(prefix)]
        columns = [data.name]
        kdf = data.to_dataframe()
        remaining_columns = []
    else:
        if isinstance(prefix, string_types):
            raise ValueError("get_dummies currently does not support prefix as string types")
        kdf = data.copy()
        if columns is None:
            columns = [column for column in kdf.columns
                       if isinstance(data._sdf.schema[column].dataType,
                                     _get_dummies_default_accept_types)]
        if len(columns) == 0:
            return kdf

        if prefix is None:
            prefix = columns

        column_set = set(columns)
        remaining_columns = [kdf[column] for column in kdf.columns if column not in column_set]

    if any(not isinstance(kdf._sdf.schema[column].dataType, _get_dummies_acceptable_types)
           for column in columns):
        raise ValueError("get_dummies currently only accept {} values"
                         .format(', '.join([t.typeName() for t in _get_dummies_acceptable_types])))

    if prefix is not None and len(columns) != len(prefix):
        raise ValueError(
            "Length of 'prefix' ({}) did not match the length of the columns being encoded ({})."
            .format(len(prefix), len(columns)))

    all_values = _reduce_spark_multi(kdf._sdf, [F.collect_set(F.col(column)).alias(column)
                                                for column in columns])
    for i, column in enumerate(columns):
        values = sorted(all_values[i])
        if drop_first:
            values = values[1:]

        def column_name(value):
            if prefix is None:
                return str(value)
            else:
                return '{}{}{}'.format(prefix[i], prefix_sep, value)

        for value in values:
            remaining_columns.append((kdf[column].notnull() & (kdf[column] == value))
                                     .astype(dtype)
                                     .rename(column_name(value)))
        if dummy_na:
            remaining_columns.append(kdf[column].isnull().astype(dtype).rename(column_name('nan')))

    return kdf[remaining_columns]


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


_get_dummies_default_accept_types = (
    DecimalType, StringType, DateType
)
_get_dummies_acceptable_types = _get_dummies_default_accept_types + (
    ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType, BooleanType, TimestampType
)
