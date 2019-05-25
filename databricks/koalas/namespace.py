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
from typing import Optional, Union
from collections.abc import Iterable
import itertools

import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.types import ByteType, ShortType, IntegerType, LongType, FloatType, \
    DoubleType, BooleanType, TimestampType, DecimalType, StringType, DateType, StructType

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.utils import default_session
from databricks.koalas.frame import DataFrame, _reduce_spark_multi
from databricks.koalas.typedef import Col, pandas_wraps
from databricks.koalas.series import Series, _col


def from_pandas(pobj: Union['pd.DataFrame', 'pd.Series']) -> Union['Series', 'DataFrame']:
    """Create a Koalas DataFrame or Series from a pandas DataFrame or Series.

    This is similar to Spark's `SparkSession.createDataFrame()` with pandas DataFrame,
    but this also works with pandas Series and picks the index.

    Parameters
    ----------
    pobj : pandas.DataFrame or pandas.Series
        pandas DataFrame or Series to read.

    Returns
    -------
    Series or DataFrame
        If a pandas Series is passed in, this function returns a Koalas Series.
        If a pandas DataFrame is passed in, this function returns a Koalas DataFrame.
    """
    if isinstance(pobj, pd.Series):
        return Series(pobj)
    elif isinstance(pobj, pd.DataFrame):
        return DataFrame(pobj)
    else:
        raise ValueError("Unknown data type: {}".format(type(pobj)))


def sql(query: str) -> DataFrame:
    """
    Execute a SQL query and return the result as a Koalas DataFrame.

    Parameters
    ----------
    query : str
        the SQL query

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> ks.sql("select * from range(10) where id > 7")
       id
    0   8
    1   9
    """
    return DataFrame(default_session().sql(query))


def range(start: int,
          end: Optional[int] = None,
          step: int = 1,
          num_partitions: Optional[int] = None) -> DataFrame:
    """
    Create a DataFrame with some range of numbers.

    The resulting DataFrame has a single int64 column named `id`, containing elements in a range
    from ``start`` to ``end`` (exclusive) with step value ``step``. If only the first parameter
    (i.e. start) is specified, we treat it as the end value with the start value being 0.

    This is similar to the range function in SparkSession and is used primarily for testing.

    Parameters
    ----------
    start : int
        the start value (inclusive)
    end : int, optional
        the end value (exclusive)
    step : int, optional, default 1
        the incremental step
    num_partitions : int, optional
        the number of partitions of the DataFrame

    Returns
    -------
    DataFrame

    Examples
    --------
    When the first parameter is specified, we generate a range of values up till that number.

    >>> ks.range(5)
       id
    0   0
    1   1
    2   2
    3   3
    4   4

    When start, end, and step are specified:

    >>> ks.range(start = 100, end = 200, step = 20)
        id
    0  100
    1  120
    2  140
    3  160
    4  180
    """
    sdf = default_session().range(start=start, end=end, step=step, numPartitions=num_partitions)
    return DataFrame(sdf)


def read_csv(path, header='infer', names=None, usecols=None,
             mangle_dupe_cols=True, parse_dates=False, comment=None):
    """Read CSV (comma-separated) file into DataFrame.

    Parameters
    ----------
    path : str
        The path string storing the CSV file to be read.
    header : int, list of int, default ‘infer’
        Whether to to use as the column names, and the start of the data.
        Default behavior is to infer the column names: if no names are passed
        the behavior is identical to `header=0` and column names are inferred from
        the first line of the file, if column names are passed explicitly then
        the behavior is identical to `header=None`. Explicitly pass `header=0` to be
        able to replace existing names
    names : array-like, optional
        List of column names to use. If file contains no header row, then you should
        explicitly pass `header=None`. Duplicates in this list will cause an error to be issued.
    usecols : list-like or callable, optional
        Return a subset of the columns. If list-like, all elements must either be
        positional (i.e. integer indices into the document columns) or strings that
        correspond to column names provided either by the user in names or inferred
        from the document header row(s).
        If callable, the callable function will be evaluated against the column names,
        returning names where the callable function evaluates to `True`.
    mangle_dupe_cols : bool, default True
        Duplicate columns will be specified as 'X0', 'X1', ... 'XN', rather
        than 'X' ... 'X'. Passing in False will cause data to be overwritten if
        there are duplicate names in the columns.
        Currently only `True` is allowed.
    parse_dates : boolean or list of ints or names or list of lists or dict, default `False`.
        Currently only `False` is allowed.
    comment: str, optional
        Indicates the line should not be parsed.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> ks.read_csv('data.csv')  # doctest: +SKIP
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
            if not isinstance(comment, str) or len(comment) != 1:
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
            elif all(isinstance(col, str) for col in usecols):
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

    Parameters
    ----------
    path : string
        File path
    columns : list, default=None
        If not None, only these columns will be read from the file.

    Returns
    -------
    DataFrame

    Examples
    --------
    >>> ks.read_parquet('data.parquet', columns=['name', 'gender'])  # doctest: +SKIP
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
    """
    Convert argument to datetime.

    Parameters
    ----------
    arg : integer, float, string, datetime, list, tuple, 1-d array, Series
           or DataFrame/dict-like

    errors : {'ignore', 'raise', 'coerce'}, default 'raise'

        - If 'raise', then invalid parsing will raise an exception
        - If 'coerce', then invalid parsing will be set as NaT
        - If 'ignore', then invalid parsing will return the input
    format : string, default None
        strftime to parse time, eg "%d/%m/%Y", note that "%f" will parse
        all the way up to nanoseconds.
    infer_datetime_format : boolean, default False
        If True and no `format` is given, attempt to infer the format of the
        datetime strings, and if it can be inferred, switch to a faster
        method of parsing them. In some cases this can increase the parsing
        speed by ~5-10x.

    Returns
    -------
    ret : datetime if parsing succeeded.
        Return type depends on input:

        - list-like: DatetimeIndex
        - Series: Series of datetime64 dtype
        - scalar: Timestamp

        In case when it is not possible to return designated types (e.g. when
        any element of input is before Timestamp.min or after Timestamp.max)
        return will have datetime.datetime type (or corresponding
        array/Series).

    Examples
    --------
    Assembling a datetime from multiple columns of a DataFrame. The keys can be
    common abbreviations like ['year', 'month', 'day', 'minute', 'second',
    'ms', 'us', 'ns']) or plurals of the same

    >>> df = ks.DataFrame({'year': [2015, 2016],
    ...                    'month': [2, 3],
    ...                    'day': [4, 5]})
    >>> ks.to_datetime(df)
    0   2015-02-04
    1   2016-03-05
    Name: _to_datetime2(arg_day=day, arg_month=month, arg_year=year), dtype: datetime64[ns]

    If a date does not meet the `timestamp limitations
    <http://pandas.pydata.org/pandas-docs/stable/timeseries.html
    #timeseries-timestamp-limits>`_, passing errors='ignore'
    will return the original input instead of raising any exception.

    Passing errors='coerce' will force an out-of-bounds date to NaT,
    in addition to forcing non-dates (or non-parseable dates) to NaT.

    >>> ks.to_datetime('13000101', format='%Y%m%d', errors='ignore')
    datetime.datetime(1300, 1, 1, 0, 0)
    >>> ks.to_datetime('13000101', format='%Y%m%d', errors='coerce')
    NaT

    Passing infer_datetime_format=True can often-times speedup a parsing
    if its not an ISO8601 format exactly, but in a regular format.

    >>> s = ks.Series(['3/11/2000', '3/12/2000', '3/13/2000'] * 1000)
    >>> s.head()
    0    3/11/2000
    1    3/12/2000
    2    3/13/2000
    3    3/11/2000
    4    3/12/2000
    Name: 0, dtype: object

    >>> import timeit
    >>> timeit.timeit(
    ...    lambda: repr(ks.to_datetime(s, infer_datetime_format=True)),
    ...    number = 1)  # doctest: +SKIP
    0.35832712500000063

    >>> timeit.timeit(
    ...    lambda: repr(ks.to_datetime(s, infer_datetime_format=False)),
    ...    number = 1)  # doctest: +SKIP
    0.8895321660000004
    """
    if isinstance(arg, Series):
        return _to_datetime1(
            arg,
            errors=errors,
            format=format,
            infer_datetime_format=infer_datetime_format)
    if isinstance(arg, DataFrame):
        return _to_datetime2(
            arg_year=arg['year'],
            arg_month=arg['month'],
            arg_day=arg['day'],
            errors=errors,
            format=format,
            infer_datetime_format=infer_datetime_format)
    if isinstance(arg, dict):
        return _to_datetime2(
            arg_year=arg['year'],
            arg_month=arg['month'],
            arg_day=arg['day'],
            errors=errors,
            format=format,
            infer_datetime_format=infer_datetime_format)
    return pd.to_datetime(
        arg, errors=errors, format=format, infer_datetime_format=infer_datetime_format)


def get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False,
                drop_first=False, dtype=None):
    """
    Convert categorical variable into dummy/indicator variables, also
    known as one hot encoding.

    Parameters
    ----------
    data : array-like, Series, or DataFrame
    prefix : string, list of strings, or dict of strings, default None
        String to append DataFrame column names.
        Pass a list with length equal to the number of columns
        when calling get_dummies on a DataFrame. Alternatively, `prefix`
        can be a dictionary mapping column names to prefixes.
    prefix_sep : string, default '_'
        If appending prefix, separator/delimiter to use. Or pass a
        list or dictionary as with `prefix.`
    dummy_na : bool, default False
        Add a column to indicate NaNs, if False NaNs are ignored.
    columns : list-like, default None
        Column names in the DataFrame to be encoded.
        If `columns` is None then all the columns with
        `object` or `category` dtype will be converted.
    sparse : bool, default False
        Whether the dummy-encoded columns should be be backed by
        a :class:`SparseArray` (True) or a regular NumPy array (False).
        In Koalas, this value must be "False".
    drop_first : bool, default False
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level.
    dtype : dtype, default np.uint8
        Data type for new columns. Only a single dtype is allowed.

    Returns
    -------
    dummies : DataFrame

    See Also
    --------
    Series.str.get_dummies

    Examples
    --------
    >>> s = ks.Series(list('abca'))

    >>> ks.get_dummies(s)
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0

    >>> df = ks.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'],
    ...                    'C': [1, 2, 3]},
    ...                   columns=['A', 'B', 'C'])

    >>> ks.get_dummies(df, prefix=['col1', 'col2'])
       C  col1_a  col1_b  col2_a  col2_b  col2_c
    0  1       1       0       0       1       0
    1  2       0       1       1       0       0
    2  3       1       0       0       0       1

    >>> ks.get_dummies(ks.Series(list('abcaa')))
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0
    4  1  0  0

    >>> ks.get_dummies(ks.Series(list('abcaa')), drop_first=True)
       b  c
    0  0  0
    1  1  0
    2  0  1
    3  0  0
    4  0  0

    >>> ks.get_dummies(ks.Series(list('abc')), dtype=float)
         a    b    c
    0  1.0  0.0  0.0
    1  0.0  1.0  0.0
    2  0.0  0.0  1.0
    """
    if sparse is not False:
        raise NotImplementedError("get_dummies currently does not support sparse")

    if isinstance(columns, str):
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
        if isinstance(prefix, str):
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


# TODO: there are many parameters to implement and support. See Pandas's pd.concat.
def concat(objs, axis=0, join='outer', ignore_index=False):
    """
    Concatenate pandas objects along a particular axis with optional set logic
    along the other axes.

    Parameters
    ----------
    objs : a sequence of Series or DataFrame
        Any None objects will be dropped silently unless
        they are all None in which case a ValueError will be raised
    axis : {0/'index'}, default 0
        The axis to concatenate along.
    join : {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis(es)
    ignore_index : boolean, default False
        If True, do not use the index values along the concatenation axis. The
        resulting axis will be labeled 0, ..., n - 1. This is useful if you are
        concatenating objects where the concatenation axis does not have
        meaningful indexing information. Note the index values on the other
        axes are still respected in the join.

    Returns
    -------
    concatenated : object, type of objs
        When concatenating all ``Series`` along the index (axis=0), a
        ``Series`` is returned. When ``objs`` contains at least one
        ``DataFrame``, a ``DataFrame`` is returned.

    See Also
    --------
    DataFrame.merge

    Examples
    --------
    Combine two ``Series``.

    >>> s1 = ks.Series(['a', 'b'])
    >>> s2 = ks.Series(['c', 'd'])
    >>> ks.concat([s1, s2])
    0    a
    1    b
    0    c
    1    d
    Name: 0, dtype: object

    Clear the existing index and reset it in the result
    by setting the ``ignore_index`` option to ``True``.

    >>> ks.concat([s1, s2], ignore_index=True)
    0    a
    1    b
    2    c
    3    d
    Name: 0, dtype: object

    Combine two ``DataFrame`` objects with identical columns.

    >>> df1 = ks.DataFrame([['a', 1], ['b', 2]],
    ...                    columns=['letter', 'number'])
    >>> df1
      letter  number
    0      a       1
    1      b       2
    >>> df2 = ks.DataFrame([['c', 3], ['d', 4]],
    ...                    columns=['letter', 'number'])
    >>> df2
      letter  number
    0      c       3
    1      d       4
    >>> ks.concat([df1, df2])
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4

    Combine ``DataFrame`` and ``Series`` objects with different columns.

    >>> ks.concat([df2, s1, s2])
          0 letter  number
    0  None      c     3.0
    1  None      d     4.0
    0     a   None     NaN
    1     b   None     NaN
    0     c   None     NaN
    1     d   None     NaN

    Combine ``DataFrame`` objects with overlapping columns
    and return everything. Columns outside the intersection will
    be filled with ``None`` values.

    >>> df3 = ks.DataFrame([['c', 3, 'cat'], ['d', 4, 'dog']],
    ...                    columns=['letter', 'number', 'animal'])
    >>> df3
      letter  number animal
    0      c       3    cat
    1      d       4    dog
    >>> ks.concat([df1, df3])
      animal letter  number
    0   None      a       1
    1   None      b       2
    0    cat      c       3
    1    dog      d       4

    Combine ``DataFrame`` objects with overlapping columns
    and return only those that are shared by passing ``inner`` to
    the ``join`` keyword argument.

    >>> ks.concat([df1, df3], join="inner")
      letter  number
    0      a       1
    1      b       2
    0      c       3
    1      d       4
    """
    if not isinstance(objs, (dict, Iterable)):
        raise TypeError('first argument must be an iterable of koalas '
                        'objects, you passed an object of type '
                        '"{name}"'.format(name=type(objs).__name__))

    if axis not in [0, 'index']:
        raise ValueError('axis should be either 0 or "index" currently.')

    if all(map(lambda obj: obj is None, objs)):
        raise ValueError("All objects passed were None")
    objs = list(filter(lambda obj: obj is not None, objs))

    for obj in objs:
        if not isinstance(obj, (Series, DataFrame)):
            raise TypeError('cannot concatenate object of type '"'{name}"'; only ks.Series '
                            'and ks.DataFrame are valid'.format(name=type(objs).__name__))

    # Series, Series ...
    # We should return Series if objects are all Series.
    should_return_series = all(map(lambda obj: isinstance(obj, Series), objs))

    # DataFrame, Series ... & Series, Series ...
    # In this case, we should return DataFrame.
    new_objs = []
    for obj in objs:
        if isinstance(obj, Series):
            obj = obj.to_dataframe()
        new_objs.append(obj)
    objs = new_objs

    # DataFrame, DataFrame, ...
    # All Series are converted into DataFrame and then compute concat.
    if not ignore_index:
        indices_of_kdfs = [kdf._metadata.index_map for kdf in objs]
        index_of_first_kdf = indices_of_kdfs[0]
        for index_of_kdf in indices_of_kdfs:
            if index_of_first_kdf != index_of_kdf:
                raise ValueError(
                    'Index type and names should be same in the objects to concatenate. '
                    'You passed different indices '
                    '{index_of_first_kdf} and {index_of_kdf}'.format(
                        index_of_first_kdf=index_of_first_kdf, index_of_kdf=index_of_kdf))

    columns_of_kdfs = [kdf._metadata.columns for kdf in objs]
    first_kdf = objs[0]
    if ignore_index:
        columns_of_first_kdf = first_kdf._metadata.data_columns
    else:
        columns_of_first_kdf = first_kdf._metadata.columns
    if all(current_kdf == columns_of_first_kdf for current_kdf in columns_of_kdfs):
        # If all columns are in the same order and values, use it.
        kdfs = objs
    else:
        if ignore_index:
            columns_to_apply = [kdf._metadata.data_columns for kdf in objs]
        else:
            columns_to_apply = [kdf._metadata.columns for kdf in objs]

        if join == "inner":
            interested_columns = set.intersection(*map(set, columns_to_apply))
            # Keep the column order with its firsts DataFrame.
            interested_columns = list(map(
                lambda c: columns_of_first_kdf[columns_of_first_kdf.index(c)],
                interested_columns))

            kdfs = []
            for kdf in objs:
                sdf = kdf._sdf.select(interested_columns)
                if ignore_index:
                    kdfs.append(DataFrame(sdf))
                else:
                    kdfs.append(DataFrame(sdf, first_kdf._metadata.copy()))
        elif join == "outer":
            # If there are columns unmatched, just sort the column names.
            merged_columns = set(
                itertools.chain.from_iterable(columns_to_apply))

            kdfs = []
            for kdf in objs:
                if ignore_index:
                    columns_to_add = merged_columns - set(kdf._metadata.data_columns)
                else:
                    columns_to_add = merged_columns - set(kdf._metadata.columns)

                # TODO: NaN and None difference for missing values. pandas seems filling NaN.
                kdf = kdf.assign(**dict(zip(columns_to_add, [None] * len(columns_to_add))))

                if ignore_index:
                    sdf = kdf._sdf.select(sorted(kdf._metadata.data_columns))
                else:
                    sdf = kdf._sdf.select(
                        kdf._metadata.index_columns + sorted(kdf._metadata.data_columns))

                kdf = DataFrame(sdf, kdf._metadata.copy(
                    data_columns=sorted(kdf._metadata.data_columns)))
                kdfs.append(kdf)
        else:
            raise ValueError(
                "Only can inner (intersect) or outer (union) join the other axis.")

    concatenated = kdfs[0]._sdf
    for kdf in kdfs[1:]:
        concatenated = concatenated.unionByName(kdf._sdf)

    if ignore_index:
        result_kdf = DataFrame(concatenated.select(kdfs[0]._metadata.data_columns))
    else:
        result_kdf = DataFrame(concatenated, kdfs[0]._metadata.copy())

    if should_return_series:
        # If all input were Series, we should return Series.
        return _col(result_kdf)
    else:
        return result_kdf


# @pandas_wraps(return_col=np.datetime64)
@pandas_wraps
def _to_datetime1(arg, errors, format, infer_datetime_format) -> Col[np.datetime64]:
    return pd.to_datetime(
        arg,
        errors=errors,
        format=format,
        infer_datetime_format=infer_datetime_format)


# @pandas_wraps(return_col=np.datetime64)
@pandas_wraps
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
