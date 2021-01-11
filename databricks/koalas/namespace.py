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
from typing import Any, Optional, Union, List, Tuple, Sized, cast
from collections import OrderedDict
from collections.abc import Iterable
from distutils.version import LooseVersion
from functools import reduce
from io import BytesIO
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_datetime64tz_dtype, is_list_like
import pyarrow as pa
import pyarrow.parquet as pq
import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    BooleanType,
    TimestampType,
    DecimalType,
    StringType,
    DateType,
    StructType,
)

from databricks import koalas as ks  # noqa: F401
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.utils import (
    align_diff_frames,
    default_session,
    is_name_like_tuple,
    name_like_string,
    same_anchor,
    scol_for,
    validate_axis,
)
from databricks.koalas.frame import DataFrame, _reduce_spark_multi
from databricks.koalas.internal import (
    InternalFrame,
    DEFAULT_SERIES_NAME,
    HIDDEN_COLUMNS,
)
from databricks.koalas.series import Series, first_series
from databricks.koalas.spark.utils import as_nullable_spark_type, force_decimal_precision_scale
from databricks.koalas.indexes import Index


__all__ = [
    "from_pandas",
    "range",
    "read_csv",
    "read_delta",
    "read_table",
    "read_spark_io",
    "read_parquet",
    "read_clipboard",
    "read_excel",
    "read_html",
    "to_datetime",
    "get_dummies",
    "concat",
    "melt",
    "isna",
    "isnull",
    "notna",
    "notnull",
    "read_sql_table",
    "read_sql_query",
    "read_sql",
    "read_json",
    "merge",
    "to_numeric",
    "broadcast",
]


def from_pandas(pobj: Union[pd.DataFrame, pd.Series, pd.Index]) -> Union[Series, DataFrame, Index]:
    """Create a Koalas DataFrame, Series or Index from a pandas DataFrame, Series or Index.

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
    elif isinstance(pobj, pd.Index):
        return DataFrame(pd.DataFrame(index=pobj)).index
    else:
        raise ValueError("Unknown data type: {}".format(type(pobj).__name__))


_range = range  # built-in range


def range(
    start: int, end: Optional[int] = None, step: int = 1, num_partitions: Optional[int] = None
) -> DataFrame:
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


def read_csv(
    path,
    sep=",",
    header="infer",
    names=None,
    index_col=None,
    usecols=None,
    squeeze=False,
    mangle_dupe_cols=True,
    dtype=None,
    nrows=None,
    parse_dates=False,
    quotechar=None,
    escapechar=None,
    comment=None,
    **options
) -> Union[DataFrame, Series]:
    """Read CSV (comma-separated) file into DataFrame or Series.

    Parameters
    ----------
    path : str
        The path string storing the CSV file to be read.
    sep : str, default ‘,’
        Delimiter to use. Must be a single character.
    header : int, list of int, default ‘infer’
        Whether to to use as the column names, and the start of the data.
        Default behavior is to infer the column names: if no names are passed
        the behavior is identical to `header=0` and column names are inferred from
        the first line of the file, if column names are passed explicitly then
        the behavior is identical to `header=None`. Explicitly pass `header=0` to be
        able to replace existing names
    names : str or array-like, optional
        List of column names to use. If file contains no header row, then you should
        explicitly pass `header=None`. Duplicates in this list will cause an error to be issued.
        If a string is given, it should be a DDL-formatted string in Spark SQL, which is
        preferred to avoid schema inference for better performance.
    index_col: str or list of str, optional, default: None
        Index column of table in Spark.
    usecols : list-like or callable, optional
        Return a subset of the columns. If list-like, all elements must either be
        positional (i.e. integer indices into the document columns) or strings that
        correspond to column names provided either by the user in names or inferred
        from the document header row(s).
        If callable, the callable function will be evaluated against the column names,
        returning names where the callable function evaluates to `True`.
    squeeze : bool, default False
        If the parsed data only contains one column then return a Series.
    mangle_dupe_cols : bool, default True
        Duplicate columns will be specified as 'X0', 'X1', ... 'XN', rather
        than 'X' ... 'X'. Passing in False will cause data to be overwritten if
        there are duplicate names in the columns.
        Currently only `True` is allowed.
    dtype : Type name or dict of column -> type, default None
        Data type for data or columns. E.g. {‘a’: np.float64, ‘b’: np.int32} Use str or object
        together with suitable na_values settings to preserve and not interpret dtype.
    nrows : int, default None
        Number of rows to read from the CSV file.
    parse_dates : boolean or list of ints or names or list of lists or dict, default `False`.
        Currently only `False` is allowed.
    quotechar : str (length 1), optional
        The character used to denote the start and end of a quoted item. Quoted items can include
        the delimiter and it will be ignored.
    escapechar : str (length 1), default None
        One-character string used to escape delimiter
    comment: str, optional
        Indicates the line should not be parsed.
    options : dict
        All other options passed directly into Spark's data source.

    Returns
    -------
    DataFrame or Series

    See Also
    --------
    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.

    Examples
    --------
    >>> ks.read_csv('data.csv')  # doctest: +SKIP
    """
    if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
        options = options.get("options")  # type: ignore

    if mangle_dupe_cols is not True:
        raise ValueError("mangle_dupe_cols can only be `True`: %s" % mangle_dupe_cols)
    if parse_dates is not False:
        raise ValueError("parse_dates can only be `False`: %s" % parse_dates)

    if usecols is not None and not callable(usecols):
        usecols = list(usecols)

    if usecols is None or callable(usecols) or len(usecols) > 0:
        reader = default_session().read
        reader.option("inferSchema", True)
        reader.option("sep", sep)

        if header == "infer":
            header = 0 if names is None else None
        if header == 0:
            reader.option("header", True)
        elif header is None:
            reader.option("header", False)
        else:
            raise ValueError("Unknown header argument {}".format(header))

        if quotechar is not None:
            reader.option("quote", quotechar)
        if escapechar is not None:
            reader.option("escape", escapechar)

        if comment is not None:
            if not isinstance(comment, str) or len(comment) != 1:
                raise ValueError("Only length-1 comment characters supported")
            reader.option("comment", comment)

        reader.options(**options)

        if isinstance(names, str):
            sdf = reader.schema(names).csv(path)
            column_labels = OrderedDict((col, col) for col in sdf.columns)
        else:
            sdf = reader.csv(path)
            if is_list_like(names):
                names = list(names)
                if len(set(names)) != len(names):
                    raise ValueError("Found non-unique column index")
                if len(names) != len(sdf.columns):
                    raise ValueError(
                        "The number of names [%s] does not match the number "
                        "of columns [%d]. Try names by a Spark SQL DDL-formatted "
                        "string." % (len(sdf.schema), len(names))
                    )
                column_labels = OrderedDict(zip(names, sdf.columns))
            elif header is None:
                column_labels = OrderedDict(enumerate(sdf.columns))
            else:
                column_labels = OrderedDict((col, col) for col in sdf.columns)

        if usecols is not None:
            if callable(usecols):
                column_labels = OrderedDict(
                    (label, col) for label, col in column_labels.items() if usecols(label)
                )
                missing = []
            elif all(isinstance(col, int) for col in usecols):
                new_column_labels = OrderedDict(
                    (label, col)
                    for i, (label, col) in enumerate(column_labels.items())
                    if i in usecols
                )
                missing = [
                    col
                    for col in usecols
                    if col >= len(column_labels)
                    or list(column_labels)[col] not in new_column_labels
                ]
                column_labels = new_column_labels
            elif all(isinstance(col, str) for col in usecols):
                new_column_labels = OrderedDict(
                    (label, col) for label, col in column_labels.items() if label in usecols
                )
                missing = [col for col in usecols if col not in new_column_labels]
                column_labels = new_column_labels
            else:
                raise ValueError(
                    "'usecols' must either be list-like of all strings, "
                    "all unicode, all integers or a callable."
                )
            if len(missing) > 0:
                raise ValueError(
                    "Usecols do not match columns, columns expected but not " "found: %s" % missing
                )

            if len(column_labels) > 0:
                sdf = sdf.select([scol_for(sdf, col) for col in column_labels.values()])
            else:
                sdf = default_session().createDataFrame([], schema=StructType())
    else:
        sdf = default_session().createDataFrame([], schema=StructType())
        column_labels = OrderedDict()

    if nrows is not None:
        sdf = sdf.limit(nrows)

    if index_col is not None:
        if isinstance(index_col, (str, int)):
            index_col = [index_col]
        for col in index_col:
            if col not in column_labels:
                raise KeyError(col)
        index_spark_column_names = [column_labels[col] for col in index_col]
        index_names = [(col,) for col in index_col]  # type: List[Tuple]
        column_labels = OrderedDict(
            (label, col) for label, col in column_labels.items() if label not in index_col
        )
    else:
        index_spark_column_names = []
        index_names = []

    kdf = DataFrame(
        InternalFrame(
            spark_frame=sdf,
            index_spark_columns=[scol_for(sdf, col) for col in index_spark_column_names],
            index_names=index_names,
            column_labels=[
                label if is_name_like_tuple(label) else (label,) for label in column_labels
            ],
            data_spark_columns=[scol_for(sdf, col) for col in column_labels.values()],
        )
    )  # type: DataFrame

    if dtype is not None:
        if isinstance(dtype, dict):
            for col, tpe in dtype.items():
                kdf[col] = kdf[col].astype(tpe)
        else:
            for col in kdf.columns:
                kdf[col] = kdf[col].astype(dtype)

    if squeeze and len(kdf.columns) == 1:
        return first_series(kdf)
    else:
        return kdf


def read_json(path: str, index_col: Optional[Union[str, List[str]]] = None, **options) -> DataFrame:
    """
    Convert a JSON string to DataFrame.

    Parameters
    ----------
    path : string
        File path
    index_col : str or list of str, optional, default: None
        Index column of table in Spark.
    options : dict
        All other options passed directly into Spark's data source.

    Examples
    --------
    >>> df = ks.DataFrame([['a', 'b'], ['c', 'd']],
    ...                   columns=['col 1', 'col 2'])

    >>> df.to_json(path=r'%s/read_json/foo.json' % path, num_files=1)
    >>> ks.read_json(
    ...     path=r'%s/read_json/foo.json' % path
    ... ).sort_values(by="col 1")
      col 1 col 2
    0     a     b
    1     c     d

    >>> df.to_json(path=r'%s/read_json/foo.json' % path, num_files=1, lineSep='___')
    >>> ks.read_json(
    ...     path=r'%s/read_json/foo.json' % path, lineSep='___'
    ... ).sort_values(by="col 1")
      col 1 col 2
    0     a     b
    1     c     d

    You can preserve the index in the roundtrip as below.

    >>> df.to_json(path=r'%s/read_json/bar.json' % path, num_files=1, index_col="index")
    >>> ks.read_json(
    ...     path=r'%s/read_json/bar.json' % path, index_col="index"
    ... ).sort_values(by="col 1")  # doctest: +NORMALIZE_WHITESPACE
          col 1 col 2
    index
    0         a     b
    1         c     d
    """
    if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
        options = options.get("options")  # type: ignore

    return read_spark_io(path, format="json", index_col=index_col, **options)


def read_delta(
    path: str,
    version: Optional[str] = None,
    timestamp: Optional[str] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    **options
) -> DataFrame:
    """
    Read a Delta Lake table on some file system and return a DataFrame.

    If the Delta Lake table is already stored in the catalog (aka the metastore), use 'read_table'.

    Parameters
    ----------
    path : string
        Path to the Delta Lake table.
    version : string, optional
        Specifies the table version (based on Delta's internal transaction version) to read from,
        using Delta's time travel feature. This sets Delta's 'versionAsOf' option.
    timestamp : string, optional
        Specifies the table version (based on timestamp) to read from,
        using Delta's time travel feature. This must be a valid date or timestamp string in Spark,
        and sets Delta's 'timestampAsOf' option.
    index_col : str or list of str, optional, default: None
        Index column of table in Spark.
    options
        Additional options that can be passed onto Delta.

    Returns
    -------
    DataFrame

    See Also
    --------
    DataFrame.to_delta
    read_table
    read_spark_io
    read_parquet

    Examples
    --------
    >>> ks.range(1).to_delta('%s/read_delta/foo' % path)
    >>> ks.read_delta('%s/read_delta/foo' % path)
       id
    0   0

    >>> ks.range(10, 15, num_partitions=1).to_delta('%s/read_delta/foo' % path, mode='overwrite')
    >>> ks.read_delta('%s/read_delta/foo' % path)
       id
    0  10
    1  11
    2  12
    3  13
    4  14

    >>> ks.read_delta('%s/read_delta/foo' % path, version=0)
       id
    0   0

    You can preserve the index in the roundtrip as below.

    >>> ks.range(10, 15, num_partitions=1).to_delta(
    ...     '%s/read_delta/bar' % path, index_col="index")
    >>> ks.read_delta('%s/read_delta/bar' % path, index_col="index")
    ... # doctest: +NORMALIZE_WHITESPACE
           id
    index
    0      10
    1      11
    2      12
    3      13
    4      14
    """
    if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
        options = options.get("options")  # type: ignore

    if version is not None:
        options["versionAsOf"] = version
    if timestamp is not None:
        options["timestampAsOf"] = timestamp
    return read_spark_io(path, format="delta", index_col=index_col, **options)


def read_table(name: str, index_col: Optional[Union[str, List[str]]] = None) -> DataFrame:
    """
    Read a Spark table and return a DataFrame.

    Parameters
    ----------
    name : string
        Table name in Spark.

    index_col : str or list of str, optional, default: None
        Index column of table in Spark.

    Returns
    -------
    DataFrame

    See Also
    --------
    DataFrame.to_table
    read_delta
    read_parquet
    read_spark_io

    Examples
    --------
    >>> ks.range(1).to_table('%s.my_table' % db)
    >>> ks.read_table('%s.my_table' % db)
       id
    0   0

    >>> ks.range(1).to_table('%s.my_table' % db, index_col="index")
    >>> ks.read_table('%s.my_table' % db, index_col="index")  # doctest: +NORMALIZE_WHITESPACE
           id
    index
    0       0
    """
    sdf = default_session().read.table(name)
    index_spark_columns, index_names = _get_index_map(sdf, index_col)

    return DataFrame(
        InternalFrame(
            spark_frame=sdf, index_spark_columns=index_spark_columns, index_names=index_names
        )
    )


def read_spark_io(
    path: Optional[str] = None,
    format: Optional[str] = None,
    schema: Union[str, "StructType"] = None,
    index_col: Optional[Union[str, List[str]]] = None,
    **options
) -> DataFrame:
    """Load a DataFrame from a Spark data source.

    Parameters
    ----------
    path : string, optional
        Path to the data source.
    format : string, optional
        Specifies the output data source format. Some common ones are:

        - 'delta'
        - 'parquet'
        - 'orc'
        - 'json'
        - 'csv'
    schema : string or StructType, optional
        Input schema. If none, Spark tries to infer the schema automatically.
        The schema can either be a Spark StructType, or a DDL-formatted string like
        `col0 INT, col1 DOUBLE`.
    index_col : str or list of str, optional, default: None
        Index column of table in Spark.
    options : dict
        All other options passed directly into Spark's data source.

    See Also
    --------
    DataFrame.to_spark_io
    DataFrame.read_table
    DataFrame.read_delta
    DataFrame.read_parquet

    Examples
    --------
    >>> ks.range(1).to_spark_io('%s/read_spark_io/data.parquet' % path)
    >>> ks.read_spark_io(
    ...     '%s/read_spark_io/data.parquet' % path, format='parquet', schema='id long')
       id
    0   0

    >>> ks.range(10, 15, num_partitions=1).to_spark_io('%s/read_spark_io/data.json' % path,
    ...                                                format='json', lineSep='__')
    >>> ks.read_spark_io(
    ...     '%s/read_spark_io/data.json' % path, format='json', schema='id long', lineSep='__')
       id
    0  10
    1  11
    2  12
    3  13
    4  14

    You can preserve the index in the roundtrip as below.

    >>> ks.range(10, 15, num_partitions=1).to_spark_io('%s/read_spark_io/data.orc' % path,
    ...                                                format='orc', index_col="index")
    >>> ks.read_spark_io(
    ...     path=r'%s/read_spark_io/data.orc' % path, format="orc", index_col="index")
    ... # doctest: +NORMALIZE_WHITESPACE
           id
    index
    0      10
    1      11
    2      12
    3      13
    4      14
    """
    if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
        options = options.get("options")  # type: ignore

    sdf = default_session().read.load(path=path, format=format, schema=schema, **options)
    index_spark_columns, index_names = _get_index_map(sdf, index_col)

    return DataFrame(
        InternalFrame(
            spark_frame=sdf, index_spark_columns=index_spark_columns, index_names=index_names
        )
    )


def read_parquet(path, columns=None, index_col=None, pandas_metadata=False, **options) -> DataFrame:
    """Load a parquet object from the file path, returning a DataFrame.

    Parameters
    ----------
    path : string
        File path
    columns : list, default=None
        If not None, only these columns will be read from the file.
    index_col : str or list of str, optional, default: None
        Index column of table in Spark.
    pandas_metadata : bool, default: False
        If True, try to respect the metadata if the Parquet file is written from pandas.
    options : dict
        All other options passed directly into Spark's data source.

    Returns
    -------
    DataFrame

    See Also
    --------
    DataFrame.to_parquet
    DataFrame.read_table
    DataFrame.read_delta
    DataFrame.read_spark_io

    Examples
    --------
    >>> ks.range(1).to_parquet('%s/read_spark_io/data.parquet' % path)
    >>> ks.read_parquet('%s/read_spark_io/data.parquet' % path, columns=['id'])
       id
    0   0

    You can preserve the index in the roundtrip as below.

    >>> ks.range(1).to_parquet('%s/read_spark_io/data.parquet' % path, index_col="index")
    >>> ks.read_parquet('%s/read_spark_io/data.parquet' % path, columns=['id'], index_col="index")
    ... # doctest: +NORMALIZE_WHITESPACE
           id
    index
    0       0
    """
    if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
        options = options.get("options")  # type: ignore

    if columns is not None:
        columns = list(columns)

    index_names = None

    if index_col is None and pandas_metadata:
        if LooseVersion(pyspark.__version__) < LooseVersion("3.0.0"):
            raise ValueError("pandas_metadata is not supported with Spark < 3.0.")

        # Try to read pandas metadata

        @pandas_udf("index_col array<string>, index_names array<string>", PandasUDFType.SCALAR)
        def read_index_metadata(pser):
            binary = pser.iloc[0]
            metadata = pq.ParquetFile(pa.BufferReader(binary)).metadata.metadata
            if b"pandas" in metadata:
                pandas_metadata = json.loads(metadata[b"pandas"].decode("utf8"))
                if all(isinstance(col, str) for col in pandas_metadata["index_columns"]):
                    index_col = []
                    index_names = []
                    for col in pandas_metadata["index_columns"]:
                        index_col.append(col)
                        for column in pandas_metadata["columns"]:
                            if column["field_name"] == col:
                                index_names.append(column["name"])
                                break
                        else:
                            index_names.append(None)
                    return pd.DataFrame({"index_col": [index_col], "index_names": [index_names]})
            return pd.DataFrame({"index_col": [None], "index_names": [None]})

        index_col, index_names = (
            default_session()
            .read.format("binaryFile")
            .load(path)
            .limit(1)
            .select(read_index_metadata("content").alias("index_metadata"))
            .select("index_metadata.*")
            .head()
        )

    kdf = read_spark_io(path=path, format="parquet", options=options, index_col=index_col)

    if columns is not None:
        new_columns = [c for c in columns if c in kdf.columns]
        if len(new_columns) > 0:
            kdf = kdf[new_columns]
        else:
            sdf = default_session().createDataFrame([], schema=StructType())
            index_spark_columns, index_names = _get_index_map(sdf, index_col)
            kdf = DataFrame(
                InternalFrame(
                    spark_frame=sdf,
                    index_spark_columns=index_spark_columns,
                    index_names=index_names,
                )
            )

    if index_names is not None:
        kdf.index.names = index_names

    return kdf


def read_clipboard(sep=r"\s+", **kwargs) -> DataFrame:
    r"""
    Read text from clipboard and pass to read_csv. See read_csv for the
    full argument list

    Parameters
    ----------
    sep : str, default '\s+'
        A string or regex delimiter. The default of '\s+' denotes
        one or more whitespace characters.

    See Also
    --------
    DataFrame.to_clipboard : Write text out to clipboard.

    Returns
    -------
    parsed : DataFrame
    """
    return cast(DataFrame, from_pandas(pd.read_clipboard(sep, **kwargs)))


def read_excel(
    io,
    sheet_name=0,
    header=0,
    names=None,
    index_col=None,
    usecols=None,
    squeeze=False,
    dtype=None,
    engine=None,
    converters=None,
    true_values=None,
    false_values=None,
    skiprows=None,
    nrows=None,
    na_values=None,
    keep_default_na=True,
    verbose=False,
    parse_dates=False,
    date_parser=None,
    thousands=None,
    comment=None,
    skipfooter=0,
    convert_float=True,
    mangle_dupe_cols=True,
    **kwds
) -> Union[DataFrame, Series, OrderedDict]:
    """
    Read an Excel file into a Koalas DataFrame or Series.

    Support both `xls` and `xlsx` file extensions from a local filesystem or URL.
    Support an option to read a single sheet or a list of sheets.

    Parameters
    ----------
    io : str, file descriptor, pathlib.Path, ExcelFile or xlrd.Book
        The string could be a URL. The value URL must be available in Spark's DataFrameReader.

        .. note::
            If the underlying Spark is below 3.0, the parameter as a string is not supported.
            You can use `ks.from_pandas(pd.read_excel(...))` as a workaround.

    sheet_name : str, int, list, or None, default 0
        Strings are used for sheet names. Integers are used in zero-indexed
        sheet positions. Lists of strings/integers are used to request
        multiple sheets. Specify None to get all sheets.

        Available cases:

        * Defaults to ``0``: 1st sheet as a `DataFrame`
        * ``1``: 2nd sheet as a `DataFrame`
        * ``"Sheet1"``: Load sheet with name "Sheet1"
        * ``[0, 1, "Sheet5"]``: Load first, second and sheet named "Sheet5"
          as a dict of `DataFrame`
        * None: All sheets.

    header : int, list of int, default 0
        Row (0-indexed) to use for the column labels of the parsed
        DataFrame. If a list of integers is passed those row positions will
        be combined into a ``MultiIndex``. Use None if there is no header.
    names : array-like, default None
        List of column names to use. If file contains no header row,
        then you should explicitly pass header=None.
    index_col : int, list of int, default None
        Column (0-indexed) to use as the row labels of the DataFrame.
        Pass None if there is no such column.  If a list is passed,
        those columns will be combined into a ``MultiIndex``.  If a
        subset of data is selected with ``usecols``, index_col
        is based on the subset.
    usecols : int, str, list-like, or callable default None
        Return a subset of the columns.

        * If None, then parse all columns.
        * If str, then indicates comma separated list of Excel column letters
          and column ranges (e.g. "A:E" or "A,C,E:F"). Ranges are inclusive of
          both sides.
        * If list of int, then indicates list of column numbers to be parsed.
        * If list of string, then indicates list of column names to be parsed.
        * If callable, then evaluate each column name against it and parse the
          column if the callable returns ``True``.
    squeeze : bool, default False
        If the parsed data only contains one column then return a Series.
    dtype : Type name or dict of column -> type, default None
        Data type for data or columns. E.g. {'a': np.float64, 'b': np.int32}
        Use `object` to preserve data as stored in Excel and not interpret dtype.
        If converters are specified, they will be applied INSTEAD
        of dtype conversion.
    engine : str, default None
        If io is not a buffer or path, this must be set to identify io.
        Acceptable values are None or xlrd.
    converters : dict, default None
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels, values are functions that take one
        input argument, the Excel cell content, and return the transformed
        content.
    true_values : list, default None
        Values to consider as True.
    false_values : list, default None
        Values to consider as False.
    skiprows : list-like
        Rows to skip at the beginning (0-indexed).
    nrows : int, default None
        Number of rows to parse.
    na_values : scalar, str, list-like, or dict, default None
        Additional strings to recognize as NA/NaN. If dict passed, specific
        per-column NA values. By default the following values are interpreted
        as NaN.
    keep_default_na : bool, default True
        If na_values are specified and keep_default_na is False the default NaN
        values are overridden, otherwise they're appended to.
    verbose : bool, default False
        Indicate number of NA values placed in non-numeric columns.
    parse_dates : bool, list-like, or dict, default False
        The behavior is as follows:

        * bool. If True -> try parsing the index.
        * list of int or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3
          each as a separate date column.
        * list of lists. e.g.  If [[1, 3]] -> combine columns 1 and 3 and parse as
          a single date column.
        * dict, e.g. {{'foo' : [1, 3]}} -> parse columns 1, 3 as date and call
          result 'foo'

        If a column or index contains an unparseable date, the entire column or
        index will be returned unaltered as an object data type. For non-standard
        datetime parsing, use ``pd.to_datetime`` after ``pd.read_csv``

        Note: A fast-path exists for iso8601-formatted dates.
    date_parser : function, optional
        Function to use for converting a sequence of string columns to an array of
        datetime instances. The default uses ``dateutil.parser.parser`` to do the
        conversion. Koalas will try to call `date_parser` in three different ways,
        advancing to the next if an exception occurs: 1) Pass one or more arrays
        (as defined by `parse_dates`) as arguments; 2) concatenate (row-wise) the
        string values from the columns defined by `parse_dates` into a single array
        and pass that; and 3) call `date_parser` once for each row using one or
        more strings (corresponding to the columns defined by `parse_dates`) as
        arguments.
    thousands : str, default None
        Thousands separator for parsing string columns to numeric.  Note that
        this parameter is only necessary for columns stored as TEXT in Excel,
        any numeric columns will automatically be parsed, regardless of display
        format.
    comment : str, default None
        Comments out remainder of line. Pass a character or characters to this
        argument to indicate comments in the input file. Any data between the
        comment string and the end of the current line is ignored.
    skipfooter : int, default 0
        Rows at the end to skip (0-indexed).
    convert_float : bool, default True
        Convert integral floats to int (i.e., 1.0 --> 1). If False, all numeric
        data will be read in as floats: Excel stores all numbers as floats
        internally.
    mangle_dupe_cols : bool, default True
        Duplicate columns will be specified as 'X', 'X.1', ...'X.N', rather than
        'X'...'X'. Passing in False will cause data to be overwritten if there
        are duplicate names in the columns.
    **kwds : optional
        Optional keyword arguments can be passed to ``TextFileReader``.

    Returns
    -------
    DataFrame or dict of DataFrames
        DataFrame from the passed in Excel file. See notes in sheet_name
        argument for more information on when a dict of DataFrames is returned.

    See Also
    --------
    DataFrame.to_excel : Write DataFrame to an Excel file.
    DataFrame.to_csv : Write DataFrame to a comma-separated values (csv) file.
    read_csv : Read a comma-separated values (csv) file into DataFrame.

    Examples
    --------
    The file can be read using the file name as string or an open file object:

    >>> ks.read_excel('tmp.xlsx', index_col=0)  # doctest: +SKIP
           Name  Value
    0   string1      1
    1   string2      2
    2  #Comment      3

    >>> ks.read_excel(open('tmp.xlsx', 'rb'),
    ...               sheet_name='Sheet3')  # doctest: +SKIP
       Unnamed: 0      Name  Value
    0           0   string1      1
    1           1   string2      2
    2           2  #Comment      3

    Index and header can be specified via the `index_col` and `header` arguments

    >>> ks.read_excel('tmp.xlsx', index_col=None, header=None)  # doctest: +SKIP
         0         1      2
    0  NaN      Name  Value
    1  0.0   string1      1
    2  1.0   string2      2
    3  2.0  #Comment      3

    Column types are inferred but can be explicitly specified

    >>> ks.read_excel('tmp.xlsx', index_col=0,
    ...               dtype={'Name': str, 'Value': float})  # doctest: +SKIP
           Name  Value
    0   string1    1.0
    1   string2    2.0
    2  #Comment    3.0

    True, False, and NA values, and thousands separators have defaults,
    but can be explicitly specified, too. Supply the values you would like
    as strings or lists of strings!

    >>> ks.read_excel('tmp.xlsx', index_col=0,
    ...               na_values=['string1', 'string2'])  # doctest: +SKIP
           Name  Value
    0      None      1
    1      None      2
    2  #Comment      3

    Comment lines in the excel input file can be skipped using the `comment` kwarg

    >>> ks.read_excel('tmp.xlsx', index_col=0, comment='#')  # doctest: +SKIP
          Name  Value
    0  string1    1.0
    1  string2    2.0
    2     None    NaN
    """

    def pd_read_excel(io_or_bin, sn, sq):
        return pd.read_excel(
            io=BytesIO(io_or_bin) if isinstance(io_or_bin, (bytes, bytearray)) else io_or_bin,
            sheet_name=sn,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            squeeze=sq,
            dtype=dtype,
            engine=engine,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            verbose=verbose,
            parse_dates=parse_dates,
            date_parser=date_parser,
            thousands=thousands,
            comment=comment,
            skipfooter=skipfooter,
            convert_float=convert_float,
            mangle_dupe_cols=mangle_dupe_cols,
            **kwds
        )

    if isinstance(io, str):
        if LooseVersion(pyspark.__version__) < LooseVersion("3.0.0"):
            raise ValueError(
                "The `io` parameter as a string is not supported if the underlying Spark is "
                "below 3.0. You can use `ks.from_pandas(pd.read_excel(...))` as a workaround"
            )
        # 'binaryFile' format is available since Spark 3.0.0.
        binaries = default_session().read.format("binaryFile").load(io).select("content").head(2)
        io_or_bin = binaries[0][0]
        single_file = len(binaries) == 1
    else:
        io_or_bin = io
        single_file = True

    pdf_or_psers = pd_read_excel(io_or_bin, sn=sheet_name, sq=squeeze)

    if single_file:
        if isinstance(pdf_or_psers, dict):
            return OrderedDict(
                [(sn, from_pandas(pdf_or_pser)) for sn, pdf_or_pser in pdf_or_psers.items()]
            )
        else:
            return cast(Union[DataFrame, Series], from_pandas(pdf_or_psers))
    else:

        def read_excel_on_spark(pdf_or_pser, sn):

            if isinstance(pdf_or_pser, pd.Series):
                pdf = pdf_or_pser.to_frame()
            else:
                pdf = pdf_or_pser

            kdf = from_pandas(pdf)
            return_schema = force_decimal_precision_scale(
                as_nullable_spark_type(kdf._internal.spark_frame.drop(*HIDDEN_COLUMNS).schema)
            )

            def output_func(pdf):
                pdf = pd.concat(
                    [pd_read_excel(bin, sn=sn, sq=False) for bin in pdf[pdf.columns[0]]]
                )

                reset_index = pdf.reset_index()
                for name, col in reset_index.iteritems():
                    dt = col.dtype
                    if is_datetime64_dtype(dt) or is_datetime64tz_dtype(dt):
                        continue
                    reset_index[name] = col.replace({np.nan: None})
                pdf = reset_index

                # Just positionally map the column names to given schema's.
                return pdf.rename(columns=dict(zip(pdf.columns, return_schema.names)))

            sdf = (
                default_session()
                .read.format("binaryFile")
                .load(io)
                .select("content")
                .mapInPandas(lambda iterator: map(output_func, iterator), schema=return_schema)
            )

            kdf = DataFrame(kdf._internal.with_new_sdf(sdf))
            if squeeze and len(kdf.columns) == 1:
                return first_series(kdf)
            else:
                return kdf

        if isinstance(pdf_or_psers, dict):
            return OrderedDict(
                [
                    (sn, read_excel_on_spark(pdf_or_pser, sn))
                    for sn, pdf_or_pser in pdf_or_psers.items()
                ]
            )
        else:
            return read_excel_on_spark(pdf_or_psers, sheet_name)


def read_html(
    io,
    match=".+",
    flavor=None,
    header=None,
    index_col=None,
    skiprows=None,
    attrs=None,
    parse_dates=False,
    thousands=",",
    encoding=None,
    decimal=".",
    converters=None,
    na_values=None,
    keep_default_na=True,
    displayed_only=True,
) -> List[DataFrame]:
    r"""Read HTML tables into a ``list`` of ``DataFrame`` objects.

    Parameters
    ----------
    io : str or file-like
        A URL, a file-like object, or a raw string containing HTML. Note that
        lxml only accepts the http, ftp and file url protocols. If you have a
        URL that starts with ``'https'`` you might try removing the ``'s'``.

    match : str or compiled regular expression, optional
        The set of tables containing text matching this regex or string will be
        returned. Unless the HTML is extremely simple you will probably need to
        pass a non-empty string here. Defaults to '.+' (match any non-empty
        string). The default value will return all tables contained on a page.
        This value is converted to a regular expression so that there is
        consistent behavior between Beautiful Soup and lxml.

    flavor : str or None, container of strings
        The parsing engine to use. 'bs4' and 'html5lib' are synonymous with
        each other, they are both there for backwards compatibility. The
        default of ``None`` tries to use ``lxml`` to parse and if that fails it
        falls back on ``bs4`` + ``html5lib``.

    header : int or list-like or None, optional
        The row (or list of rows for a :class:`~ks.MultiIndex`) to use to
        make the columns headers.

    index_col : int or list-like or None, optional
        The column (or list of columns) to use to create the index.

    skiprows : int or list-like or slice or None, optional
        0-based. Number of rows to skip after parsing the column integer. If a
        sequence of integers or a slice is given, will skip the rows indexed by
        that sequence.  Note that a single element sequence means 'skip the nth
        row' whereas an integer means 'skip n rows'.

    attrs : dict or None, optional
        This is a dictionary of attributes that you can pass to use to identify
        the table in the HTML. These are not checked for validity before being
        passed to lxml or Beautiful Soup. However, these attributes must be
        valid HTML table attributes to work correctly. For example, ::

            attrs = {'id': 'table'}

        is a valid attribute dictionary because the 'id' HTML tag attribute is
        a valid HTML attribute for *any* HTML tag as per `this document
        <http://www.w3.org/TR/html-markup/global-attributes.html>`__. ::

            attrs = {'asdf': 'table'}

        is *not* a valid attribute dictionary because 'asdf' is not a valid
        HTML attribute even if it is a valid XML attribute.  Valid HTML 4.01
        table attributes can be found `here
        <http://www.w3.org/TR/REC-html40/struct/tables.html#h-11.2>`__. A
        working draft of the HTML 5 spec can be found `here
        <http://www.w3.org/TR/html-markup/table.html>`__. It contains the
        latest information on table attributes for the modern web.

    parse_dates : bool, optional
        See :func:`~ks.read_csv` for more details.

    thousands : str, optional
        Separator to use to parse thousands. Defaults to ``','``.

    encoding : str or None, optional
        The encoding used to decode the web page. Defaults to ``None``.``None``
        preserves the previous encoding behavior, which depends on the
        underlying parser library (e.g., the parser library will try to use
        the encoding provided by the document).

    decimal : str, default '.'
        Character to recognize as decimal point (e.g. use ',' for European
        data).

    converters : dict, default None
        Dict of functions for converting values in certain columns. Keys can
        either be integers or column labels, values are functions that take one
        input argument, the cell (not column) content, and return the
        transformed content.

    na_values : iterable, default None
        Custom NA values

    keep_default_na : bool, default True
        If na_values are specified and keep_default_na is False the default NaN
        values are overridden, otherwise they're appended to

    displayed_only : bool, default True
        Whether elements with "display: none" should be parsed

    Returns
    -------
    dfs : list of DataFrames

    See Also
    --------
    read_csv
    DataFrame.to_html
    """
    pdfs = pd.read_html(
        io=io,
        match=match,
        flavor=flavor,
        header=header,
        index_col=index_col,
        skiprows=skiprows,
        attrs=attrs,
        parse_dates=parse_dates,
        thousands=thousands,
        encoding=encoding,
        decimal=decimal,
        converters=converters,
        na_values=na_values,
        keep_default_na=keep_default_na,
        displayed_only=displayed_only,
    )
    return cast(List[DataFrame], [from_pandas(pdf) for pdf in pdfs])


# TODO: add `coerce_float` and 'parse_dates' parameters
def read_sql_table(
    table_name, con, schema=None, index_col=None, columns=None, **options
) -> DataFrame:
    """
    Read SQL database table into a DataFrame.

    Given a table name and a JDBC URI, returns a DataFrame.

    Parameters
    ----------
    table_name : str
        Name of SQL table in database.
    con : str
        A JDBC URI could be provided as as str.

        .. note:: The URI must be JDBC URI instead of Python's database URI.

    schema : str, default None
        Name of SQL schema in database to query (if database flavor
        supports this). Uses default schema if None (default).
    index_col : str or list of str, optional, default: None
        Column(s) to set as index(MultiIndex).
    columns : list, default None
        List of column names to select from SQL table.
    options : dict
        All other options passed directly into Spark's JDBC data source.

    Returns
    -------
    DataFrame
        A SQL table is returned as two-dimensional data structure with labeled
        axes.

    See Also
    --------
    read_sql_query : Read SQL query into a DataFrame.
    read_sql : Read SQL query or database table into a DataFrame.

    Examples
    --------
    >>> ks.read_sql_table('table_name', 'jdbc:postgresql:db_name')  # doctest: +SKIP
    """
    if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
        options = options.get("options")  # type: ignore

    reader = default_session().read
    reader.option("dbtable", table_name)
    reader.option("url", con)
    if schema is not None:
        reader.schema(schema)
    reader.options(**options)
    sdf = reader.format("jdbc").load()
    index_spark_columns, index_names = _get_index_map(sdf, index_col)
    kdf = DataFrame(
        InternalFrame(
            spark_frame=sdf, index_spark_columns=index_spark_columns, index_names=index_names
        )
    )  # type: DataFrame
    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        kdf = kdf[columns]
    return kdf


# TODO: add `coerce_float`, `params`, and 'parse_dates' parameters
def read_sql_query(sql, con, index_col=None, **options) -> DataFrame:
    """Read SQL query into a DataFrame.

    Returns a DataFrame corresponding to the result set of the query
    string. Optionally provide an `index_col` parameter to use one of the
    columns as the index, otherwise default index will be used.

    .. note:: Some database might hit the issue of Spark: SPARK-27596

    Parameters
    ----------
    sql : string SQL query
        SQL query to be executed.
    con : str
        A JDBC URI could be provided as as str.

        .. note:: The URI must be JDBC URI instead of Python's database URI.

    index_col : string or list of strings, optional, default: None
        Column(s) to set as index(MultiIndex).
    options : dict
        All other options passed directly into Spark's JDBC data source.

    Returns
    -------
    DataFrame

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql

    Examples
    --------
    >>> ks.read_sql_query('SELECT * FROM table_name', 'jdbc:postgresql:db_name')  # doctest: +SKIP
    """
    if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
        options = options.get("options")  # type: ignore

    reader = default_session().read
    reader.option("query", sql)
    reader.option("url", con)
    reader.options(**options)
    sdf = reader.format("jdbc").load()
    index_spark_columns, index_names = _get_index_map(sdf, index_col)
    return DataFrame(
        InternalFrame(
            spark_frame=sdf, index_spark_columns=index_spark_columns, index_names=index_names
        )
    )


# TODO: add `coerce_float`, `params`, and 'parse_dates' parameters
def read_sql(sql, con, index_col=None, columns=None, **options) -> DataFrame:
    """
    Read SQL query or database table into a DataFrame.

    This function is a convenience wrapper around ``read_sql_table`` and
    ``read_sql_query`` (for backward compatibility). It will delegate
    to the specific function depending on the provided input. A SQL query
    will be routed to ``read_sql_query``, while a database table name will
    be routed to ``read_sql_table``. Note that the delegated function might
    have more specific notes about their functionality not listed here.

    .. note:: Some database might hit the issue of Spark: SPARK-27596

    Parameters
    ----------
    sql : string
        SQL query to be executed or a table name.
    con : str
        A JDBC URI could be provided as as str.

        .. note:: The URI must be JDBC URI instead of Python's database URI.

    index_col : string or list of strings, optional, default: None
        Column(s) to set as index(MultiIndex).
    columns : list, default: None
        List of column names to select from SQL table (only used when reading
        a table).
    options : dict
        All other options passed directly into Spark's JDBC data source.

    Returns
    -------
    DataFrame

    See Also
    --------
    read_sql_table : Read SQL database table into a DataFrame.
    read_sql_query : Read SQL query into a DataFrame.

    Examples
    --------
    >>> ks.read_sql('table_name', 'jdbc:postgresql:db_name')  # doctest: +SKIP
    >>> ks.read_sql('SELECT * FROM table_name', 'jdbc:postgresql:db_name')  # doctest: +SKIP
    """
    if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
        options = options.get("options")  # type: ignore

    striped = sql.strip()
    if " " not in striped:  # TODO: identify the table name or not more precisely.
        return read_sql_table(sql, con, index_col=index_col, columns=columns, **options)
    else:
        return read_sql_query(sql, con, index_col=index_col, **options)


def to_datetime(
    arg, errors="raise", format=None, unit=None, infer_datetime_format=False, origin="unix"
):
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
    unit : string, default None
        unit of the arg (D,s,ms,us,ns) denote the unit, which is an
        integer or float number. This will be based off the origin.
        Example, with unit='ms' and origin='unix' (the default), this
        would calculate the number of milliseconds to the unix epoch start.
    infer_datetime_format : boolean, default False
        If True and no `format` is given, attempt to infer the format of the
        datetime strings, and if it can be inferred, switch to a faster
        method of parsing them. In some cases this can increase the parsing
        speed by ~5-10x.
    origin : scalar, default 'unix'
        Define the reference date. The numeric values would be parsed as number
        of units (defined by `unit`) since this reference date.

        - If 'unix' (or POSIX) time; origin is set to 1970-01-01.
        - If 'julian', unit must be 'D', and origin is set to beginning of
          Julian Calendar. Julian day number 0 is assigned to the day starting
          at noon on January 1, 4713 BC.
        - If Timestamp convertible, origin is set to Timestamp identified by
          origin.

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
    dtype: datetime64[ns]

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
    dtype: object

    >>> import timeit
    >>> timeit.timeit(
    ...    lambda: repr(ks.to_datetime(s, infer_datetime_format=True)),
    ...    number = 1)  # doctest: +SKIP
    0.35832712500000063

    >>> timeit.timeit(
    ...    lambda: repr(ks.to_datetime(s, infer_datetime_format=False)),
    ...    number = 1)  # doctest: +SKIP
    0.8895321660000004

    Using a unix epoch time

    >>> ks.to_datetime(1490195805, unit='s')
    Timestamp('2017-03-22 15:16:45')
    >>> ks.to_datetime(1490195805433502912, unit='ns')
    Timestamp('2017-03-22 15:16:45.433502912')

    Using a non-unix epoch origin

    >>> ks.to_datetime([1, 2, 3], unit='D', origin=pd.Timestamp('1960-01-01'))
    DatetimeIndex(['1960-01-02', '1960-01-03', '1960-01-04'], dtype='datetime64[ns]', freq=None)
    """

    def pandas_to_datetime(pser_or_pdf) -> Series[np.datetime64]:
        if isinstance(pser_or_pdf, pd.DataFrame):
            pser_or_pdf = pser_or_pdf[["year", "month", "day"]]
        return pd.to_datetime(
            pser_or_pdf,
            errors=errors,
            format=format,
            unit=unit,
            infer_datetime_format=infer_datetime_format,
            origin=origin,
        )

    if isinstance(arg, Series):
        return arg.koalas.transform_batch(pandas_to_datetime)
    if isinstance(arg, DataFrame):
        kdf = arg[["year", "month", "day"]]
        return kdf.koalas.transform_batch(pandas_to_datetime)
    return pd.to_datetime(
        arg,
        errors=errors,
        format=format,
        unit=unit,
        infer_datetime_format=infer_datetime_format,
        origin=origin,
    )


def get_dummies(
    data,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    sparse=False,
    drop_first=False,
    dtype=None,
) -> DataFrame:
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

    if columns is not None:
        if not is_list_like(columns):
            raise TypeError("Input must be a list-like for parameter `columns`")

    if dtype is None:
        dtype = "byte"

    if isinstance(data, Series):
        if prefix is not None:
            prefix = [str(prefix)]
        kdf = data.to_frame()
        column_labels = kdf._internal.column_labels
        remaining_columns = []
    else:
        if isinstance(prefix, str):
            raise NotImplementedError(
                "get_dummies currently does not support prefix as string types"
            )
        kdf = data.copy()

        if columns is None:
            column_labels = [
                label
                for label in kdf._internal.column_labels
                if isinstance(
                    kdf._internal.spark_type_for(label), _get_dummies_default_accept_types
                )
            ]
        else:
            if is_name_like_tuple(columns):
                column_labels = [
                    label
                    for label in kdf._internal.column_labels
                    if label[: len(columns)] == columns
                ]
                if len(column_labels) == 0:
                    raise KeyError(name_like_string(columns))
                if prefix is None:
                    prefix = [
                        str(label[len(columns) :])
                        if len(label) > len(columns) + 1
                        else label[len(columns)]
                        if len(label) == len(columns) + 1
                        else ""
                        for label in column_labels
                    ]
            elif any(isinstance(col, tuple) for col in columns) and any(
                not is_name_like_tuple(col) for col in columns
            ):
                raise ValueError(
                    "Expected tuple, got {}".format(
                        type(set(col for col in columns if not is_name_like_tuple(col)).pop())
                    )
                )
            else:
                column_labels = [
                    label
                    for key in columns
                    for label in kdf._internal.column_labels
                    if label == key or label[0] == key
                ]
        if len(column_labels) == 0:
            if columns is None:
                return kdf
            raise KeyError("{} not in index".format(columns))

        if prefix is None:
            prefix = [str(label) if len(label) > 1 else label[0] for label in column_labels]

        column_labels_set = set(column_labels)
        remaining_columns = [
            (
                kdf[label]
                if kdf._internal.column_labels_level == 1
                else kdf[label].rename(name_like_string(label))
            )
            for label in kdf._internal.column_labels
            if label not in column_labels_set
        ]

    if any(
        not isinstance(kdf._internal.spark_type_for(label), _get_dummies_acceptable_types)
        for label in column_labels
    ):
        raise NotImplementedError(
            "get_dummies currently only accept {} values".format(
                ", ".join([t.typeName() for t in _get_dummies_acceptable_types])
            )
        )

    if prefix is not None and len(column_labels) != len(prefix):
        raise ValueError(
            "Length of 'prefix' ({}) did not match the length of "
            "the columns being encoded ({}).".format(len(prefix), len(column_labels))
        )
    elif isinstance(prefix, dict):
        prefix = [prefix[column_label[0]] for column_label in column_labels]

    all_values = _reduce_spark_multi(
        kdf._internal.spark_frame,
        [F.collect_set(kdf._internal.spark_column_for(label)) for label in column_labels],
    )
    for i, label in enumerate(column_labels):
        values = sorted(all_values[i])
        if drop_first:
            values = values[1:]

        def column_name(value):
            if prefix is None or prefix[i] == "":
                return value
            else:
                return "{}{}{}".format(prefix[i], prefix_sep, value)

        for value in values:
            remaining_columns.append(
                (kdf[label].notnull() & (kdf[label] == value))
                .astype(dtype)
                .rename(column_name(value))
            )
        if dummy_na:
            remaining_columns.append(kdf[label].isnull().astype(dtype).rename(column_name(np.nan)))

    return kdf[remaining_columns]


# TODO: there are many parameters to implement and support. See pandas's pd.concat.
def concat(objs, axis=0, join="outer", ignore_index=False, sort=False) -> Union[Series, DataFrame]:
    """
    Concatenate Koalas objects along a particular axis with optional set logic
    along the other axes.

    Parameters
    ----------
    objs : a sequence of Series or DataFrame
        Any None objects will be dropped silently unless
        they are all None in which case a ValueError will be raised
    axis : {0/'index', 1/'columns'}, default 0
        The axis to concatenate along.
    join : {'inner', 'outer'}, default 'outer'
        How to handle indexes on other axis (or axes).
    ignore_index : bool, default False
        If True, do not use the index values along the concatenation axis. The
        resulting axis will be labeled 0, ..., n - 1. This is useful if you are
        concatenating objects where the concatenation axis does not have
        meaningful indexing information. Note the index values on the other
        axes are still respected in the join.
    sort : bool, default False
        Sort non-concatenation axis if it is not already aligned.

    Returns
    -------
    object, type of objs
        When concatenating all ``Series`` along the index (axis=0), a
        ``Series`` is returned. When ``objs`` contains at least one
        ``DataFrame``, a ``DataFrame`` is returned. When concatenating along
        the columns (axis=1), a ``DataFrame`` is returned.

    See Also
    --------
    Series.append : Concatenate Series.
    DataFrame.join : Join DataFrames using indexes.
    DataFrame.merge : Merge DataFrames by indexes or columns.

    Examples
    --------
    >>> from databricks.koalas.config import set_option, reset_option
    >>> set_option("compute.ops_on_diff_frames", True)

    Combine two ``Series``.

    >>> s1 = ks.Series(['a', 'b'])
    >>> s2 = ks.Series(['c', 'd'])
    >>> ks.concat([s1, s2])
    0    a
    1    b
    0    c
    1    d
    dtype: object

    Clear the existing index and reset it in the result
    by setting the ``ignore_index`` option to ``True``.

    >>> ks.concat([s1, s2], ignore_index=True)
    0    a
    1    b
    2    c
    3    d
    dtype: object

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

    >>> ks.concat([df2, s1])
      letter  number     0
    0      c     3.0  None
    1      d     4.0  None
    0   None     NaN     a
    1   None     NaN     b

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
      letter  number animal
    0      a       1   None
    1      b       2   None
    0      c       3    cat
    1      d       4    dog

    Sort the columns.

    >>> ks.concat([df1, df3], sort=True)
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

    >>> df4 = ks.DataFrame([['bird', 'polly'], ['monkey', 'george']],
    ...                    columns=['animal', 'name'])

    Combine with column axis.

    >>> ks.concat([df1, df4], axis=1)
      letter  number  animal    name
    0      a       1    bird   polly
    1      b       2  monkey  george

    >>> reset_option("compute.ops_on_diff_frames")
    """
    if isinstance(objs, (DataFrame, IndexOpsMixin)) or not isinstance(
        objs, Iterable
    ):  # TODO: support dict
        raise TypeError(
            "first argument must be an iterable of Koalas "
            "objects, you passed an object of type "
            '"{name}"'.format(name=type(objs).__name__)
        )

    if len(cast(Sized, objs)) == 0:
        raise ValueError("No objects to concatenate")
    objs = list(filter(lambda obj: obj is not None, objs))
    if len(objs) == 0:
        raise ValueError("All objects passed were None")

    for obj in objs:
        if not isinstance(obj, (Series, DataFrame)):
            raise TypeError(
                "cannot concatenate object of type "
                "'{name}"
                "; only ks.Series "
                "and ks.DataFrame are valid".format(name=type(objs).__name__)
            )

    if join not in ["inner", "outer"]:
        raise ValueError("Only can inner (intersect) or outer (union) join the other axis.")

    axis = validate_axis(axis)
    if axis == 1:
        kdfs = [obj.to_frame() if isinstance(obj, Series) else obj for obj in objs]

        level = min(kdf._internal.column_labels_level for kdf in kdfs)
        kdfs = [
            DataFrame._index_normalized_frame(level, kdf)
            if kdf._internal.column_labels_level > level
            else kdf
            for kdf in kdfs
        ]

        concat_kdf = kdfs[0]
        column_labels = concat_kdf._internal.column_labels.copy()

        kdfs_not_same_anchor = []
        for kdf in kdfs[1:]:
            duplicated = [label for label in kdf._internal.column_labels if label in column_labels]
            if len(duplicated) > 0:
                pretty_names = [name_like_string(label) for label in duplicated]
                raise ValueError(
                    "Labels have to be unique; however, got duplicated labels %s." % pretty_names
                )
            column_labels.extend(kdf._internal.column_labels)

            if same_anchor(concat_kdf, kdf):
                concat_kdf = DataFrame(
                    concat_kdf._internal.with_new_columns(
                        concat_kdf._internal.data_spark_columns + kdf._internal.data_spark_columns,
                        concat_kdf._internal.column_labels + kdf._internal.column_labels,
                    )
                )
            else:
                kdfs_not_same_anchor.append(kdf)

        if len(kdfs_not_same_anchor) > 0:

            def resolve_func(kdf, this_column_labels, that_column_labels):
                raise AssertionError("This should not happen.")

            for kdf in kdfs_not_same_anchor:
                if join == "inner":
                    concat_kdf = align_diff_frames(
                        resolve_func, concat_kdf, kdf, fillna=False, how="inner",
                    )
                elif join == "outer":
                    concat_kdf = align_diff_frames(
                        resolve_func, concat_kdf, kdf, fillna=False, how="full",
                    )

            concat_kdf = concat_kdf[column_labels]

        if ignore_index:
            concat_kdf.columns = list(map(str, _range(len(concat_kdf.columns))))

        if sort:
            concat_kdf = concat_kdf.sort_index()

        return concat_kdf

    # Series, Series ...
    # We should return Series if objects are all Series.
    should_return_series = all(map(lambda obj: isinstance(obj, Series), objs))

    # DataFrame, Series ... & Series, Series ...
    # In this case, we should return DataFrame.
    new_objs = []
    num_series = 0
    series_names = set()
    for obj in objs:
        if isinstance(obj, Series):
            num_series += 1
            series_names.add(obj.name)
            obj = obj.to_frame(DEFAULT_SERIES_NAME)
        new_objs.append(obj)
    objs = new_objs

    column_labels_levels = set(obj._internal.column_labels_level for obj in objs)
    if len(column_labels_levels) != 1:
        raise ValueError("MultiIndex columns should have the same levels")

    # DataFrame, DataFrame, ...
    # All Series are converted into DataFrame and then compute concat.
    if not ignore_index:
        indices_of_kdfs = [kdf.index for kdf in objs]
        index_of_first_kdf = indices_of_kdfs[0]
        for index_of_kdf in indices_of_kdfs:
            if index_of_first_kdf.names != index_of_kdf.names:
                raise ValueError(
                    "Index type and names should be same in the objects to concatenate. "
                    "You passed different indices "
                    "{index_of_first_kdf} and {index_of_kdf}".format(
                        index_of_first_kdf=index_of_first_kdf.names, index_of_kdf=index_of_kdf.names
                    )
                )

    column_labels_of_kdfs = [kdf._internal.column_labels for kdf in objs]
    if ignore_index:
        index_names_of_kdfs = [[] for _ in objs]  # type: List
    else:
        index_names_of_kdfs = [kdf._internal.index_names for kdf in objs]

    if all(name == index_names_of_kdfs[0] for name in index_names_of_kdfs) and all(
        idx == column_labels_of_kdfs[0] for idx in column_labels_of_kdfs
    ):
        # If all columns are in the same order and values, use it.
        kdfs = objs
    else:
        if join == "inner":
            interested_columns = set.intersection(*map(set, column_labels_of_kdfs))
            # Keep the column order with its firsts DataFrame.
            merged_columns = [
                label for label in column_labels_of_kdfs[0] if label in interested_columns
            ]

            # When multi-index column, although pandas is flaky if `join="inner" and sort=False`,
            # always sort to follow the `join="outer"` case behavior.
            if (len(merged_columns) > 0 and len(merged_columns[0]) > 1) or sort:
                # FIXME: better ordering
                merged_columns = sorted(merged_columns, key=name_like_string)

            kdfs = [kdf[merged_columns] for kdf in objs]
        elif join == "outer":
            merged_columns = []
            for labels in column_labels_of_kdfs:
                merged_columns.extend(label for label in labels if label not in merged_columns)

            assert len(merged_columns) > 0

            if LooseVersion(pd.__version__) < LooseVersion("0.24"):
                # Always sort when multi-index columns, and if there are Series, never sort.
                sort = len(merged_columns[0]) > 1 or (num_series == 0 and sort)
            else:
                # Always sort when multi-index columns or there are more than two Series,
                # and if there is only one Series, never sort.
                sort = len(merged_columns[0]) > 1 or num_series > 1 or (num_series != 1 and sort)

            if sort:
                # FIXME: better ordering
                merged_columns = sorted(merged_columns, key=name_like_string)

            kdfs = []
            for kdf in objs:
                columns_to_add = list(set(merged_columns) - set(kdf._internal.column_labels))

                # TODO: NaN and None difference for missing values. pandas seems filling NaN.
                sdf = kdf._internal.resolved_copy.spark_frame
                for label in columns_to_add:
                    sdf = sdf.withColumn(name_like_string(label), F.lit(None))

                data_columns = kdf._internal.data_spark_column_names + [
                    name_like_string(label) for label in columns_to_add
                ]
                kdf = DataFrame(
                    kdf._internal.copy(
                        spark_frame=sdf,
                        index_spark_columns=[
                            scol_for(sdf, col) for col in kdf._internal.index_spark_column_names
                        ],
                        column_labels=(kdf._internal.column_labels + columns_to_add),
                        data_spark_columns=[scol_for(sdf, col) for col in data_columns],
                    )
                )

                kdfs.append(kdf[merged_columns])

    if ignore_index:
        sdfs = [kdf._internal.spark_frame.select(kdf._internal.data_spark_columns) for kdf in kdfs]
    else:
        sdfs = [
            kdf._internal.spark_frame.select(
                kdf._internal.index_spark_columns + kdf._internal.data_spark_columns
            )
            for kdf in kdfs
        ]
    concatenated = reduce(lambda x, y: x.union(y), sdfs)

    if ignore_index:
        index_spark_column_names = []
        index_names = []
    else:
        index_spark_column_names = kdfs[0]._internal.index_spark_column_names
        index_names = kdfs[0]._internal.index_names

    result_kdf = DataFrame(
        kdfs[0]._internal.copy(
            spark_frame=concatenated,
            index_spark_columns=[scol_for(concatenated, col) for col in index_spark_column_names],
            index_names=index_names,
            data_spark_columns=[
                scol_for(concatenated, col) for col in kdfs[0]._internal.data_spark_column_names
            ],
        )
    )  # type: DataFrame

    if should_return_series:
        # If all input were Series, we should return Series.
        if len(series_names) == 1:
            name = series_names.pop()
        else:
            name = None
        return first_series(result_kdf).rename(name)
    else:
        return result_kdf


def melt(frame, id_vars=None, value_vars=None, var_name=None, value_name="value") -> DataFrame:
    return DataFrame.melt(frame, id_vars, value_vars, var_name, value_name)


melt.__doc__ = DataFrame.melt.__doc__


def isna(obj):
    """
    Detect missing values for an array-like object.

    This function takes a scalar or array-like object and indicates
    whether values are missing (``NaN`` in numeric arrays, ``None`` or ``NaN``
    in object arrays).

    Parameters
    ----------
    obj : scalar or array-like
        Object to check for null or missing values.

    Returns
    -------
    bool or array-like of bool
        For scalar input, returns a scalar boolean.
        For array input, returns an array of boolean indicating whether each
        corresponding element is missing.

    See Also
    --------
    Series.isna : Detect missing values in a Series.
    Series.isnull : Detect missing values in a Series.
    DataFrame.isna : Detect missing values in a DataFrame.
    DataFrame.isnull : Detect missing values in a DataFrame.
    Index.isna : Detect missing values in an Index.
    Index.isnull : Detect missing values in an Index.

    Examples
    --------
    Scalar arguments (including strings) result in a scalar boolean.

    >>> ks.isna('dog')
    False

    >>> ks.isna(np.nan)
    True

    ndarrays result in an ndarray of booleans.

    >>> array = np.array([[1, np.nan, 3], [4, 5, np.nan]])
    >>> array
    array([[ 1., nan,  3.],
           [ 4.,  5., nan]])
    >>> ks.isna(array)
    array([[False,  True, False],
           [False, False,  True]])

    For Series and DataFrame, the same type is returned, containing booleans.

    >>> df = ks.DataFrame({'a': ['ant', 'bee', 'cat'], 'b': ['dog', None, 'fly']})
    >>> df
         a     b
    0  ant   dog
    1  bee  None
    2  cat   fly

    >>> ks.isna(df)
           a      b
    0  False  False
    1  False   True
    2  False  False

    >>> ks.isnull(df.b)
    0    False
    1     True
    2    False
    Name: b, dtype: bool
    """
    # TODO: Add back:
    #     notnull : Boolean inverse of pandas.isnull.
    #   into the See Also in the docstring. It does not find the method in the latest numpydoc.
    if isinstance(obj, (DataFrame, Series)):
        return obj.isnull()
    else:
        return pd.isnull(obj)


isnull = isna


def notna(obj):
    """
    Detect existing (non-missing) values.

    Return a boolean same-sized object indicating if the values are not NA.
    Non-missing values get mapped to True. NA values, such as None or
    :attr:`numpy.NaN`, get mapped to False values.

    Returns
    -------
    bool or array-like of bool
        Mask of bool values for each element that
        indicates whether an element is not an NA value.

    See Also
    --------
    isna : Detect missing values for an array-like object.
    Series.notna : Boolean inverse of Series.isna.
    DataFrame.notnull : Boolean inverse of DataFrame.isnull.
    Index.notna : Boolean inverse of Index.isna.
    Index.notnull : Boolean inverse of Index.isnull.

    Examples
    --------
    Show which entries in a DataFrame are not NA.

    >>> df = ks.DataFrame({'age': [5, 6, np.NaN],
    ...                    'born': [pd.NaT, pd.Timestamp('1939-05-27'),
    ...                             pd.Timestamp('1940-04-25')],
    ...                    'name': ['Alfred', 'Batman', ''],
    ...                    'toy': [None, 'Batmobile', 'Joker']})
    >>> df
       age       born    name        toy
    0  5.0        NaT  Alfred       None
    1  6.0 1939-05-27  Batman  Batmobile
    2  NaN 1940-04-25              Joker

    >>> df.notnull()
         age   born  name    toy
    0   True  False  True  False
    1   True   True  True   True
    2  False   True  True   True

    Show which entries in a Series are not NA.

    >>> ser = ks.Series([5, 6, np.NaN])
    >>> ser
    0    5.0
    1    6.0
    2    NaN
    dtype: float64

    >>> ks.notna(ser)
    0     True
    1     True
    2    False
    dtype: bool

    >>> ks.notna(ser.index)
    True
    """
    # TODO: Add back:
    #     Series.notnull :Boolean inverse of Series.isnull.
    #     DataFrame.notna :Boolean inverse of DataFrame.isna.
    #   into the See Also in the docstring. It does not find the method in the latest numpydoc.
    if isinstance(obj, (DataFrame, Series)):
        return obj.notna()
    else:
        return pd.notna(obj)


notnull = notna


def merge(
    obj,
    right: "DataFrame",
    how: str = "inner",
    on: Union[Any, List[Any], Tuple, List[Tuple]] = None,
    left_on: Union[Any, List[Any], Tuple, List[Tuple]] = None,
    right_on: Union[Any, List[Any], Tuple, List[Tuple]] = None,
    left_index: bool = False,
    right_index: bool = False,
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> "DataFrame":
    """
    Merge DataFrame objects with a database-style join.

    The index of the resulting DataFrame will be one of the following:
        - 0...n if no index is used for merging
        - Index of the left DataFrame if merged only on the index of the right DataFrame
        - Index of the right DataFrame if merged only on the index of the left DataFrame
        - All involved indices if merged using the indices of both DataFrames
            e.g. if `left` with indices (a, x) and `right` with indices (b, x), the result will
            be an index (x, a, b)

    Parameters
    ----------
    right: Object to merge with.
    how: Type of merge to be performed.
        {'left', 'right', 'outer', 'inner'}, default 'inner'

        left: use only keys from left frame, similar to a SQL left outer join; preserve key
            order.
        right: use only keys from right frame, similar to a SQL right outer join; preserve key
            order.
        outer: use union of keys from both frames, similar to a SQL full outer join; sort keys
            lexicographically.
        inner: use intersection of keys from both frames, similar to a SQL inner join;
            preserve the order of the left keys.
    on: Column or index level names to join on. These must be found in both DataFrames. If on
        is None and not merging on indexes then this defaults to the intersection of the
        columns in both DataFrames.
    left_on: Column or index level names to join on in the left DataFrame. Can also
        be an array or list of arrays of the length of the left DataFrame.
        These arrays are treated as if they are columns.
    right_on: Column or index level names to join on in the right DataFrame. Can also
        be an array or list of arrays of the length of the right DataFrame.
        These arrays are treated as if they are columns.
    left_index: Use the index from the left DataFrame as the join key(s). If it is a
        MultiIndex, the number of keys in the other DataFrame (either the index or a number of
        columns) must match the number of levels.
    right_index: Use the index from the right DataFrame as the join key. Same caveats as
        left_index.
    suffixes: Suffix to apply to overlapping column names in the left and right side,
        respectively.

    Returns
    -------
    DataFrame
        A DataFrame of the two merged objects.

    Examples
    --------

    >>> df1 = ks.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
    ...                     'value': [1, 2, 3, 5]},
    ...                    columns=['lkey', 'value'])
    >>> df2 = ks.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
    ...                     'value': [5, 6, 7, 8]},
    ...                    columns=['rkey', 'value'])
    >>> df1
      lkey  value
    0  foo      1
    1  bar      2
    2  baz      3
    3  foo      5
    >>> df2
      rkey  value
    0  foo      5
    1  bar      6
    2  baz      7
    3  foo      8

    Merge df1 and df2 on the lkey and rkey columns. The value columns have
    the default suffixes, _x and _y, appended.

    >>> merged = ks.merge(df1, df2, left_on='lkey', right_on='rkey')
    >>> merged.sort_values(by=['lkey', 'value_x', 'rkey', 'value_y'])  # doctest: +ELLIPSIS
      lkey  value_x rkey  value_y
    ...bar        2  bar        6
    ...baz        3  baz        7
    ...foo        1  foo        5
    ...foo        1  foo        8
    ...foo        5  foo        5
    ...foo        5  foo        8

    >>> left_kdf = ks.DataFrame({'A': [1, 2]})
    >>> right_kdf = ks.DataFrame({'B': ['x', 'y']}, index=[1, 2])

    >>> ks.merge(left_kdf, right_kdf, left_index=True, right_index=True).sort_index()
       A  B
    1  2  x

    >>> ks.merge(left_kdf, right_kdf, left_index=True, right_index=True, how='left').sort_index()
       A     B
    0  1  None
    1  2     x

    >>> ks.merge(left_kdf, right_kdf, left_index=True, right_index=True, how='right').sort_index()
         A  B
    1  2.0  x
    2  NaN  y

    >>> ks.merge(left_kdf, right_kdf, left_index=True, right_index=True, how='outer').sort_index()
         A     B
    0  1.0  None
    1  2.0     x
    2  NaN     y

    Notes
    -----
    As described in #263, joining string columns currently returns None for missing values
        instead of NaN.
    """
    return obj.merge(
        right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        suffixes=suffixes,
    )


def to_numeric(arg):
    """
    Convert argument to a numeric type.

    Parameters
    ----------
    arg : scalar, list, tuple, 1-d array, or Series

    Returns
    -------
    ret : numeric if parsing succeeded.

    See Also
    --------
    DataFrame.astype : Cast argument to a specified dtype.
    to_datetime : Convert argument to datetime.
    to_timedelta : Convert argument to timedelta.
    numpy.ndarray.astype : Cast a numpy array to a specified type.

    Examples
    --------

    >>> kser = ks.Series(['1.0', '2', '-3'])
    >>> kser
    0    1.0
    1      2
    2     -3
    dtype: object

    >>> ks.to_numeric(kser)
    0    1.0
    1    2.0
    2   -3.0
    dtype: float32

    If given Series contains invalid value to cast float, just cast it to `np.nan`

    >>> kser = ks.Series(['apple', '1.0', '2', '-3'])
    >>> kser
    0    apple
    1      1.0
    2        2
    3       -3
    dtype: object

    >>> ks.to_numeric(kser)
    0    NaN
    1    1.0
    2    2.0
    3   -3.0
    dtype: float32

    Also support for list, tuple, np.array, or a scalar

    >>> ks.to_numeric(['1.0', '2', '-3'])
    array([ 1.,  2., -3.])

    >>> ks.to_numeric(('1.0', '2', '-3'))
    array([ 1.,  2., -3.])

    >>> ks.to_numeric(np.array(['1.0', '2', '-3']))
    array([ 1.,  2., -3.])

    >>> ks.to_numeric('1.0')
    1.0
    """
    if isinstance(arg, Series):
        return arg._with_new_scol(arg.spark.column.cast("float"))
    else:
        return pd.to_numeric(arg)


def broadcast(obj) -> DataFrame:
    """
    Marks a DataFrame as small enough for use in broadcast joins.

    Parameters
    ----------
    obj : DataFrame

    Returns
    -------
    ret : DataFrame with broadcast hint.

    See Also
    --------
    DataFrame.merge : Merge DataFrame objects with a database-style join.
    DataFrame.join : Join columns of another DataFrame.
    DataFrame.update : Modify in place using non-NA values from another DataFrame.
    DataFrame.hint : Specifies some hint on the current DataFrame.

    Examples
    --------
    >>> df1 = ks.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo'],
    ...                     'value': [1, 2, 3, 5]},
    ...                    columns=['lkey', 'value']).set_index('lkey')
    >>> df2 = ks.DataFrame({'rkey': ['foo', 'bar', 'baz', 'foo'],
    ...                     'value': [5, 6, 7, 8]},
    ...                    columns=['rkey', 'value']).set_index('rkey')
    >>> merged = df1.merge(ks.broadcast(df2), left_index=True, right_index=True)
    >>> merged.spark.explain()  # doctest: +ELLIPSIS
    == Physical Plan ==
    ...
    ...BroadcastHashJoin...
    ...
    """
    if not isinstance(obj, DataFrame):
        raise ValueError("Invalid type : expected DataFrame got {}".format(type(obj).__name__))
    return DataFrame(obj._internal.with_new_sdf(F.broadcast(obj._internal.spark_frame)))


def _get_index_map(
    sdf: spark.DataFrame, index_col: Optional[Union[str, List[str]]] = None
) -> Tuple[Optional[List[spark.Column]], Optional[List[Tuple]]]:
    if index_col is not None:
        if isinstance(index_col, str):
            index_col = [index_col]
        sdf_columns = set(sdf.columns)
        for col in index_col:
            if col not in sdf_columns:
                raise KeyError(col)
        index_spark_columns = [
            scol_for(sdf, col) for col in index_col
        ]  # type: Optional[List[spark.Column]]
        index_names = [(col,) for col in index_col]  # type: Optional[List[Tuple]]
    else:
        index_spark_columns = None
        index_names = None

    return index_spark_columns, index_names


_get_dummies_default_accept_types = (DecimalType, StringType, DateType)
_get_dummies_acceptable_types = _get_dummies_default_accept_types + (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    BooleanType,
    TimestampType,
)
