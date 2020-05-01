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
A base class to be monkey-patched to DataFrame/Column to behave similar to pandas DataFrame/Series.
"""
import warnings
from collections import Counter, OrderedDict
from collections.abc import Iterable
from distutils.version import LooseVersion
from functools import reduce
from typing import Optional, Union, List

import numpy as np
import pandas as pd

from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.readwriter import OptionUtils
from pyspark.sql.types import DataType, DoubleType, FloatType

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.indexing import AtIndexer, iAtIndexer, iLocIndexer, LocIndexer
from databricks.koalas.internal import _InternalFrame, NATURAL_ORDER_COLUMN_NAME
from databricks.koalas.utils import (
    validate_arguments_and_invoke_function,
    scol_for,
    validate_axis,
    align_diff_frames,
    name_like_string,
)
from databricks.koalas.window import Rolling, Expanding


class _Frame(object):
    """
    The base class for both DataFrame and Series.
    """

    def __init__(self, internal: _InternalFrame):
        self._internal = internal  # type: _InternalFrame

    # TODO: add 'axis' parameter
    def cummin(self, skipna: bool = True):
        """
        Return cumulative minimum over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative minimum.

        .. note:: the current implementation of cummin uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        Returns
        -------
        DataFrame or Series

        See Also
        --------
        DataFrame.min : Return the minimum over DataFrame axis.
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        Series.min : Return the minimum over Series axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Examples
        --------
        >>> df = ks.DataFrame([[2.0, 1.0], [3.0, None], [1.0, 0.0]], columns=list('AB'))
        >>> df
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  1.0  0.0

        By default, iterates over rows and finds the minimum in each column.

        >>> df.cummin()
             A    B
        0  2.0  1.0
        1  2.0  NaN
        2  1.0  0.0

        It works identically in Series.

        >>> df.A.cummin()
        0    2.0
        1    2.0
        2    1.0
        Name: A, dtype: float64
        """
        return self._apply_series_op(lambda kser: kser._cum(F.min, skipna))  # type: ignore

    # TODO: add 'axis' parameter
    def cummax(self, skipna: bool = True):
        """
        Return cumulative maximum over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative maximum.

        .. note:: the current implementation of cummax uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        Returns
        -------
        DataFrame or Series

        See Also
        --------
        DataFrame.max : Return the maximum over DataFrame axis.
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        DataFrame.cumprod : Return cumulative product over DataFrame axis.
        Series.max : Return the maximum over Series axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Examples
        --------
        >>> df = ks.DataFrame([[2.0, 1.0], [3.0, None], [1.0, 0.0]], columns=list('AB'))
        >>> df
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  1.0  0.0

        By default, iterates over rows and finds the maximum in each column.

        >>> df.cummax()
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  3.0  1.0

        It works identically in Series.

        >>> df.B.cummax()
        0    1.0
        1    NaN
        2    1.0
        Name: B, dtype: float64
        """
        return self._apply_series_op(lambda kser: kser._cum(F.max, skipna))  # type: ignore

    # TODO: add 'axis' parameter
    def cumsum(self, skipna: bool = True):
        """
        Return cumulative sum over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative sum.

        .. note:: the current implementation of cumsum uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        Returns
        -------
        DataFrame or Series

        See Also
        --------
        DataFrame.sum : Return the sum over DataFrame axis.
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        DataFrame.cumprod : Return cumulative product over DataFrame axis.
        Series.sum : Return the sum over Series axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Examples
        --------
        >>> df = ks.DataFrame([[2.0, 1.0], [3.0, None], [1.0, 0.0]], columns=list('AB'))
        >>> df
             A    B
        0  2.0  1.0
        1  3.0  NaN
        2  1.0  0.0

        By default, iterates over rows and finds the sum in each column.

        >>> df.cumsum()
             A    B
        0  2.0  1.0
        1  5.0  NaN
        2  6.0  1.0

        It works identically in Series.

        >>> df.A.cumsum()
        0    2.0
        1    5.0
        2    6.0
        Name: A, dtype: float64
        """
        return self._apply_series_op(lambda kser: kser._cum(F.sum, skipna))  # type: ignore

    # TODO: add 'axis' parameter
    # TODO: use pandas_udf to support negative values and other options later
    #  other window except unbounded ones is supported as of Spark 3.0.
    def cumprod(self, skipna: bool = True):
        """
        Return cumulative product over a DataFrame or Series axis.

        Returns a DataFrame or Series of the same size containing the cumulative product.

        .. note:: the current implementation of cumprod uses Spark's Window without
            specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        .. note:: unlike pandas', Koalas' emulates cumulative product by ``exp(sum(log(...)))``
            trick. Therefore, it only works for positive numbers.

        Parameters
        ----------
        skipna : boolean, default True
            Exclude NA/null values. If an entire row/column is NA, the result will be NA.

        Returns
        -------
        DataFrame or Series

        See Also
        --------
        DataFrame.cummax : Return cumulative maximum over DataFrame axis.
        DataFrame.cummin : Return cumulative minimum over DataFrame axis.
        DataFrame.cumsum : Return cumulative sum over DataFrame axis.
        DataFrame.cumprod : Return cumulative product over DataFrame axis.
        Series.cummax : Return cumulative maximum over Series axis.
        Series.cummin : Return cumulative minimum over Series axis.
        Series.cumsum : Return cumulative sum over Series axis.
        Series.cumprod : Return cumulative product over Series axis.

        Raises
        ------
        Exception : If the values is equal to or lower than 0.

        Examples
        --------
        >>> df = ks.DataFrame([[2.0, 1.0], [3.0, None], [4.0, 10.0]], columns=list('AB'))
        >>> df
             A     B
        0  2.0   1.0
        1  3.0   NaN
        2  4.0  10.0

        By default, iterates over rows and finds the sum in each column.

        >>> df.cumprod()
              A     B
        0   2.0   1.0
        1   6.0   NaN
        2  24.0  10.0

        It works identically in Series.

        >>> df.A.cumprod()
        0     2.0
        1     6.0
        2    24.0
        Name: A, dtype: float64

        """
        return self._apply_series_op(lambda kser: kser._cumprod(skipna))  # type: ignore

    # TODO: Although this has removed pandas >= 1.0.0, but we're keeping this as deprecated
    # since we're using this for `DataFrame.info` internally.
    def get_dtype_counts(self):
        """
        Return counts of unique dtypes in this object.

        .. deprecated:: 0.14.0

        Returns
        -------
        dtype : pd.Series
            Series with the count of columns with each dtype.

        See Also
        --------
        dtypes : Return the dtypes in this object.

        Examples
        --------
        >>> a = [['a', 1, 1], ['b', 2, 2], ['c', 3, 3]]
        >>> df = ks.DataFrame(a, columns=['str', 'int1', 'int2'])
        >>> df
          str  int1  int2
        0   a     1     1
        1   b     2     2
        2   c     3     3

        >>> df.get_dtype_counts().sort_values()
        object    1
        int64     2
        dtype: int64

        >>> df.str.get_dtype_counts().sort_values()
        object    1
        dtype: int64
        """
        warnings.warn(
            "`get_dtype_counts` has been deprecated and will be "
            "removed in a future version. For DataFrames use "
            "`.dtypes.value_counts()",
            FutureWarning,
        )
        if not isinstance(self.dtypes, Iterable):
            dtypes = [self.dtypes]
        else:
            dtypes = self.dtypes
        return pd.Series(dict(Counter([d.name for d in list(dtypes)])))

    def pipe(self, func, *args, **kwargs):
        r"""
        Apply func(self, \*args, \*\*kwargs).

        Parameters
        ----------
        func : function
            function to apply to the DataFrame.
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the DataFrames.
        args : iterable, optional
            positional arguments passed into ``func``.
        kwargs : mapping, optional
            a dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        Notes
        -----
        Use ``.pipe`` when chaining together functions that expect
        Series, DataFrames or GroupBy objects. For example, given

        >>> df = ks.DataFrame({'category': ['A', 'A', 'B'],
        ...                    'col1': [1, 2, 3],
        ...                    'col2': [4, 5, 6]},
        ...                   columns=['category', 'col1', 'col2'])
        >>> def keep_category_a(df):
        ...    return df[df['category'] == 'A']
        >>> def add_one(df, column):
        ...    return df.assign(col3=df[column] + 1)
        >>> def multiply(df, column1, column2):
        ...    return df.assign(col4=df[column1] * df[column2])


        instead of writing

        >>> multiply(add_one(keep_category_a(df), column="col1"), column1="col2", column2="col3")
          category  col1  col2  col3  col4
        0        A     1     4     2     8
        1        A     2     5     3    15


        You can write

        >>> (df.pipe(keep_category_a)
        ...    .pipe(add_one, column="col1")
        ...    .pipe(multiply, column1="col2", column2="col3")
        ... )
          category  col1  col2  col3  col4
        0        A     1     4     2     8
        1        A     2     5     3    15


        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``f`` takes its data as ``df``:

        >>> def multiply_2(column1, df, column2):
        ...     return df.assign(col4=df[column1] * df[column2])


        Then you can write

        >>> (df.pipe(keep_category_a)
        ...    .pipe(add_one, column="col1")
        ...    .pipe((multiply_2, 'df'), column1="col2", column2="col3")
        ... )
          category  col1  col2  col3  col4
        0        A     1     4     2     8
        1        A     2     5     3    15

        You can use lambda as wel

        >>> ks.Series([1, 2, 3]).pipe(lambda x: (x + 1).rename("value"))
        0    2
        1    3
        2    4
        Name: value, dtype: int64
        """

        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                raise ValueError("%s is both the pipe target and a keyword " "argument" % target)
            kwargs[target] = self
            return func(*args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    def to_numpy(self):
        """
        A NumPy ndarray representing the values in this DataFrame or Series.

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> ks.DataFrame({"A": [1, 2], "B": [3, 4]}).to_numpy()
        array([[1, 3],
               [2, 4]])

        With heterogeneous data, the lowest common type will have to be used.

        >>> ks.DataFrame({"A": [1, 2], "B": [3.0, 4.5]}).to_numpy()
        array([[1. , 3. ],
               [2. , 4.5]])

        For a mix of numeric and non-numeric types, the output array will have object dtype.

        >>> df = ks.DataFrame({"A": [1, 2], "B": [3.0, 4.5], "C": pd.date_range('2000', periods=2)})
        >>> df.to_numpy()
        array([[1, 3.0, Timestamp('2000-01-01 00:00:00')],
               [2, 4.5, Timestamp('2000-01-02 00:00:00')]], dtype=object)

        For Series,

        >>> ks.Series(['a', 'b', 'a']).to_numpy()
        array(['a', 'b', 'a'], dtype=object)
        """
        return self.to_pandas().values

    @property
    def values(self):
        """
        Return a Numpy representation of the DataFrame or the Series.

        .. warning:: We recommend using `DataFrame.to_numpy()` or `Series.to_numpy()` instead.

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        A DataFrame where all columns are the same type (e.g., int64) results in an array of
        the same type.

        >>> df = ks.DataFrame({'age':    [ 3,  29],
        ...                    'height': [94, 170],
        ...                    'weight': [31, 115]})
        >>> df
           age  height  weight
        0    3      94      31
        1   29     170     115
        >>> df.dtypes
        age       int64
        height    int64
        weight    int64
        dtype: object
        >>> df.values
        array([[  3,  94,  31],
               [ 29, 170, 115]])

        A DataFrame with mixed type columns(e.g., str/object, int64, float32) results in an ndarray
        of the broadest type that accommodates these mixed types (e.g., object).

        >>> df2 = ks.DataFrame([('parrot',   24.0, 'second'),
        ...                     ('lion',     80.5, 'first'),
        ...                     ('monkey', np.nan, None)],
        ...                   columns=('name', 'max_speed', 'rank'))
        >>> df2.dtypes
        name          object
        max_speed    float64
        rank          object
        dtype: object
        >>> df2.values
        array([['parrot', 24.0, 'second'],
               ['lion', 80.5, 'first'],
               ['monkey', nan, None]], dtype=object)

        For Series,

        >>> ks.Series([1, 2, 3]).values
        array([1, 2, 3])

        >>> ks.Series(list('aabc')).values
        array(['a', 'a', 'b', 'c'], dtype=object)
        """
        warnings.warn("We recommend using `{}.to_numpy()` instead.".format(type(self).__name__))
        return self.to_numpy()

    def to_csv(
        self,
        path=None,
        sep=",",
        na_rep="",
        columns=None,
        header=True,
        quotechar='"',
        date_format=None,
        escapechar=None,
        num_files=None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options
    ):
        r"""
        Write object to a comma-separated values (csv) file.

        .. note:: Koalas `to_csv` writes files to a path or URI. Unlike pandas', Koalas
            respects HDFS's property such as 'fs.default.name'.

        .. note:: Koalas writes CSV files into the directory, `path`, and writes
            multiple `part-...` files in the directory when `path` is specified.
            This behaviour was inherited from Apache Spark. The number of files can
            be controlled by `num_files`.

        Parameters
        ----------
        path : str, default None
            File path. If None is provided the result is returned as a string.
        sep : str, default ','
            String of length 1. Field delimiter for the output file.
        na_rep : str, default ''
            Missing data representation.
        columns : sequence, optional
            Columns to write.
        header : bool or list of str, default True
            Write out the column names. If a list of strings is given it is
            assumed to be aliases for the column names.
        quotechar : str, default '\"'
            String of length 1. Character used to quote fields.
        date_format : str, default None
            Format string for datetime objects.
        escapechar : str, default None
            String of length 1. Character used to escape `sep` and `quotechar`
            when appropriate.
        num_files : the number of files to be written in `path` directory when
            this is a path.
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        options: keyword arguments for additional options specific to PySpark.
            This kwargs are specific to PySpark's CSV options to pass. Check
            the options in PySpark's API documentation for spark.write.csv(...).
            It has higher priority and overwrites all other options.
            This parameter only works when `path` is specified.

        See Also
        --------
        read_csv
        DataFrame.to_delta
        DataFrame.to_table
        DataFrame.to_parquet
        DataFrame.to_spark_io

        Examples
        --------
        >>> df = ks.DataFrame(dict(
        ...    date=list(pd.date_range('2012-1-1 12:00:00', periods=3, freq='M')),
        ...    country=['KR', 'US', 'JP'],
        ...    code=[1, 2 ,3]), columns=['date', 'country', 'code'])
        >>> df.sort_values(by="date")  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                           date country  code
        ... 2012-01-31 12:00:00      KR     1
        ... 2012-02-29 12:00:00      US     2
        ... 2012-03-31 12:00:00      JP     3

        >>> print(df.to_csv())  # doctest: +NORMALIZE_WHITESPACE
        date,country,code
        2012-01-31 12:00:00,KR,1
        2012-02-29 12:00:00,US,2
        2012-03-31 12:00:00,JP,3

        >>> df.to_csv(path=r'%s/to_csv/foo.csv' % path, num_files=1)
        >>> ks.read_csv(
        ...    path=r'%s/to_csv/foo.csv' % path
        ... ).sort_values(by="date")  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                           date country  code
        ... 2012-01-31 12:00:00      KR     1
        ... 2012-02-29 12:00:00      US     2
        ... 2012-03-31 12:00:00      JP     3

        In case of Series,

        >>> print(df.date.to_csv())  # doctest: +NORMALIZE_WHITESPACE
        date
        2012-01-31 12:00:00
        2012-02-29 12:00:00
        2012-03-31 12:00:00

        >>> df.date.to_csv(path=r'%s/to_csv/foo.csv' % path, num_files=1)
        >>> ks.read_csv(
        ...     path=r'%s/to_csv/foo.csv' % path
        ... ).sort_values(by="date")  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                           date
        ... 2012-01-31 12:00:00
        ... 2012-02-29 12:00:00
        ... 2012-03-31 12:00:00

        You can preserve the index in the roundtrip as below.

        >>> df.set_index("country", append=True, inplace=True)
        >>> df.date.to_csv(
        ...     path=r'%s/to_csv/bar.csv' % path,
        ...     num_files=1,
        ...     index_col=["index1", "index2"])
        >>> ks.read_csv(
        ...     path=r'%s/to_csv/bar.csv' % path, index_col=["index1", "index2"]
        ... ).sort_values(by="date")  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                                     date
        index1 index2
        ...    ...    2012-01-31 12:00:00
        ...    ...    2012-02-29 12:00:00
        ...    ...    2012-03-31 12:00:00
        """
        if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
            options = options.get("options")  # type: ignore

        if path is None:
            # If path is none, just collect and use pandas's to_csv.
            kdf_or_ser = self
            if (LooseVersion("0.24") > LooseVersion(pd.__version__)) and isinstance(
                self, ks.Series
            ):
                # 0.23 seems not having 'columns' parameter in Series' to_csv.
                return kdf_or_ser.to_pandas().to_csv(  # type: ignore
                    None,
                    sep=sep,
                    na_rep=na_rep,
                    header=header,
                    date_format=date_format,
                    index=False,
                )
            else:
                return kdf_or_ser.to_pandas().to_csv(  # type: ignore
                    None,
                    sep=sep,
                    na_rep=na_rep,
                    columns=columns,
                    header=header,
                    quotechar=quotechar,
                    date_format=date_format,
                    escapechar=escapechar,
                    index=False,
                )

        kdf = self
        if isinstance(self, ks.Series):
            kdf = self.to_frame()

        if columns is None:
            column_labels = kdf._internal.column_labels
        elif isinstance(columns, str):
            column_labels = [(columns,)]
        elif isinstance(columns, tuple):
            column_labels = [columns]
        else:
            column_labels = [
                lb if isinstance(lb, tuple) else (lb,) for lb in columns  # type: ignore
            ]

        if isinstance(index_col, str):
            index_cols = [index_col]
        elif index_col is None:
            index_cols = []
        else:
            index_cols = index_col

        if header is True and kdf._internal.column_labels_level > 1:
            raise ValueError("to_csv only support one-level index column now")
        elif isinstance(header, list):
            sdf = kdf.to_spark(index_col)  # type: ignore
            sdf = sdf.select(
                [scol_for(sdf, name_like_string(label)) for label in index_cols]
                + [
                    self._internal.spark_column_for(label).alias(new_name)
                    for (label, new_name) in zip(column_labels, header)
                ]
            )
            header = True
        else:
            sdf = kdf.to_spark(index_col)  # type: ignore
            sdf = sdf.select(
                [scol_for(sdf, name_like_string(label)) for label in index_cols]
                + [kdf._internal.spark_column_for(label) for label in column_labels]
            )

        if num_files is not None:
            sdf = sdf.repartition(num_files)

        builder = sdf.write.mode("overwrite")
        OptionUtils._set_opts(
            builder,
            path=path,
            sep=sep,
            nullValue=na_rep,
            header=header,
            quote=quotechar,
            dateFormat=date_format,
            charToEscapeQuoteEscaping=escapechar,
        )
        builder.options(**options).format("csv").save(path)

    def to_json(
        self,
        path=None,
        compression="uncompressed",
        num_files=None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options
    ):
        """
        Convert the object to a JSON string.

        .. note:: Koalas `to_json` writes files to a path or URI. Unlike pandas', Koalas
            respects HDFS's property such as 'fs.default.name'.

        .. note:: Koalas writes JSON files into the directory, `path`, and writes
            multiple `part-...` files in the directory when `path` is specified.
            This behaviour was inherited from Apache Spark. The number of files can
            be controlled by `num_files`.

        .. note:: output JSON format is different from pandas'. It always use `orient='records'`
            for its output. This behaviour might have to change in the near future.

        Note NaN's and None will be converted to null and datetime objects
        will be converted to UNIX timestamps.

        Parameters
        ----------
        path : string, optional
            File path. If not specified, the result is returned as
            a string.
        compression : {'gzip', 'bz2', 'xz', None}
            A string representing the compression to use in the output file,
            only used when the first argument is a filename. By default, the
            compression is inferred from the filename.
        num_files : the number of files to be written in `path` directory when
            this is a path.
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        options: keyword arguments for additional options specific to PySpark.
            It is specific to PySpark's JSON options to pass. Check
            the options in PySpark's API documentation for `spark.write.json(...)`.
            It has a higher priority and overwrites all other options.
            This parameter only works when `path` is specified.

        Examples
        --------
        >>> df = ks.DataFrame([['a', 'b'], ['c', 'd']],
        ...                   columns=['col 1', 'col 2'])
        >>> df.to_json()
        '[{"col 1":"a","col 2":"b"},{"col 1":"c","col 2":"d"}]'

        >>> df['col 1'].to_json()
        '[{"col 1":"a"},{"col 1":"c"}]'

        >>> df.to_json(path=r'%s/to_json/foo.json' % path, num_files=1)
        >>> ks.read_json(
        ...     path=r'%s/to_json/foo.json' % path
        ... ).sort_values(by="col 1")
          col 1 col 2
        0     a     b
        1     c     d

        >>> df['col 1'].to_json(path=r'%s/to_json/foo.json' % path, num_files=1, index_col="index")
        >>> ks.read_json(
        ...     path=r'%s/to_json/foo.json' % path, index_col="index"
        ... ).sort_values(by="col 1")  # doctest: +NORMALIZE_WHITESPACE
              col 1
        index
        0         a
        1         c
        """
        if "options" in options and isinstance(options.get("options"), dict) and len(options) == 1:
            options = options.get("options")  # type: ignore

        if path is None:
            # If path is none, just collect and use pandas's to_json.
            kdf_or_ser = self
            pdf = kdf_or_ser.to_pandas()  # type: ignore
            if isinstance(self, ks.Series):
                pdf = pdf.to_frame()
            # To make the format consistent and readable by `read_json`, convert it to pandas' and
            # use 'records' orient for now.
            return pdf.to_json(orient="records")

        kdf = self
        if isinstance(self, ks.Series):
            kdf = self.to_frame()
        sdf = kdf.to_spark(index_col=index_col)  # type: ignore

        if num_files is not None:
            sdf = sdf.repartition(num_files)

        builder = sdf.write.mode("overwrite")
        OptionUtils._set_opts(builder, compression=compression)
        builder.options(**options).format("json").save(path)

    def to_excel(
        self,
        excel_writer,
        sheet_name="Sheet1",
        na_rep="",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        startrow=0,
        startcol=0,
        engine=None,
        merge_cells=True,
        encoding=None,
        inf_rep="inf",
        verbose=True,
        freeze_panes=None,
    ):
        """
        Write object to an Excel sheet.

        .. note:: This method should only be used if the resulting DataFrame is expected
                  to be small, as all the data is loaded into the driver's memory.

        To write a single object to an Excel .xlsx file it is only necessary to
        specify a target file name. To write to multiple sheets it is necessary to
        create an `ExcelWriter` object with a target file name, and specify a sheet
        in the file to write to.

        Multiple sheets may be written to by specifying unique `sheet_name`.
        With all data written to the file it is necessary to save the changes.
        Note that creating an `ExcelWriter` object with a file name that already
        exists will result in the contents of the existing file being erased.

        Parameters
        ----------
        excel_writer : str or ExcelWriter object
            File path or existing ExcelWriter.
        sheet_name : str, default 'Sheet1'
            Name of sheet which will contain DataFrame.
        na_rep : str, default ''
            Missing data representation.
        float_format : str, optional
            Format string for floating point numbers. For example
            ``float_format="%%.2f"`` will format 0.1234 to 0.12.
        columns : sequence or list of str, optional
            Columns to write.
        header : bool or list of str, default True
            Write out the column names. If a list of string is given it is
            assumed to be aliases for the column names.
        index : bool, default True
            Write row names (index).
        index_label : str or sequence, optional
            Column label for index column(s) if desired. If not specified, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the DataFrame uses MultiIndex.
        startrow : int, default 0
            Upper left cell row to dump data frame.
        startcol : int, default 0
            Upper left cell column to dump data frame.
        engine : str, optional
            Write engine to use, 'openpyxl' or 'xlsxwriter'. You can also set this
            via the options ``io.excel.xlsx.writer``, ``io.excel.xls.writer``, and
            ``io.excel.xlsm.writer``.
        merge_cells : bool, default True
            Write MultiIndex and Hierarchical Rows as merged cells.
        encoding : str, optional
            Encoding of the resulting excel file. Only necessary for xlwt,
            other writers support unicode natively.
        inf_rep : str, default 'inf'
            Representation for infinity (there is no native representation for
            infinity in Excel).
        verbose : bool, default True
            Display more information in the error logs.
        freeze_panes : tuple of int (length 2), optional
            Specifies the one-based bottommost row and rightmost column that
            is to be frozen.

        Notes
        -----
        Once a workbook has been saved it is not possible write further data
        without rewriting the whole workbook.

        See Also
        --------
        read_excel : Read Excel file.

        Examples
        --------
        Create, write to and save a workbook:

        >>> df1 = ks.DataFrame([['a', 'b'], ['c', 'd']],
        ...                    index=['row 1', 'row 2'],
        ...                    columns=['col 1', 'col 2'])
        >>> df1.to_excel("output.xlsx")  # doctest: +SKIP

        To specify the sheet name:

        >>> df1.to_excel("output.xlsx")  # doctest: +SKIP
        >>> df1.to_excel("output.xlsx",
        ...              sheet_name='Sheet_name_1')  # doctest: +SKIP

        If you wish to write to more than one sheet in the workbook, it is
        necessary to specify an ExcelWriter object:

        >>> with pd.ExcelWriter('output.xlsx') as writer:  # doctest: +SKIP
        ...      df1.to_excel(writer, sheet_name='Sheet_name_1')
        ...      df2.to_excel(writer, sheet_name='Sheet_name_2')

        To set the library that is used to write the Excel file,
        you can pass the `engine` keyword (the default engine is
        automatically chosen depending on the file extension):

        >>> df1.to_excel('output1.xlsx', engine='xlsxwriter')  # doctest: +SKIP
        """
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        kdf = self

        if isinstance(self, ks.DataFrame):
            f = pd.DataFrame.to_excel
        elif isinstance(self, ks.Series):
            f = pd.Series.to_excel
        else:
            raise TypeError(
                "Constructor expects DataFrame or Series; however, " "got [%s]" % (self,)
            )
        return validate_arguments_and_invoke_function(
            kdf._to_internal_pandas(), self.to_excel, f, args
        )

    def mean(self, axis=None, numeric_only=True):
        """
        Return the mean of the values.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : bool, default True
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        mean : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.mean()
        a    2.0
        b    0.2
        Name: 0, dtype: float64

        >>> df.mean(axis=1)
        0    0.55
        1    1.10
        2    1.65
        3     NaN
        Name: 0, dtype: float64

        On a Series:

        >>> df['a'].mean()
        2.0
        """
        return self._reduce_for_stat_function(
            F.mean, name="mean", numeric_only=numeric_only, axis=axis
        )

    def sum(self, axis=None, numeric_only=True):
        """
        Return the sum of the values.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : bool, default True
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        sum : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.sum()
        a    6.0
        b    0.6
        Name: 0, dtype: float64

        >>> df.sum(axis=1)
        0    1.1
        1    2.2
        2    3.3
        3    0.0
        Name: 0, dtype: float64

        On a Series:

        >>> df['a'].sum()
        6.0
        """
        return self._reduce_for_stat_function(
            F.sum, name="sum", numeric_only=numeric_only, axis=axis
        )

    def skew(self, axis=None, numeric_only=True):
        """
        Return unbiased skew normalized by N-1.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : bool, default True
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        skew : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.skew()  # doctest: +SKIP
        a    0.000000e+00
        b   -3.319678e-16
        dtype: float64

        On a Series:

        >>> df['a'].skew()
        0.0
        """
        return self._reduce_for_stat_function(
            F.skewness, name="skew", numeric_only=numeric_only, axis=axis
        )

    def kurtosis(self, axis=None, numeric_only=True):
        """
        Return unbiased kurtosis using Fisher’s definition of kurtosis (kurtosis of normal == 0.0).
        Normalized by N-1.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : bool, default True
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        kurt : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.kurtosis()
        a   -1.5
        b   -1.5
        Name: 0, dtype: float64

        On a Series:

        >>> df['a'].kurtosis()
        -1.5
        """
        return self._reduce_for_stat_function(
            F.kurtosis, name="kurtosis", numeric_only=numeric_only, axis=axis
        )

    kurt = kurtosis

    def min(self, axis=None, numeric_only=None):
        """
        Return the minimum of the values.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : bool, default None
            If True, include only float, int, boolean columns. This parameter is mainly for
            pandas compatibility. False is supported; however, the columns should
            be all numeric or all non-numeric.

        Returns
        -------
        min : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.min()
        a    1.0
        b    0.1
        Name: 0, dtype: float64

        >>> df.min(axis=1)
        0    0.1
        1    0.2
        2    0.3
        3    NaN
        Name: 0, dtype: float64

        On a Series:

        >>> df['a'].min()
        1.0
        """
        return self._reduce_for_stat_function(
            F.min, name="min", numeric_only=numeric_only, axis=axis
        )

    def max(self, axis=None, numeric_only=None):
        """
        Return the maximum of the values.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : bool, default None
            If True, include only float, int, boolean columns. This parameter is mainly for
            pandas compatibility. False is supported; however, the columns should
            be all numeric or all non-numeric.

        Returns
        -------
        max : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.max()
        a    3.0
        b    0.3
        Name: 0, dtype: float64

        >>> df.max(axis=1)
        0    1.0
        1    2.0
        2    3.0
        3    NaN
        Name: 0, dtype: float64

        On a Series:

        >>> df['a'].max()
        3.0
        """
        return self._reduce_for_stat_function(
            F.max, name="max", numeric_only=numeric_only, axis=axis
        )

    def std(self, axis=None, numeric_only=True):
        """
        Return sample standard deviation.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : bool, default True
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        std : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.std()
        a    1.0
        b    0.1
        Name: 0, dtype: float64

        >>> df.std(axis=1)
        0    0.636396
        1    1.272792
        2    1.909188
        3         NaN
        Name: 0, dtype: float64

        On a Series:

        >>> df['a'].std()
        1.0
        """
        return self._reduce_for_stat_function(
            F.stddev, name="std", numeric_only=numeric_only, axis=axis
        )

    def var(self, axis=None, numeric_only=True):
        """
        Return unbiased variance.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : bool, default True
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.

        Returns
        -------
        var : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.var()
        a    1.00
        b    0.01
        Name: 0, dtype: float64

        >>> df.var(axis=1)
        0    0.405
        1    1.620
        2    3.645
        3      NaN
        Name: 0, dtype: float64

        On a Series:

        >>> df['a'].var()
        1.0
        """
        return self._reduce_for_stat_function(
            F.variance, name="var", numeric_only=numeric_only, axis=axis
        )

    @property
    def size(self) -> int:
        """
        Return an int representing the number of elements in this object.

        Return the number of rows if Series. Otherwise return the number of
        rows times number of columns if DataFrame.

        Examples
        --------
        >>> s = ks.Series({'a': 1, 'b': 2, 'c': None})
        >>> s.size
        3

        >>> df = ks.DataFrame({'col1': [1, 2, None], 'col2': [3, 4, None]})
        >>> df.size
        3
        """
        return len(self)  # type: ignore

    def abs(self):
        """
        Return a Series/DataFrame with absolute numeric value of each element.

        Returns
        -------
        abs : Series/DataFrame containing the absolute value of each element.

        Examples
        --------

        Absolute numeric values in a Series.

        >>> s = ks.Series([-1.10, 2, -3.33, 4])
        >>> s.abs()
        0    1.10
        1    2.00
        2    3.33
        3    4.00
        Name: 0, dtype: float64

        Absolute numeric values in a DataFrame.

        >>> df = ks.DataFrame({
        ...     'a': [4, 5, 6, 7],
        ...     'b': [10, 20, 30, 40],
        ...     'c': [100, 50, -30, -50]
        ...   },
        ...   columns=['a', 'b', 'c'])
        >>> df.abs()
           a   b    c
        0  4  10  100
        1  5  20   50
        2  6  30   30
        3  7  40   50
        """
        # TODO: The first example above should not have "Name: 0".
        return self._apply_series_op(
            lambda kser: kser._with_new_scol(F.abs(kser.spark_column)).rename(kser.name)
        )

    # TODO: by argument only support the grouping name and as_index only for now. Documentation
    # should be updated when it's supported.
    def groupby(self, by, axis=0, as_index: bool = True):
        """
        Group DataFrame or Series using a Series of columns.

        A groupby operation involves some combination of splitting the
        object, applying a function, and combining the results. This can be
        used to group large amounts of data and compute operations on these
        groups.

        Parameters
        ----------
        by : Series, label, or list of labels
            Used to determine the groups for the groupby.
            If Series is passed, the Series or dict VALUES
            will be used to determine the groups. A label or list of
            labels may be passed to group by the columns in ``self``.
        axis : int, default 0 or 'index'
            Can only be set to 0 at the moment.
        as_index : bool, default True
            For aggregated output, return object with group labels as the
            index. Only relevant for DataFrame input. as_index=False is
            effectively "SQL-style" grouped output.

        Returns
        -------
        DataFrameGroupBy or SeriesGroupBy
            Depends on the calling object and returns groupby object that
            contains information about the groups.

        See Also
        --------
        koalas.groupby.GroupBy

        Examples
        --------
        >>> df = ks.DataFrame({'Animal': ['Falcon', 'Falcon',
        ...                               'Parrot', 'Parrot'],
        ...                    'Max Speed': [380., 370., 24., 26.]},
        ...                   columns=['Animal', 'Max Speed'])
        >>> df
           Animal  Max Speed
        0  Falcon      380.0
        1  Falcon      370.0
        2  Parrot       24.0
        3  Parrot       26.0

        >>> df.groupby(['Animal']).mean().sort_index()  # doctest: +NORMALIZE_WHITESPACE
                Max Speed
        Animal
        Falcon      375.0
        Parrot       25.0

        >>> df.groupby(['Animal'], as_index=False).mean().sort_values('Animal')
        ... # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
           Animal  Max Speed
        ...Falcon      375.0
        ...Parrot       25.0
        """
        from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy

        df_or_s = self
        if isinstance(by, ks.DataFrame):
            raise ValueError("Grouper for '{}' not 1-dimensional".format(type(by)))
        elif isinstance(by, str):
            if isinstance(df_or_s, ks.Series):
                raise KeyError(by)
            by = [(by,)]
        elif isinstance(by, tuple):
            if isinstance(df_or_s, ks.Series):
                for key in by:
                    if isinstance(key, str):
                        raise KeyError(key)
            for key in by:
                if isinstance(key, ks.DataFrame):
                    raise ValueError("Grouper for '{}' not 1-dimensional".format(type(key)))
            by = [by]
        elif isinstance(by, ks.Series):
            by = [by]
        elif isinstance(by, Iterable):
            if isinstance(df_or_s, ks.Series):
                for key in by:
                    if isinstance(key, str):
                        raise KeyError(key)
            by = [key if isinstance(key, (tuple, ks.Series)) else (key,) for key in by]
        else:
            raise ValueError("Grouper for '{}' not 1-dimensional".format(type(by)))
        if not len(by):
            raise ValueError("No group keys passed!")
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')

        if isinstance(df_or_s, ks.DataFrame):
            df = df_or_s  # type: ks.DataFrame

            # This is to match the output with pandas'. Pandas seems returning different results
            # when given series is from different dataframes. It only applies when as_index is
            # False.
            should_drop_index = any(
                isinstance(col_or_s, ks.Series) and df is not col_or_s._kdf for col_or_s in by
            )

            kdf, col_by = self._resolve_grouping_series(by)
            return DataFrameGroupBy(
                kdf, col_by, as_index=as_index, should_drop_index=should_drop_index
            )
        if isinstance(df_or_s, ks.Series):
            col = df_or_s  # type: ks.Series
            _, col_by = self._resolve_grouping_series(by)
            return SeriesGroupBy(col, col_by, as_index=as_index)
        raise TypeError(
            "Constructor expects DataFrame or Series; however, " "got [%s]" % (df_or_s,)
        )

    def _resolve_grouping_series(self, by):
        should_use_name = False
        if isinstance(self, ks.Series):
            kdf = self._kdf
        else:
            kdf = self
        for col_or_s in by:
            if isinstance(col_or_s, ks.Series) and kdf is not col_or_s._kdf:

                def assign_columns(kdf, this_column_labels, that_column_labels):
                    raise NotImplementedError(
                        "Duplicated labels with groupby() and "
                        "'compute.ops_on_diff_frames' option are not supported currently "
                        "Please use unique labels in series and frames."
                    )

                kdf = align_diff_frames(assign_columns, kdf, col_or_s, fillna=False, how="inner")
                # Should use name to search series because now the anchor is different
                should_use_name = True

        new_by_series = []
        for col_or_s in by:
            if isinstance(col_or_s, ks.Series) and should_use_name:
                new_by_series.append(kdf[col_or_s.name])
            elif isinstance(col_or_s, ks.Series):
                new_by_series.append(col_or_s)
            elif isinstance(col_or_s, tuple):
                new_by_series.append(kdf[col_or_s])
            else:
                raise ValueError(col_or_s)

        return kdf, new_by_series

    def bool(self):
        """
        Return the bool of a single element in the current object.

        This must be a boolean scalar value, either True or False. Raise a ValueError if
        the object does not have exactly 1 element, or that element is not boolean

        Examples
        --------
        >>> ks.DataFrame({'a': [True]}).bool()
        True

        >>> ks.Series([False]).bool()
        False

        If there are non-boolean or multiple values exist, it raises an exception in all
        cases as below.

        >>> ks.DataFrame({'a': ['a']}).bool()
        Traceback (most recent call last):
          ...
        ValueError: bool cannot act on a non-boolean single element DataFrame

        >>> ks.DataFrame({'a': [True], 'b': [False]}).bool()  # doctest: +NORMALIZE_WHITESPACE
        Traceback (most recent call last):
          ...
        ValueError: The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(),
        a.item(), a.any() or a.all().

        >>> ks.Series([1]).bool()
        Traceback (most recent call last):
          ...
        ValueError: bool cannot act on a non-boolean single element DataFrame
        """
        if isinstance(self, ks.DataFrame):
            df = self
        elif isinstance(self, ks.Series):
            df = self.to_dataframe()
        else:
            raise TypeError("bool() expects DataFrame or Series; however, " "got [%s]" % (self,))
        return df.head(2)._to_internal_pandas().bool()

    def first_valid_index(self):
        """
        Retrieves the index of the first valid value.

        Returns
        -------
        idx_first_valid : type of index

        Examples
        --------

        Support for DataFrame

        >>> kdf = ks.DataFrame({'a': [None, 2, 3, 2],
        ...                     'b': [None, 2.0, 3.0, 1.0],
        ...                     'c': [None, 200, 400, 200]},
        ...                     index=['Q', 'W', 'E', 'R'])
        >>> kdf
             a    b      c
        Q  NaN  NaN    NaN
        W  2.0  2.0  200.0
        E  3.0  3.0  400.0
        R  2.0  1.0  200.0

        >>> kdf.first_valid_index()
        'W'

        Support for MultiIndex columns

        >>> kdf.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> kdf
             a    b      c
             x    y      z
        Q  NaN  NaN    NaN
        W  2.0  2.0  200.0
        E  3.0  3.0  400.0
        R  2.0  1.0  200.0

        >>> kdf.first_valid_index()
        'W'

        Support for Series.

        >>> s = ks.Series([None, None, 3, 4, 5], index=[100, 200, 300, 400, 500])
        >>> s
        100    NaN
        200    NaN
        300    3.0
        400    4.0
        500    5.0
        Name: 0, dtype: float64

        >>> s.first_valid_index()
        300

        Support for MultiIndex

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = ks.Series([None, None, None, None, 250, 1.5, 320, 1, 0.3], index=midx)
        >>> s
        lama    speed       NaN
                weight      NaN
                length      NaN
        cow     speed       NaN
                weight    250.0
                length      1.5
        falcon  speed     320.0
                weight      1.0
                length      0.3
        Name: 0, dtype: float64

        >>> s.first_valid_index()
        ('cow', 'weight')
        """
        sdf = self._internal.spark_frame
        data_spark_columns = self._internal.data_spark_columns
        cond = reduce(lambda x, y: x & y, map(lambda x: x.isNotNull(), data_spark_columns))

        first_valid_row = sdf.drop(NATURAL_ORDER_COLUMN_NAME).filter(cond).first()
        first_valid_idx = tuple(
            first_valid_row[idx_col] for idx_col in self._internal.index_spark_column_names
        )

        if len(first_valid_idx) == 1:
            first_valid_idx = first_valid_idx[0]

        return first_valid_idx

    def median(self, accuracy=10000):
        """
        Return the median of the values for the requested axis.

        .. note:: Unlike pandas', the median in Koalas is an approximated median based upon
            approximate percentile computation because computing median across a large dataset
            is extremely expensive.

        Parameters
        ----------
        accuracy : int, optional
            Default accuracy of approximation. Larger value means better accuracy.
            The relative error can be deduced by 1.0 / accuracy.

        Returns
        -------
        median : scalar or Series

        Examples
        --------
        >>> df = ks.DataFrame({
        ...     'a': [24., 21., 25., 33., 26.], 'b': [1, 2, 3, 4, 5]}, columns=['a', 'b'])
        >>> df
              a  b
        0  24.0  1
        1  21.0  2
        2  25.0  3
        3  33.0  4
        4  26.0  5

        On a DataFrame:

        >>> df.median()
        a    25.0
        b     3.0
        Name: 0, dtype: float64

        On a Series:

        >>> df['a'].median()
        25.0
        >>> (df['a'] + 100).median()
        125.0

        For multi-index columns,

        >>> df.columns = pd.MultiIndex.from_tuples([('x', 'a'), ('y', 'b')])
        >>> df
              x  y
              a  b
        0  24.0  1
        1  21.0  2
        2  25.0  3
        3  33.0  4
        4  26.0  5

        On a DataFrame:

        >>> df.median()
        x  a    25.0
        y  b     3.0
        Name: 0, dtype: float64

        On a Series:

        >>> df[('x', 'a')].median()
        25.0
        >>> (df[('x', 'a')] + 100).median()
        125.0
        """
        if not isinstance(accuracy, int):
            raise ValueError("accuracy must be an integer; however, got [%s]" % type(accuracy))

        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series, _col

        kdf_or_kser = self
        if isinstance(kdf_or_kser, Series):
            kser = _col(kdf_or_kser.to_frame())
            return kser._reduce_for_stat_function(
                lambda _: F.expr(
                    "approx_percentile(`%s`, 0.5, %s)"
                    % (kser._internal.data_spark_column_names[0], accuracy)
                ),
                name="median",
            )
        assert isinstance(kdf_or_kser, DataFrame)

        # This code path cannot reuse `_reduce_for_stat_function` since there looks no proper way
        # to get a column name from Spark column but we need it to pass it through `expr`.
        kdf = kdf_or_kser
        sdf = kdf._sdf.select(kdf._internal.spark_columns)
        median = lambda name: F.expr("approx_percentile(`%s`, 0.5, %s)" % (name, accuracy))
        sdf = sdf.select([median(col).alias(col) for col in kdf._internal.data_spark_column_names])

        # Attach a dummy column for index to avoid default index.
        sdf = _InternalFrame.attach_distributed_column(sdf, "__DUMMY__")

        # This is expected to be small so it's fine to transpose.
        return (
            DataFrame(
                kdf._internal.copy(
                    spark_frame=sdf,
                    index_map=OrderedDict({"__DUMMY__": None}),
                    data_spark_columns=[
                        scol_for(sdf, col) for col in kdf._internal.data_spark_column_names
                    ],
                )
            )
            ._to_internal_pandas()
            .transpose()
            .iloc[:, 0]
        )

    # TODO: 'center', 'win_type', 'on', 'axis' parameter should be implemented.
    def rolling(self, window, min_periods=None):
        """
        Provide rolling transformations.

        .. note:: 'min_periods' in Koalas works as a fixed window size unlike pandas.
            Unlike pandas, NA is also counted as the period. This might be changed
            in the near future.

        Parameters
        ----------
        window : int, or offset
            Size of the moving window.
            This is the number of observations used for calculating the statistic.
            Each window will be a fixed size.

        min_periods : int, default None
            Minimum number of observations in window required to have a value
            (otherwise result is NA).
            For a window that is specified by an offset, min_periods will default to 1.
            Otherwise, min_periods will default to the size of the window.

        Returns
        -------
        a Window sub-classed for the particular operation
        """
        return Rolling(self, window=window, min_periods=min_periods)

    # TODO: 'center' and 'axis' parameter should be implemented.
    #   'axis' implementation, refer https://github.com/databricks/koalas/pull/607
    def expanding(self, min_periods=1):
        """
        Provide expanding transformations.

        .. note:: 'min_periods' in Koalas works as a fixed window size unlike pandas.
            Unlike pandas, NA is also counted as the period. This might be changed
            in the near future.

        Parameters
        ----------
        min_periods : int, default 1
            Minimum number of observations in window required to have a value
            (otherwise result is NA).

        Returns
        -------
        a Window sub-classed for the particular operation
        """
        return Expanding(self, min_periods=min_periods)

    def get(self, key, default=None):
        """
        Get item from object for given key (DataFrame column, Panel slice,
        etc.). Returns default value if not found.

        Parameters
        ----------
        key : object

        Returns
        -------
        value : same type as items contained in object

        Examples
        --------
        >>> df = ks.DataFrame({'x':range(3), 'y':['a','b','b'], 'z':['a','b','b']},
        ...                   columns=['x', 'y', 'z'], index=[10, 20, 20])
        >>> df
            x  y  z
        10  0  a  a
        20  1  b  b
        20  2  b  b

        >>> df.get('x')
        10    0
        20    1
        20    2
        Name: x, dtype: int64

        >>> df.get(['x', 'y'])
            x  y
        10  0  a
        20  1  b
        20  2  b

        >>> df.x.get(10)
        0

        >>> df.x.get(20)
        20    1
        20    2
        Name: x, dtype: int64

        >>> df.x.get(15, -1)
        -1
        """
        try:
            return self[key]
        except (KeyError, ValueError, IndexError):
            return default

    def squeeze(self, axis=None):
        """
        Squeeze 1 dimensional axis objects into scalars.

        Series or DataFrames with a single element are squeezed to a scalar.
        DataFrames with a single column or a single row are squeezed to a
        Series. Otherwise the object is unchanged.

        This method is most useful when you don't know if your
        object is a Series or DataFrame, but you do know it has just a single
        column. In that case you can safely call `squeeze` to ensure you have a
        Series.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns', None}, default None
            A specific axis to squeeze. By default, all length-1 axes are
            squeezed.

        Returns
        -------
        DataFrame, Series, or scalar
            The projection after squeezing `axis` or all the axes.

        See Also
        --------
        Series.iloc : Integer-location based indexing for selecting scalars.
        DataFrame.iloc : Integer-location based indexing for selecting Series.
        Series.to_frame : Inverse of DataFrame.squeeze for a
            single-column DataFrame.

        Examples
        --------
        >>> primes = ks.Series([2, 3, 5, 7])

        Slicing might produce a Series with a single value:

        >>> even_primes = primes[primes % 2 == 0]
        >>> even_primes
        0    2
        Name: 0, dtype: int64

        >>> even_primes.squeeze()
        2

        Squeezing objects with more than one value in every axis does nothing:

        >>> odd_primes = primes[primes % 2 == 1]
        >>> odd_primes
        1    3
        2    5
        3    7
        Name: 0, dtype: int64

        >>> odd_primes.squeeze()
        1    3
        2    5
        3    7
        Name: 0, dtype: int64

        Squeezing is even more effective when used with DataFrames.

        >>> df = ks.DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
        >>> df
           a  b
        0  1  2
        1  3  4

        Slicing a single column will produce a DataFrame with the columns
        having only one value:

        >>> df_a = df[['a']]
        >>> df_a
           a
        0  1
        1  3

        So the columns can be squeezed down, resulting in a Series:

        >>> df_a.squeeze('columns')
        0    1
        1    3
        Name: a, dtype: int64

        Slicing a single row from a single column will produce a single
        scalar DataFrame:

        >>> df_0a = df.loc[[0], ['a']]
        >>> df_0a
           a
        0  1

        Squeezing the rows produces a single scalar Series:

        >>> df_0a.squeeze('rows')
        a    1
        Name: 0, dtype: int64

        Squeezing all axes will project directly into a scalar:

        >>> df_0a.squeeze()
        1
        """
        if axis is not None:
            axis = "index" if axis == "rows" else axis
            axis = validate_axis(axis)

        if isinstance(self, ks.DataFrame):
            from databricks.koalas.series import _col

            is_squeezable = len(self.columns[:2]) == 1
            # If DataFrame has multiple columns, there is no change.
            if not is_squeezable:
                return self
            series_from_column = _col(self)
            has_single_value = len(series_from_column.head(2)) == 1
            # If DataFrame has only a single value, use pandas API directly.
            if has_single_value:
                result = self._to_internal_pandas().squeeze(axis)
                return ks.Series(result) if isinstance(result, pd.Series) else result
            elif axis == 0:
                return self
            else:
                return series_from_column
        elif isinstance(self, ks.Series):
            # The case of Series is simple.
            # If Series has only a single value, just return it as a scalar.
            # Otherwise, there is no change.
            self_top_two = self.head(2)
            has_single_value = len(self_top_two) == 1
            return self_top_two[0] if has_single_value else self

    def truncate(self, before=None, after=None, axis=None, copy=True):
        """
        Truncate a Series or DataFrame before and after some index value.

        This is a useful shorthand for boolean indexing based on index
        values above or below certain thresholds.

        .. note:: This API is dependent on :meth:`Index.is_monotonic_increasing`
            which can be expensive.

        Parameters
        ----------
        before : date, str, int
            Truncate all rows before this index value.
        after : date, str, int
            Truncate all rows after this index value.
        axis : {0 or 'index', 1 or 'columns'}, optional
            Axis to truncate. Truncates the index (rows) by default.
        copy : bool, default is True,
            Return a copy of the truncated section.

        Returns
        -------
        type of caller
            The truncated Series or DataFrame.

        See Also
        --------
        DataFrame.loc : Select a subset of a DataFrame by label.
        DataFrame.iloc : Select a subset of a DataFrame by position.

        Examples
        --------
        >>> df = ks.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
        ...                    'B': ['f', 'g', 'h', 'i', 'j'],
        ...                    'C': ['k', 'l', 'm', 'n', 'o']},
        ...                   index=[1, 2, 3, 4, 5])
        >>> df
           A  B  C
        1  a  f  k
        2  b  g  l
        3  c  h  m
        4  d  i  n
        5  e  j  o

        >>> df.truncate(before=2, after=4)
           A  B  C
        2  b  g  l
        3  c  h  m
        4  d  i  n

        The columns of a DataFrame can be truncated.

        >>> df.truncate(before="A", after="B", axis="columns")
           A  B
        1  a  f
        2  b  g
        3  c  h
        4  d  i
        5  e  j

        For Series, only rows can be truncated.

        >>> df['A'].truncate(before=2, after=4)
        2    b
        3    c
        4    d
        Name: A, dtype: object

        A Series has index that sorted integers.

        >>> s = ks.Series([10, 20, 30, 40, 50, 60, 70],
        ...               index=[1, 2, 3, 4, 5, 6, 7])
        >>> s
        1    10
        2    20
        3    30
        4    40
        5    50
        6    60
        7    70
        Name: 0, dtype: int64

        >>> s.truncate(2, 5)
        2    20
        3    30
        4    40
        5    50
        Name: 0, dtype: int64

        A Series has index that sorted strings.

        >>> s = ks.Series([10, 20, 30, 40, 50, 60, 70],
        ...               index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])
        >>> s
        a    10
        b    20
        c    30
        d    40
        e    50
        f    60
        g    70
        Name: 0, dtype: int64

        >>> s.truncate('b', 'e')
        b    20
        c    30
        d    40
        e    50
        Name: 0, dtype: int64
        """
        from databricks.koalas.series import _col

        axis = validate_axis(axis)
        indexes = self.index
        indexes_increasing = indexes.is_monotonic_increasing
        if not indexes_increasing and not indexes.is_monotonic_decreasing:
            raise ValueError("truncate requires a sorted index")
        if (before is None) and (after is None):
            return self.copy() if copy else self
        if (before is not None and after is not None) and before > after:
            raise ValueError("Truncate: %s must be after %s" % (after, before))

        if isinstance(self, ks.Series):
            result = _col(self.to_frame().loc[before:after])
        elif isinstance(self, ks.DataFrame):
            if axis == 0:
                result = self.loc[before:after]
            elif axis == 1:
                result = self.loc[:, before:after]

        return result.copy() if copy else result

    @property
    def at(self):
        return AtIndexer(self)

    at.__doc__ = AtIndexer.__doc__

    @property
    def iat(self):
        return iAtIndexer(self)

    iat.__doc__ = iAtIndexer.__doc__

    @property
    def iloc(self):
        return iLocIndexer(self)

    iloc.__doc__ = iLocIndexer.__doc__

    @property
    def loc(self):
        return LocIndexer(self)

    loc.__doc__ = LocIndexer.__doc__

    def compute(self):
        """Alias of `to_pandas()` to mimic dask for easily porting tests."""
        return self.toPandas()

    @staticmethod
    def _count_expr(col: spark.Column, spark_type: DataType) -> spark.Column:
        # Special handle floating point types because Spark's count treats nan as a valid value,
        # whereas Pandas count doesn't include nan.
        if isinstance(spark_type, (FloatType, DoubleType)):
            return F.count(F.nanvl(col, F.lit(None)))
        else:
            return F.count(col)
