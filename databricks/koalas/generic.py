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
A base class of DataFrame/Column to behave similar to pandas DataFrame/Series.
"""
from abc import ABCMeta, abstractmethod
from collections import Counter
from collections.abc import Iterable
from distutils.version import LooseVersion
from functools import reduce
from typing import Any, List, Optional, Tuple, Union, TYPE_CHECKING, cast
import warnings

import numpy as np  # noqa: F401
import pandas as pd
from pandas.api.types import is_list_like

import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, DoubleType, FloatType

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.indexing import AtIndexer, iAtIndexer, iLocIndexer, LocIndexer
from databricks.koalas.internal import InternalFrame
from databricks.koalas.spark import functions as SF
from databricks.koalas.typedef import Scalar
from databricks.koalas.utils import (
    is_name_like_tuple,
    is_name_like_value,
    name_like_string,
    scol_for,
    validate_arguments_and_invoke_function,
    validate_axis,
)
from databricks.koalas.window import Rolling, Expanding

if TYPE_CHECKING:
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy
    from databricks.koalas.series import Series


class Frame(object, metaclass=ABCMeta):
    """
    The base class for both DataFrame and Series.
    """

    @abstractmethod
    def __getitem__(self, key):
        pass

    @property
    @abstractmethod
    def _internal(self) -> InternalFrame:
        pass

    @abstractmethod
    def _apply_series_op(self, op, should_resolve: bool = False):
        pass

    @abstractmethod
    def _reduce_for_stat_function(self, sfun, name, axis=None, numeric_only=True):
        pass

    @property
    @abstractmethod
    def dtypes(self):
        pass

    @abstractmethod
    def to_pandas(self):
        pass

    @property
    @abstractmethod
    def index(self):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def _to_internal_pandas(self):
        pass

    @abstractmethod
    def head(self, n: int = 5):
        pass

    # TODO: add 'axis' parameter
    def cummin(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
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
        return self._apply_series_op(
            lambda kser: kser._cum(F.min, skipna), should_resolve=True
        )  # type: ignore

    # TODO: add 'axis' parameter
    def cummax(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
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
        return self._apply_series_op(
            lambda kser: kser._cum(F.max, skipna), should_resolve=True
        )  # type: ignore

    # TODO: add 'axis' parameter
    def cumsum(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
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
        return self._apply_series_op(
            lambda kser: kser._cum(F.sum, skipna), should_resolve=True
        )  # type: ignore

    # TODO: add 'axis' parameter
    # TODO: use pandas_udf to support negative values and other options later
    #  other window except unbounded ones is supported as of Spark 3.0.
    def cumprod(self, skipna: bool = True) -> Union["Series", "DataFrame"]:
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
        return self._apply_series_op(
            lambda kser: kser._cumprod(skipna), should_resolve=True
        )  # type: ignore

    # TODO: Although this has removed pandas >= 1.0.0, but we're keeping this as deprecated
    # since we're using this for `DataFrame.info` internally.
    # We can drop it once our minimal pandas version becomes 1.0.0.
    def get_dtype_counts(self) -> pd.Series:
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
            dtypes = list(self.dtypes)
        return pd.Series(dict(Counter([d.name for d in dtypes])))

    def pipe(self, func, *args, **kwargs) -> Any:
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
        ...     return df[df['category'] == 'A']
        >>> def add_one(df, column):
        ...     return df.assign(col3=df[column] + 1)
        >>> def multiply(df, column1, column2):
        ...     return df.assign(col4=df[column1] * df[column2])


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

    def to_numpy(self) -> np.ndarray:
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
    def values(self) -> np.ndarray:
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
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options
    ) -> Optional[str]:
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
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'},
            default 'overwrite'. Specifies the behavior of the save operation when the
            destination exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        options: keyword arguments for additional options specific to PySpark.
            This kwargs are specific to PySpark's CSV options to pass. Check
            the options in PySpark's API documentation for spark.write.csv(...).
            It has higher priority and overwrites all other options.
            This parameter only works when `path` is specified.

        Returns
        -------
        str or None

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

        >>> df.cummax().to_csv(path=r'%s/to_csv/foo.csv' % path, num_files=1)
        >>> ks.read_csv(
        ...    path=r'%s/to_csv/foo.csv' % path
        ... ).sort_values(by="date")  # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
                           date country  code
        ... 2012-01-31 12:00:00      KR     1
        ... 2012-02-29 12:00:00      US     2
        ... 2012-03-31 12:00:00      US     3

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
        else:
            column_labels = []
            for label in columns:
                if not is_name_like_tuple(label):
                    label = (label,)
                if label not in kdf._internal.column_labels:
                    raise KeyError(name_like_string(label))
                column_labels.append(label)

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
                    scol_for(sdf, str(i) if label is None else name_like_string(label)).alias(
                        new_name
                    )
                    for i, (label, new_name) in enumerate(zip(column_labels, header))
                ]
            )
            header = True
        else:
            sdf = kdf.to_spark(index_col)  # type: ignore
            sdf = sdf.select(
                [scol_for(sdf, name_like_string(label)) for label in index_cols]
                + [
                    scol_for(sdf, str(i) if label is None else name_like_string(label))
                    for i, label in enumerate(column_labels)
                ]
            )

        if num_files is not None:
            sdf = sdf.repartition(num_files)

        builder = sdf.write.mode(mode)
        if partition_cols is not None:
            builder.partitionBy(partition_cols)
        builder._set_opts(
            sep=sep,
            nullValue=na_rep,
            header=header,
            quote=quotechar,
            dateFormat=date_format,
            charToEscapeQuoteEscaping=escapechar,
        )
        builder.options(**options).format("csv").save(path)
        return None

    def to_json(
        self,
        path=None,
        compression="uncompressed",
        num_files=None,
        mode: str = "overwrite",
        partition_cols: Optional[Union[str, List[str]]] = None,
        index_col: Optional[Union[str, List[str]]] = None,
        **options
    ) -> Optional[str]:
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
        mode : str {'append', 'overwrite', 'ignore', 'error', 'errorifexists'},
            default 'overwrite'. Specifies the behavior of the save operation when the
            destination exists already.

            - 'append': Append the new data to existing data.
            - 'overwrite': Overwrite existing data.
            - 'ignore': Silently ignore this operation if data already exists.
            - 'error' or 'errorifexists': Throw an exception if data already exists.

        partition_cols : str or list of str, optional, default None
            Names of partitioning columns
        index_col: str or list of str, optional, default: None
            Column names to be used in Spark to represent Koalas' index. The index name
            in Koalas is ignored. By default, the index is always lost.
        options: keyword arguments for additional options specific to PySpark.
            It is specific to PySpark's JSON options to pass. Check
            the options in PySpark's API documentation for `spark.write.json(...)`.
            It has a higher priority and overwrites all other options.
            This parameter only works when `path` is specified.

        Returns
        --------
        str or None

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

        builder = sdf.write.mode(mode)
        if partition_cols is not None:
            builder.partitionBy(partition_cols)
        builder._set_opts(compression=compression)
        builder.options(**options).format("json").save(path)
        return None

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
    ) -> None:
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

    def mean(self, axis=None, numeric_only=True) -> Union[Scalar, "Series"]:
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
        dtype: float64

        >>> df.mean(axis=1)
        0    0.55
        1    1.10
        2    1.65
        3     NaN
        dtype: float64

        On a Series:

        >>> df['a'].mean()
        2.0
        """
        return self._reduce_for_stat_function(
            F.mean, name="mean", numeric_only=numeric_only, axis=axis
        )

    def sum(self, axis=None, numeric_only=True) -> Union[Scalar, "Series"]:
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
        dtype: float64

        >>> df.sum(axis=1)
        0    1.1
        1    2.2
        2    3.3
        3    0.0
        dtype: float64

        On a Series:

        >>> df['a'].sum()
        6.0
        """
        return self._reduce_for_stat_function(
            F.sum, name="sum", numeric_only=numeric_only, axis=axis
        )

    def skew(self, axis=None, numeric_only=True) -> Union[Scalar, "Series"]:
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

    def kurtosis(self, axis=None, numeric_only=True) -> Union[Scalar, "Series"]:
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
        dtype: float64

        On a Series:

        >>> df['a'].kurtosis()
        -1.5
        """
        return self._reduce_for_stat_function(
            F.kurtosis, name="kurtosis", numeric_only=numeric_only, axis=axis
        )

    kurt = kurtosis

    def min(self, axis=None, numeric_only=None) -> Union[Scalar, "Series"]:
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
        dtype: float64

        >>> df.min(axis=1)
        0    0.1
        1    0.2
        2    0.3
        3    NaN
        dtype: float64

        On a Series:

        >>> df['a'].min()
        1.0
        """
        return self._reduce_for_stat_function(
            F.min, name="min", numeric_only=numeric_only, axis=axis
        )

    def max(self, axis=None, numeric_only=None) -> Union[Scalar, "Series"]:
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
        dtype: float64

        >>> df.max(axis=1)
        0    1.0
        1    2.0
        2    3.0
        3    NaN
        dtype: float64

        On a Series:

        >>> df['a'].max()
        3.0
        """
        return self._reduce_for_stat_function(
            F.max, name="max", numeric_only=numeric_only, axis=axis
        )

    def std(self, axis=None, numeric_only=True) -> Union[Scalar, "Series"]:
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
        dtype: float64

        >>> df.std(axis=1)
        0    0.636396
        1    1.272792
        2    1.909188
        3         NaN
        dtype: float64

        On a Series:

        >>> df['a'].std()
        1.0
        """
        return self._reduce_for_stat_function(
            F.stddev, name="std", numeric_only=numeric_only, axis=axis
        )

    def var(self, axis=None, numeric_only=True) -> Union[Scalar, "Series"]:
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
        dtype: float64

        >>> df.var(axis=1)
        0    0.405
        1    1.620
        2    3.645
        3      NaN
        dtype: float64

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
        6

        >>> df = ks.DataFrame(index=[1, 2, None])
        >>> df.size
        0
        """
        num_columns = len(self._internal.data_spark_columns)
        if num_columns == 0:
            return 0
        else:
            return len(self) * num_columns  # type: ignore

    def abs(self) -> Union["DataFrame", "Series"]:
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
        dtype: float64

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
        return self._apply_series_op(lambda kser: kser._with_new_scol(F.abs(kser.spark.column)))

    # TODO: by argument only support the grouping name and as_index only for now. Documentation
    # should be updated when it's supported.
    def groupby(
        self, by, axis=0, as_index: bool = True, dropna: bool = True
    ) -> Union["DataFrameGroupBy", "SeriesGroupBy"]:
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
        dropna : bool, default True
            If True, and if group keys contain NA values,
            NA values together with row/column will be dropped.
            If False, NA values will also be treated as the key in groups.

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

        We can also choose to include NA in group keys or not by setting dropna parameter,
        the default setting is True:

        >>> l = [[1, 2, 3], [1, None, 4], [2, 1, 3], [1, 2, 2]]
        >>> df = ks.DataFrame(l, columns=["a", "b", "c"])
        >>> df.groupby(by=["b"]).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
             a  c
        b
        1.0  2  3
        2.0  2  5

        >>> df.groupby(by=["b"], dropna=False).sum().sort_index()  # doctest: +NORMALIZE_WHITESPACE
             a  c
        b
        1.0  2  3
        2.0  2  5
        NaN  1  4
        """
        from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy

        if isinstance(by, ks.DataFrame):
            raise ValueError("Grouper for '{}' not 1-dimensional".format(type(by).__name__))
        elif isinstance(by, ks.Series):
            by = [by]
        elif is_name_like_tuple(by):
            if isinstance(self, ks.Series):
                raise KeyError(by)
            by = [by]
        elif is_name_like_value(by):
            if isinstance(self, ks.Series):
                raise KeyError(by)
            by = [(by,)]
        elif is_list_like(by):
            new_by = []  # type: List[Union[Tuple, ks.Series]]
            for key in by:
                if isinstance(key, ks.DataFrame):
                    raise ValueError(
                        "Grouper for '{}' not 1-dimensional".format(type(key).__name__)
                    )
                elif isinstance(key, ks.Series):
                    new_by.append(key)
                elif is_name_like_tuple(key):
                    if isinstance(self, ks.Series):
                        raise KeyError(key)
                    new_by.append(key)
                elif is_name_like_value(key):
                    if isinstance(self, ks.Series):
                        raise KeyError(key)
                    new_by.append((key,))
                else:
                    raise ValueError(
                        "Grouper for '{}' not 1-dimensional".format(type(key).__name__)
                    )
            by = new_by
        else:
            raise ValueError("Grouper for '{}' not 1-dimensional".format(type(by).__name__))
        if not len(by):
            raise ValueError("No group keys passed!")
        axis = validate_axis(axis)
        if axis != 0:
            raise NotImplementedError('axis should be either 0 or "index" currently.')

        if isinstance(self, ks.DataFrame):
            return DataFrameGroupBy._build(self, by, as_index=as_index, dropna=dropna)
        elif isinstance(self, ks.Series):
            return SeriesGroupBy._build(self, by, as_index=as_index, dropna=dropna)
        else:
            raise TypeError(
                "Constructor expects DataFrame or Series; however, " "got [%s]" % (self,)
            )

    def bool(self) -> bool:
        """
        Return the bool of a single element in the current object.

        This must be a boolean scalar value, either True or False. Raise a ValueError if
        the object does not have exactly 1 element, or that element is not boolean

        Returns
        --------
        bool

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

    def first_valid_index(self) -> Optional[Union[Scalar, Tuple[Scalar, ...]]]:
        """
        Retrieves the index of the first valid value.

        Returns
        -------
        scalar, tuple, or None

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
        dtype: float64

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
        dtype: float64

        >>> s.first_valid_index()
        ('cow', 'weight')
        """
        data_spark_columns = self._internal.data_spark_columns

        if len(data_spark_columns) == 0:
            return None

        cond = reduce(lambda x, y: x & y, map(lambda x: x.isNotNull(), data_spark_columns))

        first_valid_row = (
            self._internal.spark_frame.filter(cond)
            .select(self._internal.index_spark_columns)
            .first()
        )

        # For Empty Series or DataFrame, returns None.
        if first_valid_row is None:
            return None

        if len(first_valid_row) == 1:
            return first_valid_row[0]
        else:
            return tuple(first_valid_row)

    def last_valid_index(self) -> Optional[Union[Scalar, Tuple[Scalar, ...]]]:
        """
        Return index for last non-NA/null value.

        Returns
        -------
        scalar, tuple, or None

        Notes
        -----
        This API only works with PySpark >= 3.0.

        Examples
        --------

        Support for DataFrame

        >>> kdf = ks.DataFrame({'a': [1, 2, 3, None],
        ...                     'b': [1.0, 2.0, 3.0, None],
        ...                     'c': [100, 200, 400, None]},
        ...                     index=['Q', 'W', 'E', 'R'])
        >>> kdf
             a    b      c
        Q  1.0  1.0  100.0
        W  2.0  2.0  200.0
        E  3.0  3.0  400.0
        R  NaN  NaN    NaN

        >>> kdf.last_valid_index()  # doctest: +SKIP
        'E'

        Support for MultiIndex columns

        >>> kdf.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> kdf
             a    b      c
             x    y      z
        Q  1.0  1.0  100.0
        W  2.0  2.0  200.0
        E  3.0  3.0  400.0
        R  NaN  NaN    NaN

        >>> kdf.last_valid_index()  # doctest: +SKIP
        'E'

        Support for Series.

        >>> s = ks.Series([1, 2, 3, None, None], index=[100, 200, 300, 400, 500])
        >>> s
        100    1.0
        200    2.0
        300    3.0
        400    NaN
        500    NaN
        dtype: float64

        >>> s.last_valid_index()  # doctest: +SKIP
        300

        Support for MultiIndex

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       ['speed', 'weight', 'length']],
        ...                      [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> s = ks.Series([250, 1.5, 320, 1, 0.3, None, None, None, None], index=midx)
        >>> s
        lama    speed     250.0
                weight      1.5
                length    320.0
        cow     speed       1.0
                weight      0.3
                length      NaN
        falcon  speed       NaN
                weight      NaN
                length      NaN
        dtype: float64

        >>> s.last_valid_index()  # doctest: +SKIP
        ('cow', 'weight')
        """
        if LooseVersion(pyspark.__version__) < LooseVersion("3.0"):
            raise RuntimeError("last_valid_index can be used in PySpark >= 3.0")

        data_spark_columns = self._internal.data_spark_columns

        if len(data_spark_columns) == 0:
            return None

        cond = reduce(lambda x, y: x & y, map(lambda x: x.isNotNull(), data_spark_columns))

        last_valid_rows = (
            self._internal.spark_frame.filter(cond)
            .select(self._internal.index_spark_columns)
            .tail(1)
        )

        # For Empty Series or DataFrame, returns None.
        if len(last_valid_rows) == 0:
            return None

        last_valid_row = last_valid_rows[0]

        if len(last_valid_row) == 1:
            return last_valid_row[0]
        else:
            return tuple(last_valid_row)

    def median(self, axis=None, numeric_only=True, accuracy=10000) -> Union[Scalar, "Series"]:
        """
        Return the median of the values for the requested axis.

        .. note:: Unlike pandas', the median in Koalas is an approximated median based upon
            approximate percentile computation because computing median across a large dataset
            is extremely expensive.

        Parameters
        ----------
        axis : {index (0), columns (1)}
            Axis for the function to be applied on.
        numeric_only : bool, default True
            Include only float, int, boolean columns. False is not supported. This parameter
            is mainly for pandas compatibility.
        accuracy : int, optional
            Default accuracy of approximation. Larger value means better accuracy.
            The relative error can be deduced by 1.0 / accuracy.

        Returns
        -------
        median : scalar or Series

        Examples
        --------
        >>> df = ks.DataFrame({
        ...     'a': [24., 21., 25., 33., 26.], 'b': [1., 2., 3., 4., 5.]}, columns=['a', 'b'])
        >>> df
              a    b
        0  24.0  1.0
        1  21.0  2.0
        2  25.0  3.0
        3  33.0  4.0
        4  26.0  5.0

        On a DataFrame:

        >>> df.median()
        a    25.0
        b     3.0
        dtype: float64

        On a Series:

        >>> df['a'].median()
        25.0
        >>> (df['a'] + 100).median()
        125.0

        For multi-index columns,

        >>> df.columns = pd.MultiIndex.from_tuples([('x', 'a'), ('y', 'b')])
        >>> df
              x    y
              a    b
        0  24.0  1.0
        1  21.0  2.0
        2  25.0  3.0
        3  33.0  4.0
        4  26.0  5.0

        On a DataFrame:

        >>> df.median()
        x  a    25.0
        y  b     3.0
        dtype: float64

        >>> df.median(axis=1)
        0    12.5
        1    11.5
        2    14.0
        3    18.5
        4    15.5
        dtype: float64

        On a Series:

        >>> df[('x', 'a')].median()
        25.0
        >>> (df[('x', 'a')] + 100).median()
        125.0
        """
        if not isinstance(accuracy, int):
            raise ValueError(
                "accuracy must be an integer; however, got [%s]" % type(accuracy).__name__
            )

        return self._reduce_for_stat_function(
            lambda scol: SF.percentile_approx(scol, 0.5, accuracy),
            name="median",
            numeric_only=numeric_only,
            axis=axis,
        )

    # TODO: 'center', 'win_type', 'on', 'axis' parameter should be implemented.
    def rolling(self, window, min_periods=None) -> Rolling:
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
    def expanding(self, min_periods=1) -> Expanding:
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

    def get(self, key, default=None) -> Any:
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

    def squeeze(self, axis=None) -> Union[Scalar, "DataFrame", "Series"]:
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
        dtype: int64

        >>> even_primes.squeeze()
        2

        Squeezing objects with more than one value in every axis does nothing:

        >>> odd_primes = primes[primes % 2 == 1]
        >>> odd_primes
        1    3
        2    5
        3    7
        dtype: int64

        >>> odd_primes.squeeze()
        1    3
        2    5
        3    7
        dtype: int64

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

        >>> df_1a = df.loc[[1], ['a']]
        >>> df_1a
           a
        1  3

        Squeezing the rows produces a single scalar Series:

        >>> df_1a.squeeze('rows')
        a    3
        Name: 1, dtype: int64

        Squeezing all axes will project directly into a scalar:

        >>> df_1a.squeeze()
        3
        """
        if axis is not None:
            axis = "index" if axis == "rows" else axis
            axis = validate_axis(axis)

        if isinstance(self, ks.DataFrame):
            from databricks.koalas.series import first_series

            is_squeezable = len(self.columns[:2]) == 1
            # If DataFrame has multiple columns, there is no change.
            if not is_squeezable:
                return self
            series_from_column = first_series(self)
            has_single_value = len(series_from_column.head(2)) == 1
            # If DataFrame has only a single value, use pandas API directly.
            if has_single_value:
                result = self._to_internal_pandas().squeeze(axis)
                return ks.Series(result) if isinstance(result, pd.Series) else result
            elif axis == 0:
                return self
            else:
                return series_from_column
        else:
            # The case of Series is simple.
            # If Series has only a single value, just return it as a scalar.
            # Otherwise, there is no change.
            self_top_two = self.head(2)
            has_single_value = len(self_top_two) == 1
            return cast(Union[Scalar, ks.Series], self_top_two[0] if has_single_value else self)

    def truncate(
        self, before=None, after=None, axis=None, copy=True
    ) -> Union["DataFrame", "Series"]:
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
        dtype: int64

        >>> s.truncate(2, 5)
        2    20
        3    30
        4    40
        5    50
        dtype: int64

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
        dtype: int64

        >>> s.truncate('b', 'e')
        b    20
        c    30
        d    40
        e    50
        dtype: int64
        """
        from databricks.koalas.series import first_series

        axis = validate_axis(axis)
        indexes = self.index
        indexes_increasing = indexes.is_monotonic_increasing
        if not indexes_increasing and not indexes.is_monotonic_decreasing:
            raise ValueError("truncate requires a sorted index")
        if (before is None) and (after is None):
            return cast(Union[ks.DataFrame, ks.Series], self.copy() if copy else self)
        if (before is not None and after is not None) and before > after:
            raise ValueError("Truncate: %s must be after %s" % (after, before))

        if isinstance(self, ks.Series):
            if indexes_increasing:
                result = first_series(self.to_frame().loc[before:after]).rename(self.name)
            else:
                result = first_series(self.to_frame().loc[after:before]).rename(self.name)
        elif isinstance(self, ks.DataFrame):
            if axis == 0:
                if indexes_increasing:
                    result = self.loc[before:after]
                else:
                    result = self.loc[after:before]
            elif axis == 1:
                result = self.loc[:, before:after]

        return cast(Union[ks.DataFrame, ks.Series], result.copy() if copy else result)

    def to_markdown(self, buf=None, mode=None) -> str:
        """
        Print Series or DataFrame in Markdown-friendly format.

        .. note:: This method should only be used if the resulting pandas object is expected
                  to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        buf : writable buffer, defaults to sys.stdout
            Where to send the output. By default, the output is printed to
            sys.stdout. Pass a writable buffer if you need to further process
            the output.
        mode : str, optional
            Mode in which file is opened.
        **kwargs
            These parameters will be passed to `tabulate`.

        Returns
        -------
        str
            Series or DataFrame in Markdown-friendly format.

        Examples
        --------
        >>> kser = ks.Series(["elk", "pig", "dog", "quetzal"], name="animal")
        >>> print(kser.to_markdown())  # doctest: +SKIP
        |    | animal   |
        |---:|:---------|
        |  0 | elk      |
        |  1 | pig      |
        |  2 | dog      |
        |  3 | quetzal  |

        >>> kdf = ks.DataFrame(
        ...     data={"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]}
        ... )
        >>> print(kdf.to_markdown())  # doctest: +SKIP
        |    | animal_1   | animal_2   |
        |---:|:-----------|:-----------|
        |  0 | elk        | dog        |
        |  1 | pig        | quetzal    |
        """
        # `to_markdown` is supported in pandas >= 1.0.0 since it's newly added in pandas 1.0.0.
        if LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            raise NotImplementedError(
                "`to_markdown()` only supported in Koalas with pandas >= 1.0.0"
            )
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        kser_or_kdf = self
        internal_pandas = kser_or_kdf._to_internal_pandas()
        return validate_arguments_and_invoke_function(
            internal_pandas, self.to_markdown, type(internal_pandas).to_markdown, args
        )

    @abstractmethod
    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None):
        pass

    # TODO: add 'downcast' when value parameter exists
    def bfill(self, axis=None, inplace=False, limit=None) -> Union["DataFrame", "Series"]:
        """
        Synonym for `DataFrame.fillna()` or `Series.fillna()` with ``method=`bfill```.

        .. note:: the current implementation of 'bfill' uses Spark's Window
            without specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        axis : {0 or `index`}
            1 and `columns` are not supported.
        inplace : boolean, default False
            Fill in place (do not create a new object)
        limit : int, default None
            If method is specified, this is the maximum number of consecutive NaN values to
            forward/backward fill. In other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. If method is not specified,
            this is the maximum number of entries along the entire axis where NaNs will be filled.
            Must be greater than 0 if not None

        Returns
        -------
        DataFrame or Series
            DataFrame or Series with NA entries filled.

        Examples
        --------
        >>> kdf = ks.DataFrame({
        ...     'A': [None, 3, None, None],
        ...     'B': [2, 4, None, 3],
        ...     'C': [None, None, None, 1],
        ...     'D': [0, 1, 5, 4]
        ...     },
        ...     columns=['A', 'B', 'C', 'D'])
        >>> kdf
             A    B    C  D
        0  NaN  2.0  NaN  0
        1  3.0  4.0  NaN  1
        2  NaN  NaN  NaN  5
        3  NaN  3.0  1.0  4

        Propagate non-null values backward.

        >>> kdf.bfill()
             A    B    C  D
        0  3.0  2.0  1.0  0
        1  3.0  4.0  1.0  1
        2  NaN  3.0  1.0  5
        3  NaN  3.0  1.0  4

        For Series

        >>> kser = ks.Series([None, None, None, 1])
        >>> kser
        0    NaN
        1    NaN
        2    NaN
        3    1.0
        dtype: float64

        >>> kser.bfill()
        0    1.0
        1    1.0
        2    1.0
        3    1.0
        dtype: float64
        """
        return self.fillna(method="bfill", axis=axis, inplace=inplace, limit=limit)

    backfill = bfill

    # TODO: add 'downcast' when value parameter exists
    def ffill(self, axis=None, inplace=False, limit=None) -> Union["DataFrame", "Series"]:
        """
        Synonym for `DataFrame.fillna()` or `Series.fillna()` with ``method=`ffill```.

        .. note:: the current implementation of 'ffill' uses Spark's Window
            without specifying partition specification. This leads to move all data into
            single partition in single machine and could cause serious
            performance degradation. Avoid this method against very large dataset.

        Parameters
        ----------
        axis : {0 or `index`}
            1 and `columns` are not supported.
        inplace : boolean, default False
            Fill in place (do not create a new object)
        limit : int, default None
            If method is specified, this is the maximum number of consecutive NaN values to
            forward/backward fill. In other words, if there is a gap with more than this number of
            consecutive NaNs, it will only be partially filled. If method is not specified,
            this is the maximum number of entries along the entire axis where NaNs will be filled.
            Must be greater than 0 if not None

        Returns
        -------
        DataFrame or Series
            DataFrame or Series with NA entries filled.

        Examples
        --------
        >>> kdf = ks.DataFrame({
        ...     'A': [None, 3, None, None],
        ...     'B': [2, 4, None, 3],
        ...     'C': [None, None, None, 1],
        ...     'D': [0, 1, 5, 4]
        ...     },
        ...     columns=['A', 'B', 'C', 'D'])
        >>> kdf
             A    B    C  D
        0  NaN  2.0  NaN  0
        1  3.0  4.0  NaN  1
        2  NaN  NaN  NaN  5
        3  NaN  3.0  1.0  4

        Propagate non-null values forward.

        >>> kdf.ffill()
             A    B    C  D
        0  NaN  2.0  NaN  0
        1  3.0  4.0  NaN  1
        2  3.0  4.0  NaN  5
        3  3.0  3.0  1.0  4

        For Series

        >>> kser = ks.Series([2, 4, None, 3])
        >>> kser
        0    2.0
        1    4.0
        2    NaN
        3    3.0
        dtype: float64

        >>> kser.ffill()
        0    2.0
        1    4.0
        2    4.0
        3    3.0
        dtype: float64
        """
        return self.fillna(method="ffill", axis=axis, inplace=inplace, limit=limit)

    pad = ffill

    @property
    def at(self) -> AtIndexer:
        return AtIndexer(self)

    at.__doc__ = AtIndexer.__doc__

    @property
    def iat(self) -> iAtIndexer:
        return iAtIndexer(self)

    iat.__doc__ = iAtIndexer.__doc__

    @property
    def iloc(self) -> iLocIndexer:
        return iLocIndexer(self)

    iloc.__doc__ = iLocIndexer.__doc__

    @property
    def loc(self) -> LocIndexer:
        return LocIndexer(self)

    loc.__doc__ = LocIndexer.__doc__

    def __bool__(self):
        raise ValueError(
            "The truth value of a {0} is ambiguous. "
            "Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(self.__class__.__name__)
        )

    @staticmethod
    def _count_expr(col: spark.Column, spark_type: DataType) -> spark.Column:
        # Special handle floating point types because Spark's count treats nan as a valid value,
        # whereas pandas count doesn't include nan.
        if isinstance(spark_type, (FloatType, DoubleType)):
            return F.count(F.nanvl(col, F.lit(None)))
        else:
            return F.count(col)
