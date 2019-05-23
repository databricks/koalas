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
from collections.abc import Iterable

import numpy as np
import pandas as pd

from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, DoubleType, FloatType

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.indexing import ILocIndexer, LocIndexer
from databricks.koalas.utils import validate_arguments_and_invoke_function

max_display_count = 1000


class _Frame(object):
    """
    The base class for both DataFrame and Series.
    """

    @property
    def values(self):
        raise NotImplementedError("Koalas does not support the 'values' property. " +
                                  "If you want to collect your data as an NumPy array, " +
                                  "use 'to_numpy()' instead.")

    def to_numpy(self):
        """
        A NumPy ndarray representing the values in this DataFrame or Series.

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.

        Returns
        -------
        numpy.ndarray
        """
        return self.to_pandas().values

    def to_csv(self, path_or_buf=None, sep=",", na_rep='', float_format=None,
               columns=None, header=True, index=True, index_label=None,
               mode='w', encoding=None, compression='infer', quoting=None,
               quotechar='"', line_terminator="\n", chunksize=None,
               tupleize_cols=None, date_format=None, doublequote=True,
               escapechar=None, decimal='.'):
        """
        Write object to a comma-separated values (csv) file.

        .. note:: This method should only be used if the resulting CSV is expected
            to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        path_or_buf : str or file handle, default None
            File path or object, if None is provided the result is returned as
            a string.  If a file object is passed it should be opened with
            `newline=''`, disabling universal newlines.

        sep : str, default ','
            String of length 1. Field delimiter for the output file.
        na_rep : str, default ''
            Missing data representation.
        float_format : str, default None
            Format string for floating point numbers.
        columns : sequence, optional
            Columns to write.
        header : bool or list of str, default True
            Write out the column names. If a list of strings is given it is
            assumed to be aliases for the column names.
        index : bool, default True
            Write row names (index).
        index_label : str or sequence, or False, default None
            Column label for index column(s) if desired. If None is given, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the object uses MultiIndex. If
            False do not print fields for index names. Use index_label=False
            for easier importing in R.
        mode : str
            Python write mode, default 'w'.
        encoding : str, optional
            A string representing the encoding to use in the output file,
            defaults to 'ascii' on Python 2 and 'utf-8' on Python 3.
        compression : str, default 'infer'
            Compression mode among the following possible values: {'infer',
            'gzip', 'bz2', 'zip', 'xz', None}. If 'infer' and `path_or_buf`
            is path-like, then detect compression from the following
            extensions: '.gz', '.bz2', '.zip' or '.xz'. (otherwise no
            compression).
        quoting : optional constant from csv module
            Defaults to csv.QUOTE_MINIMAL. If you have set a `float_format`
            then floats are converted to strings and thus csv.QUOTE_NONNUMERIC
            will treat them as non-numeric.
        quotechar : str, default '\"'
            String of length 1. Character used to quote fields.
        line_terminator : string, default '\\n'
            The newline character or character sequence to use in the output
            file. Defaults to `os.linesep`, which depends on the OS in which
            this method is called ('\n' for linux, '\r\n' for Windows, i.e.).
        chunksize : int or None
            Rows to write at a time.
        tupleize_cols : bool, default False
            Write MultiIndex columns as a list of tuples (if True) or in
            the new, expanded format, where each MultiIndex column is a row
            in the CSV (if False).
        date_format : str, default None
            Format string for datetime objects.
        doublequote : bool, default True
            Control quoting of `quotechar` inside a field.
        escapechar : str, default None
            String of length 1. Character used to escape `sep` and `quotechar`
            when appropriate.
        decimal : str, default '.'
            Character recognized as decimal separator. E.g. use ',' for
            European data.

        Returns
        -------
        None or str
            If path_or_buf is None, returns the resulting csv format as a
            string. Otherwise returns None.

        Examples
        --------
        >>> df = ks.DataFrame({'name': ['Raphael', 'Donatello'],
        ...                    'mask': ['red', 'purple'],
        ...                    'weapon': ['sai', 'bo staff']},
        ...                     columns=['name', 'mask', 'weapon'])
        >>> df.to_csv(index=False)
        'name,mask,weapon\\nRaphael,red,sai\\nDonatello,purple,bo staff\\n'
        >>> df.name.to_csv()  # doctest: +ELLIPSIS
        '...Raphael\\n1,Donatello\\n'
        """

        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        kdf = self

        if isinstance(self, ks.DataFrame):
            f = pd.DataFrame.to_csv
        elif isinstance(self, ks.Series):
            f = pd.Series.to_csv
        else:
            raise TypeError('Constructor expects DataFrame or Series; however, '
                            'got [%s]' % (self,))

        return validate_arguments_and_invoke_function(
            kdf.to_pandas(), self.to_csv, f, args)

    def to_json(self, path_or_buf=None, orient=None, date_format=None,
                double_precision=10, force_ascii=True, date_unit='ms',
                default_handler=None, lines=False, compression='infer',
                index=True):
        """
        Convert the object to a JSON string.

        Note NaN's and None will be converted to null and datetime objects
        will be converted to UNIX timestamps.

        .. note:: This method should only be used if the resulting JSON is expected
            to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        path_or_buf : string or file handle, optional
            File path or object. If not specified, the result is returned as
            a string.
        orient : string
            Indication of expected JSON string format.

            * Series

              - default is 'index'
              - allowed values are: {'split','records','index','table'}

            * DataFrame

              - default is 'columns'
              - allowed values are:
                {'split','records','index','columns','values','table'}

            * The format of the JSON string

              - 'split' : dict like {'index' -> [index],
                'columns' -> [columns], 'data' -> [values]}
              - 'records' : list like
                [{column -> value}, ... , {column -> value}]
              - 'index' : dict like {index -> {column -> value}}
              - 'columns' : dict like {column -> {index -> value}}
              - 'values' : just the values array
              - 'table' : dict like {'schema': {schema}, 'data': {data}}
                describing the data, and the data component is
                like ``orient='records'``.
        date_format : {None, 'epoch', 'iso'}
            Type of date conversion. 'epoch' = epoch milliseconds,
            'iso' = ISO8601. The default depends on the `orient`. For
            ``orient='table'``, the default is 'iso'. For all other orients,
            the default is 'epoch'.
        double_precision : int, default 10
            The number of decimal places to use when encoding
            floating point values.
        force_ascii : bool, default True
            Force encoded string to be ASCII.
        date_unit : string, default 'ms' (milliseconds)
            The time unit to encode to, governs timestamp and ISO8601
            precision.  One of 's', 'ms', 'us', 'ns' for second, millisecond,
            microsecond, and nanosecond respectively.
        default_handler : callable, default None
            Handler to call if object cannot otherwise be converted to a
            suitable format for JSON. Should receive a single argument which is
            the object to convert and return a serialisable object.
        lines : bool, default False
            If 'orient' is 'records' write out line delimited json format. Will
            throw ValueError if incorrect 'orient' since others are not list
            like.
        compression : {'infer', 'gzip', 'bz2', 'zip', 'xz', None}
            A string representing the compression to use in the output file,
            only used when the first argument is a filename. By default, the
            compression is inferred from the filename.
        index : bool, default True
            Whether to include the index values in the JSON string. Not
            including the index (``index=False``) is only supported when
            orient is 'split' or 'table'.

        Examples
        --------

        >>> df = ks.DataFrame([['a', 'b'], ['c', 'd']],
        ...                   index=['row 1', 'row 2'],
        ...                   columns=['col 1', 'col 2'])
        >>> df.to_json(orient='split')
        '{"columns":["col 1","col 2"],\
"index":["row 1","row 2"],\
"data":[["a","b"],["c","d"]]}'

        >>> df['col 1'].to_json(orient='split')
        '{"name":"col 1","index":["row 1","row 2"],"data":["a","c"]}'

        Encoding/decoding a Dataframe using ``'records'`` formatted JSON.
        Note that index labels are not preserved with this encoding.

        >>> df.to_json(orient='records')
        '[{"col 1":"a","col 2":"b"},{"col 1":"c","col 2":"d"}]'

        >>> df['col 1'].to_json(orient='records')
        '["a","c"]'

        Encoding/decoding a Dataframe using ``'index'`` formatted JSON:

        >>> df.to_json(orient='index')
        '{"row 1":{"col 1":"a","col 2":"b"},"row 2":{"col 1":"c","col 2":"d"}}'

        >>> df['col 1'].to_json(orient='index')
        '{"row 1":"a","row 2":"c"}'

        Encoding/decoding a Dataframe using ``'columns'`` formatted JSON:

        >>> df.to_json(orient='columns')
        '{"col 1":{"row 1":"a","row 2":"c"},"col 2":{"row 1":"b","row 2":"d"}}'

        >>> df['col 1'].to_json(orient='columns')
        '{"row 1":"a","row 2":"c"}'

        Encoding/decoding a Dataframe using ``'values'`` formatted JSON:

        >>> df.to_json(orient='values')
        '[["a","b"],["c","d"]]'

        >>> df['col 1'].to_json(orient='values')
        '["a","c"]'

        Encoding with Table Schema

        >>> df.to_json(orient='table')  # doctest: +SKIP
        '{"schema": {"fields":[{"name":"index","type":"string"},\
{"name":"col 1","type":"string"},\
{"name":"col 2","type":"string"}],\
"primaryKey":["index"],\
"pandas_version":"0.20.0"}, \
"data": [{"index":"row 1","col 1":"a","col 2":"b"},\
{"index":"row 2","col 1":"c","col 2":"d"}]}'

        >>> df['col 1'].to_json(orient='table')  # doctest: +SKIP
        '{"schema": {"fields":[{"name":"index","type":"string"},\
{"name":"col 1","type":"string"}],"primaryKey":["index"],"pandas_version":"0.20.0"}, \
"data": [{"index":"row 1","col 1":"a"},{"index":"row 2","col 1":"c"}]}'
        """
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        kdf = self

        if isinstance(self, ks.DataFrame):
            f = pd.DataFrame.to_json
        elif isinstance(self, ks.Series):
            f = pd.Series.to_json
        else:
            raise TypeError('Constructor expects DataFrame or Series; however, '
                            'got [%s]' % (self,))

        return validate_arguments_and_invoke_function(
            kdf.to_pandas(), self.to_json, f, args)

    def to_excel(self, excel_writer, sheet_name="Sheet1", na_rep="", float_format=None,
                 columns=None, header=True, index=True, index_label=None, startrow=0,
                 startcol=0, engine=None, merge_cells=True, encoding=None, inf_rep="inf",
                 verbose=True, freeze_panes=None):
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
            raise TypeError('Constructor expects DataFrame or Series; however, '
                            'got [%s]' % (self,))
        return validate_arguments_and_invoke_function(
            kdf.to_pandas(), self.to_excel, f, args)

    def mean(self):
        """
        Return the mean of the values.

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

        On a Series:

        >>> df['a'].mean()
        2.0
        """
        return self._reduce_for_stat_function(F.mean)

    def sum(self):
        """
        Return the sum of the values.

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

        On a Series:

        >>> df['a'].sum()
        6.0
        """
        return self._reduce_for_stat_function(F.sum)

    def skew(self):
        """
        Return unbiased skew normalized by N-1.

        Returns
        -------
        skew : scalar for a Series, and a Series for a DataFrame.

        Examples
        --------

        >>> df = ks.DataFrame({'a': [1, 2, 3, np.nan], 'b': [0.1, 0.2, 0.3, np.nan]},
        ...                   columns=['a', 'b'])

        On a DataFrame:

        >>> df.skew()
        a    0.000000e+00
        b   -3.319678e-16
        dtype: float64

        On a Series:

        >>> df['a'].skew()
        0.0
        """
        return self._reduce_for_stat_function(F.skewness)

    def kurtosis(self):
        """
        Return unbiased kurtosis using Fisher’s definition of kurtosis (kurtosis of normal == 0.0).
        Normalized by N-1.

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
        return self._reduce_for_stat_function(F.kurtosis)

    kurt = kurtosis

    def min(self):
        """
        Return the minimum of the values.

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

        On a Series:

        >>> df['a'].min()
        1.0
        """
        return self._reduce_for_stat_function(F.min)

    def max(self):
        """
        Return the maximum of the values.

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

        On a Series:

        >>> df['a'].max()
        3.0
        """
        return self._reduce_for_stat_function(F.max)

    def std(self):
        """
        Return sample standard deviation.

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

        On a Series:

        >>> df['a'].std()
        1.0
        """
        return self._reduce_for_stat_function(F.stddev)

    def var(self):
        """
        Return unbiased variance.

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

        On a Series:

        >>> df['a'].var()
        1.0
        """
        return self._reduce_for_stat_function(F.variance)

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
        Name: abs(0), dtype: float64

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
        # TODO: The first example above should not have "Name: abs(0)".
        return _spark_col_apply(self, F.abs)

    # TODO: by argument only support the grouping name only for now. Documentation should
    # be updated when it's supported.
    def groupby(self, by):
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
        >>> df.groupby(['Animal']).mean()  # doctest: +NORMALIZE_WHITESPACE
                Max Speed
        Animal
        Falcon      375.0
        Parrot       25.0
        """
        from databricks.koalas.frame import DataFrame
        from databricks.koalas.series import Series
        from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy

        df_or_s = self
        if isinstance(by, str):
            by = [by]
        elif isinstance(by, Series):
            by = [by]
        elif isinstance(by, Iterable):
            by = list(by)
        else:
            raise ValueError('Not a valid index: TODO')
        if not len(by):
            raise ValueError('No group keys passed!')
        if isinstance(df_or_s, DataFrame):
            df = df_or_s  # type: DataFrame
            col_by = [_resolve_col(df, col_or_s) for col_or_s in by]
            return DataFrameGroupBy(df_or_s, col_by)
        if isinstance(df_or_s, Series):
            col = df_or_s  # type: Series
            anchor = df_or_s._kdf
            col_by = [_resolve_col(anchor, col_or_s) for col_or_s in by]
            return SeriesGroupBy(col, col_by)
        raise TypeError('Constructor expects DataFrame or Series; however, '
                        'got [%s]' % (df_or_s,))

    @property
    def iloc(self):
        return ILocIndexer(self)

    iloc.__doc__ = ILocIndexer.__doc__

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


def _resolve_col(kdf, col_like):
    if isinstance(col_like, ks.Series):
        assert kdf == col_like._kdf, \
            "Cannot combine column argument because it comes from a different dataframe"
        return col_like
    elif isinstance(col_like, str):
        return kdf[col_like]
    else:
        raise ValueError(col_like)


def _spark_col_apply(kdf_or_ks, sfun):
    """
    Performs a function to all cells on a dataframe, the function being a known sql function.
    """
    from databricks.koalas.frame import DataFrame
    from databricks.koalas.series import Series
    if isinstance(kdf_or_ks, Series):
        ks = kdf_or_ks
        return Series(sfun(kdf_or_ks._scol), anchor=ks._kdf, index=ks._index_map)
    assert isinstance(kdf_or_ks, DataFrame)
    kdf = kdf_or_ks
    sdf = kdf._sdf
    sdf = sdf.select([sfun(sdf[col]).alias(col) for col in kdf.columns])
    return DataFrame(sdf)
