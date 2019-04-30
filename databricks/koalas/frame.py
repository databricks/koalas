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
A wrapper class for Spark DataFrame to behave similar to pandas DataFrame.
"""
from decorator import dispatch_on
from functools import partial, reduce

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_datetime64tz_dtype
from pyspark import sql as spark
from pyspark.sql import functions as F, Column
from pyspark.sql.types import DataType, DoubleType, FloatType, StructField, StructType, \
    to_arrow_type
from pyspark.sql.utils import AnalysisException

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.utils import default_session
from databricks.koalas.dask.compatibility import string_types
from databricks.koalas.dask.utils import derived_from
from databricks.koalas.generic import _Frame, max_display_count
from databricks.koalas.metadata import Metadata
from databricks.koalas.missing.frame import _MissingPandasLikeDataFrame
from databricks.koalas.ml import corr
from databricks.koalas.selection import SparkDataFrameLocator
from databricks.koalas.typedef import infer_pd_series_spark_type


class DataFrame(_Frame):
    """
    Koala DataFrame that corresponds to Pandas DataFrame logically. This holds Spark DataFrame
    internally.

    :ivar _sdf: Spark Column instance
    :ivar _metadata: Metadata related to column names and index information.
    """

    @derived_from(pd.DataFrame)
    @dispatch_on('data')
    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        pdf = pd.DataFrame(data=data, index=index, columns=columns, dtype=dtype, copy=copy)
        self._init_from_pandas(pdf)

    @__init__.register(pd.DataFrame)
    def _init_from_pandas(self, pdf, *args):
        metadata = Metadata.from_pandas(pdf)
        reset_index = pdf.reset_index()
        reset_index.columns = metadata.all_fields
        schema = StructType([StructField(name, infer_pd_series_spark_type(col),
                                         nullable=bool(col.isnull().any()))
                             for name, col in reset_index.iteritems()])
        for name, col in reset_index.iteritems():
            dt = col.dtype
            if is_datetime64_dtype(dt) or is_datetime64tz_dtype(dt):
                continue
            reset_index[name] = col.replace({np.nan: None})
        self._init_from_spark(default_session().createDataFrame(reset_index, schema=schema),
                              metadata)

    @__init__.register(spark.DataFrame)
    def _init_from_spark(self, sdf, metadata=None, *args):
        self._sdf = sdf
        if metadata is None:
            self._metadata = Metadata(column_fields=self._sdf.schema.fieldNames())
        else:
            self._metadata = metadata

    @property
    def _index_columns(self):
        return [self._sdf.__getitem__(field)
                for field in self._metadata.index_fields]

    def _reduce_for_stat_function(self, sfun):
        """
        Applies sfun to each column and returns a pd.Series where the number of rows equal the
        number of columns.

        :param sfun: either an 1-arg function that takes a Column and returns a Column, or
        a 2-arg function that takes a Column and its DataType and returns a Column.
        """
        from inspect import signature
        exprs = []
        num_args = len(signature(sfun).parameters)
        for col in self.columns:
            if num_args == 1:
                # Only pass in the column if sfun accepts only one arg
                exprs.append(sfun(self._sdf[col]).alias(col))
            else:  # must be 2
                assert num_args == 2
                # Pass in both the column and its data type if sfun accepts two args
                exprs.append(sfun(self._sdf[col], self._sdf.schema[col].dataType).alias(col))

        sdf = self._sdf.select(*exprs)
        pdf = sdf.toPandas()
        assert len(pdf) == 1, (sdf, pdf)
        row = pdf.iloc[0]
        row.name = None
        return row  # Return first row as a Series

    def corr(self, method='pearson'):
        """
        Compute pairwise correlation of columns, excluding NA/null values.

        Parameters
        ----------
        method : {'pearson', 'spearman'}
            * pearson : standard correlation coefficient
            * spearman : Spearman rank correlation

        Returns
        -------
        y : pandas.DataFrame

        See Also
        --------
        Series.corr

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df.corr('pearson')
                  dogs      cats
        dogs  1.000000 -0.851064
        cats -0.851064  1.000000

        >>> df.corr('spearman')
                  dogs      cats
        dogs  1.000000 -0.948683
        cats -0.948683  1.000000

        Notes
        -----
        There are behavior differences between Koalas and pandas.

        * the `method` argument only accepts 'pearson', 'spearman'
        * the data should not contain NaNs. Koalas will return an error.
        * Koalas doesn't support the following argument(s).

          * `min_periods` argument is not supported
        """
        return corr(self, method)

    @derived_from(pd.DataFrame)
    def iteritems(self):
        cols = list(self.columns)
        return list((col_name, self[col_name]) for col_name in cols)

    def to_html(self, buf=None, columns=None, col_space=None, header=True, index=True,
                na_rep='NaN', formatters=None, float_format=None, sparsify=None, index_names=True,
                justify=None, max_rows=None, max_cols=None, show_dimensions=False, decimal='.',
                bold_rows=True, classes=None, escape=True, notebook=False, border=None,
                table_id=None, render_links=False):
        """
        Render a DataFrame as an HTML table.

        .. note:: This method should only be used if the resulting Pandas DataFrame is expected
            to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        buf : StringIO-like, optional
            Buffer to write to.
        columns : sequence, optional, default None
            The subset of columns to write. Writes all columns by default.
        col_space : int, optional
            The minimum width of each column.
        header : bool, optional
            %(header)s.
        index : bool, optional, default True
            Whether to print index (row) labels.
        na_rep : str, optional, default 'NaN'
            String representation of NAN to use.
        formatters : list or dict of one-param. functions, optional
            Formatter functions to apply to columns' elements by position or
            name.
            The result of each function must be a unicode string.
            List must be of length equal to the number of columns.
        float_format : one-parameter function, optional, default None
            Formatter function to apply to columns' elements if they are
            floats. The result of this function must be a unicode string.
        sparsify : bool, optional, default True
            Set to False for a DataFrame with a hierarchical index to print
            every multiindex key at each row.
        index_names : bool, optional, default True
            Prints the names of the indexes.
        justify : str, default None
            How to justify the column labels. If None uses the option from
            the print configuration (controlled by set_option), 'right' out
            of the box. Valid values are

            * left
            * right
            * center
            * justify
            * justify-all
            * start
            * end
            * inherit
            * match-parent
            * initial
            * unset.
        max_rows : int, optional
            Maximum number of rows to display in the console.
        max_cols : int, optional
            Maximum number of columns to display in the console.
        show_dimensions : bool, default False
            Display DataFrame dimensions (number of rows by number of columns).
        decimal : str, default '.'
            Character recognized as decimal separator, e.g. ',' in Europe.
        bold_rows : bool, default True
            Make the row labels bold in the output.
        classes : str or list or tuple, default None
            CSS class(es) to apply to the resulting html table.
        escape : bool, default True
            Convert the characters <, >, and & to HTML-safe sequences.
        notebook : {True, False}, default False
            Whether the generated HTML is for IPython Notebook.
        border : int
            A ``border=border`` attribute is included in the opening
            `<table>` tag. Default ``pd.options.html.border``.
        table_id : str, optional
            A css id is included in the opening `<table>` tag if specified.
        render_links : bool, default False
            Convert URLs to HTML links.

        Returns
        -------
        str (or unicode, depending on data and options)
            String representation of the dataframe.

        See Also
        --------
        to_string : Convert DataFrame to a string.
        """
        if max_rows is not None:
            kdf = self.head(max_rows)
        else:
            kdf = self

        return kdf.to_pandas().to_html(
            buf=buf, columns=columns, col_space=col_space, header=header, index=index,
            na_rep=na_rep, formatters=formatters, float_format=float_format, sparsify=sparsify,
            index_names=index_names, justify=justify, max_rows=max_rows, max_cols=max_cols,
            show_dimensions=show_dimensions, decimal=decimal, bold_rows=bold_rows, classes=classes,
            escape=escape, notebook=notebook, border=border, table_id=table_id,
            render_links=render_links)

    @property
    def index(self):
        """The index (row labels) Column of the DataFrame.

        Currently supported only when the DataFrame has a single index.
        """
        from databricks.koalas.series import Series
        if len(self._metadata.index_info) != 1:
            raise KeyError('Currently supported only when the DataFrame has a single index.')
        return Series(self._index_columns[0], self, [])

    def set_index(self, keys, drop=True, append=False, inplace=False):
        """Set the DataFrame index (row labels) using one or more existing columns. By default
        yields a new object.

        :param keys: column label or list of column labels / arrays
        :param drop: boolean, default True
                     Delete columns to be used as the new index
        :param append: boolean, default False
                       Whether to append columns to existing index
        :param inplace: boolean, default False
                        Modify the DataFrame in place (do not create a new object)
        :return: :class:`DataFrame`
        """
        if isinstance(keys, string_types):
            keys = [keys]
        else:
            keys = list(keys)
        for key in keys:
            if key not in self.columns:
                raise KeyError(key)

        if drop:
            columns = [column for column in self._metadata.column_fields if column not in keys]
        else:
            columns = self._metadata.column_fields
        if append:
            index_info = self._metadata.index_info + [(column, column) for column in keys]
        else:
            index_info = [(column, column) for column in keys]

        metadata = self._metadata.copy(column_fields=columns, index_info=index_info)
        if inplace:
            self._metadata = metadata
        else:
            kdf = self.copy()
            kdf._metadata = metadata
            return kdf

    def reset_index(self, level=None, drop=False, inplace=False):
        """For DataFrame with multi-level index, return new DataFrame with labeling information in
        the columns under the index names, defaulting to 'level_0', 'level_1', etc. if any are None.
        For a standard index, the index name will be used (if set), otherwise a default 'index' or
        'level_0' (if 'index' is already taken) will be used.

        :param level: int, str, tuple, or list, default None
                      Only remove the given levels from the index. Removes all levels by default
        :param drop: boolean, default False
                     Do not try to insert index into dataframe columns. This resets the index to the
                     default integer index.
        :param inplace: boolean, default False
                        Modify the DataFrame in place (do not create a new object)
        :return: :class:`DataFrame`
        """
        if len(self._metadata.index_info) == 0:
            raise NotImplementedError('Can\'t reset index because there is no index.')

        multi_index = len(self._metadata.index_info) > 1
        if multi_index:
            rename = lambda i: 'level_{}'.format(i)
        else:
            rename = lambda i: \
                'index' if 'index' not in self._metadata.column_fields else 'level_{}'.fomat(i)

        if level is None:
            index_columns = [(column, name if name is not None else rename(i))
                             for i, (column, name) in enumerate(self._metadata.index_info)]
            index_info = []
        else:
            if isinstance(level, (int, string_types)):
                level = [level]
            level = list(level)

            if all(isinstance(l, int) for l in level):
                for l in level:
                    if l >= len(self._metadata.index_info):
                        raise IndexError('Too many levels: Index has only {} level, not {}'
                                         .format(len(self._metadata.index_info), l + 1))
                idx = level
            elif all(isinstance(l, string_types) for l in level):
                idx = []
                for l in level:
                    try:
                        i = self._metadata.index_fields.index(l)
                        idx.append(i)
                    except ValueError:
                        if multi_index:
                            raise KeyError('Level unknown not found')
                        else:
                            raise KeyError('Level unknown must be same as name ({})'
                                           .format(self._metadata.index_fields[0]))
            else:
                raise ValueError('Level should be all int or all string.')
            idx.sort()

            index_columns = []
            index_info = self._metadata.index_info.copy()
            for i in idx:
                info = self._metadata.index_info[i]
                column_field, index_name = info
                index_columns.append((column_field,
                                      index_name if index_name is not None else rename(index_name)))
                index_info.remove(info)

        if drop:
            index_columns = []

        metadata = self._metadata.copy(
            column_fields=[column for column, _ in index_columns] + self._metadata.column_fields,
            index_info=index_info)
        columns = [name for _, name in index_columns] + self._metadata.column_fields
        if inplace:
            self._metadata = metadata
            self.columns = columns
        else:
            kdf = self.copy()
            kdf._metadata = metadata
            kdf.columns = columns
            return kdf

    @derived_from(pd.DataFrame)
    def isnull(self):
        kdf = self.copy()
        for name, ks in kdf.iteritems():
            kdf[name] = ks.isnull()
        return kdf

    isna = isnull

    @derived_from(pd.DataFrame)
    def notnull(self):
        kdf = self.copy()
        for name, ks in kdf.iteritems():
            kdf[name] = ks.notnull()
        return kdf

    notna = notnull

    def to_koalas(self):
        """
        Converts the existing DataFrame into a Koalas DataFrame.

        This method is monkey-patched into Spark's DataFrame and can be used
        to convert a Spark DataFrame into a Koalas DataFrame. If running on
        an existing Koalas DataFrame, the method returns itself.

        If a Koalas DataFrame is converted to a Spark DataFrame and then back
        to Koalas, it will lose the index information and the original index
        will be turned into a normal column.

        See Also
        --------
        DataFrame.to_spark

        Examples
        --------
        >>> df = ks.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df
           col1  col2
        0     1     3
        1     2     4

        >>> spark_df = df.to_spark()
        >>> spark_df
        DataFrame[__index_level_0__: bigint, col1: bigint, col2: bigint]

        >>> kdf = spark_df.to_koalas()
        >>> kdf
           __index_level_0__  col1  col2
        0                  0     1     3
        1                  1     2     4
        """
        if isinstance(self, DataFrame):
            return self
        else:
            return DataFrame(self)

    def to_spark(self):
        """
        Return the current DataFrame as a Spark DataFrame.

        See Also
        --------
        DataFrame.to_koalas
        """
        return self._sdf

    def to_pandas(self):
        """
        Return a Pandas DataFrame.

        .. note:: This method should only be used if the resulting Pandas DataFrame is expected
            to be small, as all the data is loaded into the driver's memory.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'])
        >>> df.to_pandas()
           dogs  cats
        0   0.2   0.3
        1   0.0   0.6
        2   0.6   0.0
        3   0.2   0.1
        """
        sdf = self._sdf.select(['`{}`'.format(name) for name in self._metadata.all_fields])
        pdf = sdf.toPandas()
        if len(pdf) == 0 and len(sdf.schema) > 0:
            # TODO: push to OSS
            pdf = pdf.astype({field.name: to_arrow_type(field.dataType).to_pandas_dtype()
                              for field in sdf.schema})
        if len(self._metadata.index_info) > 0:
            append = False
            for index_field in self._metadata.index_fields:
                drop = index_field not in self._metadata.column_fields
                pdf = pdf.set_index(index_field, drop=drop, append=append)
                append = True
            pdf = pdf[self._metadata.column_fields]
        index_names = self._metadata.index_names
        if len(index_names) > 0:
            if isinstance(pdf.index, pd.MultiIndex):
                pdf.index.names = index_names
            else:
                pdf.index.name = index_names[0]
        return pdf

    # Alias to maintain backward compatibility with Spark
    toPandas = to_pandas

    @derived_from(pd.DataFrame)
    def assign(self, **kwargs):
        from databricks.koalas.series import Series
        for k, v in kwargs.items():
            if not (isinstance(v, (Series, spark.Column)) or
                    callable(v) or pd.api.types.is_scalar(v)):
                raise TypeError("Column assignment doesn't support type "
                                "{0}".format(type(v).__name__))
            if callable(v):
                kwargs[k] = v(self)

        pairs = list(kwargs.items())
        sdf = self._sdf
        for (name, c) in pairs:
            if isinstance(c, Series):
                sdf = sdf.withColumn(name, c._scol)
            elif isinstance(c, Column):
                sdf = sdf.withColumn(name, c)
            else:
                sdf = sdf.withColumn(name, F.lit(c))

        metadata = self._metadata.copy(
            column_fields=(self._metadata.column_fields +
                           [name for name, _ in pairs if name not in self._metadata.column_fields]))
        return DataFrame(sdf, metadata)

    @property
    def loc(self):
        return SparkDataFrameLocator(self)

    def copy(self):
        return DataFrame(self._sdf, self._metadata.copy())

    @derived_from(pd.DataFrame)
    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False):
        if axis == 0 or axis == 'index':
            if subset is not None:
                if isinstance(subset, string_types):
                    columns = [subset]
                else:
                    columns = list(subset)
                invalids = [column for column in columns
                            if column not in self._metadata.column_fields]
                if len(invalids) > 0:
                    raise KeyError(invalids)
            else:
                columns = list(self.columns)

            cnt = reduce(lambda x, y: x + y,
                         [F.when(self[column].notna()._scol, 1).otherwise(0)
                          for column in columns],
                         F.lit(0))
            if thresh is not None:
                pred = cnt >= F.lit(int(thresh))
            elif how == 'any':
                pred = cnt == F.lit(len(columns))
            elif how == 'all':
                pred = cnt > F.lit(0)
            else:
                if how is not None:
                    raise ValueError('invalid how option: {h}'.format(h=how))
                else:
                    raise TypeError('must specify how or thresh')

            sdf = self._sdf.filter(pred)
            if inplace:
                self._sdf = sdf
            else:
                return DataFrame(sdf, self._metadata.copy())

        else:
            raise NotImplementedError("dropna currently only works for axis=0 or axis='index'")

    def head(self, n=5):
        """
        Return the first `n` rows.

        This function returns the first `n` rows for the object based
        on position. It is useful for quickly testing if your object
        has the right type of data in it.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        obj_head : same type as caller
            The first `n` rows of the caller object.

        Examples
        --------
        >>> df = ks.DataFrame({'animal':['alligator', 'bee', 'falcon', 'lion',
        ...                    'monkey', 'parrot', 'shark', 'whale', 'zebra']})
        >>> df
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey
        5     parrot
        6      shark
        7      whale
        8      zebra

        Viewing the first 5 lines

        >>> df.head()
              animal
        0  alligator
        1        bee
        2     falcon
        3       lion
        4     monkey

        Viewing the first `n` lines (three in this case)

        >>> df.head(3)
              animal
        0  alligator
        1        bee
        2     falcon
        """

        return DataFrame(self._sdf.limit(n), self._metadata.copy())

    @property
    def columns(self):
        """The column labels of the DataFrame."""
        return pd.Index(self._metadata.column_fields)

    @columns.setter
    def columns(self, names):
        old_names = self._metadata.column_fields
        if len(old_names) != len(names):
            raise ValueError(
                "Length mismatch: Expected axis has %d elements, new values have %d elements"
                % (len(old_names), len(names)))
        sdf = self._sdf.select(self._metadata.index_fields +
                               [self[old_name]._scol.alias(new_name)
                                for (old_name, new_name) in zip(old_names, names)])
        self._sdf = sdf
        self._metadata = self._metadata.copy(column_fields=names)

    @property
    def dtypes(self):
        """Return the dtypes in the DataFrame.

        This returns a Series with the data type of each column. The result's index is the original
        DataFrame's columns. Columns with mixed types are stored with the object dtype.

        :return: :class:`pd.Series` The data type of each column.

        Examples
        --------
        >>> df = ks.DataFrame({'a': list('abc'),
        ...                    'b': list(range(1, 4)),
        ...                    'c': np.arange(3, 6).astype('i1'),
        ...                    'd': np.arange(4.0, 7.0, dtype='float64'),
        ...                    'e': [True, False, True],
        ...                    'f': pd.date_range('20130101', periods=3)})
        >>> df.dtypes
        a            object
        b             int64
        c              int8
        d           float64
        e              bool
        f    datetime64[ns]
        dtype: object
        """
        return pd.Series([self[col].dtype for col in self._metadata.column_fields],
                         index=self._metadata.column_fields)

    def count(self):
        """
        Count non-NA cells for each column.

        The values `None`, `NaN` are considered NA.

        Returns
        -------
        pandas.Series

        See Also
        --------
        Series.count: Number of non-NA elements in a Series.
        DataFrame.shape: Number of DataFrame rows and columns (including NA
            elements).
        DataFrame.isna: Boolean same-sized DataFrame showing places of NA
            elements.

        Examples
        --------
        Constructing DataFrame from a dictionary:

        >>> df = ks.DataFrame({"Person":
        ...                    ["John", "Myla", "Lewis", "John", "Myla"],
        ...                    "Age": [24., np.nan, 21., 33, 26],
        ...                    "Single": [False, True, True, True, False]})
        >>> df
          Person   Age  Single
        0   John  24.0   False
        1   Myla   NaN    True
        2  Lewis  21.0    True
        3   John  33.0    True
        4   Myla  26.0   False

        Notice the uncounted NA values:

        >>> df.count()
        Person    5
        Age       4
        Single    5
        dtype: int64
        """
        return self._reduce_for_stat_function(_Frame._count_expr)

    def unique(self):
        sdf = self._sdf
        return DataFrame(spark.DataFrame(sdf._jdf.distinct(), sdf.sql_ctx), self._metadata.copy())

    @derived_from(pd.DataFrame)
    def drop(self, labels, axis=0, errors='raise'):
        axis = self._validate_axis(axis)
        if axis == 1:
            if isinstance(labels, list):
                sdf = self._sdf.drop(*labels)
                metadata = self._metadata.copy(
                    column_fields=[column for column in self._metadata.column_fields
                                   if column not in labels])
            else:
                sdf = self._sdf.drop(labels)
                metadata = self._metadata.copy(
                    column_fields=[column for column in self._metadata.column_fields
                                   if column != labels])
            return DataFrame(sdf, metadata)
        raise NotImplementedError("Drop currently only works for axis=1")

    @derived_from(pd.DataFrame)
    def get(self, key, default=None):
        try:
            return self._pd_getitem(key)
        except (KeyError, ValueError, IndexError):
            return default

    def sort_values(self, by):
        return DataFrame(self._sdf.sort(by), self._metadata.copy())

    def groupby(self, by):
        from databricks.koalas.groups import PandasLikeGroupBy
        gp = self._sdf.groupby(by)
        return PandasLikeGroupBy(self, gp, None)

    @derived_from(pd.DataFrame)
    def pipe(self, func, *args, **kwargs):
        # Taken from pandas:
        # https://github.com/pydata/pandas/blob/master/pandas/core/generic.py#L2698-L2707
        if isinstance(func, tuple):
            func, target = func
            if target in kwargs:
                raise ValueError('%s is both the pipe target and a keyword '
                                 'argument' % target)
            kwargs[target] = self
            return func(*args, **kwargs)
        else:
            return func(self, *args, **kwargs)

    @property
    def shape(self):
        """
        Return a tuple representing the dimensionality of the DataFrame.

        Examples
        --------
        >>> df = ks.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        >>> df.shape
        (2, 2)

        >>> df = ks.DataFrame({'col1': [1, 2], 'col2': [3, 4],
        ...                    'col3': [5, 6]})
        >>> df.shape
        (2, 3)
        """
        return len(self), len(self.columns)

    def _pd_getitem(self, key):
        from databricks.koalas.series import Series
        if key is None:
            raise KeyError("none key")
        if isinstance(key, string_types):
            try:
                return Series(self._sdf.__getitem__(key), self, self._metadata.index_info)
            except AnalysisException:
                raise KeyError(key)
        if np.isscalar(key) or isinstance(key, (tuple, string_types)):
            raise NotImplementedError(key)
        elif isinstance(key, slice):
            return self.loc[key]

        if isinstance(key, (pd.Series, np.ndarray, pd.Index)):
            raise NotImplementedError(key)
        if isinstance(key, list):
            return self.loc[:, key]
        if isinstance(key, DataFrame):
            # TODO Should not implement alignment, too dangerous?
            return Series(self._sdf.__getitem__(key), self, self._metadata.index_info)
        if isinstance(key, Series):
            # TODO Should not implement alignment, too dangerous?
            # It is assumed to be only a filter, otherwise .loc should be used.
            bcol = key._scol.cast("boolean")
            return DataFrame(self._sdf.filter(bcol), self._metadata.copy())
        raise NotImplementedError(key)

    def __repr__(self):
        return repr(self.toPandas())

    def __getitem__(self, key):
        return self._pd_getitem(key)

    def __setitem__(self, key, value):
        from databricks.koalas.series import Series
        # For now, we don't support realignment against different dataframes.
        # This is too expensive in Spark.
        # Are we assigning against a column?
        if isinstance(value, Series):
            assert value._kdf is self, \
                "Cannot combine column argument because it comes from a different dataframe"
        if isinstance(key, (tuple, list)):
            assert isinstance(value.schema, StructType)
            field_names = value.schema.fieldNames()
            kdf = self.assign(**{k: value[c] for k, c in zip(key, field_names)})
        else:
            kdf = self.assign(**{key: value})

        self._sdf: spark.DataFrame = kdf._sdf
        self._metadata = kdf._metadata

    def __getattr__(self, key):
        from databricks.koalas.series import Series
        if key.startswith("__") or key.startswith("_pandas_") or key.startswith("_spark_"):
            raise AttributeError(key)
        if hasattr(_MissingPandasLikeDataFrame, key):
            return partial(getattr(_MissingPandasLikeDataFrame, key), self)
        return Series(self._sdf.__getattr__(key), self, self._metadata.index_info)

    def __iter__(self):
        return self.toPandas().__iter__()

    def __len__(self):
        return self._sdf.count()

    def __dir__(self):
        fields = [f for f in self._sdf.schema.fieldNames() if ' ' not in f]
        return super(DataFrame, self).__dir__() + fields

    def _repr_html_(self):
        return self.head(max_display_count).toPandas()._repr_html_()

    @classmethod
    def _validate_axis(cls, axis=0):
        if axis not in (0, 1, 'index', 'columns', None):
            raise ValueError('No axis named {0}'.format(axis))
        # convert to numeric axis
        return {None: 0, 'index': 0, 'columns': 1}.get(axis, axis)


def _reduce_spark_multi(sdf, aggs):
    """
    Performs a reduction on a dataframe, the functions being known sql aggregate functions.
    """
    assert isinstance(sdf, spark.DataFrame)
    sdf0 = sdf.agg(*aggs)
    l = sdf0.head(2)
    assert len(l) == 1, (sdf, l)
    row = l[0]
    l2 = list(row)
    assert len(l2) == len(aggs), (row, l2)
    return l2
