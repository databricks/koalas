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
Base classes to be monkey-patched to DataFrame/Column to behave similar to pandas DataFrame/Series.
"""
from functools import reduce

import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Column
from pyspark.sql.types import FloatType, DoubleType, StructType, to_arrow_type
from pyspark.sql.utils import AnalysisException

from .metadata import Metadata
from .selection import SparkDataFrameLocator
from ._dask_stubs.utils import derived_from
from ._dask_stubs.compatibility import string_types


__all__ = ['PandasLikeSeries', 'PandasLikeDataFrame', 'SparkSessionPatches', 'anchor_wrap']

max_display_count = 1000


class SparkSessionPatches(object):
    """
    Methods for :class:`SparkSession`.
    """

    def from_pandas(self, pdf):
        if isinstance(pdf, pd.Series):
            return _col(self.from_pandas(pd.DataFrame(pdf)))
        metadata = Metadata.from_pandas(pdf)
        reset_index = pdf.reset_index()
        reset_index.columns = metadata.all_fields
        df = self.createDataFrame(reset_index)
        df._metadata = metadata
        return df

    def read_csv(self, path, header='infer', names=None, usecols=None,
                 mangle_dupe_cols=True, parse_dates=False, comment=None):
        if mangle_dupe_cols is not True:
            raise ValueError("mangle_dupe_cols can only be `True`: %s" % mangle_dupe_cols)
        if parse_dates is not False:
            raise ValueError("parse_dates can only be `False`: %s" % parse_dates)

        if usecols is not None and not callable(usecols):
            usecols = list(usecols)
        if usecols is None or callable(usecols) or len(usecols) > 0:
            reader = self.read.option("inferSchema", "true")

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

            df = reader.csv(path)

            if header is None:
                df = df._spark_selectExpr(*["`%s` as `%s`" % (field.name, i)
                                            for i, field in enumerate(df.schema)])
            if names is not None:
                names = list(names)
                if len(set(names)) != len(names):
                    raise ValueError('Found non-unique column index')
                if len(names) != len(df.schema):
                    raise ValueError('Names do not match the number of columns: %d' % len(names))
                df = df._spark_selectExpr(*["`%s` as `%s`" % (field.name, name)
                                            for field, name in zip(df.schema, names)])

            if usecols is not None:
                if callable(usecols):
                    cols = [field.name for field in df.schema if usecols(field.name)]
                    missing = []
                elif all(isinstance(col, int) for col in usecols):
                    cols = [field.name for i, field in enumerate(df.schema) if i in usecols]
                    missing = [col for col in usecols
                               if col >= len(df.schema) or df.schema[col].name not in cols]
                elif all(isinstance(col, string_types) for col in usecols):
                    cols = [field.name for field in df.schema if field.name in usecols]
                    missing = [col for col in usecols if col not in cols]
                else:
                    raise ValueError("'usecols' must either be list-like of all strings, "
                                     "all unicode, all integers or a callable.")
                if len(missing) > 0:
                    raise ValueError('Usecols do not match columns, columns expected but not '
                                     'found: %s' % missing)

                if len(cols) > 0:
                    df = df._spark_select(cols)
                else:
                    df = self.createDataFrame([], schema=StructType())
        else:
            df = self.createDataFrame([], schema=StructType())
        return df

    def read_parquet(self, path, columns=None):
        if columns is not None:
            columns = list(columns)
        if columns is None or len(columns) > 0:
            df = self.read.parquet(path)
            if columns is not None:
                fields = [field.name for field in df.schema]
                cols = [col for col in columns if col in fields]
                if len(cols) > 0:
                    df = df._spark_select(cols)
                else:
                    df = self.createDataFrame([], schema=StructType())
        else:
            df = self.createDataFrame([], schema=StructType())
        return df


class _Frame(object):
    """
    The base class for both dataframes and series.
    """

    def max(self):
        return _reduce_spark(self, F.max)

    @derived_from(pd.DataFrame)
    def abs(self):
        """
        Return a Series/DataFrame with absolute numeric value of each element.

        :return: :class:`Series` or :class:`DataFrame` with the absolute value of each element.
        """
        return _spark_col_apply(self, F.abs)

    def compute(self):
        """Alias of `toPandas()` to mimic dask for easily porting tests."""
        return self.toPandas()


class PandasLikeSeries(_Frame):
    """
    Methods that are appropriate for distributed series.
    """

    def __init__(self):
        """ Define additional private fields.

        * ``_pandas_metadata``: The metadata which stores column fields, and index fields and names.
        * ``_spark_ref_dataframe``: The reference to DataFraem anchored to this Column.
        * ``_pandas_schema``: The schema when representing this Column as a DataFrame.
        """
        self._pandas_metadata = None
        self._spark_ref_dataframe = None
        self._pandas_schema = None

    @property
    def dtype(self):
        from .typing import as_python_type
        return as_python_type(self.schema.fields[-1].dataType)

    def astype(self, tpe):
        from .typing import as_spark_type
        spark_type = as_spark_type(tpe)
        if not spark_type:
            raise ValueError("Type {} not understood".format(tpe))
        return anchor_wrap(self, self._spark_cast(spark_type))

    def getField(self, name):
        if not isinstance(self.schema, StructType):
            raise AttributeError("Not a struct: {}".format(self.schema))
        else:
            fnames = self.schema.fieldNames()
            if name not in fnames:
                raise AttributeError(
                    "Field {} not found, possible values are {}".format(name, ", ".join(fnames)))
            return anchor_wrap(self, self._spark_getField(name))

    # TODO: automate the process here
    def alias(self, name):
        return self.rename(name)

    @property
    def schema(self):
        if not hasattr(self, '_pandas_schema') or self._pandas_schema is None:
            self._pandas_schema = self.to_dataframe().schema
        return self._pandas_schema

    @property
    def shape(self):
        return len(self),

    @property
    def name(self):
        return self._metadata.column_fields[0]

    @name.setter
    def name(self, name):
        self.rename(name, inplace=True)

    def rename(self, name, inplace=False):
        col = self._spark_alias(name)
        if inplace:
            self._jc = col._jc
            self._pandas_schema = None
            self._pandas_metadata = None
            return self
        else:
            return anchor_wrap(self, col)

    @property
    def _metadata(self):
        if not hasattr(self, '_pandas_metadata') or self._pandas_metadata is None:
            self._pandas_metadata = self.to_dataframe()._metadata
        return self._pandas_metadata

    def _set_metadata(self, metadata):
        self._pandas_metadata = metadata

    @property
    def index(self):
        """The index (axis labels) Column of the Series.

        Currently supported only when the DataFrame has a single index.
        """
        if len(self._metadata.index_info) != 1:
            raise KeyError('Currently supported only when the Column has a single index.')
        return self._pandas_anchor.index

    @derived_from(pd.Series)
    def reset_index(self, level=None, drop=False, name=None, inplace=False):
        if inplace and not drop:
            raise TypeError('Cannot reset_index inplace on a Series to create a DataFrame')

        if name is not None:
            df = self.rename(name).to_dataframe()
        else:
            df = self.to_dataframe()
        df = df.reset_index(level=level, drop=drop)
        if drop:
            col = _col(df)
            if inplace:
                anchor_wrap(col, self)
                self._jc = col._jc
                self._pandas_schema = None
                self._pandas_metadata = None
            else:
                return col
        else:
            return df

    @property
    def loc(self):
        return SparkDataFrameLocator(self)

    def to_dataframe(self):
        ref = self._pandas_anchor
        df = ref._spark_select(ref._metadata.index_fields + [self])
        df._metadata = ref._metadata.copy(column_fields=df._metadata.column_fields[-1:])
        return df

    def toPandas(self):
        return _col(self.to_dataframe().toPandas())

    @derived_from(pd.Series)
    def isnull(self):
        if isinstance(self.schema[self.name].dataType, (FloatType, DoubleType)):
            return anchor_wrap(self, self._spark_isNull() | F._spark_isnan(self))
        else:
            return anchor_wrap(self, self._spark_isNull())

    isna = isnull

    @derived_from(pd.Series)
    def notnull(self):
        return ~self.isnull()

    notna = notnull

    @derived_from(pd.Series)
    def dropna(self, axis=0, inplace=False, **kwargs):
        col = _col(self.to_dataframe().dropna(axis=axis, inplace=False))
        if inplace:
            anchor_wrap(col, self)
            self._jc = col._jc
            self._pandas_schema = None
            self._pandas_metadata = None
        else:
            return col

    def head(self, n=5):
        return _col(self.to_dataframe().head(n))

    def unique(self):
        # Pandas wants a series/array-like object
        return _col(self.to_dataframe().unique())

    @derived_from(pd.Series)
    def value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
        if bins is not None:
            raise NotImplementedError("value_counts currently does not support bins")

        if dropna:
            df_dropna = self.to_dataframe()._spark_filter(self.notna())
        else:
            df_dropna = self.to_dataframe()
        df = df_dropna._spark_groupby(self).count()
        if sort:
            if ascending:
                df = df._spark_orderBy(F._spark_col('count'))
            else:
                df = df._spark_orderBy(F._spark_col('count')._spark_desc())

        if normalize:
            sum = df_dropna._spark_count()
            df = df._spark_withColumn('count', F._spark_col('count') / F._spark_lit(sum))

        index_name = 'index' if self.name != 'index' else 'level_0'
        df.columns = [index_name, self.name]
        df._metadata = Metadata(column_fields=[self.name], index_info=[(index_name, None)])
        return _col(df)

    @property
    def _pandas_anchor(self) -> DataFrame:
        """
        The anchoring dataframe for this column (if any).
        :return:
        """
        if hasattr(self, "_spark_ref_dataframe"):
            return self._spark_ref_dataframe
        n = self._pandas_orig_repr()
        raise ValueError("No reference to a dataframe for column {}".format(n))

    def __len__(self):
        return len(self.to_dataframe())

    def __getitem__(self, key):
        return anchor_wrap(self, self._spark_getitem(key))

    def __getattr__(self, item):
        if item.startswith("__") or item.startswith("_pandas_") or item.startswith("_spark_"):
            raise AttributeError(item)
        return anchor_wrap(self, self.getField(item))

    def __invert__(self):
        return anchor_wrap(self, self.astype(bool) == F._spark_lit(False))

    def __str__(self):
        return self._pandas_orig_repr()

    def __repr__(self):
        return repr(self.head(max_display_count).toPandas())

    def __dir__(self):
        if not isinstance(self.schema, StructType):
            fields = []
        else:
            fields = [f for f in self.schema.fieldNames() if ' ' not in f]
        return super(Column, self).__dir__() + fields

    def _pandas_orig_repr(self):
        # TODO: figure out how to reuse the original one.
        return 'Column<%s>' % self._jc.toString().encode('utf8')


class PandasLikeDataFrame(_Frame):
    """
    Methods that are relevant to dataframes.
    """

    def __init__(self):
        """ Define additional private fields.

        * ``_pandas_metadata``: The metadata which stores column fields, and index fields and names.
        """
        self._pandas_metadata = None

    @property
    def _metadata(self):
        if not hasattr(self, '_pandas_metadata') or self._pandas_metadata is None:
            self._pandas_metadata = Metadata(column_fields=self.schema.fieldNames())
        return self._pandas_metadata

    @_metadata.setter
    def _metadata(self, metadata):
        self._pandas_metadata = metadata

    @property
    def _index_columns(self):
        return [anchor_wrap(self, self._spark_getitem(field))
                for field in self._metadata.index_fields]

    @derived_from(pd.DataFrame)
    def iteritems(self):
        cols = list(self.columns)
        return list((col_name, self[col_name]) for col_name in cols)

    @derived_from(pd.DataFrame)
    def to_html(self, index=True, classes=None):
        return self.toPandas().to_html(index=index, classes=classes)

    @property
    def index(self):
        """The index (row labels) Column of the DataFrame.

        Currently supported only when the DataFrame has a single index.
        """
        if len(self._metadata.index_info) != 1:
            raise KeyError('Currently supported only when the DataFrame has a single index.')
        col = self._index_columns[0]
        col._set_metadata(col._metadata.copy(index_info=[]))
        return col

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
            df = self.copy()
            df._metadata = metadata
            return df

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
            df = self.copy()
            df._metadata = metadata
            df.columns = columns
            return df

    @derived_from(pd.DataFrame)
    def isnull(self):
        df = self.copy()
        for name, col in df.iteritems():
            df[name] = col.isnull()
        return df

    isna = isnull

    @derived_from(pd.DataFrame)
    def notnull(self):
        df = self.copy()
        for name, col in df.iteritems():
            df[name] = col.notnull()
        return df

    notna = notnull

    @derived_from(DataFrame)
    def toPandas(self):
        df = self._spark_select(['`{}`'.format(name) for name in self._metadata.all_fields])
        pdf = df._spark_toPandas()
        if len(pdf) == 0 and len(df.schema) > 0:
            # TODO: push to OSS
            pdf = pdf.astype({field.name: to_arrow_type(field.dataType).to_pandas_dtype()
                              for field in df.schema})
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

    @derived_from(pd.DataFrame)
    def assign(self, **kwargs):
        for k, v in kwargs.items():
            if not (isinstance(v, (Column,)) or
                    callable(v) or pd.api.types.is_scalar(v)):
                raise TypeError("Column assignment doesn't support type "
                                "{0}".format(type(v).__name__))
            if callable(v):
                kwargs[k] = v(self)

        pairs = list(kwargs.items())
        df = self
        for (name, c) in pairs:
            df = df._spark_withColumn(name, c)
        df._metadata = self._metadata.copy(
            column_fields=(self._metadata.column_fields +
                           [name for name, _ in pairs if name not in self._metadata.column_fields]))
        return df

    @property
    def loc(self):
        return SparkDataFrameLocator(self)

    def copy(self):
        df = DataFrame(self._jdf, self.sql_ctx)
        df._metadata = self._metadata.copy()
        return df

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
                         [F._spark_when(self[column].notna(), 1)._spark_otherwise(0)
                          for column in columns],
                         F._spark_lit(0))
            if thresh is not None:
                pred = cnt >= F._spark_lit(int(thresh))
            elif how == 'any':
                pred = cnt == F._spark_lit(len(columns))
            elif how == 'all':
                pred = cnt > F._spark_lit(0)
            else:
                if how is not None:
                    raise ValueError('invalid how option: {h}'.format(h=how))
                else:
                    raise TypeError('must specify how or thresh')

            df = self._spark_filter(pred)
            df._metadata = self._metadata.copy()
            if inplace:
                _reassign_jdf(self, df)
            else:
                return df

        else:
            raise NotImplementedError("dropna currently only works for axis=0 or axis='index'")

    def head(self, n=5):
        df = self._spark_limit(n)
        df._metadata = self._metadata.copy()
        return df

    @property
    def columns(self):
        return pd.Index(self._metadata.column_fields)

    @columns.setter
    def columns(self, names):
        old_names = self._metadata.column_fields
        if len(old_names) != len(names):
            raise ValueError(
                "Length mismatch: Expected axis has %d elements, new values have %d elements"
                % (len(old_names), len(names)))
        df = self._spark_select(self._metadata.index_fields +
                                [self[old_name]._spark_alias(new_name)
                                 for (old_name, new_name) in zip(old_names, names)])
        df._metadata = self._metadata.copy(column_fields=names)

        _reassign_jdf(self, df)

    def count(self):
        return self._spark_count()

    def unique(self):
        return DataFrame(self._jdf.distinct(), self.sql_ctx)

    @derived_from(pd.DataFrame)
    def drop(self, labels, axis=0, errors='raise'):
        axis = self._validate_axis(axis)
        if axis == 1:
            if isinstance(labels, list):
                df = self._spark_drop(*labels)
                df._metadata = self._metadata.copy(
                    column_fields=[column for column in self._metadata.column_fields
                                   if column not in labels])
            else:
                df = self._spark_drop(labels)
                df._metadata = self._metadata.copy(
                    column_fields=[column for column in self._metadata.column_fields
                                   if column != labels])
            return df
            # return self.map_partitions(M.drop, labels, axis=axis, errors=errors)
        raise NotImplementedError("Drop currently only works for axis=1")

    @derived_from(pd.DataFrame)
    def get(self, key, default=None):
        try:
            return anchor_wrap(self, self._pd_getitem(key))
        except (KeyError, ValueError, IndexError):
            return default

    def sort_values(self, by):
        df = self._spark_sort(by)
        df._metadata = self._metadata
        return df

    def groupby(self, by):
        gp = self._spark_groupby(by)
        from .groups import PandasLikeGroupBy
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
        return len(self), len(self.columns)

    def _pd_getitem(self, key):
        if key is None:
            raise KeyError("none key")
        if isinstance(key, string_types):
            try:
                return self._spark_getitem(key)
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
            return self._spark_getitem(key)
        if isinstance(key, Column):
            # TODO Should not implement alignment, too dangerous?
            # It is assumed to be only a filter, otherwise .loc should be used.
            bcol = key.cast("boolean")
            df = self._spark_getitem(bcol)
            df._metadata = self._metadata
            return anchor_wrap(self, df)
        raise NotImplementedError(key)

    def __getitem__(self, key):
        return anchor_wrap(self, self._pd_getitem(key))

    def __setitem__(self, key, value):
        # For now, we don't support realignment against different dataframes.
        # This is too expensive in Spark.
        # Are we assigning against a column?
        if isinstance(value, Column):
            assert value._pandas_anchor is self,\
                "Cannot combine column argument because it comes from a different dataframe"
        if isinstance(key, (tuple, list)):
            assert isinstance(value.schema, StructType)
            field_names = value.schema.fieldNames()
            df = self.assign(**{k: value[c]
                                for k, c in zip(key, field_names)})
        else:
            df = self.assign(**{key: value})

        _reassign_jdf(self, df)

    def __getattr__(self, key):
        if key.startswith("__") or key.startswith("_pandas_") or key.startswith("_spark_"):
            raise AttributeError(key)
        return anchor_wrap(self, self._spark_getattr(key))

    def __iter__(self):
        return self.toPandas().__iter__()

    def __len__(self):
        return self._spark_count()

    def __dir__(self):
        fields = [f for f in self.schema.fieldNames() if ' ' not in f]
        return super(DataFrame, self).__dir__() + fields

    def _repr_html_(self):
        return self.head(max_display_count).toPandas()._repr_html_()

    @classmethod
    def _validate_axis(cls, axis=0):
        if axis not in (0, 1, 'index', 'columns', None):
            raise ValueError('No axis named {0}'.format(axis))
        # convert to numeric axis
        return {None: 0, 'index': 0, 'columns': 1}.get(axis, axis)


def _reassign_jdf(target_df: DataFrame, new_df: DataFrame):
    """
    Reassigns the java df contont of a dataframe.
    """
    target_df._jdf = new_df._jdf
    target_df._metadata = new_df._metadata
    # Reset the cached variables
    target_df._schema = None
    target_df._lazy_rdd = None


def _spark_col_apply(col_or_df, sfun):
    """
    Performs a function to all cells on a dataframe, the function being a known sql function.
    """
    if isinstance(col_or_df, Column):
        return sfun(col_or_df)
    assert isinstance(col_or_df, DataFrame)
    df = col_or_df
    df = df._spark_select([sfun(df[col]).alias(col) for col in df.columns])
    return df


def _reduce_spark(col_or_df, sfun):
    """
    Performs a reduction on a dataframe, the function being a known sql function.
    """
    if isinstance(col_or_df, Column):
        col = col_or_df
        df0 = col._spark_ref_dataframe._spark_select(sfun(col))
    else:
        assert isinstance(col_or_df, DataFrame)
        df = col_or_df
        df0 = df._spark_select(sfun("*"))
    return _unpack_scalar(df0)


def _unpack_scalar(df):
    """
    Takes a dataframe that is supposed to contain a single row with a single scalar value,
    and returns this value.
    """
    l = df.head(2).collect()
    assert len(l) == 1, (df, l)
    row = l[0]
    l2 = list(row.asDict().values())
    assert len(l2) == 1, (row, l2)
    return l2[0]


def _reduce_spark_multi(df, aggs):
    """
    Performs a reduction on a dataframe, the functions being known sql aggregate functions.
    """
    assert isinstance(df, DataFrame)
    df0 = df._spark_agg(*aggs)
    l = df0.head(2).collect()
    assert len(l) == 1, (df, l)
    row = l[0]
    l2 = list(row)
    assert len(l2) == len(aggs), (row, l2)
    return l2


def anchor_wrap(df, col):
    """
    Ensures that the column has an anchoring reference to the dataframe.

    This is required to get self-representable columns.
    :param df: dataframe or column
    :param col: a column
    :return: column
    """
    if isinstance(col, Column):
        if isinstance(df, Column):
            ref = df._pandas_anchor
        else:
            assert isinstance(df, DataFrame), type(df)
            ref = df
        col._spark_ref_dataframe = ref
    return col


def _col(df):
    assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
    return df[df.columns[0]]
