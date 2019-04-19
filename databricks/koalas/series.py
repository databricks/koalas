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
A base classes to be monkey-patched to Column to behave similar to pandas Series.
"""
import pandas as pd
from pyspark.sql import Column, DataFrame, functions as F
from pyspark.sql.types import FloatType, DoubleType, StructType

from databricks.koalas.dask.utils import derived_from
from databricks.koalas.generic import _Frame, anchor_wrap, max_display_count
from databricks.koalas.metadata import Metadata
from databricks.koalas.missing.series import _MissingPandasLikeSeries
from databricks.koalas.selection import SparkDataFrameLocator


class PandasLikeSeries(_Frame, _MissingPandasLikeSeries):
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
        from databricks.koalas.typing import as_python_type
        return as_python_type(self.schema.fields[-1].dataType)

    def astype(self, dtype):
        from databricks.koalas.typing import as_spark_type
        spark_type = as_spark_type(dtype)
        if not spark_type:
            raise ValueError("Type {} not understood".format(dtype))
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

    @derived_from(pd.Series)
    def rename(self, index=None, **kwargs):
        if index is None:
            return self
        col = self._spark_alias(index)
        if kwargs.get('inplace', False):
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
            df_dropna = self._pandas_anchor._spark_filter(self.notna())
        else:
            df_dropna = self._pandas_anchor
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


def _col(df):
    assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
    return df[df.columns[0]]
