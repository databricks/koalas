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
A wrapper class for Spark Column to behave similar to pandas Series.
"""
from decorator import decorator, dispatch_on

import pandas as pd
from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType, DoubleType, StructType

from databricks.koalas.dask.utils import derived_from
from databricks.koalas.frame import DataFrame
from databricks.koalas.generic import _Frame, max_display_count
from databricks.koalas.metadata import Metadata
from databricks.koalas.missing.series import _MissingPandasLikeSeries
from databricks.koalas.selection import SparkDataFrameLocator


@decorator
def _column_op(f, self, *args):
    assert all((not isinstance(arg, Series)) or (arg._kdf is self._kdf) for arg in args), \
        "Cannot combine column argument because it comes from a different dataframe"

    args = [arg._scol if isinstance(arg, Series) else arg for arg in args]
    scol = f(self._scol, *args)
    return Series(scol, self._kdf)


class Series(_Frame, _MissingPandasLikeSeries):

    @derived_from(pd.Series)
    @dispatch_on('data')
    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        s = pd.Series(data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath)
        self._init__from_pandas(s)

    @__init__.register(pd.Series)
    def _init_from_pandas(self, s, *args):
        kdf = DataFrame(pd.DataFrame(s))
        self._init_from_spark(kdf._sdf[kdf._metadata.column_fields[0]], kdf)

    @__init__.register(spark.Column)
    def _init_from_spark(self, scol, kdf, metadata=None, *args):
        self._scol = scol
        self._kdf = kdf
        self._pandas_metadata = metadata
        self._pandas_schema = None

    # arithmetic operators
    __neg__ = _column_op(spark.Column.__neg__)
    __add__ = _column_op(spark.Column.__add__)
    __sub__ = _column_op(spark.Column.__sub__)
    __mul__ = _column_op(spark.Column.__mul__)
    __div__ = _column_op(spark.Column.__div__)
    __truediv__ = _column_op(spark.Column.__truediv__)
    __mod__ = _column_op(spark.Column.__mod__)
    __radd__ = _column_op(spark.Column.__radd__)
    __rsub__ = _column_op(spark.Column.__rsub__)
    __rmul__ = _column_op(spark.Column.__rmul__)
    __rdiv__ = _column_op(spark.Column.__rdiv__)
    __rtruediv__ = _column_op(spark.Column.__rtruediv__)
    __rmod__ = _column_op(spark.Column.__rmod__)
    __pow__ = _column_op(spark.Column.__pow__)
    __rpow__ = _column_op(spark.Column.__rpow__)

    # logistic operators
    __eq__ = _column_op(spark.Column.__eq__)
    __ne__ = _column_op(spark.Column.__ne__)
    __lt__ = _column_op(spark.Column.__lt__)
    __le__ = _column_op(spark.Column.__le__)
    __ge__ = _column_op(spark.Column.__ge__)
    __gt__ = _column_op(spark.Column.__gt__)

    # `and`, `or`, `not` cannot be overloaded in Python,
    # so use bitwise operators as boolean operators
    __and__ = _column_op(spark.Column.__and__)
    __or__ = _column_op(spark.Column.__or__)
    __invert__ = _column_op(spark.Column.__invert__)
    __rand__ = _column_op(spark.Column.__rand__)
    __ror__ = _column_op(spark.Column.__ror__)

    @property
    def dtype(self):
        from databricks.koalas.typing import as_python_type
        return as_python_type(self.schema.fields[-1].dataType)

    def astype(self, dtype):
        from databricks.koalas.typing import as_spark_type
        spark_type = as_spark_type(dtype)
        if not spark_type:
            raise ValueError("Type {} not understood".format(dtype))
        return Series(self._scol.cast(spark_type), self._kdf)

    def getField(self, name):
        if not isinstance(self.schema, StructType):
            raise AttributeError("Not a struct: {}".format(self.schema))
        else:
            fnames = self.schema.fieldNames()
            if name not in fnames:
                raise AttributeError(
                    "Field {} not found, possible values are {}".format(name, ", ".join(fnames)))
            return Series(self._scol.getField(name), self._kdf)

    # TODO: automate the process here
    def alias(self, name):
        return self.rename(name)

    @property
    def schema(self):
        if self._pandas_schema is None:
            self._pandas_schema = self.to_dataframe()._sdf.schema
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
        scol = self._scol.alias(index)
        if kwargs.get('inplace', False):
            self._scol = scol
            self._pandas_schema = None
            self._pandas_metadata = self._metadata.copy(column_fields=[index])
            return self
        else:
            return Series(scol, self._kdf, self._metadata.copy(column_fields=[index]))

    @property
    def _metadata(self):
        if self._pandas_metadata is None:
            self._pandas_metadata = self.to_dataframe()._metadata
        return self._pandas_metadata

    @property
    def index(self):
        """The index (axis labels) Column of the Series.

        Currently supported only when the DataFrame has a single index.
        """
        if len(self._metadata.index_info) != 1:
            raise KeyError('Currently supported only when the Column has a single index.')
        return self._kdf.index

    @derived_from(pd.Series)
    def reset_index(self, level=None, drop=False, name=None, inplace=False):
        if inplace and not drop:
            raise TypeError('Cannot reset_index inplace on a Series to create a DataFrame')

        if name is not None:
            kdf = self.rename(name).to_dataframe()
        else:
            kdf = self.to_dataframe()
        kdf = kdf.reset_index(level=level, drop=drop)
        if drop:
            col = _col(kdf)
            if inplace:
                self._kdf = kdf
                self._scol = col._scol
                self._pandas_metadata = None
            else:
                return col
        else:
            return kdf

    @property
    def loc(self):
        return SparkDataFrameLocator(self)

    def to_dataframe(self):
        kdf = self._kdf
        if self._pandas_metadata is not None:
            metadata = self._pandas_metadata
        else:
            metadata = kdf._metadata
        sdf = kdf._sdf.select(metadata.index_fields + [self._scol])
        metadata = metadata.copy(column_fields=[sdf.schema[-1].name])
        return DataFrame(sdf, metadata)

    def toPandas(self):
        return _col(self.to_dataframe().toPandas())

    @derived_from(pd.Series)
    def isnull(self):
        if isinstance(self.schema[self.name].dataType, (FloatType, DoubleType)):
            return Series(self._scol.isNull() | F.isnan(self._scol), self._kdf)
        else:
            return Series(self._scol.isNull(), self._kdf)

    isna = isnull

    @derived_from(pd.Series)
    def notnull(self):
        return ~self.isnull()

    notna = notnull

    @derived_from(pd.Series)
    def dropna(self, axis=0, inplace=False, **kwargs):
        ks = _col(self.to_dataframe().dropna(axis=axis, inplace=False))
        if inplace:
            self._kdf = ks._kdf
            self._scol = ks._scol
        else:
            return ks

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
            sdf_dropna = self._kdf._sdf.filter(self.notna()._scol)
        else:
            sdf_dropna = self._kdf._sdf
        sdf = sdf_dropna.groupby(self._scol).count()
        if sort:
            if ascending:
                sdf = sdf.orderBy(F.col('count'))
            else:
                sdf = sdf.orderBy(F.col('count').desc())

        if normalize:
            sum = sdf_dropna.count()
            sdf = sdf.withColumn('count', F.col('count') / F.lit(sum))

        index_name = 'index' if self.name != 'index' else 'level_0'
        kdf = DataFrame(sdf)
        kdf.columns = [index_name, self.name]
        kdf._pandas_metadata = Metadata(column_fields=[self.name], index_info=[(index_name, None)])
        return _col(kdf)

    @derived_from(pd.Series, ua_args=['level'])
    def count(self):
        return self._reduce_for_stat_function(F.count)

    def _reduce_for_stat_function(self, sfun):
        return _unpack_scalar(self._kdf._sdf.select(sfun(self._scol)))

    def __len__(self):
        return len(self.to_dataframe())

    def __getitem__(self, key):
        return Series(self._scol.__getitem__(key), self._kdf)

    def __getattr__(self, item):
        if item.startswith("__") or item.startswith("_pandas_") or item.startswith("_spark_"):
            raise AttributeError(item)
        return self.getField(item)

    def __str__(self):
        return self._pandas_orig_repr()

    def __repr__(self):
        return repr(self.head(max_display_count).toPandas())

    def __dir__(self):
        if not isinstance(self.schema, StructType):
            fields = []
        else:
            fields = [f for f in self.schema.fieldNames() if ' ' not in f]
        return super(spark.Column, self).__dir__() + fields

    def _pandas_orig_repr(self):
        # TODO: figure out how to reuse the original one.
        return 'Column<%s>' % self._scol._jc.toString().encode('utf8')


def _unpack_scalar(sdf):
    """
    Takes a dataframe that is supposed to contain a single row with a single scalar value,
    and returns this value.
    """
    l = sdf.head(2)
    assert len(l) == 1, (sdf, l)
    row = l[0]
    l2 = list(row.asDict().values())
    assert len(l2) == 1, (row, l2)
    return l2[0]


def _col(df):
    assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
    return df[df.columns[0]]
