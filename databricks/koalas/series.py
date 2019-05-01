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
from functools import partial

import numpy as np
import pandas as pd

from pyspark import sql as spark
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, FloatType, DoubleType, LongType, StructType, \
    TimestampType, to_arrow_type

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.dask.utils import derived_from
from databricks.koalas.frame import DataFrame
from databricks.koalas.generic import _Frame, max_display_count
from databricks.koalas.metadata import Metadata
from databricks.koalas.missing.series import _MissingPandasLikeSeries
from databricks.koalas.selection import SparkDataFrameLocator
from databricks.koalas.utils import validate_arguments_and_invoke_function


@decorator
def _column_op(f, self, *args):
    """
    A decorator that wraps APIs taking/returning Spark Column so that Koalas Series can be
    supported too. If this decorator is used for the `f` function that takes Spark Column and
    returns Spark Column, decorated `f` takes Koalas Series as well and returns Koalas
    Series.

    :param f: a function that takes Spark Column and returns Spark Column.
    :param self: Koalas Series
    :param args: arguments that the function `f` takes.
    """

    assert all((not isinstance(arg, Series)) or (arg._kdf is self._kdf) for arg in args), \
        "Cannot combine column argument because it comes from a different dataframe"

    # It is possible for the function `f` takes other arguments than Spark Column.
    # To cover this case, explicitly check if the argument is Koalas Series and
    # extract Spark Column. For other arguments, they are used as are.
    args = [arg._scol if isinstance(arg, Series) else arg for arg in args]
    scol = f(self._scol, *args)
    return Series(scol, self._kdf, self._index_info)


@decorator
def _numpy_column_op(f, self, *args):
    # PySpark does not support NumPy type out of the box. For now, we convert NumPy types
    # into some primitive types understandable in PySpark.
    new_args = []
    for arg in args:
        # TODO: This is a quick hack to support NumPy type. We should revisit this.
        if isinstance(self.spark_type, LongType) and isinstance(arg, np.timedelta64):
            new_args.append(float(arg / np.timedelta64(1, 's')))
        else:
            new_args.append(arg)
    return _column_op(f)(self, *new_args)


class Series(_Frame):
    """
    Koala Series that corresponds to Pandas Series logically. This holds Spark Column
    internally.

    :ivar _scol: Spark Column instance
    :ivar _kdf: Parent's Koalas DataFrame
    :ivar _index_info: Each pair holds the index field name which exists in Spark fields,
      and the index name.
    """

    @derived_from(pd.Series)
    @dispatch_on('data')
    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):
        s = pd.Series(data=data, index=index, dtype=dtype, name=name, copy=copy, fastpath=fastpath)
        self._init_from_pandas(s)

    @__init__.register(pd.Series)
    def _init_from_pandas(self, s, *args):
        """
        Creates Koalas Series from Pandas Series.

        :param s: Pandas Series
        """

        kdf = DataFrame(pd.DataFrame(s))
        self._init_from_spark(kdf._sdf[kdf._metadata.column_fields[0]],
                              kdf, kdf._metadata.index_info)

    @__init__.register(spark.Column)
    def _init_from_spark(self, scol, kdf, index_info, *args):
        """
        Creates Koalas Series from Spark Column.

        :param scol: Spark Column
        :param kdf: Koalas DataFrame that should have the `scol`.
        :param index_info: index information of this Series.
        """
        assert index_info is not None
        self._scol = scol
        self._kdf = kdf
        self._index_info = index_info

    # arithmetic operators
    __neg__ = _column_op(spark.Column.__neg__)
    __add__ = _column_op(spark.Column.__add__)

    def __sub__(self, other):
        # Note that timestamp subtraction casts arguments to integer. This is to mimic Pandas's
        # behaviors. Pandas returns 'timedelta64[ns]' from 'datetime64[ns]'s subtraction.
        if isinstance(other, Series) and isinstance(self.spark_type, TimestampType):
            if not isinstance(other.spark_type, TimestampType):
                raise TypeError('datetime subtraction can only be applied to datetime series.')
            return self.astype('bigint') - other.astype('bigint')
        else:
            return _column_op(spark.Column.__sub__)(self, other)

    __mul__ = _column_op(spark.Column.__mul__)
    __div__ = _numpy_column_op(spark.Column.__div__)
    __truediv__ = _numpy_column_op(spark.Column.__truediv__)
    __mod__ = _column_op(spark.Column.__mod__)
    __radd__ = _column_op(spark.Column.__radd__)
    __rsub__ = _column_op(spark.Column.__rsub__)
    __rmul__ = _column_op(spark.Column.__rmul__)
    __rdiv__ = _numpy_column_op(spark.Column.__rdiv__)
    __rtruediv__ = _numpy_column_op(spark.Column.__rtruediv__)
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
        """Return the dtype object of the underlying data.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3])
        >>> s.dtype
        dtype('int64')

        >>> s = ks.Series(list('abc'))
        >>> s.dtype
        dtype('O')

        >>> s = ks.Series(pd.date_range('20130101', periods=3))
        >>> s.dtype
        dtype('<M8[ns]')
        """
        if type(self.spark_type) == TimestampType:
            return np.dtype('datetime64[ns]')
        else:
            return np.dtype(to_arrow_type(self.spark_type).to_pandas_dtype())

    @property
    def spark_type(self):
        """ Returns the data type as defined by Spark, as a Spark DataType object."""
        return self.schema.fields[-1].dataType

    def astype(self, dtype):
        from databricks.koalas.typedef import as_spark_type
        spark_type = as_spark_type(dtype)
        if not spark_type:
            raise ValueError("Type {} not understood".format(dtype))
        return Series(self._scol.cast(spark_type), self._kdf, self._index_info)

    def getField(self, name):
        if not isinstance(self.schema, StructType):
            raise AttributeError("Not a struct: {}".format(self.schema))
        else:
            fnames = self.schema.fieldNames()
            if name not in fnames:
                raise AttributeError(
                    "Field {} not found, possible values are {}".format(name, ", ".join(fnames)))
            return Series(self._scol.getField(name), self._kdf, self._index_info)

    def alias(self, name):
        """An alias for :meth:`Series.rename`."""
        return self.rename(name)

    @property
    def schema(self):
        return self.to_dataframe()._sdf.schema

    @property
    def shape(self):
        """Return a tuple of the shape of the underlying data."""
        return len(self),

    @property
    def name(self):
        return self._metadata.column_fields[0]

    @name.setter
    def name(self, name):
        self.rename(name, inplace=True)

    # TODO: Functionality and documentation should be matched. Currently, changing index labels
    # taking dictionary and function to change index are not supported.
    def rename(self, index=None, **kwargs):
        """
        Alter Series name.

        Parameters
        ----------
        index : scalar
            Scalar will alter the ``Series.name`` attribute.
        inplace : bool, default False
            Whether to return a new Series. If True then value of copy is
            ignored.

        Returns
        -------
        Series
            Series with name altered.

        Examples
        --------
        >>> s = ks.Series([1, 2, 3])
        >>> s
        0    1
        1    2
        2    3
        Name: 0, dtype: int64
        >>> s.rename("my_name")  # scalar, changes Series.name
        0    1
        1    2
        2    3
        Name: my_name, dtype: int64
        """
        if index is None:
            return self
        scol = self._scol.alias(index)
        if kwargs.get('inplace', False):
            self._scol = scol
            return self
        else:
            return Series(scol, self._kdf, self._index_info)

    @property
    def _metadata(self):
        return self.to_dataframe()._metadata

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
            s = _col(kdf)
            if inplace:
                self._kdf = kdf
                self._scol = s._scol
                self._index_info = s._index_info
            else:
                return s
        else:
            return kdf

    @property
    def loc(self):
        return SparkDataFrameLocator(self)

    def to_dataframe(self):
        sdf = self._kdf._sdf.select([field for field, _ in self._index_info] + [self._scol])
        metadata = Metadata(column_fields=[sdf.schema[-1].name], index_info=self._index_info)
        return DataFrame(sdf, metadata)

    def to_string(self, buf=None, na_rep='NaN', float_format=None, header=True,
                  index=True, length=False, dtype=False, name=False,
                  max_rows=None):
        """
        Render a string representation of the Series.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory. If the input
                  is large, set max_rows parameter.

        Parameters
        ----------
        buf : StringIO-like, optional
            buffer to write to
        na_rep : string, optional
            string representation of NAN to use, default 'NaN'
        float_format : one-parameter function, optional
            formatter function to apply to columns' elements if they are floats
            default None
        header : boolean, default True
            Add the Series header (index name)
        index : bool, optional
            Add index (row) labels, default True
        length : boolean, default False
            Add the Series length
        dtype : boolean, default False
            Add the Series dtype
        name : boolean, default False
            Add the Series name if not None
        max_rows : int, optional
            Maximum number of rows to show before truncating. If None, show
            all.

        Returns
        -------
        formatted : string (if not buffer passed)

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)], columns=['dogs', 'cats'])
        >>> print(df['dogs'].to_string())
        0    0.2
        1    0.0
        2    0.6
        3    0.2

        >>> print(df['dogs'].to_string(max_rows=2))
        0    0.2
        1    0.0
        """
        # Make sure locals() call is at the top of the function so we don't capture local variables.
        args = locals()
        if max_rows is not None:
            kseries = self.head(max_rows)
        else:
            kseries = self

        return validate_arguments_and_invoke_function(
            kseries.to_pandas(), self.to_string, pd.Series.to_string, args)

    def to_pandas(self):
        """
        Return a pandas Series.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory. If the input
                  is large, set max_rows parameter.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)], columns=['dogs', 'cats'])
        >>> df['dogs'].to_pandas()
        0    0.2
        1    0.0
        2    0.6
        3    0.2
        Name: dogs, dtype: float64
        """
        return _col(self.to_dataframe().toPandas())

    # Alias to maintain backward compatibility with Spark
    toPandas = to_pandas

    @derived_from(pd.Series)
    def isnull(self):
        if isinstance(self.schema[self.name].dataType, (FloatType, DoubleType)):
            return Series(self._scol.isNull() | F.isnan(self._scol), self._kdf, self._index_info)
        else:
            return Series(self._scol.isNull(), self._kdf, self._index_info)

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

    @derived_from(DataFrame)
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
        kdf._metadata = Metadata(column_fields=[self.name], index_info=[(index_name, None)])
        return _col(kdf)

    def corr(self, other, method='pearson'):
        """
        Compute correlation with `other` Series, excluding missing values.

        Parameters
        ----------
        other : Series
        method : {'pearson', 'spearman'}
            * pearson : standard correlation coefficient
            * spearman : Spearman rank correlation

        Returns
        -------
        correlation : float

        Examples
        --------
        >>> df = ks.DataFrame({'s1': [.2, .0, .6, .2],
        ...                    's2': [.3, .6, .0, .1]})
        >>> s1 = df.s1
        >>> s2 = df.s2
        >>> s1.corr(s2, method='pearson')  # doctest: +ELLIPSIS
        -0.851064...

        >>> s1.corr(s2, method='spearman')  # doctest: +ELLIPSIS
        -0.948683...

        Notes
        -----
        There are behavior differences between Koalas and pandas.

        * the `method` argument only accepts 'pearson', 'spearman'
        * the data should not contain NaNs. Koalas will return an error.
        * Koalas doesn't support the following argument(s).

          * `min_periods` argument is not supported
        """
        # This implementation is suboptimal because it computes more than necessary,
        # but it should be a start
        df = self._kdf.assign(corr_arg1=self, corr_arg2=other)[["corr_arg1", "corr_arg2"]]
        c = df.corr(method=method)
        return c.loc["corr_arg1", "corr_arg2"]

    def count(self):
        """
        Return number of non-NA/null observations in the Series.

        Returns
        -------
        nobs : int

        Examples
        --------
        Constructing DataFrame from a dictionary:

        >>> df = ks.DataFrame({"Person":
        ...                    ["John", "Myla", "Lewis", "John", "Myla"],
        ...                    "Age": [24., np.nan, 21., 33, 26]})

        Notice the uncounted NA values:

        >>> df['Person'].count()
        5

        >>> df['Age'].count()
        4
        """
        return self._reduce_for_stat_function(_Frame._count_expr)

    def _reduce_for_stat_function(self, sfun):
        from inspect import signature
        num_args = len(signature(sfun).parameters)
        col_sdf = self._scol
        col_type = self.schema[self.name].dataType
        if isinstance(col_type, BooleanType) and sfun.__name__ not in ('min', 'max'):
            # Stat functions cannot be used with boolean values by default
            # Thus, cast to integer (true to 1 and false to 0)
            # Exclude the min and max methods though since those work with booleans
            col_sdf = col_sdf.cast('integer')
        if num_args == 1:
            # Only pass in the column if sfun accepts only one arg
            col_sdf = sfun(col_sdf)
        else:  # must be 2
            assert num_args == 2
            # Pass in both the column and its data type if sfun accepts two args
            col_sdf = sfun(col_sdf, col_type)
        return _unpack_scalar(self._kdf._sdf.select(col_sdf))

    def __len__(self):
        return len(self.to_dataframe())

    def __getitem__(self, key):
        return Series(self._scol.__getitem__(key), self._kdf, self._index_info)

    def __getattr__(self, item):
        if item.startswith("__") or item.startswith("_pandas_") or item.startswith("_spark_"):
            raise AttributeError(item)
        if hasattr(_MissingPandasLikeSeries, item):
            return partial(getattr(_MissingPandasLikeSeries, item), self)
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
        return super(Series, self).__dir__() + fields

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
