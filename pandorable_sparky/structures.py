import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Column
from pyspark.sql.types import StructType

from ._dask_stubs.utils import derived_from
from ._dask_stubs.compatibility import string_types

__all__ = ['PandasLikeSeries', 'PandasLikeDataFrame', 'anchor_wrap']

max_display_count = 1000


class _Frame(object):
    """
    The base class for both dataframes and series.
    """

    def max(self):
        return _reduce_spark(self, F.max)

    def compute(self):
        return self.toPandas()


class PandasLikeSeries(_Frame):
    """
    Methods that are appropriate for distributed series.
    """

    def __init__(self):
        self._pandas_schema = None

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

    @property
    def schema(self):
        if hasattr(self, "_pandas_schema") and self._pandas_schema is not None:
            return self._pandas_schema
        df = self.to_dataframe()
        self._pandas_schema = df.schema
        return df.schema

    @property
    def shape(self):
        return len(self),

    @property
    def name(self):
        return self._jc.toString()

    @name.setter
    def name(self, name):
        col = _col(self.to_dataframe().select(self.alias(name)))
        anchor_wrap(col, self)
        self._jc = col._jc

    def rename(self, name, inplace=False):
        if inplace:
            self.name = name
            return self
        else:
            return _col(self.to_dataframe().select(self.alias(name)))

    def to_dataframe(self):
        if hasattr(self, "_spark_ref_dataframe"):
            return self._spark_ref_dataframe.select(self)
        n = self._pandas_orig_repr()
        raise ValueError("No reference to a dataframe for column {}".format(n))

    def toPandas(self):
        return _col(self.to_dataframe().toPandas())

    def head(self, n=5):
        return _col(self.to_dataframe().head(n))

    def unique(self):
        # Pandas wants a series/array-like object
        return _col(self.to_dataframe().unique())

    def _pandas_anchor(self) -> DataFrame:
        """
        The anchoring dataframe for this column (if any).
        :return:
        """
        return getattr(self, "_spark_ref_dataframe", None)

    # DANGER: will materialize.
    def __iter__(self):
        print("__iter__", self)
        return self.toPandas().__iter__()

    def __len__(self):
        return len(self.to_dataframe())

    def __getitem__(self, key):
        res = anchor_wrap(self, self._spark_getitem(key))
        print("series:getitem:", key, self.schema, res)
        return res

    def __getattr__(self, item):
        if item.startswith("__") or item.startswith("_pandas_") or item.startswith("_spark_"):
            raise AttributeError(item)
        return anchor_wrap(self, self.getField(item))

    def __invert__(self):
        return self.cast("boolean") == False  # noqa: disable=E712

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
            df = df.withColumn(name, c)
        return df

    def head(self, n=5):
        l = self.take(n)
        df0 = self.sql_ctx.createDataFrame(l, schema=self.schema)
        return df0

    @property
    def columns(self):
        return pd.Index(self.schema.fieldNames())

    @columns.setter
    def columns(self, names):
        renamed = _rename(self, names)
        _reassign_jdf(self, renamed)

    def count(self):
        return self._spark_count()

    def unique(self):
        return DataFrame(self._jdf.distinct(), self.sql_ctx)

    @derived_from(pd.DataFrame)
    def drop(self, labels, axis=0, errors='raise'):
        axis = self._validate_axis(axis)
        if axis == 1:
            if isinstance(labels, list):
                return self._spark_drop(*labels)
            else:
                return self._spark_drop(labels)
            # return self.map_partitions(M.drop, labels, axis=axis, errors=errors)
        raise NotImplementedError("Drop currently only works for axis=1")

    @derived_from(pd.DataFrame)
    def get(self, key, default=None):
        try:
            return anchor_wrap(self, self._pd_getitem(key))
        except (KeyError, ValueError, IndexError):
            return default

    def groupby(self, *cols):
        gp = self._spark_groupby(*cols)
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
        # print("__getitem__:key", key, type(key))
        if key is None:
            raise KeyError("none key")
        if isinstance(key, string_types):
            return self._spark_getattr(key)
        if np.isscalar(key) or isinstance(key, (tuple, string_types)):
            raise NotImplementedError(key)
        elif isinstance(key, slice):
            return self.loc[key]

        if isinstance(key, (pd.Series, np.ndarray, pd.Index, list)):
            raise NotImplementedError(key)
        if isinstance(key, DataFrame):
            # TODO Should not implement alignment, too dangerous?
            return self._spark_getitem(key)
        if isinstance(key, Column):
            # TODO Should not implement alignment, too dangerous?
            # It is assumed to be only a filter, otherwise .loc should be used.
            bcol = key.cast("boolean")
            return anchor_wrap(self, self._spark_getitem(bcol))
        raise NotImplementedError(key)

    def __getitem__(self, key):
        return anchor_wrap(self, self._pd_getitem(key))

    def __setitem__(self, key, value):
        # For now, we don't support realignment against different dataframes.
        # This is too expensive in Spark.
        # Are we assigning against a column?
        if isinstance(value, Column):
            assert value._pandas_anchor() is self,\
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
        return anchor_wrap(self, self._spark_getattr(key))

    def __iter__(self):
        print("df__iter__", self)
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
    # Reset the cached variables
    target_df._schema = None
    target_df._lazy_rdd = None


def _rename(frame, names):
    # assert isinstance(frame, _Frame) # TODO: injection does not fix hierarchy
    if isinstance(frame, Column):
        assert isinstance(frame.schema, StructType)
    old_names = frame.schema.fieldNames()
    if len(old_names) != len(names):
        raise ValueError(
            "Length mismatch: Expected axis has %d elements, new values have %d elements"
            % (len(old_names), len(names)))
    for (old_name, new_name) in zip(old_names, names):
        frame = frame.withColumnRenamed(old_name, new_name)
    return frame


def _reduce_spark(col_or_df, sfun):
    """
    Performs a reduction on a dataframe, the function being a known sql function.
    """
    if isinstance(col_or_df, Column):
        col = col_or_df
        df0 = col._spark_ref_dataframe.select(sfun(col))
    else:
        assert isinstance(col_or_df, DataFrame)
        df = col_or_df
        df0 = df.select(sfun("*"))
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
            ref = df._pandas_anchor()
        else:
            assert isinstance(df, DataFrame), type(df)
            ref = df
        col._spark_ref_dataframe = ref
        col._pandas_schema = None
    return col


def _col(df):
    assert isinstance(df, (DataFrame, pd.DataFrame)), type(df)
    return df[df.columns[0]]
