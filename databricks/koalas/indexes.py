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
Wrappers for Indexes to behave similar to pandas Index, MultiIndex.
"""
from collections import OrderedDict
from distutils.version import LooseVersion
from functools import partial
from typing import Any, List, Tuple, Union
import warnings

import pandas as pd
import numpy as np
from pandas.api.types import (
    is_list_like,
    is_interval_dtype,
    is_bool_dtype,
    is_categorical_dtype,
    is_integer_dtype,
    is_float_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from pandas.io.formats.printing import pprint_thing

import pyspark
from pyspark import sql as spark
from pyspark.sql import functions as F, Window
from pyspark.sql.types import BooleanType, NumericType, StringType, TimestampType

from databricks import koalas as ks  # For running doctests and reference resolution in PyCharm.
from databricks.koalas.config import get_option, option_context
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.base import IndexOpsMixin
from databricks.koalas.frame import DataFrame
from databricks.koalas.missing.indexes import _MissingPandasLikeIndex, _MissingPandasLikeMultiIndex
from databricks.koalas.series import Series, _col
from databricks.koalas.utils import (
    compare_allow_null,
    compare_disallow_null,
    compare_null_first,
    compare_null_last,
    default_session,
    name_like_string,
    scol_for,
    verify_temp_column_name,
    validate_bool_kwarg,
)
from databricks.koalas.internal import _InternalFrame, NATURAL_ORDER_COLUMN_NAME


class Index(IndexOpsMixin):
    """
    Koalas Index that corresponds to Pandas Index logically. This might hold Spark Column
    internally.

    :ivar _kdf: The parent dataframe
    :type _kdf: DataFrame
    :ivar _scol: Spark Column instance
    :type _scol: pyspark.Column

    Parameters
    ----------
    data : DataFrame or list
        Index can be created by DataFrame or list
    dtype : dtype, default None
        Data type to force. Only a single dtype is allowed. If None, infer
    name : name of index, hashable

    See Also
    --------
    MultiIndex : A multi-level, or hierarchical, Index.

    Examples
    --------
    >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[1, 2, 3]).index
    Int64Index([1, 2, 3], dtype='int64')

    >>> ks.DataFrame({'a': [1, 2, 3]}, index=list('abc')).index
    Index(['a', 'b', 'c'], dtype='object')

    >>> Index([1, 2, 3])
    Int64Index([1, 2, 3], dtype='int64')

    >>> Index(list('abc'))
    Index(['a', 'b', 'c'], dtype='object')
    """

    def __init__(self, data: Union[DataFrame, list], dtype=None, name=None) -> None:
        if isinstance(data, DataFrame):
            assert dtype is None
            assert name is None
            kdf = data
        else:
            kdf = DataFrame(index=pd.Index(data=data, dtype=dtype, name=name))
        internal = kdf._internal.copy(
            spark_column=kdf._internal.index_spark_columns[0],
            column_labels=kdf._internal.index_names,
            column_label_names=None,
        )
        IndexOpsMixin.__init__(self, internal, kdf)

    def _with_new_scol(self, scol: spark.Column) -> "Index":
        """
        Copy Koalas Index with the new Spark Column.

        :param scol: the new Spark Column
        :return: the copied Index
        """
        sdf = self._internal.spark_frame.select(scol)  # type: ignore
        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=OrderedDict(zip(sdf.columns, self._internal.index_names)),  # type: ignore
        )
        return DataFrame(internal).index

    # This method is used via `DataFrame.info` API internally.
    def _summary(self, name=None):
        """
        Return a summarized representation.

        Parameters
        ----------
        name : str
            name to use in the summary representation

        Returns
        -------
        String with a summarized representation of the index
        """
        head, tail, total_count = self._kdf._sdf.select(
            F.first(self.spark_column), F.last(self.spark_column), F.count(F.expr("*"))
        ).first()

        if total_count > 0:
            index_summary = ", %s to %s" % (pprint_thing(head), pprint_thing(tail))
        else:
            index_summary = ""

        if name is None:
            name = type(self).__name__
        return "%s: %s entries%s" % (name, total_count, index_summary)

    @property
    def size(self) -> int:
        """
        Return an int representing the number of elements in this object.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=list('abcd'))
        >>> df.index.size
        4

        >>> df.set_index('dogs', append=True).index.size
        4
        """
        return len(self._kdf)  # type: ignore

    @property
    def shape(self) -> tuple:
        """
        Return a tuple of the shape of the underlying data.

        Examples
        --------
        >>> idx = ks.Index(['a', 'b', 'c'])
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')
        >>> idx.shape
        (3,)

        >>> midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )

        >>> midx.shape
        (3,)
        """
        return (len(self._kdf),)

    def identical(self, other):
        """
        Similar to equals, but check that other comparable attributes are
        also equal.

        Returns
        -------
        bool
            If two Index objects have equal elements and same type True,
            otherwise False.

        Examples
        --------

        >>> from databricks.koalas.config import option_context
        >>> idx = ks.Index(['a', 'b', 'c'])
        >>> midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])

        For Index

        >>> idx.identical(idx)
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     idx.identical(ks.Index(['a', 'b', 'c']))
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     idx.identical(ks.Index(['b', 'b', 'a']))
        False
        >>> idx.identical(midx)
        False

        For MultiIndex

        >>> midx.identical(midx)
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     midx.identical(ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')]))
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     midx.identical(ks.MultiIndex.from_tuples([('c', 'z'), ('b', 'y'), ('a', 'x')]))
        False
        >>> midx.identical(idx)
        False
        """
        self_name = self.names if isinstance(self, MultiIndex) else self.name
        other_name = other.names if isinstance(other, MultiIndex) else other.name

        return (self is other) or (
            type(self) == type(other)
            and self_name == other_name  # to support non-index comparison by short-circuiting.
            and self.equals(other)
        )

    def equals(self, other):
        """
        Determine if two Index objects contain the same elements.

        Returns
        -------
        bool
            True if "other" is an Index and it has the same elements as calling
            index; False otherwise.

        Examples
        --------

        >>> from databricks.koalas.config import option_context
        >>> idx = ks.Index(['a', 'b', 'c'])
        >>> idx.name = "name"
        >>> midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> midx.names = ("nameA", "nameB")

        For Index

        >>> idx.equals(idx)
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     idx.equals(ks.Index(['a', 'b', 'c']))
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     idx.equals(ks.Index(['b', 'b', 'a']))
        False
        >>> idx.equals(midx)
        False

        For MultiIndex

        >>> midx.equals(midx)
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     midx.equals(ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')]))
        True
        >>> with option_context('compute.ops_on_diff_frames', True):
        ...     midx.equals(ks.MultiIndex.from_tuples([('c', 'z'), ('b', 'y'), ('a', 'x')]))
        False
        >>> midx.equals(idx)
        False
        """
        # TODO: avoid using default index?
        with option_context("compute.default_index_type", "distributed-sequence"):
            # Directly using Series from both self and other seems causing
            # some exceptions when 'compute.ops_on_diff_frames' is enabled.
            # Working around for now via using frame.
            return (self is other) or (
                type(self) == type(other)
                and (
                    self.to_series().rename("self").to_frame().reset_index()["self"]
                    == other.to_series().rename("other").to_frame().reset_index()["other"]
                ).all()
            )

    def transpose(self):
        """
        Return the transpose, For index, It will be index itself.

        Examples
        --------
        >>> idx = ks.Index(['a', 'b', 'c'])
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')

        >>> idx.transpose()
        Index(['a', 'b', 'c'], dtype='object')

        For MultiIndex

        >>> midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )

        >>> midx.transpose()  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )
        """
        return self

    T = property(transpose)

    def to_pandas(self) -> pd.Index:
        """
        Return a pandas Index.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=list('abcd'))
        >>> df['dogs'].index.to_pandas()
        Index(['a', 'b', 'c', 'd'], dtype='object')
        """
        return self._internal.to_pandas_frame.index  # type: ignore

    toPandas = to_pandas

    def to_numpy(self, dtype=None, copy=False):
        """
        A NumPy ndarray representing the values in this Index or MultiIndex.

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.

        Parameters
        ----------
        dtype : str or numpy.dtype, optional
            The dtype to pass to :meth:`numpy.asarray`
        copy : bool, default False
            Whether to ensure that the returned value is a not a view on
            another array. Note that ``copy=False`` does not *ensure* that
            ``to_numpy()`` is no-copy. Rather, ``copy=True`` ensure that
            a copy is made, even if not strictly necessary.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> ks.Series([1, 2, 3, 4]).index.to_numpy()
        array([0, 1, 2, 3])
        >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[[1, 2, 3], [4, 5, 6]]).index.to_numpy()
        array([(1, 4), (2, 5), (3, 6)], dtype=object)
        """
        result = np.asarray(self.to_pandas()._values, dtype=dtype)
        if copy:
            result = result.copy()
        return result

    @property
    def values(self):
        """
        Return an array representing the data in the Index.

        .. warning:: We recommend using `Index.to_numpy()` instead.

        .. note:: This method should only be used if the resulting NumPy ndarray is expected
            to be small, as all the data is loaded into the driver's memory.

        Returns
        -------
        numpy.ndarray

        Examples
        --------
        >>> ks.Series([1, 2, 3, 4]).index.values
        array([0, 1, 2, 3])
        >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[[1, 2, 3], [4, 5, 6]]).index.values
        array([(1, 4), (2, 5), (3, 6)], dtype=object)
        """
        warnings.warn("We recommend using `{}.to_numpy()` instead.".format(type(self).__name__))
        return self.to_numpy()

    @property
    def spark_type(self):
        """ Returns the data type as defined by Spark, as a Spark DataType object."""
        return self.to_series().spark_type

    @property
    def has_duplicates(self) -> bool:
        """
        If index has duplicates, return True, otherwise False.

        Examples
        --------
        >>> kdf = ks.DataFrame({'a': [1, 2, 3]}, index=list('aac'))
        >>> kdf.index.has_duplicates
        True

        >>> kdf = ks.DataFrame({'a': [1, 2, 3]}, index=[list('abc'), list('def')])
        >>> kdf.index.has_duplicates
        False

        >>> kdf = ks.DataFrame({'a': [1, 2, 3]}, index=[list('aac'), list('eef')])
        >>> kdf.index.has_duplicates
        True
        """
        df = self._kdf._sdf.select(self.spark_column)
        col = df.columns[0]

        return df.select(F.count(col) != F.countDistinct(col)).first()[0]

    @property
    def name(self) -> Union[str, Tuple[str, ...]]:
        """Return name of the Index."""
        return self.names[0]

    @name.setter
    def name(self, name: Union[str, Tuple[str, ...]]) -> None:
        self.names = [name]

    @property
    def names(self) -> List[Union[str, Tuple[str, ...]]]:
        """Return names of the Index."""
        return [
            name if name is None or len(name) > 1 else name[0]
            for name in self._internal.index_names  # type: ignore
        ]

    @names.setter
    def names(self, names: List[Union[str, Tuple[str, ...]]]) -> None:
        if not is_list_like(names):
            raise ValueError("Names must be a list-like")
        self.rename(names, inplace=True)

    @property
    def nlevels(self) -> int:
        """
        Number of levels in Index & MultiIndex.

        Examples
        --------
        >>> kdf = ks.DataFrame({"a": [1, 2, 3]}, index=pd.Index(['a', 'b', 'c'], name="idx"))
        >>> kdf.index.nlevels
        1

        >>> kdf = ks.DataFrame({'a': [1, 2, 3]}, index=[list('abc'), list('def')])
        >>> kdf.index.nlevels
        2
        """
        return len(self._kdf._internal.index_spark_column_names)

    def rename(
        self,
        name: Union[str, Tuple[str, ...], List[Union[str, Tuple[str, ...]]]],
        inplace: bool = False,
    ):
        """
        Alter Index or MultiIndex name.
        Able to set new names without level. Defaults to returning new index.

        Parameters
        ----------
        name : label or list of labels
            Name(s) to set.
        inplace : boolean, default False
            Modifies the object directly, instead of creating a new Index or MultiIndex.

        Returns
        -------
        Index or MultiIndex
            The same type as the caller or None if inplace is True.

        Examples
        --------
        >>> df = ks.DataFrame({'a': ['A', 'C'], 'b': ['A', 'B']}, columns=['a', 'b'])
        >>> df.index.rename("c")
        Int64Index([0, 1], dtype='int64', name='c')

        >>> df.set_index("a", inplace=True)
        >>> df.index.rename("d")
        Index(['A', 'C'], dtype='object', name='d')

        You can also change the index name in place.

        >>> df.index.rename("e", inplace=True)
        >>> df.index
        Index(['A', 'C'], dtype='object', name='e')

        >>> df  # doctest: +NORMALIZE_WHITESPACE
           b
        e
        A  A
        C  B

        Support for MultiIndex

        >>> kidx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y')])
        >>> kidx.names = ['hello', 'koalas']
        >>> kidx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   names=['hello', 'koalas'])

        >>> kidx.rename(['aloha', 'databricks'])  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   names=['aloha', 'databricks'])
        """
        names = self._verify_for_rename(name)

        if inplace:
            kdf = self._kdf
        else:
            kdf = self._kdf.copy()

        kdf._internal = kdf._internal.copy(
            index_map=OrderedDict(zip(kdf._internal.index_spark_column_names, names))
        )

        idx = kdf.index
        idx._internal = idx._internal.copy(spark_column=self.spark_column)
        if inplace:
            self._internal = idx._internal
        else:
            return idx

    def _verify_for_rename(self, name):
        if name is None or isinstance(name, tuple):
            return [name]
        elif isinstance(name, str):
            return [(name,)]
        elif is_list_like(name):
            if len(self._internal.index_map) != len(name):
                raise ValueError(
                    "Length of new names must be {}, got {}".format(
                        len(self._internal.index_map), len(name)
                    )
                )
            return [n if n is None or isinstance(n, tuple) else (n,) for n in name]
        else:
            raise TypeError("name must be a hashable type")

    # TODO: add downcast parameter for fillna function
    def fillna(self, value):
        """
        Fill NA/NaN values with the specified value.

        Parameters
        ----------
        value : scalar
            Scalar value to use to fill holes (e.g. 0). This value cannot be a list-likes.

        Returns
        -------
        Index :
            filled with value

        Examples
        --------
        >>> ki = ks.DataFrame({'a': ['a', 'b', 'c']}, index=[1, 2, None]).index
        >>> ki
        Float64Index([1.0, 2.0, nan], dtype='float64')

        >>> ki.fillna(0)
        Float64Index([1.0, 2.0, 0.0], dtype='float64')
        """
        if not isinstance(value, (float, int, str, bool)):
            raise TypeError("Unsupported type %s" % type(value))
        sdf = self._internal.spark_frame.fillna(value)
        result = DataFrame(self._kdf._internal.with_new_sdf(sdf)).index
        return result

    # TODO: ADD keep parameter
    def drop_duplicates(self):
        """
        Return Index with duplicate values removed.

        Returns
        -------
        deduplicated : Index

        See Also
        --------
        Series.drop_duplicates : Equivalent method on Series.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.

        Examples
        --------
        Generate an pandas.Index with duplicate values.

        >>> idx = ks.Index(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'])

        >>> idx.drop_duplicates().sort_values()
        Index(['beetle', 'cow', 'hippo', 'lama'], dtype='object')
        """
        sdf = self._internal.spark_frame.select(
            self._internal.index_spark_columns
        ).drop_duplicates()
        internal = _InternalFrame(spark_frame=sdf, index_map=self._kdf._internal.index_map)
        result = DataFrame(internal).index
        return result

    def to_series(self, name: Union[str, Tuple[str, ...]] = None) -> Series:
        """
        Create a Series with both index and values equal to the index keys
        useful with map for returning an indexer based on an index.

        Parameters
        ----------
        name : string, optional
            name of resulting Series. If None, defaults to name of original
            index

        Returns
        -------
        Series : dtype will be based on the type of the Index values.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=list('abcd'))
        >>> df['dogs'].index.to_series()
        a    a
        b    b
        c    c
        d    d
        Name: 0, dtype: object
        """
        kdf = self._kdf
        scol = self.spark_column
        if name is not None:
            scol = scol.alias(name_like_string(name))
        column_labels = [None] if len(kdf._internal.index_map) > 1 else kdf._internal.index_names
        return Series(
            kdf._internal.copy(
                spark_column=scol, column_labels=column_labels, column_label_names=None
            ),
            anchor=kdf,
        )

    def to_frame(self, index=True, name=None) -> DataFrame:
        """
        Create a DataFrame with a column containing the Index.

        Parameters
        ----------
        index : boolean, default True
            Set the index of the returned DataFrame as the original Index.
        name : object, default None
            The passed name should substitute for the index name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame containing the original Index data.

        See Also
        --------
        Index.to_series : Convert an Index to a Series.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = ks.Index(['Ant', 'Bear', 'Cow'], name='animal')
        >>> idx.to_frame()  # doctest: +NORMALIZE_WHITESPACE
               animal
        animal
        Ant       Ant
        Bear     Bear
        Cow       Cow

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
          animal
        0    Ant
        1   Bear
        2    Cow

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(name='zoo')  # doctest: +NORMALIZE_WHITESPACE
                 zoo
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
        """
        if name is None:
            if self._internal.index_names[0] is None:
                name = ("0",)
            else:
                name = self._internal.index_names[0]
        elif isinstance(name, str):
            name = (name,)
        scol = self.spark_column.alias(name_like_string(name))

        sdf = self._internal.spark_frame.select(scol, NATURAL_ORDER_COLUMN_NAME)

        if index:
            index_map = OrderedDict({name_like_string(name): self._internal.index_names[0]})
        else:
            index_map = None  # type: ignore

        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=index_map,
            column_labels=[name],
            data_spark_columns=[scol_for(sdf, name_like_string(name))],
        )
        return DataFrame(internal)

    def is_boolean(self):
        """
        Return if the current index type is a boolean type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[True]).index.is_boolean()
        True
        """
        return is_bool_dtype(self.dtype)

    def is_categorical(self):
        """
        Return if the current index type is a categorical type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[1]).index.is_categorical()
        False
        """
        return is_categorical_dtype(self.dtype)

    def is_floating(self):
        """
        Return if the current index type is a floating type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[1]).index.is_floating()
        False
        """
        return is_float_dtype(self.dtype)

    def is_integer(self):
        """
        Return if the current index type is a integer type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[1]).index.is_integer()
        True
        """
        return is_integer_dtype(self.dtype)

    def is_interval(self):
        """
        Return if the current index type is an interval type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[1]).index.is_interval()
        False
        """
        return is_interval_dtype(self.dtype)

    def is_numeric(self):
        """
        Return if the current index type is a numeric type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=[1]).index.is_numeric()
        True
        """
        return is_numeric_dtype(self.dtype)

    def is_object(self):
        """
        Return if the current index type is a object type.

        Examples
        --------
        >>> ks.DataFrame({'a': [1]}, index=["a"]).index.is_object()
        True
        """
        return is_object_dtype(self.dtype)

    def dropna(self):
        """
        Return Index or MultiIndex without NA/NaN values

        Examples
        --------

        >>> df = ks.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=['cobra', 'viper', None],
        ...                   columns=['max_speed', 'shield'])
        >>> df
               max_speed  shield
        cobra          1       2
        viper          4       5
        NaN            7       8

        >>> df.index.dropna()
        Index(['cobra', 'viper'], dtype='object')

        Also support for MultiIndex

        >>> midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                       [None, 'weight', 'length']],
        ...                      [[0, 1, 1, 1, 1, 1, 2, 2, 2],
        ...                       [0, 1, 1, 0, 1, 2, 1, 1, 2]])
        >>> s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, None],
        ...               index=midx)
        >>> s
        lama    NaN        45.0
        cow     weight    200.0
                weight      1.2
                NaN        30.0
                weight    250.0
                length      1.5
        falcon  weight    320.0
                weight      1.0
                length      NaN
        Name: 0, dtype: float64

        >>> s.index.dropna()  # doctest: +SKIP
        MultiIndex([(   'cow', 'weight'),
                    (   'cow', 'weight'),
                    (   'cow', 'weight'),
                    (   'cow', 'length'),
                    ('falcon', 'weight'),
                    ('falcon', 'weight'),
                    ('falcon', 'length')],
                   )
        """
        kdf = self._kdf.copy()
        sdf = kdf._internal.spark_frame.select(self._internal.index_spark_columns).dropna()
        internal = _InternalFrame(spark_frame=sdf, index_map=self._internal.index_map)
        return DataFrame(internal).index

    def unique(self, level=None):
        """
        Return unique values in the index.

        Be aware the order of unique values might be different than pandas.Index.unique

        Parameters
        ----------
        level : int or str, optional, default is None

        Returns
        -------
        Index without duplicates

        See Also
        --------
        Series.unique
        groupby.SeriesGroupBy.unique

        Examples
        --------
        >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[1, 1, 3]).index.unique().sort_values()
        Int64Index([1, 3], dtype='int64')

        >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=['d', 'e', 'e']).index.unique().sort_values()
        Index(['d', 'e'], dtype='object')

        MultiIndex

        >>> ks.MultiIndex.from_tuples([("A", "X"), ("A", "Y"), ("A", "X")]).unique()
        ... # doctest: +SKIP
        MultiIndex([('A', 'X'),
                    ('A', 'Y')],
                   )
        """
        if level is not None:
            self._validate_index_level(level)
        scols = self._internal.index_spark_columns
        scol_names = self._internal.index_spark_column_names
        scols = [scol.alias(scol_name) for scol, scol_name in zip(scols, scol_names)]
        sdf = self._kdf._sdf.select(scols).distinct()
        return DataFrame(
            _InternalFrame(spark_frame=sdf, index_map=self._kdf._internal.index_map)
        ).index

    # TODO: add error parameter
    def drop(self, labels):
        """
        Make new Index with passed list of labels deleted.

        Parameters
        ----------
        labels : array-like

        Returns
        -------
        dropped : Index

        Examples
        --------
        >>> index = ks.Index([1, 2, 3])
        >>> index
        Int64Index([1, 2, 3], dtype='int64')

        >>> index.drop([1])
        Int64Index([2, 3], dtype='int64')
        """
        if not isinstance(labels, (tuple, list)):
            labels = [labels]
        sdf = self._internal.spark_frame[~self._internal.index_spark_columns[0].isin(labels)]
        return Index(
            DataFrame(_InternalFrame(spark_frame=sdf, index_map=self._kdf._internal.index_map))
        )

    def _validate_index_level(self, level):
        """
        Validate index level.
        For single-level Index getting level number is a no-op, but some
        verification must be done like in MultiIndex.
        """
        if isinstance(level, int):
            if level < 0 and level != -1:
                raise IndexError(
                    "Too many levels: Index has only 1 level,"
                    " %d is not a valid level number" % (level,)
                )
            elif level > 0:
                raise IndexError("Too many levels:" " Index has only 1 level, not %d" % (level + 1))
        elif level != self.name:
            raise KeyError(
                "Requested level ({}) does not match index name ({})".format(level, self.name)
            )

    def copy(self, name=None, deep=None):
        """
        Make a copy of this object. name sets those attributes on the new object.

        Parameters
        ----------
        name : string, optional
            to set name of index
        deep : None
            this parameter is not supported but just dummy parameter to match pandas.

        Examples
        --------
        >>> df = ks.DataFrame([[1, 2], [4, 5], [7, 8]],
        ...                   index=['cobra', 'viper', 'sidewinder'],
        ...                   columns=['max_speed', 'shield'])
        >>> df
                    max_speed  shield
        cobra               1       2
        viper               4       5
        sidewinder          7       8
        >>> df.index
        Index(['cobra', 'viper', 'sidewinder'], dtype='object')

        Copy index

        >>> df.index.copy()
        Index(['cobra', 'viper', 'sidewinder'], dtype='object')

        Copy index with name

        >>> df.index.copy(name='snake')
        Index(['cobra', 'viper', 'sidewinder'], dtype='object', name='snake')
        """
        result = Index(self._kdf.copy())
        if name:
            result.name = name
        return result

    def droplevel(self, level):
        """
        Return index with requested level(s) removed.
        If resulting index has only 1 level left, the result will be
        of Index type, not MultiIndex.

        Parameters
        ----------
        level : int, str, tuple, or list-like, default 0
            If a string is given, must be the name of a level
            If list-like, elements must be names or indexes of levels.

        Returns
        -------
        Index or MultiIndex

        Examples
        --------
        >>> midx = ks.DataFrame({'a': ['a', 'b']}, index=[['a', 'x'], ['b', 'y'], [1, 2]]).index
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 'b', 1),
                    ('x', 'y', 2)],
                   )
        >>> midx.droplevel([0, 1])  # doctest: +SKIP
        Int64Index([1, 2], dtype='int64')
        >>> midx.droplevel(0)  # doctest: +SKIP
        MultiIndex([('b', 1),
                    ('y', 2)],
                   )
        >>> midx.names = [("a", "b"), "b", "c"]
        >>> midx.droplevel([('a', 'b')])  # doctest: +SKIP
        MultiIndex([('b', 1),
                    ('y', 2)],
                   names=['b', 'c'])
        """
        names = self.names
        nlevels = self.nlevels
        if not isinstance(level, (tuple, list)):
            level = [level]

        for n in level:
            if isinstance(n, int) and (n > nlevels - 1):
                raise IndexError(
                    "Too many levels: Index has only {} levels, not {}".format(nlevels, n + 1)
                )
            if isinstance(n, (str, tuple)) and (n not in names):
                raise KeyError("Level {} not found".format(n))

        if len(level) >= nlevels:
            raise ValueError(
                "Cannot remove {} levels from an index with {} "
                "levels: at least one level must be "
                "left.".format(len(level), nlevels)
            )

        int_level = [n if isinstance(n, int) else names.index(n) for n in level]
        index_map = list(self._internal.index_map.items())
        index_map = OrderedDict(index_map[c] for c in range(0, nlevels) if c not in int_level)
        sdf = self._internal.spark_frame
        sdf = sdf.select(*index_map.keys())
        result = _InternalFrame(spark_frame=sdf, index_map=index_map)
        return DataFrame(result).index

    def symmetric_difference(self, other, result_name=None, sort=None):
        """
        Compute the symmetric difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        result_name : str
        sort : True or None, default None
            Whether to sort the resulting index.
            * True : Attempt to sort the result.
            * None : Do not sort the result.

        Returns
        -------
        symmetric_difference : Index

        Notes
        -----
        ``symmetric_difference`` contains elements that appear in either
        ``idx1`` or ``idx2`` but not both. Equivalent to the Index created by
        ``idx1.difference(idx2) | idx2.difference(idx1)`` with duplicates
        dropped.

        Examples
        --------
        >>> s1 = ks.Series([1, 2, 3, 4], index=[1, 2, 3, 4])
        >>> s2 = ks.Series([1, 2, 3, 4], index=[2, 3, 4, 5])

        >>> s1.index.symmetric_difference(s2.index)
        Int64Index([5, 1], dtype='int64')

        You can set name of result Index.

        >>> s1.index.symmetric_difference(s2.index, result_name='koalas')
        Int64Index([5, 1], dtype='int64', name='koalas')

        You can set sort to `True`, if you want to sort the resulting index.

        >>> s1.index.symmetric_difference(s2.index, sort=True)
        Int64Index([1, 5], dtype='int64')

        You can also use the ``^`` operator:

        >>> s1.index ^ s2.index
        Int64Index([5, 1], dtype='int64')
        """
        if type(self) != type(other):
            raise NotImplementedError(
                "Doesn't support symmetric_difference between Index & MultiIndex for now"
            )

        sdf_self = self._kdf._sdf.select(self._internal.index_spark_columns)
        sdf_other = other._kdf._sdf.select(other._internal.index_spark_columns)

        sdf_symdiff = sdf_self.union(sdf_other).subtract(sdf_self.intersect(sdf_other))

        if sort:
            sdf_symdiff = sdf_symdiff.sort(self._internal.index_spark_columns)

        internal = _InternalFrame(spark_frame=sdf_symdiff, index_map=self._internal.index_map)
        result = Index(DataFrame(internal))

        if result_name:
            result.name = result_name

        return result

    # TODO: return_indexer
    def sort_values(self, ascending=True):
        """
        Return a sorted copy of the index.

        .. note:: This method is not supported for pandas when index has NaN value.
                  pandas raises unexpected TypeError, but we support treating NaN
                  as the smallest value.

        Parameters
        ----------
        ascending : bool, default True
            Should the index values be sorted in an ascending order.

        Returns
        -------
        sorted_index : ks.Index or ks.MultiIndex
            Sorted copy of the index.

        See Also
        --------
        Series.sort_values : Sort values of a Series.
        DataFrame.sort_values : Sort values in a DataFrame.

        Examples
        --------
        >>> idx = ks.Index([10, 100, 1, 1000])
        >>> idx
        Int64Index([10, 100, 1, 1000], dtype='int64')

        Sort values in ascending order (default behavior).

        >>> idx.sort_values()
        Int64Index([1, 10, 100, 1000], dtype='int64')

        Sort values in descending order.

        >>> idx.sort_values(ascending=False)
        Int64Index([1000, 100, 10, 1], dtype='int64')

        Support for MultiIndex.

        >>> kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('c', 'y', 2), ('b', 'z', 3)])
        >>> kidx  # doctest: +SKIP
        MultiIndex([('a', 'x', 1),
                    ('c', 'y', 2),
                    ('b', 'z', 3)],
                   )

        >>> kidx.sort_values()  # doctest: +SKIP
        MultiIndex([('a', 'x', 1),
                    ('b', 'z', 3),
                    ('c', 'y', 2)],
                   )

        >>> kidx.sort_values(ascending=False)  # doctest: +SKIP
        MultiIndex([('c', 'y', 2),
                    ('b', 'z', 3),
                    ('a', 'x', 1)],
                   )
        """
        sdf = self._internal.spark_frame
        sdf = sdf.orderBy(self._internal.index_spark_columns, ascending=ascending)

        internal = _InternalFrame(
            spark_frame=sdf.select(self._internal.index_spark_columns),
            index_map=self._internal.index_map,
        )
        return DataFrame(internal).index

    def sort(self, *args, **kwargs):
        """
        Use sort_values instead.
        """
        raise TypeError("cannot sort an Index object in-place, use sort_values instead")

    def min(self):
        """
        Return the minimum value of the Index.

        Returns
        -------
        scalar
            Minimum value.

        See Also
        --------
        Index.max : Return the maximum value of the object.
        Series.min : Return the minimum value in a Series.
        DataFrame.min : Return the minimum values in a DataFrame.

        Examples
        --------
        >>> idx = ks.Index([3, 2, 1])
        >>> idx.min()
        1

        >>> idx = ks.Index(['c', 'b', 'a'])
        >>> idx.min()
        'a'

        For a MultiIndex, the maximum is determined lexicographically.

        >>> idx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2)])
        >>> idx.min()
        ('a', 'x', 1)
        """
        sdf = self._internal.spark_frame
        min_row = sdf.select(F.min(F.struct(self._internal.index_spark_columns))).head()
        result = tuple(min_row[0])

        return result if len(result) > 1 else result[0]

    def max(self):
        """
        Return the maximum value of the Index.

        Returns
        -------
        scalar
            Maximum value.

        See Also
        --------
        Index.min : Return the minimum value in an Index.
        Series.max : Return the maximum value in a Series.
        DataFrame.max : Return the maximum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.max()
        3

        >>> idx = pd.Index(['c', 'b', 'a'])
        >>> idx.max()
        'c'

        For a MultiIndex, the maximum is determined lexicographically.

        >>> idx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2)])
        >>> idx.max()
        ('b', 'y', 2)
        """
        sdf = self._internal.spark_frame
        max_row = sdf.select(F.max(F.struct(self._internal.index_spark_columns))).head()
        result = tuple(max_row[0])

        return result if len(result) > 1 else result[0]

    def append(self, other):
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index

        Returns
        -------
        appended : Index

        Examples
        --------
        >>> kidx = ks.Index([10, 5, 0, 5, 10, 5, 0, 10])
        >>> kidx
        Int64Index([10, 5, 0, 5, 10, 5, 0, 10], dtype='int64')

        >>> kidx.append(kidx)
        Int64Index([10, 5, 0, 5, 10, 5, 0, 10, 10, 5, 0, 5, 10, 5, 0, 10], dtype='int64')

        Support for MiltiIndex

        >>> kidx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y')])
        >>> kidx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   )

        >>> kidx.append(kidx)  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('a', 'x'),
                    ('b', 'y')],
                   )
        """
        if type(self) is not type(other):
            raise NotImplementedError(
                "append() between Index & MultiIndex currently is not supported"
            )

        sdf_self = self._internal.spark_frame.select(self._internal.index_spark_columns)
        sdf_other = other._internal.spark_frame.select(other._internal.index_spark_columns)
        sdf_appended = sdf_self.union(sdf_other)

        # names should be kept when MultiIndex, but Index wouldn't keep its name.
        if isinstance(self, MultiIndex):
            index_map = self._internal.index_map
        else:
            index_map = OrderedDict(
                (idx_col, None) for idx_col in self._internal.index_spark_column_names
            )

        internal = _InternalFrame(spark_frame=sdf_appended, index_map=index_map)

        return DataFrame(internal).index

    def argmax(self):
        """
        Return a maximum argument indexer.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        maximum argument indexer

        Examples
        --------
        >>> kidx = ks.Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3])
        >>> kidx
        Int64Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3], dtype='int64')

        >>> kidx.argmax()
        4
        """
        sdf = self._internal.spark_frame.select(self.spark_column)
        sequence_col = verify_temp_column_name(sdf, "__distributed_sequence_column__")
        sdf = _InternalFrame.attach_distributed_sequence_column(sdf, column_name=sequence_col)
        # spark_frame here looks like below
        # +-----------------+---------------+
        # |__index_level_0__|__index_value__|
        # +-----------------+---------------+
        # |                0|             10|
        # |                4|            100|
        # |                2|              8|
        # |                3|              7|
        # |                6|              4|
        # |                5|              5|
        # |                7|              3|
        # |                8|            100|
        # |                1|              9|
        # +-----------------+---------------+

        return sdf.orderBy(self.spark_column.desc(), F.col(sequence_col).asc()).first()[0]

    def argmin(self):
        """
        Return a minimum argument indexer.

        Parameters
        ----------
        skipna : bool, default True

        Returns
        -------
        minimum argument indexer

        Examples
        --------
        >>> kidx = ks.Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3])
        >>> kidx
        Int64Index([10, 9, 8, 7, 100, 5, 4, 3, 100, 3], dtype='int64')

        >>> kidx.argmin()
        7
        """
        sdf = self._internal.spark_frame.select(self.spark_column)
        sequence_col = verify_temp_column_name(sdf, "__distributed_sequence_column__")
        sdf = _InternalFrame.attach_distributed_sequence_column(sdf, column_name=sequence_col)

        return sdf.orderBy(self.spark_column.asc(), F.col(sequence_col).asc()).first()[0]

    def set_names(self, names, level=None, inplace=False):
        """
        Set Index or MultiIndex name.
        Able to set new names partially and by level.

        Parameters
        ----------
        names : label or list of label
            Name(s) to set.
        level : int, label or list of int or label, optional
            If the index is a MultiIndex, level(s) to set (None for all
            levels). Otherwise level must be None.
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index
            The same type as the caller or None if inplace is True.

        See Also
        --------
        Index.rename : Able to set new names without level.

        Examples
        --------
        >>> idx = ks.Index([1, 2, 3, 4])
        >>> idx
        Int64Index([1, 2, 3, 4], dtype='int64')

        >>> idx.set_names('quarter')
        Int64Index([1, 2, 3, 4], dtype='int64', name='quarter')

        For MultiIndex

        >>> idx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y')])
        >>> idx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   )

        >>> idx.set_names(['kind', 'year'], inplace=True)
        >>> idx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   names=['kind', 'year'])

        >>> idx.set_names('species', level=0)  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y')],
                   names=['species', 'year'])
        """
        if isinstance(self, MultiIndex):
            if level is not None:
                self_names = self.names
                self_names[level] = names
                names = self_names
        return self.rename(name=names, inplace=inplace)

    def difference(self, other, sort=None):
        """
        Return a new Index with elements from the index that are not in
        `other`.

        This is the set difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : True or None, default None
            Whether to sort the resulting index.
            * True : Attempt to sort the result.
            * None : Do not sort the result.

        Returns
        -------
        difference : Index

        Examples
        --------

        >>> idx1 = ks.Index([2, 1, 3, 4])
        >>> idx2 = ks.Index([3, 4, 5, 6])
        >>> idx1.difference(idx2, sort=True)
        Int64Index([1, 2], dtype='int64')

        MultiIndex

        >>> midx1 = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        >>> midx2 = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'z', 2), ('k', 'z', 3)])
        >>> midx1.difference(midx2)  # doctest: +SKIP
        MultiIndex([('b', 'y', 2),
                    ('c', 'z', 3)],
                   )
        """
        if not is_list_like(other):
            raise TypeError("Input must be Index or array-like")
        if not isinstance(sort, (type(None), type(True))):
            raise ValueError(
                "The 'sort' keyword only takes the values of None or True; {} was passed.".format(
                    sort
                )
            )
        # Handling MultiIndex
        if isinstance(self, ks.MultiIndex) and not isinstance(other, ks.MultiIndex):
            if not all([isinstance(item, tuple) for item in other]):
                raise TypeError("other must be a MultiIndex or a list of tuples")
            other = ks.MultiIndex.from_tuples(other)

        if not isinstance(other, ks.Index):
            other = ks.Index(other)

        sdf_self = self._internal.spark_frame
        sdf_other = other._internal.spark_frame
        idx_self = self._internal.index_spark_columns
        idx_other = other._internal.index_spark_columns
        sdf_diff = sdf_self.select(idx_self).subtract(sdf_other.select(idx_other))
        internal = _InternalFrame(spark_frame=sdf_diff, index_map=self._internal.index_map)
        result = DataFrame(internal).index
        # Name(s) will be kept when only name(s) of (Multi)Index are the same.
        if isinstance(self, type(other)) and isinstance(self, ks.MultiIndex):
            if self.names == other.names:
                result.names = self.names
        elif isinstance(self, type(other)) and not isinstance(self, ks.MultiIndex):
            if self.name == other.name:
                result.name = self.name
        return result if sort is None else result.sort_values()

    @property
    def is_all_dates(self):
        """
        Return if all data types of the index are datetime.
        remember that since Koalas does not support multiple data types in an index,
        so it returns True if any type of data is datetime.

        Examples
        --------
        >>> from datetime import datetime

        >>> idx = ks.Index([datetime(2019, 1, 1, 0, 0, 0), datetime(2019, 2, 3, 0, 0, 0)])
        >>> idx
        DatetimeIndex(['2019-01-01', '2019-02-03'], dtype='datetime64[ns]', freq=None)

        >>> idx.is_all_dates
        True

        >>> idx = ks.Index([datetime(2019, 1, 1, 0, 0, 0), None])
        >>> idx
        DatetimeIndex(['2019-01-01', 'NaT'], dtype='datetime64[ns]', freq=None)

        >>> idx.is_all_dates
        True

        >>> idx = ks.Index([0, 1, 2])
        >>> idx
        Int64Index([0, 1, 2], dtype='int64')

        >>> idx.is_all_dates
        False
        """
        return isinstance(self.spark_type, TimestampType)

    def repeat(self, repeats: int) -> "Index":
        """
        Repeat elements of a Index/MultiIndex.

        Returns a new Index/MultiIndex where each element of the current Index/MultiIndex
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            Index.

        Returns
        -------
        repeated_index : Index/MultiIndex
            Newly created Index/MultiIndex with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.

        Examples
        --------
        >>> idx = ks.Index(['a', 'b', 'c'])
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')
        >>> idx.repeat(2)
        Index(['a', 'b', 'c', 'a', 'b', 'c'], dtype='object')

        For MultiIndex,

        >>> midx = ks.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'b'),
                    ('y', 'c')],
                   )
        >>> midx.repeat(2)  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'b'),
                    ('y', 'c'),
                    ('x', 'a'),
                    ('x', 'b'),
                    ('y', 'c')],
                   )
        >>> midx.repeat(0)  # doctest: +SKIP
        MultiIndex([], )
        """
        if not isinstance(repeats, int):
            raise ValueError("`repeats` argument must be integer, but got {}".format(type(repeats)))
        elif repeats < 0:
            raise ValueError("negative dimensions are not allowed")

        sdf = self._internal.spark_frame.select(self._internal.index_spark_columns)
        internal = _InternalFrame(
            spark_frame=sdf, index_map=OrderedDict(zip(sdf.columns, self._internal.index_names))
        )
        kdf = DataFrame(internal)  # type: DataFrame
        if repeats == 0:
            return DataFrame(kdf._internal.with_filter(F.lit(False))).index
        else:
            return ks.concat([kdf] * repeats).index

    def asof(self, label):
        """
        Return the label from the index, or, if not present, the previous one.

        Assuming that the index is sorted, return the passed index label if it
        is in the index, or return the previous index label if the passed one
        is not in the index.

        .. note:: This API is dependent on :meth:`Index.is_monotonic_increasing`
            which can be expensive.

        Parameters
        ----------
        label : object
            The label up to which the method returns the latest index label.

        Returns
        -------
        object
            The passed label if it is in the index. The previous label if the
            passed label is not in the sorted index or `NaN` if there is no
            such label.

        Examples
        --------
        `Index.asof` returns the latest index label up to the passed label.

        >>> idx = ks.Index(['2013-12-31', '2014-01-02', '2014-01-03'])
        >>> idx.asof('2014-01-01')
        '2013-12-31'

        If the label is in the index, the method returns the passed label.

        >>> idx.asof('2014-01-02')
        '2014-01-02'

        If all of the labels in the index are later than the passed label,
        NaN is returned.

        >>> idx.asof('1999-01-02')
        nan
        """
        sdf = self._internal._sdf
        if self.is_monotonic_increasing:
            sdf = sdf.where(self.spark_column <= label).select(F.max(self.spark_column))
        elif self.is_monotonic_decreasing:
            sdf = sdf.where(self.spark_column >= label).select(F.min(self.spark_column))
        else:
            raise ValueError("index must be monotonic increasing or decreasing")
        result = sdf.head()[0]
        return result if result is not None else np.nan

    def union(self, other, sort=None):
        """
        Form the union of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : bool or None, default None
            Whether to sort the resulting Index.

        Returns
        -------
        union : Index

        Examples
        --------

        Index

        >>> idx1 = ks.Index([1, 2, 3, 4])
        >>> idx2 = ks.Index([3, 4, 5, 6])
        >>> idx1.union(idx2).sort_values()
        Int64Index([1, 2, 3, 4, 5, 6], dtype='int64')

        MultiIndex

        >>> midx1 = ks.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("x", "c"), ("x", "d")])
        >>> midx2 = ks.MultiIndex.from_tuples([("x", "c"), ("x", "d"), ("x", "e"), ("x", "f")])
        >>> midx1.union(midx2).sort_values()  # doctest: +SKIP
        MultiIndex([('x', 'a'),
                    ('x', 'b'),
                    ('x', 'c'),
                    ('x', 'd'),
                    ('x', 'e'),
                    ('x', 'f')],
                   )
        """
        sort = True if sort is None else sort
        sort = validate_bool_kwarg(sort, "sort")
        if type(self) is not type(other):
            if isinstance(self, MultiIndex):
                if not isinstance(other, list) or not all(
                    [isinstance(item, tuple) for item in other]
                ):
                    raise TypeError("other must be a MultiIndex or a list of tuples")
                other = MultiIndex.from_tuples(other)
            else:
                if isinstance(other, MultiIndex):
                    # TODO: We can't support different type of values in a single column for now.
                    raise NotImplementedError(
                        "Union between Index and MultiIndex is not yet supported"
                    )
                elif isinstance(other, Series):
                    other = other.to_frame().set_index(other.name).index
                elif isinstance(other, DataFrame):
                    raise ValueError("Index data must be 1-dimensional")
                else:
                    other = Index(other)
        sdf_self = self._internal._sdf.select(self._internal.index_spark_columns)
        sdf_other = other._internal._sdf.select(other._internal.index_spark_columns)
        sdf = sdf_self.union(sdf_other.subtract(sdf_self))
        if isinstance(self, MultiIndex):
            sdf = sdf.drop_duplicates()
        if sort:
            sdf = sdf.sort(self._internal.index_spark_columns)
        internal = _InternalFrame(spark_frame=sdf, index_map=self._internal.index_map)

        return DataFrame(internal).index

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeIndex, item):
            property_or_func = getattr(_MissingPandasLikeIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError("'Index' object has no attribute '{}'".format(item))

    def __repr__(self):
        max_display_count = get_option("display.max_rows")
        if max_display_count is None:
            return repr(self.to_pandas())

        pindex = (
            self._kdf.head(max_display_count + 1)
            .index._with_new_scol(self.spark_column)
            .to_pandas()
        )

        pindex_length = len(pindex)
        repr_string = repr(pindex[:max_display_count])

        if pindex_length > max_display_count:
            footer = "\nShowing only the first {}".format(max_display_count)
            return repr_string + footer
        return repr_string

    def __iter__(self):
        return _MissingPandasLikeIndex.__iter__(self)

    def __xor__(self, other):
        return self.symmetric_difference(other)

    def __len__(self):
        return self.size


class MultiIndex(Index):
    """
    Koalas MultiIndex that corresponds to Pandas MultiIndex logically. This might hold Spark Column
    internally.

    :ivar _kdf: The parent dataframe
    :type _kdf: DataFrame
    :ivar _scol: Spark Column instance
    :type _scol: pyspark.Column

    See Also
    --------
    Index : A single-level Index.

    Examples
    --------
    >>> ks.DataFrame({'a': ['a', 'b', 'c']}, index=[[1, 2, 3], [4, 5, 6]]).index  # doctest: +SKIP
    MultiIndex([(1, 4),
                (2, 5),
                (3, 6)],
               )

    >>> ks.DataFrame({'a': [1, 2, 3]}, index=[list('abc'), list('def')]).index  # doctest: +SKIP
    MultiIndex([('a', 'd'),
                ('b', 'e'),
                ('c', 'f')],
               )
    """

    def __init__(self, kdf: DataFrame):
        assert len(kdf._internal._index_map) > 1
        scol = F.struct(kdf._internal.index_spark_columns)
        data_columns = kdf._sdf.select(scol).columns
        internal = kdf._internal.copy(
            spark_column=scol,
            column_labels=[(col, None) for col in data_columns],
            column_label_names=None,
        )
        IndexOpsMixin.__init__(self, internal, kdf)

    def _with_new_scol(self, scol: spark.Column):
        raise NotImplementedError("Not supported for type MultiIndex")

    def any(self, *args, **kwargs):
        raise TypeError("cannot perform any with this index type: MultiIndex")

    def all(self, *args, **kwargs):
        raise TypeError("cannot perform all with this index type: MultiIndex")

    @staticmethod
    def from_tuples(tuples, sortorder=None, names=None):
        """
        Convert list of tuples to MultiIndex.

        Parameters
        ----------
        tuples : list / sequence of tuple-likes
            Each tuple is the index of one row/column.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index : MultiIndex

        Examples
        --------

        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> ks.MultiIndex.from_tuples(tuples, names=('number', 'color'))  # doctest: +SKIP
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        return DataFrame(
            index=pd.MultiIndex.from_tuples(tuples=tuples, sortorder=sortorder, names=names)
        ).index

    @staticmethod
    def from_arrays(arrays, sortorder=None, names=None):
        """
        Convert arrays to MultiIndex.

        Parameters
        ----------
        arrays: list / sequence of array-likes
            Each array-like gives one level’s value for each data point. len(arrays)
            is the number of levels.
        sortorder: int or None
            Level of sortedness (must be lexicographically sorted by that level).
        names: list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index: MultiIndex

        Examples
        --------

        >>> arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        >>> ks.MultiIndex.from_arrays(arrays, names=('number', 'color'))  # doctest: +SKIP
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        """
        return DataFrame(
            index=pd.MultiIndex.from_arrays(arrays=arrays, sortorder=sortorder, names=names)
        ).index

    @staticmethod
    def from_product(iterables, sortorder=None, names=None):
        """
        Make a MultiIndex from the cartesian product of multiple iterables.

        Parameters
        ----------
        iterables : list / sequence of iterables
            Each iterable has unique labels for each level of the index.
        sortorder : int or None
            Level of sortedness (must be lexicographically sorted by that
            level).
        names : list / sequence of str, optional
            Names for the levels in the index.

        Returns
        -------
        index : MultiIndex

        See Also
        --------
        MultiIndex.from_arrays : Convert list of arrays to MultiIndex.
        MultiIndex.from_tuples : Convert list of tuples to MultiIndex.

        Examples
        --------
        >>> numbers = [0, 1, 2]
        >>> colors = ['green', 'purple']
        >>> ks.MultiIndex.from_product([numbers, colors],
        ...                            names=['number', 'color'])  # doctest: +SKIP
        MultiIndex([(0,  'green'),
                    (0, 'purple'),
                    (1,  'green'),
                    (1, 'purple'),
                    (2,  'green'),
                    (2, 'purple')],
                   names=['number', 'color'])
        """
        return DataFrame(
            index=pd.MultiIndex.from_product(iterables=iterables, sortorder=sortorder, names=names)
        ).index

    @property
    def name(self) -> str:
        raise PandasNotImplementedError(class_name="pd.MultiIndex", property_name="name")

    @name.setter
    def name(self, name: str) -> None:
        raise PandasNotImplementedError(class_name="pd.MultiIndex", property_name="name")

    def _verify_for_rename(self, name):
        if is_list_like(name):
            if len(self._internal.index_map) != len(name):
                raise ValueError(
                    "Length of new names must be {}, got {}".format(
                        len(self._internal.index_map), len(name)
                    )
                )
            return [n if n is None or isinstance(n, tuple) else (n,) for n in name]
        else:
            raise TypeError("Must pass list-like as `names`.")

    def swaplevel(self, i=-2, j=-1):
        """
        Swap level i with level j.
        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int, str, default -2
            First level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.
        j : int, str, default -1
            Second level of index to be swapped. Can pass level name as string.
            Type of parameters can be mixed.

        Returns
        -------
        MultiIndex
            A new MultiIndex.

        Examples
        --------
        >>> midx = ks.MultiIndex.from_arrays([['a', 'b'], [1, 2]], names = ['word', 'number'])
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 1),
                    ('b', 2)],
                   names=['word', 'number'])

        >>> midx.swaplevel(0, 1)  # doctest: +SKIP
        MultiIndex([(1, 'a'),
                    (2, 'b')],
                   names=['number', 'word'])

        >>> midx.swaplevel('number', 'word')  # doctest: +SKIP
        MultiIndex([(1, 'a'),
                    (2, 'b')],
                   names=['number', 'word'])
        """
        for index in (i, j):
            if not isinstance(index, int) and index not in self.names:
                raise KeyError("Level %s not found" % index)

        i = i if isinstance(i, int) else self.names.index(i)
        j = j if isinstance(j, int) else self.names.index(j)

        for index in (i, j):
            if index >= len(self.names) or index < -len(self.names):
                raise IndexError(
                    "Too many levels: Index has only %s levels, "
                    "%s is not a valid level number" % (len(self.names), index)
                )

        index_map = list(self._internal.index_map.items())
        index_map[i], index_map[j], = index_map[j], index_map[i]
        result = DataFrame(self._kdf._internal.copy(index_map=OrderedDict(index_map))).index
        return result

    @property
    def levshape(self):
        """
        A tuple with the length of each level.

        Examples
        --------
        >>> midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> midx  # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )

        >>> midx.levshape
        (3, 3)
        """
        internal = self._internal
        result = internal._sdf.agg(
            *(F.countDistinct(c) for c in internal.index_spark_columns)
        ).collect()[0]
        return tuple(result)

    @staticmethod
    def _comparator_for_monotonic_increasing(data_type):
        if isinstance(data_type, BooleanType):
            return compare_allow_null
        else:
            return compare_null_last

    def _is_monotonic(self, order):
        if order == "increasing":
            return self._is_monotonic_increasing().all()
        else:
            return self._is_monotonic_decreasing().all()

    def _is_monotonic_increasing(self):
        scol = self.spark_column
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-1, -1)
        prev = F.lag(scol, 1).over(window)

        cond = F.lit(True)
        for field in self.spark_type[::-1]:
            left = scol.getField(field.name)
            right = prev.getField(field.name)
            compare = MultiIndex._comparator_for_monotonic_increasing(field.dataType)
            cond = F.when(left.eqNullSafe(right), cond).otherwise(
                compare(left, right, spark.Column.__gt__)
            )

        cond = prev.isNull() | cond

        internal = _InternalFrame(
            spark_frame=self._internal.spark_frame.select(
                self._internal.index_spark_columns + [cond]
            ),
            index_map=self._internal.index_map,
        )

        return _col(DataFrame(internal))

    @staticmethod
    def _comparator_for_monotonic_decreasing(data_type):
        if isinstance(data_type, StringType):
            return compare_disallow_null
        elif isinstance(data_type, BooleanType):
            return compare_allow_null
        elif isinstance(data_type, NumericType):
            return compare_null_last
        else:
            return compare_null_first

    def _is_monotonic_decreasing(self):
        scol = self.spark_column
        window = Window.orderBy(NATURAL_ORDER_COLUMN_NAME).rowsBetween(-1, -1)
        prev = F.lag(scol, 1).over(window)

        cond = F.lit(True)
        for field in self.spark_type[::-1]:
            left = scol.getField(field.name)
            right = prev.getField(field.name)
            compare = MultiIndex._comparator_for_monotonic_decreasing(field.dataType)
            cond = F.when(left.eqNullSafe(right), cond).otherwise(
                compare(left, right, spark.Column.__lt__)
            )

        cond = prev.isNull() | cond

        internal = _InternalFrame(
            spark_frame=self._internal.spark_frame.select(
                self._internal.index_spark_columns + [cond]
            ),
            index_map=self._internal.index_map,
        )

        return _col(DataFrame(internal))

    def to_frame(self, index=True, name=None) -> DataFrame:
        """
        Create a DataFrame with the levels of the MultiIndex as columns.
        Column ordering is determined by the DataFrame constructor with data as
        a dict.

        Parameters
        ----------
        index : boolean, default True
            Set the index of the returned DataFrame as the original MultiIndex.
        name : list / sequence of strings, optional
            The passed names should substitute index level names.

        Returns
        -------
        DataFrame : a DataFrame containing the original MultiIndex data.

        See Also
        --------
        DataFrame

        Examples
        --------
        >>> tuples = [(1, 'red'), (1, 'blue'),
        ...           (2, 'red'), (2, 'blue')]
        >>> idx = ks.MultiIndex.from_tuples(tuples, names=('number', 'color'))
        >>> idx  # doctest: +SKIP
        MultiIndex([(1,  'red'),
                    (1, 'blue'),
                    (2,  'red'),
                    (2, 'blue')],
                   names=['number', 'color'])
        >>> idx.to_frame()  # doctest: +NORMALIZE_WHITESPACE
                      number color
        number color
        1      red         1   red
               blue        1  blue
        2      red         2   red
               blue        2  blue

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
           number color
        0       1   red
        1       1  blue
        2       2   red
        3       2  blue

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(name=['n', 'c'])  # doctest: +NORMALIZE_WHITESPACE
                      n     c
        number color
        1      red    1   red
               blue   1  blue
        2      red    2   red
               blue   2  blue
        """
        if name is None:
            name = [
                name if name is not None else (str(i),)
                for i, name in enumerate(self._internal.index_names)
            ]
        elif is_list_like(name):
            if len(name) != len(self._internal.index_map):
                raise ValueError("'name' should have same length as number of levels on index.")
            name = [n if isinstance(n, tuple) else (n,) for n in name]
        else:
            raise TypeError("'name' must be a list / sequence of column names.")

        sdf = self._internal.spark_frame.select(
            [
                scol.alias(name_like_string(label))
                for scol, label in zip(self._internal.index_spark_columns, name)
            ]
            + [NATURAL_ORDER_COLUMN_NAME]
        )

        if index:
            index_map = OrderedDict(
                (name_like_string(label), n) for label, n in zip(name, self._internal.index_names)
            )
        else:
            index_map = None  # type: ignore

        internal = _InternalFrame(
            spark_frame=sdf,
            index_map=index_map,
            column_labels=name,
            data_spark_columns=[scol_for(sdf, name_like_string(label)) for label in name],
        )
        return DataFrame(internal)

    def to_pandas(self) -> pd.MultiIndex:
        """
        Return a pandas MultiIndex.

        .. note:: This method should only be used if the resulting Pandas object is expected
                  to be small, as all the data is loaded into the driver's memory.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=[list('abcd'), list('efgh')])
        >>> df['dogs'].index.to_pandas()  # doctest: +SKIP
        MultiIndex([('a', 'e'),
                    ('b', 'f'),
                    ('c', 'g'),
                    ('d', 'h')],
                   )
        """
        # TODO: We might need to handle internal state change.
        # So far, we don't have any functions to change the internal state of MultiIndex except for
        # series-like operations. In that case, it creates new Index object instead of MultiIndex.
        return self._kdf[[]]._to_internal_pandas().index

    toPandas = to_pandas

    def nunique(self, dropna=True):
        raise NotImplementedError("isna is not defined for MultiIndex")

    # TODO: add 'name' parameter after pd.MultiIndex.name is implemented
    def copy(self, deep=None):
        """
        Make a copy of this object.

        Parameters
        ----------
        deep : None
            this parameter is not supported but just dummy parameter to match pandas.

        Examples
        --------
        >>> df = ks.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
        ...                   columns=['dogs', 'cats'],
        ...                   index=[list('abcd'), list('efgh')])
        >>> df['dogs'].index  # doctest: +SKIP
        MultiIndex([('a', 'e'),
                    ('b', 'f'),
                    ('c', 'g'),
                    ('d', 'h')],
                   )

        Copy index

        >>> df.index.copy()  # doctest: +SKIP
        MultiIndex([('a', 'e'),
                    ('b', 'f'),
                    ('c', 'g'),
                    ('d', 'h')],
                   )
        """
        return MultiIndex(self._kdf.copy())

    def symmetric_difference(self, other, result_name=None, sort=None):
        """
        Compute the symmetric difference of two MultiIndex objects.

        Parameters
        ----------
        other : Index or array-like
        result_name : list
        sort : True or None, default None
            Whether to sort the resulting index.
            * True : Attempt to sort the result.
            * None : Do not sort the result.

        Returns
        -------
        symmetric_difference : MiltiIndex

        Notes
        -----
        ``symmetric_difference`` contains elements that appear in either
        ``idx1`` or ``idx2`` but not both. Equivalent to the Index created by
        ``idx1.difference(idx2) | idx2.difference(idx1)`` with duplicates
        dropped.

        Examples
        --------
        >>> midx1 = pd.MultiIndex([['lama', 'cow', 'falcon'],
        ...                        ['speed', 'weight', 'length']],
        ...                       [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                        [0, 0, 0, 0, 1, 2, 0, 1, 2]])
        >>> midx2 = pd.MultiIndex([['koalas', 'cow', 'falcon'],
        ...                        ['speed', 'weight', 'length']],
        ...                       [[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                        [0, 0, 0, 0, 1, 2, 0, 1, 2]])
        >>> s1 = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...                index=midx1)
        >>> s2 = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
        ...              index=midx2)

        >>> s1.index.symmetric_difference(s2.index)  # doctest: +SKIP
        MultiIndex([('koalas', 'speed'),
                    (  'lama', 'speed')],
                   )

        You can set names of result Index.

        >>> s1.index.symmetric_difference(s2.index, result_name=['a', 'b'])  # doctest: +SKIP
        MultiIndex([('koalas', 'speed'),
                    (  'lama', 'speed')],
                   names=['a', 'b'])

        You can set sort to `True`, if you want to sort the resulting index.

        >>> s1.index.symmetric_difference(s2.index, sort=True)  # doctest: +SKIP
        MultiIndex([('koalas', 'speed'),
                    (  'lama', 'speed')],
                   )

        You can also use the ``^`` operator:

        >>> s1.index ^ s2.index  # doctest: +SKIP
        MultiIndex([('koalas', 'speed'),
                    (  'lama', 'speed')],
                   )
        """
        if type(self) != type(other):
            raise NotImplementedError(
                "Doesn't support symmetric_difference between Index & MultiIndex for now"
            )

        sdf_self = self._kdf._sdf.select(self._internal.index_spark_columns)
        sdf_other = other._kdf._sdf.select(other._internal.index_spark_columns)

        sdf_symdiff = sdf_self.union(sdf_other).subtract(sdf_self.intersect(sdf_other))

        if sort:
            sdf_symdiff = sdf_symdiff.sort(self._internal.index_spark_columns)

        internal = _InternalFrame(spark_frame=sdf_symdiff, index_map=self._internal.index_map)
        result = MultiIndex(DataFrame(internal))

        if result_name:
            result.names = result_name

        return result

    # TODO: ADD error parameter
    def drop(self, codes, level=None):
        """
        Make new MultiIndex with passed list of labels deleted

        Parameters
        ----------
        codes : array-like
            Must be a list of tuples
        level : int or level name, default None

        Returns
        -------
        dropped : MultiIndex

        Examples
        --------
        >>> index = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        >>> index # doctest: +SKIP
        MultiIndex([('a', 'x'),
                    ('b', 'y'),
                    ('c', 'z')],
                   )

        >>> index.drop(['a']) # doctest: +SKIP
        MultiIndex([('b', 'y'),
                    ('c', 'z')],
                   )

        >>> index.drop(['x', 'y'], level=1) # doctest: +SKIP
        MultiIndex([('c', 'z')],
                   )
        """
        sdf = self._internal.spark_frame
        index_scols = self._internal.index_spark_columns
        if level is None:
            scol = index_scols[0]
        elif isinstance(level, int):
            scol = index_scols[level]
        else:
            spark_column_name = None
            for index_spark_column_name, index_name in self._internal.index_map.items():
                if not isinstance(level, tuple):
                    level = (level,)
                if level == index_name:
                    if spark_column_name is not None:
                        raise ValueError(
                            "The name {} occurs multiple times, use a level number".format(
                                name_like_string(level)
                            )
                        )
                    spark_column_name = index_spark_column_name
            if spark_column_name is None:
                raise KeyError("Level {} not found".format(name_like_string(level)))
            scol = scol_for(sdf, spark_column_name)
        sdf = sdf[~scol.isin(codes)]
        return MultiIndex(
            DataFrame(_InternalFrame(spark_frame=sdf, index_map=self._kdf._internal.index_map))
        )

    def value_counts(self, normalize=False, sort=True, ascending=False, bins=None, dropna=True):
        if (
            LooseVersion(pyspark.__version__) < LooseVersion("2.4")
            and default_session().conf.get("spark.sql.execution.arrow.enabled") == "true"
            and isinstance(self, MultiIndex)
        ):
            raise RuntimeError(
                "if you're using pyspark < 2.4, set conf "
                "'spark.sql.execution.arrow.enabled' to 'false' "
                "for using this function with MultiIndex"
            )
        return super(MultiIndex, self).value_counts(
            normalize=normalize, sort=sort, ascending=ascending, bins=bins, dropna=dropna
        )

    value_counts.__doc__ = IndexOpsMixin.value_counts.__doc__

    def argmax(self):
        raise TypeError("reduction operation 'argmax' not allowed for this dtype")

    def argmin(self):
        raise TypeError("reduction operation 'argmin' not allowed for this dtype")

    def asof(self, label):
        raise NotImplementedError(
            "only the default get_loc method is currently supported for MultiIndex"
        )

    @property
    def is_all_dates(self):
        """
        is_all_dates always returns False for MultiIndex

        Examples
        --------
        >>> from datetime import datetime

        >>> idx = ks.MultiIndex.from_tuples(
        ...     [(datetime(2019, 1, 1, 0, 0, 0), datetime(2019, 1, 1, 0, 0, 0)),
        ...      (datetime(2019, 1, 1, 0, 0, 0), datetime(2019, 1, 1, 0, 0, 0))])
        >>> idx  # doctest: +SKIP
        MultiIndex([('2019-01-01', '2019-01-01'),
                    ('2019-01-01', '2019-01-01')],
                   )

        >>> idx.is_all_dates
        False
        """
        return False

    def __getattr__(self, item: str) -> Any:
        if hasattr(_MissingPandasLikeMultiIndex, item):
            property_or_func = getattr(_MissingPandasLikeMultiIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError("'MultiIndex' object has no attribute '{}'".format(item))

    def __repr__(self):
        max_display_count = get_option("display.max_rows")
        if max_display_count is None:
            return repr(self.to_pandas())

        pindex = self._kdf.head(max_display_count + 1).index.to_pandas()

        pindex_length = len(pindex)
        repr_string = repr(pindex[:max_display_count])

        if pindex_length > max_display_count:
            footer = "\nShowing only the first {}".format(max_display_count)
            return repr_string + footer
        return repr_string

    def __iter__(self):
        return _MissingPandasLikeMultiIndex.__iter__(self)
