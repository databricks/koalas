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
from functools import partial
from typing import Any, Callable, Union

import pandas as pd
import numpy as np
from pandas.api.types import is_hashable, CategoricalDtype

from databricks import koalas as ks
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeCategoricalIndex
from databricks.koalas.series import Series


class CategoricalIndex(Index):
    """
    Index based on an underlying `Categorical`.

    CategoricalIndex can only take on a limited,
    and usually fixed, number of possible values (`categories`). Also,
    it might have an order, but numerical operations
    (additions, divisions, ...) are not possible.

    Parameters
    ----------
    data : array-like (1-dimensional)
        The values of the categorical. If `categories` are given, values not in
        `categories` will be replaced with NaN.
    categories : index-like, optional
        The categories for the categorical. Items need to be unique.
        If the categories are not given here (and also not in `dtype`), they
        will be inferred from the `data`.
    ordered : bool, optional
        Whether or not this categorical is treated as an ordered
        categorical. If not given here or in `dtype`, the resulting
        categorical will be unordered.
    dtype : CategoricalDtype or "category", optional
        If :class:`CategoricalDtype`, cannot be used together with
        `categories` or `ordered`.
    copy : bool, default False
        Make a copy of input ndarray.
    name : object, optional
        Name to be stored in the index.

    See Also
    --------
    Index : The base Koalas Index type.

    Examples
    --------
    >>> ks.CategoricalIndex(["a", "b", "c", "a", "b", "c"])  # doctest: +NORMALIZE_WHITESPACE
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')

    ``CategoricalIndex`` can also be instantiated from a ``Categorical``:

    >>> c = pd.Categorical(["a", "b", "c", "a", "b", "c"])
    >>> ks.CategoricalIndex(c)  # doctest: +NORMALIZE_WHITESPACE
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')

    Ordered ``CategoricalIndex`` can have a min and max value.

    >>> ci = ks.CategoricalIndex(
    ...     ["a", "b", "c", "a", "b", "c"], ordered=True, categories=["c", "b", "a"]
    ... )
    >>> ci  # doctest: +NORMALIZE_WHITESPACE
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['c', 'b', 'a'], ordered=True, dtype='category')

    From a Series:

    >>> s = ks.Series(["a", "b", "c", "a", "b", "c"], index=[10, 20, 30, 40, 50, 60])
    >>> ks.CategoricalIndex(s)  # doctest: +NORMALIZE_WHITESPACE
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')

    From an Index:

    >>> idx = ks.Index(["a", "b", "c", "a", "b", "c"])
    >>> ks.CategoricalIndex(idx)  # doctest: +NORMALIZE_WHITESPACE
    CategoricalIndex(['a', 'b', 'c', 'a', 'b', 'c'],
                     categories=['a', 'b', 'c'], ordered=False, dtype='category')
    """

    def __new__(cls, data=None, categories=None, ordered=None, dtype=None, copy=False, name=None):
        if not is_hashable(name):
            raise TypeError("Index.name must be a hashable type")

        if isinstance(data, (Series, Index)):
            if dtype is None:
                dtype = "category"
            return Index(data, dtype=dtype, ordered=ordered, copy=copy, name=name)

        return ks.from_pandas(
            pd.CategoricalIndex(
                data=data, categories=categories, ordered=ordered, dtype=dtype, name=name
            )
        )

    @property
    def codes(self) -> Index:
        """
        The category codes of this categorical.

        Codes are an Index of integers which are the positions of the actual
        values in the categories Index.

        There is no setter, use the other categorical methods and the normal item
        setter to change values in the categorical.

        Returns
        -------
        Index
            A non-writable view of the `codes` Index.

        Examples
        --------
        >>> idx = ks.CategoricalIndex(list("abbccc"))
        >>> idx  # doctest: +NORMALIZE_WHITESPACE
        CategoricalIndex(['a', 'b', 'b', 'c', 'c', 'c'],
                         categories=['a', 'b', 'c'], ordered=False, dtype='category')

        >>> idx.codes
        Int64Index([0, 1, 1, 2, 2, 2], dtype='int64')
        """
        return self._with_new_scol(self.spark.column).rename(None)

    @property
    def categories(self) -> pd.Index:
        """
        The categories of this categorical.

        Examples
        --------
        >>> idx = ks.CategoricalIndex(list("abbccc"))
        >>> idx  # doctest: +NORMALIZE_WHITESPACE
        CategoricalIndex(['a', 'b', 'b', 'c', 'c', 'c'],
                         categories=['a', 'b', 'c'], ordered=False, dtype='category')

        >>> idx.categories
        Index(['a', 'b', 'c'], dtype='object')
        """
        return self.dtype.categories

    @categories.setter
    def categories(self, categories):
        raise NotImplementedError()

    @property
    def ordered(self) -> bool:
        """
        Whether the categories have an ordered relationship.

        Examples
        --------
        >>> idx = ks.CategoricalIndex(list("abbccc"))
        >>> idx  # doctest: +NORMALIZE_WHITESPACE
        CategoricalIndex(['a', 'b', 'b', 'c', 'c', 'c'],
                         categories=['a', 'b', 'c'], ordered=False, dtype='category')

        >>> idx.ordered
        False
        """
        return self.dtype.ordered

    def map(
        self, mapper: Union[Callable[[Any], Any], dict, pd.Series], na_action: Any = None
    ) -> "CategoricalIndex":
        """
        Map values using input correspondence (a dict, Series, or function).
        Maps the values (their categories, not the codes) of the index to new
        categories.
        If a `dict` or :class:`~pandas.Series` is used any unmapped category is
        mapped to the original value.
        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        Returns
        -------
        CategoricalIndex
            Mapped index.
        Examples
        --------
        >>> kidx = ks.CategoricalIndex(['a', 'b', 'c'])
        >>> kidx # doctest: +NORMALIZE_WHITESPACE
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                         ordered=False, dtype='category')
        >>> kidx.map(lambda x: x.upper()) # doctest: +NORMALIZE_WHITESPACE
        CategoricalIndex(['A', 'B', 'C'], categories=['A', 'B', 'C'],
                         ordered=False, dtype='category')
        >>> kidx.map({'a': 'first', 'b': 'second', 'c': 'third'}) # doctest: +NORMALIZE_WHITESPACE
        CategoricalIndex(['first', 'second', 'third'], categories=['first', 'second', 'third'], ordered=False, dtype='category')
        If the mapping is one-to-one the ordering of the categories is
        preserved:
        >>> kidx = ks.CategoricalIndex(['a', 'b', 'c'], ordered=True)
        >>> kidx # doctest: +NORMALIZE_WHITESPACE
        CategoricalIndex(['a', 'b', 'c'], categories=['a', 'b', 'c'],
                         ordered=True, dtype='category')
        >>> kidx.map({'a': 3, 'b': 2, 'c': 1}) # doctest: +NORMALIZE_WHITESPACE
        CategoricalIndex([3, 2, 1], categories=[3, 2, 1], ordered=True, dtype='category')
        >>> kidx.map({'a': 'first', 'b': 'second', 'c': 'first'}) # doctest: +NORMALIZE_WHITESPACE
        CategoricalIndex(['first', 'second', 'first'], categories=['first', 'second'],
                         ordered=True, dtype='category')
        If a `dict` is used, all unmapped categories are mapped to the original value
        >>> idx.map({'a': 'first', 'b': 'second'}) # doctest: +NORMALIZE_WHITESPACE
        CategoricalIndex(['first', 'second', 'c'], categories=['first', 'second', 'c'],
                         ordered=True, dtype='category')
        """

        from databricks.koalas.indexes.extension import MapExtension, getOrElse

        extension_mapper = MapExtension(index=self, na_action=na_action)
        return_type = extension_mapper._mapper_return_type(mapper)

        # Pull unique elements of series and sort
        if isinstance(mapper, pd.Series):
            mapper = pd.Series(np.sort(mapper.unique()))

        unique_categories = self.dtype.categories.values
        pos_dict = {}
        for i in range(len(unique_categories)):
            if isinstance(mapper, dict):
                category = unique_categories[i]
                pos_dict[i] = mapper.get(category, return_type(category))  # type: ignore
            elif isinstance(mapper, pd.Series):
                pos_dict[i] = getOrElse(mapper, i, return_type, default_value=unique_categories[i])
            elif isinstance(mapper, ks.Series):
                raise NotImplementedError(
                    "Currently do not support input of ks.Series in CategoricalIndex.map"
                )
            else:
                pos_dict[i] = mapper(unique_categories[i])

        new_index = CategoricalIndex(extension_mapper.map(pos_dict))

        return CategoricalIndex(
            new_index.astype(
                CategoricalDtype(categories=new_index.dtype.categories, ordered=self.ordered)
            )
        )

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeCategoricalIndex, item):
            property_or_func = getattr(MissingPandasLikeCategoricalIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError("'CategoricalIndex' object has no attribute '{}'".format(item))
