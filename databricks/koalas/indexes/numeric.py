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
import pandas as pd
from pandas.api.types import is_hashable

from databricks import koalas as ks
from databricks.koalas.indexes.base import Index


class NumericIndex(Index):
    """
    Provide numeric type operations.
    This is an abstract class.
    """

    pass


_class_descr = """
    Immutable ndarray implementing an ordered, sliceable set. The basic object
    storing axis labels for all pandas objects. {klass}s is a special case
    of `Index` with purely {ltype}s labels.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype (default: {dtype}s)
    copy : bool
        Make a copy of input ndarray.
    name : object
        Name to be stored in the index.

    See Also
    --------
    Index : The base pandas Index type.

    Notes
    -----
    An Index instance can **only** contain hashable objects.
"""


class IntegerIndex(NumericIndex):
    """
    This is an abstract class for Int64Index.
    """

    pass


class Int64Index(IntegerIndex):
    __doc__ = (
        _class_descr.format(klass="Int64Index", ltype="integer", dtype="int64")
        + """

    Examples
    --------
    >>> ks.Int64Index([1, 2, 3])
    Int64Index([1, 2, 3], dtype='int64')
    """
    )

    def __new__(cls, data=None, dtype=None, copy=False, name=None):
        if not is_hashable(name):
            raise TypeError("Index.name must be a hashable type")

        return ks.from_pandas(pd.Int64Index(data=data, dtype=dtype, copy=copy, name=name))


class Float64Index(NumericIndex):
    __doc__ = (
        _class_descr.format(klass="Float64Index", ltype="float", dtype="float64")
        + """

    Examples
    --------
    >>> ks.Float64Index([1.0, 2.0, 3.0])
    Float64Index([1.0, 2.0, 3.0], dtype='float64')
    """
    )

    def __new__(cls, data=None, dtype=None, copy=False, name=None):
        if not is_hashable(name):
            raise TypeError("Index.name must be a hashable type")

        return ks.from_pandas(pd.Float64Index(data=data, dtype=dtype, copy=copy, name=name))
