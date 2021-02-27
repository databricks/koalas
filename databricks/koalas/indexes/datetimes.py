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
from typing import Any

import pandas as pd
from pandas.api.types import is_hashable
from pyspark._globals import _NoValue

from databricks import koalas as ks
from databricks.koalas.indexes.base import Index
from databricks.koalas.missing.indexes import MissingPandasLikeDatetimeIndex


class DatetimeIndex(Index):
    """
    Immutable ndarray-like of datetime64 data.

    Represented internally as int64, and which can be boxed to Timestamp objects
    that are subclasses of datetime and carry metadata.

    Parameters
    ----------
    data : array-like (1-dimensional), optional
        Optional datetime-like data to construct index with.
    freq : str or pandas offset object, optional
        One of pandas date offset strings or corresponding objects. The string
        'infer' can be passed in order to set the frequency of the index as the
        inferred frequency upon creation.
    normalize : bool, default False
        Normalize start/end dates to midnight before generating date range.
    closed : {'left', 'right'}, optional
        Set whether to include `start` and `end` that are on the
        boundary. The default includes boundary points on either end.
    ambiguous : 'infer', bool-ndarray, 'NaT', default 'raise'
        When clocks moved backward due to DST, ambiguous times may arise.
        For example in Central European Time (UTC+01), when going from 03:00
        DST to 02:00 non-DST, 02:30:00 local time occurs both at 00:30:00 UTC
        and at 01:30:00 UTC. In such a situation, the `ambiguous` parameter
        dictates how ambiguous times should be handled.

        - 'infer' will attempt to infer fall dst-transition hours based on
          order
        - bool-ndarray where True signifies a DST time, False signifies a
          non-DST time (note that this flag is only applicable for ambiguous
          times)
        - 'NaT' will return NaT where there are ambiguous times
        - 'raise' will raise an AmbiguousTimeError if there are ambiguous times.
    dayfirst : bool, default False
        If True, parse dates in `data` with the day first order.
    yearfirst : bool, default False
        If True parse dates in `data` with the year first order.
    dtype : numpy.dtype or DatetimeTZDtype or str, default None
        Note that the only NumPy dtype allowed is ‘datetime64[ns]’.
    copy : bool, default False
        Make a copy of input ndarray.
    name : label, default None
        Name to be stored in the index.

    See Also
    --------
    Index : The base pandas Index type.
    to_datetime : Convert argument to datetime.

    Examples
    --------
    >>> ks.DatetimeIndex(['1970-01-01', '1970-01-01', '1970-01-01'])
    DatetimeIndex(['1970-01-01', '1970-01-01', '1970-01-01'], dtype='datetime64[ns]', freq=None)
    """

    def __new__(
        cls,
        data=None,
        freq=_NoValue,
        normalize=False,
        closed=None,
        ambiguous="raise",
        dayfirst=False,
        yearfirst=False,
        dtype=None,
        copy=False,
        name=None,
    ):
        if not is_hashable(name):
            raise TypeError("Index.name must be a hashable type")

        kwargs = dict(
            data=data,
            normalize=normalize,
            closed=closed,
            ambiguous=ambiguous,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            dtype=dtype,
            copy=copy,
            name=name,
        )
        if freq is not _NoValue:
            kwargs["freq"] = freq
        return ks.from_pandas(pd.DatetimeIndex(**kwargs))

    def __getattr__(self, item: str) -> Any:
        if hasattr(MissingPandasLikeDatetimeIndex, item):
            property_or_func = getattr(MissingPandasLikeDatetimeIndex, item)
            if isinstance(property_or_func, property):
                return property_or_func.fget(self)  # type: ignore
            else:
                return partial(property_or_func, self)
        raise AttributeError("'DatetimeIndex' object has no attribute '{}'".format(item))
