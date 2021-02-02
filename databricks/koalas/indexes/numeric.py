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

from databricks.koalas.indexes.base import Index


class NumericIndex(Index):
    """
    Provide numeric type operations.
    This is an abstract class.
    """

    pass


class IntegerIndex(NumericIndex):
    """
    This is an abstract class for Int64Index.
    """

    pass


class Int64Index(IntegerIndex):
    """
    Immutable sequence used for indexing and alignment. The basic object
    storing axis labels for all pandas objects. Int64Index is a special case
    of `Index` with purely integer labels.

    See Also
    --------
    Index : The base pandas Index type.

    Notes
    -----
    An Index instance can **only** contain hashable objects.
    """

    pass


class Float64Index(NumericIndex):
    """
    Immutable sequence used for indexing and alignment. The basic object
    storing axis labels for all pandas objects. Float64Index is a special case
    of `Index` with purely float labels.

    See Also
    --------
    Index : The base pandas Index type.

    Notes
    -----
    An Index instance can **only** contain hashable objects.
    """

    pass
