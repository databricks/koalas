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
A class of string methods on Koalas Series.
"""
from functools import partial
from pyspark.sql.types import StringType, BinaryType

from databricks.koalas import series


class StringMethods:

    def __init__(self, koalas_series):
        self._series = koalas_series

    def capitalize(self):
        return series._pandas_column_op(lambda x: x.str.capitalize(), StringType(),
                                        'capitalize')(self._series)

    def encode(self, encoding):
        return series._pandas_column_op(partial(lambda x, e: x.str.encode(e), e=encoding),
                                        BinaryType(), 'encode')(self._series)

    def decode(self, encoding):
        return series._pandas_column_op(partial(lambda x, e: x.str.decode(e), e=encoding),
                                        StringType(), 'decode')(self._series)
