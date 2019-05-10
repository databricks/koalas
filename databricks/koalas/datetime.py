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
Date/Time related functions on Koalas Series
"""

import functools
from pyspark.sql.types import DateType, TimestampType, LongType, StringType

from databricks.koalas.series import Series
from databricks.koalas.utils import lazy_property

import pyspark.sql.functions as F


def defer_to_pandas(output_type):
    """Wraps a function that operates on pd.Series to work with DatetimeMethods.

    The origin function operates on pd.Series and the wrapped function operates
    on DatetimeMethods.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args):
            scol = F.pandas_udf(
                lambda series: func(series, *args),
                output_type
            )(self._data._scol)
            return Series(
                scol, anchor=self._data._kdf, index=self._data._index_info)
        return wrapper
    return decorator


def defer_to_spark(output_type):
    """Wraps a function that operates on spark.sql.Column to work with DatetimeMethods.

    The origin function operates on spark.sql.Column and the wrapped function
    operates on DatetimeMethods.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args):
            scol = func(self._data._scol, *args).cast(output_type)
            return Series(scol, anchor=self._data._kdf, index=self._data._index_info)
        return wrapper
    return decorator


class DatetimeMethods:
    """Date/Time methods for Koalas Series"""
    def __init__(self, series):
        if not isinstance(series.spark_type, (DateType, TimestampType)):
            raise ValueError(
                "Cannot call DatetimeMethods on type {}"
                .format(series.spark_type))
        self._data = series

    # Properties
    @lazy_property
    @defer_to_pandas(output_type=DateType())
    def date(s):
        return s.dt.date

    @lazy_property
    def time(self):
        raise NotImplementedError()

    @lazy_property
    def timetz(self):
        raise NotImplementedError()

    @lazy_property
    @defer_to_spark(output_type=LongType())
    def year(col):
        return F.year(col)

    @lazy_property
    @defer_to_spark(output_type=LongType())
    def month(col):
        return F.month(col)

    @lazy_property
    @defer_to_spark(output_type=LongType())
    def day(col):
        return F.dayofmonth(col)

    @lazy_property
    @defer_to_spark(output_type=LongType())
    def hour(col):
        return F.hour(col)

    @lazy_property
    @defer_to_spark(output_type=LongType())
    def minute(col):
        return F.minute(col)

    @lazy_property
    @defer_to_spark(output_type=LongType())
    def second(col):
        return F.second(col)

    @lazy_property
    @defer_to_spark(output_type=LongType())
    def millisecond(col):
        return F.millisecond(col)

    @lazy_property
    @defer_to_pandas(output_type=LongType())
    def microsecond(s):
        return s.dt.microsecond

    @lazy_property
    def nanosecond(self):
        raise NotImplementedError()

    @lazy_property
    @defer_to_spark(output_type=LongType())
    def week(col):
        return F.weekofyear(col)

    @lazy_property
    @defer_to_spark(output_type=LongType())
    def weekofyear(col):
        return F.weekofyear(col)

    @lazy_property
    @defer_to_pandas(output_type=LongType())
    def dayofweek(s):
        return s.dt.dayofweek

    @lazy_property
    @defer_to_spark(output_type=LongType())
    def dayofyear(col):
        return F.dayofyear(col)

    # Methods
    @defer_to_pandas(output_type=StringType())
    def strftime(s, date_format):
        return s.dt.strftime(date_format)
