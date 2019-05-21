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

import databricks.koalas as ks
import pyspark.sql.functions as F
from pyspark.sql.types import DateType, TimestampType, LongType, StringType
from databricks.koalas.series import (
    Series,
    _wrap_accessor_pandas,
    _wrap_accessor_spark
)
from databricks.koalas.utils import lazy_property


class DatetimeMethods(object):
    """Date/Time methods for Koalas Series"""
    def __init__(self, series):
        if not isinstance(series.spark_type, (DateType, TimestampType)):
            raise ValueError(
                "Cannot call DatetimeMethods on type {}"
                .format(series.spark_type))
        self._data = series

    # Properties
    @lazy_property
    def date(self) -> ks.Series:
        """
        The date part of the datetime.
        """
        # TODO: Hit a weird exception
        # syntax error in attribute name: `to_date(`start_date`)` with alias
        return _wrap_accessor_spark(
            self, lambda col: F.to_date(col).alias('date')
        )

    @lazy_property
    def time(self) -> ks.Series:
        raise NotImplementedError()

    @lazy_property
    def timetz(self) -> ks.Series:
        raise NotImplementedError()

    @lazy_property
    def year(self) -> ks.Series:
        """
        The year of the datetime.
        `"""
        return _wrap_accessor_spark(self, F.year, LongType())

    @lazy_property
    def month(self) -> ks.Series:
        """
        The month of the timestamp as January = 1 December = 12.
        """
        return _wrap_accessor_spark(self, F.month, LongType())

    @lazy_property
    def day(self) -> ks.Series:
        """
        The days of the datetime.
        """
        return _wrap_accessor_spark(self, F.dayofmonth, LongType())

    @lazy_property
    def hour(self) -> ks.Series:
        """
        The hours of the datetime.
        """
        return _wrap_accessor_spark(self, F.hour, LongType())

    @lazy_property
    def minute(self) -> ks.Series:
        """
        The minutes of the datetime.
        """
        return _wrap_accessor_spark(self, F.minute, LongType())

    @lazy_property
    def second(self) -> ks.Series:
        """
        The seconds of the datetime.
        """
        return _wrap_accessor_spark(self, F.second, LongType())

    @lazy_property
    def millisecond(self) -> ks.Series:
        """
        The milliseconds of the datetime.
        """
        return _wrap_accessor_pandas(
            self, lambda x: x.dt.millisecond, LongType())

    @lazy_property
    def microsecond(self) -> ks.Series:
        """
        The microseconds of the datetime.
        """
        return _wrap_accessor_pandas(
            self, lambda x: x.dt.microsecond, LongType())

    @lazy_property
    def nanosecond(self) -> ks.Series:
        raise NotImplementedError()

    @lazy_property
    def week(self) -> ks.Series:
        """
        The week ordinal of the year.
        """
        return _wrap_accessor_spark(self, F.weekofyear, LongType())

    @lazy_property
    def weekofyear(self) -> ks.Series:
        """
        The week ordinal of the year.
        """
        return _wrap_accessor_spark(self, F.weekofyear, LongType())

    @lazy_property
    def dayofweek(self) -> ks.Series:
        """
        The day of the week with Monday=0, Sunday=6.
        """
        return _wrap_accessor_pandas(self, lambda s: s.dt.dayofweek, LongType())

    @lazy_property
    def dayofyear(self) -> ks.Series:
        """
        The day ordinal of the year.
        """
        return _wrap_accessor_pandas(self, lambda s: s.dt.dayofyear, LongType())

    # Methods
    def strftime(self, date_format) -> ks.Series:
        """
        Convert to a String Series using specified date_format.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.dt.strftime(date_format),
            StringType()
        )
