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
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.types import DateType, TimestampType, LongType, StringType

import databricks.koalas as ks
from databricks.koalas.base import _wrap_accessor_pandas, _wrap_accessor_spark


class DatetimeMethods(object):
    """Date/Time methods for Koalas Series"""
    def __init__(self, series: ks.Series):
        if not isinstance(series.spark_type, (DateType, TimestampType)):
            raise ValueError(
                "Cannot call DatetimeMethods on type {}"
                .format(series.spark_type))
        self._data = series
        self.name = self._data.name

    # Properties
    @property
    def date(self) -> ks.Series:
        """
        Returns a Series of python datetime.date objects (namely, the date
        part of Timestamps without timezone information).
        """
        # TODO: Hit a weird exception
        # syntax error in attribute name: `to_date(`start_date`)` with alias
        return _wrap_accessor_spark(
            self, lambda col: F.to_date(col)).alias(self.name)

    @property
    def time(self) -> ks.Series:
        raise NotImplementedError()

    @property
    def timetz(self) -> ks.Series:
        raise NotImplementedError()

    @property
    def year(self) -> ks.Series:
        """
        The year of the datetime.
        `"""
        return _wrap_accessor_spark(self, F.year, LongType()).alias(self.name)

    @property
    def month(self) -> ks.Series:
        """
        The month of the timestamp as January = 1 December = 12.
        """
        return _wrap_accessor_spark(self, F.month, LongType()).alias(self.name)

    @property
    def day(self) -> ks.Series:
        """
        The days of the datetime.
        """
        return _wrap_accessor_spark(
            self, F.dayofmonth, LongType()).alias(self.name)

    @property
    def hour(self) -> ks.Series:
        """
        The hours of the datetime.
        """
        return _wrap_accessor_spark(self, F.hour, LongType()).alias(self.name)

    @property
    def minute(self) -> ks.Series:
        """
        The minutes of the datetime.
        """
        return _wrap_accessor_spark(self, F.minute, LongType()).alias(self.name)

    @property
    def second(self) -> ks.Series:
        """
        The seconds of the datetime.
        """
        return _wrap_accessor_spark(self, F.second, LongType()).alias(self.name)

    @property
    def millisecond(self) -> ks.Series:
        """
        The milliseconds of the datetime.
        """
        return _wrap_accessor_pandas(
            self, lambda x: x.dt.millisecond, LongType()).alias(self.name)

    @property
    def microsecond(self) -> ks.Series:
        """
        The microseconds of the datetime.
        """
        return _wrap_accessor_pandas(
            self, lambda x: x.dt.microsecond, LongType()).alias(self.name)

    @property
    def nanosecond(self) -> ks.Series:
        raise NotImplementedError()

    @property
    def week(self) -> ks.Series:
        """
        The week ordinal of the year.
        """
        return _wrap_accessor_spark(self, F.weekofyear, LongType()).alias(self.name)

    @property
    def weekofyear(self) -> ks.Series:
        return self.week

    weekofyear.__doc__ = week.__doc__

    @property
    def dayofweek(self) -> ks.Series:
        """
        The day of the week with Monday=0, Sunday=6.

        Return the day of the week. It is assumed the week starts on
        Monday, which is denoted by 0 and ends on Sunday which is denoted
        by 6. This method is available on both Series with datetime
        values (using the `dt` accessor) or DatetimeIndex.

        Returns
        -------
        Series or Index
            Containing integers indicating the day number.

        See Also
        --------
        Series.dt.dayofweek : Alias.
        Series.dt.weekday : Alias.
        Series.dt.day_name : Returns the name of the day of the week.

        Examples
        --------
        >>> s = ks.from_pandas(pd.date_range('2016-12-31', '2017-01-08', freq='D').to_series())
        >>> s.dt.dayofweek
        2016-12-31    5
        2017-01-01    6
        2017-01-02    0
        2017-01-03    1
        2017-01-04    2
        2017-01-05    3
        2017-01-06    4
        2017-01-07    5
        2017-01-08    6
        Name: 0, dtype: int64
        """
        return _wrap_accessor_pandas(
            self, lambda s: s.dt.dayofweek, LongType()).alias(self._data.name)

    @property
    def weekday(self) -> ks.Series:
        return self.dayofweek

    weekday.__doc__ = dayofweek.__doc__

    @property
    def dayofyear(self) -> ks.Series:
        """
        The ordinal day of the year.
        """
        return _wrap_accessor_pandas(
            self, lambda s: s.dt.dayofyear, LongType()).alias(self._data.name)

    # Methods
    def strftime(self, date_format) -> ks.Series:
        """
        Convert to a String Series using specified date_format.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.dt.strftime(date_format),
            StringType()
        ).alias(self.name)
