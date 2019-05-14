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
from databricks.koalas.series import Series, _column_op
from databricks.koalas.typedef import pandas_wraps
from databricks.koalas.utils import lazy_property


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
    def week(self) -> ks.Series:
        """
        The week ordinal of the year.
        :return:
        """
        return _column_op(
            lambda col: F.weekofyear(col).cast(LongType())
        )(self._data)

    @lazy_property
    def weekofyear(self) -> ks.Series:
        """
        The week ordinal of the year.
        """
        return _column_op(
            lambda col: F.weekofyear(col).cast(LongType())
        )(self._data)

    @lazy_property
    def dayofweek(self) -> ks.Series:
        """
        The day of the week with Monday=0, Sunday=6.
        """
        return pandas_wraps(
            function=lambda s: s.dt.dayofyear,
            return_col=StringType()
        )(self._data)

    @lazy_property
    def dayofyear(self) -> ks.Series:
        """
        The ordinal day of the year.
        """
        return pandas_wraps(
            function=lambda s: s.dt.dayofyear,
            return_col=StringType()
        )(self._data)

    # Methods
    def strftime(self, date_format) -> ks.Series:
        """
        Convert to String Series using specified date_format.
        """
        return pandas_wraps(
            function=lambda x: x.dt.strftime(date_format),
            return_col=StringType()
        )(self._data)
