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
String functions on Koalas Series
"""
from typing import TYPE_CHECKING

import numpy as np

from pyspark.sql.types import StringType, BinaryType, BooleanType

from databricks.koalas.base import _wrap_accessor_pandas

if TYPE_CHECKING:
    import databricks.koalas as ks


class StringMethods(object):
    """String methods for Koalas Series"""
    def __init__(self, series: 'ks.Series'):
        if not isinstance(series.spark_type, (StringType, BinaryType)):
            raise ValueError(
                "Cannot call StringMethods on type {}"
                .format(series.spark_type))
        self._data = series
        self.name = self._data.name

    # Methods
    def capitalize(self) -> 'ks.Series':
        """
        Convert Strings in the series to be capitalized.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.capitalize(),
            StringType()
        ).alias(self.name)

    def lower(self) -> 'ks.Series':
        """
        Convert strings in the Series/Index to all lowercase.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.lower(),
            StringType()
        ).alias(self.name)

    def upper(self) -> 'ks.Series':
        """
        Convert strings in the Series/Index to all uppercase.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.upper(),
            StringType()
        ).alias(self.name)

    def swapcase(self) -> 'ks.Series':
        """
        Convert strings in the Series/Index to be swapcased.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.swapcase(),
            StringType()
        ).alias(self.name)

    def startswith(self, pattern, na=np.NaN) -> 'ks.Series':
        """
        Test if the start of each string element matches a pattern.

        Equivalent to :func:`str.startswith`.

        Parameters
        ----------
        pattern : str
            Character sequence. Regular expressions are not accepted.
        na : object, defulat NaN
            Object shown if element is not a string.

        Returns
        -------
        Series of bool
            Koalas Series of booleans indicating whether the given pattern
            matches the start of each string element.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.startswith(pattern, na),
            BooleanType()
        ).alias(self.name)

    def endswith(self, pattern, na=np.NaN) -> 'ks.Series':
        """
        Test if the end of each string element matches a pattern.

        Equivalent to :func:`str.endswith`.

        Parameters
        ----------
        pattern : str
            Character sequence. Regular expressions are not accepted.
        na : object, defulat NaN
            Object shown if element is not a string.

        Returns
        -------
        Series of bool
            Koalas Series of booleans indicating whether the given pattern
            matches the end of each string element.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.endswith(pattern, na),
            BooleanType()
        ).alias(self.name)

    def strip(self, to_strip=None) -> 'ks.Series':
        """
        Remove leading and trailing characters.

        Strip whitespaces (including newlines) or a set of specified
        characters from each string in the Series/Index from left and
        right sides. Equivalent to :func:`str.strip`.

        Parameters
        ----------
        to_strip : str
            Specifying the set of characters to be removed. All combinations
            of this set of characters will be stripped. If None then
            whitespaces are removed.

        Returns
        -------
        Series of str
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.strip(to_strip),
            StringType()
        ).alias(self.name)

    def lstrip(self, to_strip=None) -> 'ks.Series':
        """
        Remove leading characters.

        Strip whitespaces (including newlines) or a set of specified
        characters from each string in the Series/Index from left side.
        Equivalent to :func:`str.lstrip`.

        Parameters
        ----------
        to_strip : str
            Specifying the set of characters to be removed. All combinations
            of this set of characters will be stripped. If None then
            whitespaces are removed.

        Returns
        -------
        Series of str
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.lstrip(to_strip),
            StringType()
        ).alias(self.name)

    def rstrip(self, to_strip=None) -> 'ks.Series':
        """
        Remove trailing characters.

        Strip whitespaces (including newlines) or a set of specified
        characters from each string in the Series/Index from right side.
        Equivalent to :func:`str.rstrip`.

        Parameters
        ----------
        to_strip : str
            Specifying the set of characters to be removed. All combinations
            of this set of characters will be stripped. If None then
            whitespaces are removed.

        Returns
        -------
        Series of str
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.rstrip(to_strip),
            StringType()
        ).alias(self.name)

    def get(self, i) -> 'ks.Series':
        """
        Extract element from each string in the Series/Index at the
        specified position.

        Parameters
        ----------
        i : int
            Position of element to extract.

        Returns
        -------
        Series of objects
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.get(i),
            StringType()
        ).alias(self.name)

    def isalnum(self) -> 'ks.Series':
        """
        Check whether all characters in each string are alphanumeric.

        This is equivalent to running the Python string method
        :func:`str.isalnum` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.isalnum(),
            BooleanType()
        ).alias(self.name)

    def isalpha(self) -> 'ks.Series':
        """
        Check whether all characters in each string are alphabetic.

        This is equivalent to running the Python string method
        :func:`str.isalpha` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.isalpha(),
            BooleanType()
        ).alias(self.name)

    def isdigit(self) -> 'ks.Series':
        """
        Check whether all characters in each string are digits.

        This is equivalent to running the Python string method
        :func:`str.isdigit` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.isdigit(),
            BooleanType()
        ).alias(self.name)

    def isspace(self) -> 'ks.Series':
        """
        Check whether all characters in each string are whitespaces.

        This is equivalent to running the Python string method
        :func:`str.isspace` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.isspace(),
            BooleanType()
        ).alias(self.name)

    def islower(self) -> 'ks.Series':
        """
        Check whether all characters in each string are lowercase.

        This is equivalent to running the Python string method
        :func:`str.islower` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.islower(),
            BooleanType()
        ).alias(self.name)

    def isupper(self) -> 'ks.Series':
        """
        Check whether all characters in each string are uppercase.

        This is equivalent to running the Python string method
        :func:`str.isupper` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.isupper(),
            BooleanType()
        ).alias(self.name)

    def istitle(self) -> 'ks.Series':
        """
        Check whether all characters in each string are titlecase.

        This is equivalent to running the Python string method
        :func:`str.istitle` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.istitle(),
            BooleanType()
        ).alias(self.name)

    def isnumeric(self) -> 'ks.Series':
        """
        Check whether all characters in each string are numeric.

        This is equivalent to running the Python string method
        :func:`str.isnumeric` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.isnumeric(),
            BooleanType()
        ).alias(self.name)

    def isdecimal(self) -> 'ks.Series':
        """
        Check whether all characters in each string are decimals.

        This is equivalent to running the Python string method
        :func:`str.isdecimal` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.isdecimal(),
            BooleanType()
        ).alias(self.name)
