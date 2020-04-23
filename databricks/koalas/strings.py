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
from typing import Union, TYPE_CHECKING

import numpy as np

from pyspark.sql.types import StringType, BinaryType, ArrayType, LongType, MapType
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType

from databricks.koalas.base import _column_op

if TYPE_CHECKING:
    import databricks.koalas as ks


class StringMethods(object):
    """String methods for Koalas Series"""

    def __init__(self, series: "ks.Series"):
        if not isinstance(series.spark_type, (StringType, BinaryType, ArrayType)):
            raise ValueError("Cannot call StringMethods on type {}".format(series.spark_type))
        self._data = series
        self.name = self._data.name

    # Methods
    def capitalize(self) -> "ks.Series":
        """
        Convert Strings in the series to be capitalized.

        Examples
        --------
        >>> s = ks.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        Name: 0, dtype: object

        >>> s.str.capitalize()
        0                 Lower
        1              Capitals
        2    This is a sentence
        3              Swapcase
        Name: 0, dtype: object
        """

        def pandas_capitalize(s) -> "ks.Series[str]":
            return s.str.capitalize()

        return self._data.transform_batch(pandas_capitalize)

    def title(self) -> "ks.Series":
        """
        Convert Strings in the series to be titlecase.

        Examples
        --------
        >>> s = ks.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        Name: 0, dtype: object

        >>> s.str.title()
        0                 Lower
        1              Capitals
        2    This Is A Sentence
        3              Swapcase
        Name: 0, dtype: object
        """

        def pandas_title(s) -> "ks.Series[str]":
            return s.str.title()

        return self._data.transform_batch(pandas_title)

    def lower(self) -> "ks.Series":
        """
        Convert strings in the Series/Index to all lowercase.

        Examples
        --------
        >>> s = ks.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        Name: 0, dtype: object

        >>> s.str.lower()
        0                 lower
        1              capitals
        2    this is a sentence
        3              swapcase
        Name: 0, dtype: object
        """
        return _column_op(lambda c: F.lower(c))(self._data).alias(self._data.name)

    def upper(self) -> "ks.Series":
        """
        Convert strings in the Series/Index to all uppercase.

        Examples
        --------
        >>> s = ks.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        Name: 0, dtype: object

        >>> s.str.upper()
        0                 LOWER
        1              CAPITALS
        2    THIS IS A SENTENCE
        3              SWAPCASE
        Name: 0, dtype: object
        """
        return _column_op(lambda c: F.upper(c))(self._data).alias(self._data.name)

    def swapcase(self) -> "ks.Series":
        """
        Convert strings in the Series/Index to be swapcased.

        Examples
        --------
        >>> s = ks.Series(['lower', 'CAPITALS', 'this is a sentence', 'SwApCaSe'])
        >>> s
        0                 lower
        1              CAPITALS
        2    this is a sentence
        3              SwApCaSe
        Name: 0, dtype: object

        >>> s.str.swapcase()
        0                 LOWER
        1              capitals
        2    THIS IS A SENTENCE
        3              sWaPcAsE
        Name: 0, dtype: object
        """

        def pandas_swapcase(s) -> "ks.Series[str]":
            return s.str.swapcase()

        return self._data.transform_batch(pandas_swapcase)

    def startswith(self, pattern, na=None) -> "ks.Series":
        """
        Test if the start of each string element matches a pattern.

        Equivalent to :func:`str.startswith`.

        Parameters
        ----------
        pattern : str
            Character sequence. Regular expressions are not accepted.
        na : object, default None
            Object shown if element is not a string. NaN converted to None.

        Returns
        -------
        Series of bool or object
            Koalas Series of booleans indicating whether the given pattern
            matches the start of each string element.

        Examples
        --------
        >>> s = ks.Series(['bat', 'Bear', 'cat', np.nan])
        >>> s
        0     bat
        1    Bear
        2     cat
        3    None
        Name: 0, dtype: object

        >>> s.str.startswith('b')
        0     True
        1    False
        2    False
        3     None
        Name: 0, dtype: object

        Specifying na to be False instead of None.

        >>> s.str.startswith('b', na=False)
        0     True
        1    False
        2    False
        3    False
        Name: 0, dtype: bool
        """

        def pandas_startswith(s) -> "ks.Series[bool]":
            return s.str.startswith(pattern, na)

        return self._data.transform_batch(pandas_startswith)

    def endswith(self, pattern, na=None) -> "ks.Series":
        """
        Test if the end of each string element matches a pattern.

        Equivalent to :func:`str.endswith`.

        Parameters
        ----------
        pattern : str
            Character sequence. Regular expressions are not accepted.
        na : object, default None
            Object shown if element is not a string. NaN converted to None.

        Returns
        -------
        Series of bool or object
            Koalas Series of booleans indicating whether the given pattern
            matches the end of each string element.

        Examples
        --------
        >>> s = ks.Series(['bat', 'Bear', 'cat', np.nan])
        >>> s
        0     bat
        1    Bear
        2     cat
        3    None
        Name: 0, dtype: object

        >>> s.str.endswith('t')
        0     True
        1    False
        2     True
        3     None
        Name: 0, dtype: object

        Specifying na to be False instead of None.

        >>> s.str.endswith('t', na=False)
        0     True
        1    False
        2     True
        3    False
        Name: 0, dtype: bool
        """

        def pandas_endswith(s) -> "ks.Series[bool]":
            return s.str.endswith(pattern, na)

        return self._data.transform_batch(pandas_endswith)

    def strip(self, to_strip=None) -> "ks.Series":
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
        Series of objects

        Examples
        --------
        >>> s = ks.Series(['1. Ant.', '2. Bee!\\t', None])
        >>> s
        0      1. Ant.
        1    2. Bee!\\t
        2         None
        Name: 0, dtype: object

        >>> s.str.strip()
        0    1. Ant.
        1    2. Bee!
        2       None
        Name: 0, dtype: object

        >>> s.str.strip('12.')
        0        Ant
        1     Bee!\\t
        2       None
        Name: 0, dtype: object

        >>> s.str.strip('.!\\t')
        0    1. Ant
        1    2. Bee
        2      None
        Name: 0, dtype: object
        """

        def pandas_strip(s) -> "ks.Series[str]":
            return s.str.strip(to_strip)

        return self._data.transform_batch(pandas_strip)

    def lstrip(self, to_strip=None) -> "ks.Series":
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
        Series of object

        Examples
        --------
        >>> s = ks.Series(['1. Ant.', '2. Bee!\\t', None])
        >>> s
        0      1. Ant.
        1    2. Bee!\\t
        2         None
        Name: 0, dtype: object

        >>> s.str.lstrip('12.')
        0       Ant.
        1     Bee!\\t
        2       None
        Name: 0, dtype: object
        """

        def pandas_lstrip(s) -> "ks.Series[str]":
            return s.str.lstrip(to_strip)

        return self._data.transform_batch(pandas_lstrip)

    def rstrip(self, to_strip=None) -> "ks.Series":
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
        Series of object

        Examples
        --------
        >>> s = ks.Series(['1. Ant.', '2. Bee!\\t', None])
        >>> s
        0      1. Ant.
        1    2. Bee!\\t
        2         None
        Name: 0, dtype: object

        >>> s.str.rstrip('.!\\t')
        0    1. Ant
        1    2. Bee
        2      None
        Name: 0, dtype: object
        """

        def pandas_rstrip(s) -> "ks.Series[str]":
            return s.str.rstrip(to_strip)

        return self._data.transform_batch(pandas_rstrip)

    def get(self, i) -> "ks.Series":
        """
        Extract element from each string or string list/tuple in the Series
        at the specified position.

        Parameters
        ----------
        i : int
            Position of element to extract.

        Returns
        -------
        Series of objects

        Examples
        --------
        >>> s1 = ks.Series(["String", "123"])
        >>> s1
        0    String
        1       123
        Name: 0, dtype: object

        >>> s1.str.get(1)
        0    t
        1    2
        Name: 0, dtype: object

        >>> s1.str.get(-1)
        0    g
        1    3
        Name: 0, dtype: object

        >>> s2 = ks.Series([["a", "b", "c"], ["x", "y"]])
        >>> s2
        0    [a, b, c]
        1       [x, y]
        Name: 0, dtype: object

        >>> s2.str.get(0)
        0    a
        1    x
        Name: 0, dtype: object

        >>> s2.str.get(2)
        0       c
        1    None
        Name: 0, dtype: object
        """

        def pandas_get(s) -> "ks.Series[str]":
            return s.str.get(i)

        return self._data.transform_batch(pandas_get)

    def isalnum(self) -> "ks.Series":
        """
        Check whether all characters in each string are alphanumeric.

        This is equivalent to running the Python string method
        :func:`str.isalnum` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s1 = ks.Series(['one', 'one1', '1', ''])

        >>> s1.str.isalnum()
        0     True
        1     True
        2     True
        3    False
        Name: 0, dtype: bool

        Note that checks against characters mixed with any additional
        punctuation or whitespace will evaluate to false for an alphanumeric
        check.

        >>> s2 = ks.Series(['A B', '1.5', '3,000'])
        >>> s2.str.isalnum()
        0    False
        1    False
        2    False
        Name: 0, dtype: bool
        """

        def pandas_isalnum(s) -> "ks.Series[bool]":
            return s.str.isalnum()

        return self._data.transform_batch(pandas_isalnum)

    def isalpha(self) -> "ks.Series":
        """
        Check whether all characters in each string are alphabetic.

        This is equivalent to running the Python string method
        :func:`str.isalpha` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s1 = ks.Series(['one', 'one1', '1', ''])

        >>> s1.str.isalpha()
        0     True
        1    False
        2    False
        3    False
        Name: 0, dtype: bool
        """

        def pandas_isalpha(s) -> "ks.Series[bool]":
            return s.str.isalpha()

        return self._data.transform_batch(pandas_isalpha)

    def isdigit(self) -> "ks.Series":
        """
        Check whether all characters in each string are digits.

        This is equivalent to running the Python string method
        :func:`str.isdigit` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s = ks.Series(['23', '³', '⅕', ''])

        The s.str.isdecimal method checks for characters used to form numbers
        in base 10.

        >>> s.str.isdecimal()
        0     True
        1    False
        2    False
        3    False
        Name: 0, dtype: bool

        The s.str.isdigit method is the same as s.str.isdecimal but also
        includes special digits, like superscripted and subscripted digits in
        unicode.

        >>> s.str.isdigit()
        0     True
        1     True
        2    False
        3    False
        Name: 0, dtype: bool

        The s.str.isnumeric method is the same as s.str.isdigit but also
        includes other characters that can represent quantities such as unicode
        fractions.

        >>> s.str.isnumeric()
        0     True
        1     True
        2     True
        3    False
        Name: 0, dtype: bool
        """

        def pandas_isdigit(s) -> "ks.Series[bool]":
            return s.str.isdigit()

        return self._data.transform_batch(pandas_isdigit)

    def isspace(self) -> "ks.Series":
        """
        Check whether all characters in each string are whitespaces.

        This is equivalent to running the Python string method
        :func:`str.isspace` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s = ks.Series([' ', '\\t\\r\\n ', ''])
        >>> s.str.isspace()
        0     True
        1     True
        2    False
        Name: 0, dtype: bool
        """

        def pandas_isspace(s) -> "ks.Series[bool]":
            return s.str.isspace()

        return self._data.transform_batch(pandas_isspace)

    def islower(self) -> "ks.Series":
        """
        Check whether all characters in each string are lowercase.

        This is equivalent to running the Python string method
        :func:`str.islower` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s = ks.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])
        >>> s.str.islower()
        0     True
        1    False
        2    False
        3    False
        Name: 0, dtype: bool
        """

        def pandas_isspace(s) -> "ks.Series[bool]":
            return s.str.islower()

        return self._data.transform_batch(pandas_isspace)

    def isupper(self) -> "ks.Series":
        """
        Check whether all characters in each string are uppercase.

        This is equivalent to running the Python string method
        :func:`str.isupper` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s = ks.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])
        >>> s.str.isupper()
        0    False
        1    False
        2     True
        3    False
        Name: 0, dtype: bool
        """

        def pandas_isspace(s) -> "ks.Series[bool]":
            return s.str.isupper()

        return self._data.transform_batch(pandas_isspace)

    def istitle(self) -> "ks.Series":
        """
        Check whether all characters in each string are titlecase.

        This is equivalent to running the Python string method
        :func:`str.istitle` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s = ks.Series(['leopard', 'Golden Eagle', 'SNAKE', ''])

        The s.str.istitle method checks for whether all words are in title
        case (whether only the first letter of each word is capitalized).
        Words are assumed to be as any sequence of non-numeric characters
        separated by whitespace characters.

        >>> s.str.istitle()
        0    False
        1     True
        2    False
        3    False
        Name: 0, dtype: bool
        """

        def pandas_istitle(s) -> "ks.Series[bool]":
            return s.str.istitle()

        return self._data.transform_batch(pandas_istitle)

    def isnumeric(self) -> "ks.Series":
        """
        Check whether all characters in each string are numeric.

        This is equivalent to running the Python string method
        :func:`str.isnumeric` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s1 = ks.Series(['one', 'one1', '1', ''])
        >>> s1.str.isnumeric()
        0    False
        1    False
        2     True
        3    False
        Name: 0, dtype: bool

        >>> s2 = ks.Series(['23', '³', '⅕', ''])

        The s2.str.isdecimal method checks for characters used to form numbers
        in base 10.

        >>> s2.str.isdecimal()
        0     True
        1    False
        2    False
        3    False
        Name: 0, dtype: bool

        The s2.str.isdigit method is the same as s2.str.isdecimal but also
        includes special digits, like superscripted and subscripted digits in
        unicode.

        >>> s2.str.isdigit()
        0     True
        1     True
        2    False
        3    False
        Name: 0, dtype: bool

        The s2.str.isnumeric method is the same as s2.str.isdigit but also
        includes other characters that can represent quantities such as unicode
        fractions.

        >>> s2.str.isnumeric()
        0     True
        1     True
        2     True
        3    False
        Name: 0, dtype: bool
        """

        def pandas_isnumeric(s) -> "ks.Series[bool]":
            return s.str.isnumeric()

        return self._data.transform_batch(pandas_isnumeric)

    def isdecimal(self) -> "ks.Series":
        """
        Check whether all characters in each string are decimals.

        This is equivalent to running the Python string method
        :func:`str.isdecimal` for each element of the Series/Index.
        If a string has zero characters, False is returned for that check.

        Examples
        --------
        >>> s = ks.Series(['23', '³', '⅕', ''])

        The s.str.isdecimal method checks for characters used to form numbers
        in base 10.

        >>> s.str.isdecimal()
        0     True
        1    False
        2    False
        3    False
        Name: 0, dtype: bool

        The s.str.isdigit method is the same as s.str.isdecimal but also
        includes special digits, like superscripted and subscripted digits in
        unicode.

        >>> s.str.isdigit()
        0     True
        1     True
        2    False
        3    False
        Name: 0, dtype: bool

        The s.str.isnumeric method is the same as s.str.isdigit but also
        includes other characters that can represent quantities such as unicode
        fractions.

        >>> s.str.isnumeric()
        0     True
        1     True
        2     True
        3    False
        Name: 0, dtype: bool
        """

        def pandas_isdecimal(s) -> "ks.Series[bool]":
            return s.str.isdecimal()

        return self._data.transform_batch(pandas_isdecimal)

    def cat(self, others=None, sep=None, na_rep=None, join=None) -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def center(self, width, fillchar=" ") -> "ks.Series":
        """
        Filling left and right side of strings in the Series/Index with an
        additional character. Equivalent to :func:`str.center`.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with fillchar.
        fillchar : str
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series of objects

        Examples
        --------
        >>> s = ks.Series(["caribou", "tiger"])
        >>> s
        0    caribou
        1      tiger
        Name: 0, dtype: object

        >>> s.str.center(width=10, fillchar='-')
        0    -caribou--
        1    --tiger---
        Name: 0, dtype: object
        """

        def pandas_center(s) -> "ks.Series[str]":
            return s.str.center(width, fillchar)

        return self._data.transform_batch(pandas_center)

    def contains(self, pat, case=True, flags=0, na=None, regex=True) -> "ks.Series":
        """
        Test if pattern or regex is contained within a string of a Series.

        Return boolean Series based on whether a given pattern or regex is
        contained within a string of a Series.

        Analogous to :func:`match`, but less strict, relying on
        :func:`re.search` instead of :func:`re.match`.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Flags to pass through to the re module, e.g. re.IGNORECASE.
        na : default None
            Fill value for missing values. NaN converted to None.
        regex : bool, default True
            If True, assumes the pat is a regular expression.
            If False, treats the pat as a literal string.


        Returns
        -------
        Series of boolean values or object
            A Series of boolean values indicating whether the given pattern is
            contained within the string of each element of the Series.

        Examples
        --------
        Returning a Series of booleans using only a literal pattern.

        >>> s1 = ks.Series(['Mouse', 'dog', 'house and parrot', '23', np.NaN])
        >>> s1.str.contains('og', regex=False)
        0    False
        1     True
        2    False
        3    False
        4     None
        Name: 0, dtype: object

        Specifying case sensitivity using case.

        >>> s1.str.contains('oG', case=True, regex=True)
        0    False
        1    False
        2    False
        3    False
        4     None
        Name: 0, dtype: object

        Specifying na to be False instead of NaN replaces NaN values with
        False. If Series does not contain NaN values the resultant dtype will
        be bool, otherwise, an object dtype.

        >>> s1.str.contains('og', na=False, regex=True)
        0    False
        1     True
        2    False
        3    False
        4    False
        Name: 0, dtype: bool

        Returning ‘house’ or ‘dog’ when either expression occurs in a string.

        >>> s1.str.contains('house|dog', regex=True)
        0    False
        1     True
        2     True
        3    False
        4     None
        Name: 0, dtype: object

        Ignoring case sensitivity using flags with regex.

        >>> import re
        >>> s1.str.contains('PARROT', flags=re.IGNORECASE, regex=True)
        0    False
        1    False
        2     True
        3    False
        4     None
        Name: 0, dtype: object

        Returning any digit using regular expression.

        >>> s1.str.contains('[0-9]', regex=True)
        0    False
        1    False
        2    False
        3     True
        4     None
        Name: 0, dtype: object

        Ensure pat is a not a literal pattern when regex is set to True.
        Note in the following example one might expect only s2[1] and s2[3]
        to return True. However, ‘.0’ as a regex matches any character followed
        by a 0.

        >>> s2 = ks.Series(['40','40.0','41','41.0','35'])
        >>> s2.str.contains('.0', regex=True)
        0     True
        1     True
        2    False
        3     True
        4    False
        Name: 0, dtype: bool
        """

        def pandas_contains(s) -> "ks.Series[bool]":
            return s.str.contains(pat, case, flags, na, regex)

        return self._data.transform_batch(pandas_contains)

    def count(self, pat, flags=0) -> "ks.Series":
        """
        Count occurrences of pattern in each string of the Series.

        This function is used to count the number of times a particular regex
        pattern is repeated in each of the string elements of the Series.

        Parameters
        ----------
        pat : str
            Valid regular expression.
        flags : int, default 0 (no flags)
            Flags for the re module.

        Returns
        -------
        Series of int
            A Series containing the integer counts of pattern matches.

        Examples
        --------
        >>> s = ks.Series(['A', 'B', 'Aaba', 'Baca', np.NaN, 'CABA', 'cat'])
        >>> s.str.count('a')
        0    0.0
        1    0.0
        2    2.0
        3    2.0
        4    NaN
        5    0.0
        6    1.0
        Name: 0, dtype: float64

        Escape '$' to find the literal dollar sign.

        >>> s = ks.Series(['$', 'B', 'Aab$', '$$ca', 'C$B$', 'cat'])
        >>> s.str.count('\\$')
        0    1
        1    0
        2    1
        3    2
        4    2
        5    0
        Name: 0, dtype: int32
        """

        def pandas_count(s) -> "ks.Series[int]":
            return s.str.count(pat, flags)

        return self._data.transform_batch(pandas_count)

    def decode(self, encoding, errors="strict") -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def encode(self, encoding, errors="strict") -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def extract(self, pat, flags=0, expand=True) -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def extractall(self, pat, flags=0) -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def find(self, sub, start=0, end=None) -> "ks.Series":
        """
        Return lowest indexes in each strings in the Series where the
        substring is fully contained between [start:end].

        Return -1 on failure. Equivalent to standard :func:`str.find`.

        Parameters
        ----------
        sub : str
            Substring being searched.
        start : int
            Left edge index.
        end : int
            Right edge index.

        Returns
        -------
        Series of int
            Series of lowest matching indexes.

        Examples
        --------
        >>> s = ks.Series(['apple', 'oranges', 'bananas'])

        >>> s.str.find('a')
        0    0
        1    2
        2    1
        Name: 0, dtype: int32

        >>> s.str.find('a', start=2)
        0   -1
        1    2
        2    3
        Name: 0, dtype: int32

        >>> s.str.find('a', end=1)
        0    0
        1   -1
        2   -1
        Name: 0, dtype: int32

        >>> s.str.find('a', start=2, end=2)
        0   -1
        1   -1
        2   -1
        Name: 0, dtype: int32
        """

        def pandas_find(s) -> "ks.Series[int]":
            return s.str.find(sub, start, end)

        return self._data.transform_batch(pandas_find)

    def findall(self, pat, flags=0) -> "ks.Series":
        """
        Find all occurrences of pattern or regular expression in the Series.

        Equivalent to applying :func:`re.findall` to all the elements in
        the Series.

        Parameters
        ----------
        pat : str
            Pattern or regular expression.
        flags : int, default 0 (no flags)
            `re` module flags, e.g. `re.IGNORECASE`.

        Returns
        -------
        Series of object
            All non-overlapping matches of pattern or regular expression in
            each string of this Series.

        Examples
        --------
        >>> s = ks.Series(['Lion', 'Monkey', 'Rabbit'])

        The search for the pattern ‘Monkey’ returns one match:

        >>> s.str.findall('Monkey')
        0          []
        1    [Monkey]
        2          []
        Name: 0, dtype: object

        On the other hand, the search for the pattern ‘MONKEY’ doesn’t return
        any match:

        >>> s.str.findall('MONKEY')
        0    []
        1    []
        2    []
        Name: 0, dtype: object

        Flags can be added to the pattern or regular expression. For instance,
        to find the pattern ‘MONKEY’ ignoring the case:

        >>> import re
        >>> s.str.findall('MONKEY', flags=re.IGNORECASE)
        0          []
        1    [Monkey]
        2          []
        Name: 0, dtype: object

        When the pattern matches more than one string in the Series, all
        matches are returned:

        >>> s.str.findall('on')
        0    [on]
        1    [on]
        2      []
        Name: 0, dtype: object

        Regular expressions are supported too. For instance, the search for all
        the strings ending with the word ‘on’ is shown next:

        >>> s.str.findall('on$')
        0    [on]
        1      []
        2      []
        Name: 0, dtype: object

        If the pattern is found more than once in the same string, then a list
        of multiple strings is returned:

        >>> s.str.findall('b')
        0        []
        1        []
        2    [b, b]
        Name: 0, dtype: object
        """
        # type hint does not support to specify array type yet.
        pudf = pandas_udf(
            lambda s: s.str.findall(pat, flags),
            returnType=ArrayType(StringType(), containsNull=True),
            functionType=PandasUDFType.SCALAR,
        )
        return self._data._with_new_scol(scol=pudf(self._data.spark_column)).rename(self.name)

    def index(self, sub, start=0, end=None) -> "ks.Series":
        """
        Return lowest indexes in each strings where the substring is fully
        contained between [start:end].

        This is the same as :func:`str.find` except instead of returning -1,
        it raises a ValueError when the substring is not found. Equivalent to
        standard :func:`str.index`.

        Parameters
        ----------
        sub : str
            Substring being searched.
        start : int
            Left edge index.
        end : int
            Right edge index.

        Returns
        -------
        Series of int
            Series of lowest matching indexes.

        Examples
        --------
        >>> s = ks.Series(['apple', 'oranges', 'bananas'])

        >>> s.str.index('a')
        0    0
        1    2
        2    1
        Name: 0, dtype: int64

        The following expression throws an exception:

        >>> s.str.index('a', start=2) # doctest: +SKIP
        """

        def pandas_index(s) -> "ks.Series[np.int64]":
            return s.str.index(sub, start, end)

        return self._data.transform_batch(pandas_index)

    def join(self, sep) -> "ks.Series":
        """
        Join lists contained as elements in the Series with passed delimiter.

        If the elements of a Series are lists themselves, join the content of
        these lists using the delimiter passed to the function. This function
        is an equivalent to calling :func:`str.join` on the lists.

        Parameters
        ----------
        sep : str
            Delimiter to use between list entries.

        Returns
        -------
        Series of object
            Series with list entries concatenated by intervening occurrences of
            the delimiter.

        See Also
        --------
        str.split : Split strings around given separator/delimiter.
        str.rsplit : Splits string around given separator/delimiter,
            starting from the right.

        Examples
        --------
        Example with a list that contains a None element.

        >>> s = ks.Series([['lion', 'elephant', 'zebra'],
        ...                ['cat', None, 'dog']])
        >>> s
        0    [lion, elephant, zebra]
        1           [cat, None, dog]
        Name: 0, dtype: object

        Join all lists using a ‘-‘. The list containing None will produce None.

        >>> s.str.join('-')
        0    lion-elephant-zebra
        1                   None
        Name: 0, dtype: object
        """

        def pandas_join(s) -> "ks.Series[str]":
            return s.str.join(sep)

        return self._data.transform_batch(pandas_join)

    def len(self) -> "ks.Series":
        """
        Computes the length of each element in the Series.

        The element may be a sequence (such as a string, tuple or list).

        Returns
        -------
        Series of int
            A Series of integer values indicating the length of each element in
            the Series.

        Examples
        --------
        Returns the length (number of characters) in a string. Returns the
        number of entries for lists or tuples.

        >>> s1 = ks.Series(['dog', 'monkey'])
        >>> s1.str.len()
        0    3
        1    6
        Name: 0, dtype: int64

        >>> s2 = ks.Series([["a", "b", "c"], []])
        >>> s2.str.len()
        0    3
        1    0
        Name: 0, dtype: int64
        """
        if isinstance(self._data.spark_type, (ArrayType, MapType)):
            return _column_op(lambda c: F.size(c).cast(LongType()))(self._data).alias(
                self._data.name
            )
        else:
            return _column_op(lambda c: F.length(c).cast(LongType()))(self._data).alias(
                self._data.name
            )

    def ljust(self, width, fillchar=" ") -> "ks.Series":
        """
        Filling right side of strings in the Series with an additional
        character. Equivalent to :func:`str.ljust`.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with `fillchar`.
        fillchar : str
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series of object

        Examples
        --------
        >>> s = ks.Series(["caribou", "tiger"])
        >>> s
        0    caribou
        1      tiger
        Name: 0, dtype: object

        >>> s.str.ljust(width=10, fillchar='-')
        0    caribou---
        1    tiger-----
        Name: 0, dtype: object
        """

        def pandas_ljust(s) -> "ks.Series[str]":
            return s.str.ljust(width, fillchar)

        return self._data.transform_batch(pandas_ljust)

    def match(self, pat, case=True, flags=0, na=np.NaN) -> "ks.Series":
        """
        Determine if each string matches a regular expression.

        Analogous to :func:`contains`, but more strict, relying on
        :func:`re.match` instead of :func:`re.search`.

        Parameters
        ----------
        pat : str
            Character sequence or regular expression.
        case : bool, default True
            If True, case sensitive.
        flags : int, default 0 (no flags)
            Flags to pass through to the re module, e.g. re.IGNORECASE.
        na : default NaN
            Fill value for missing values.

        Returns
        -------
        Series of boolean values or object
            A Series of boolean values indicating whether the given pattern can
            be matched in the string of each element of the Series.

        Examples
        --------
        >>> s = ks.Series(['Mouse', 'dog', 'house and parrot', '23', np.NaN])
        >>> s.str.match('dog')
        0    False
        1     True
        2    False
        3    False
        4     None
        Name: 0, dtype: object

        >>> s.str.match('mouse|dog', case=False)
        0     True
        1     True
        2    False
        3    False
        4     None
        Name: 0, dtype: object

        >>> s.str.match('.+and.+', na=True)
        0    False
        1    False
        2     True
        3    False
        4     True
        Name: 0, dtype: bool

        >>> import re
        >>> s.str.match('MOUSE', flags=re.IGNORECASE)
        0     True
        1    False
        2    False
        3    False
        4     None
        Name: 0, dtype: object
        """

        def pandas_match(s) -> "ks.Series[bool]":
            return s.str.match(pat, case, flags, na)

        return self._data.transform_batch(pandas_match)

    def normalize(self, form) -> "ks.Series":
        """
        Return the Unicode normal form for the strings in the Series.

        For more information on the forms, see the
        :func:`unicodedata.normalize`.

        Parameters
        ----------
        form : {‘NFC’, ‘NFKC’, ‘NFD’, ‘NFKD’}
            Unicode form.

        Returns
        -------
        Series of objects
            A Series of normalized strings.
        """

        def pandas_normalize(s) -> "ks.Series[str]":
            return s.str.normalize(form)

        return self._data.transform_batch(pandas_normalize)

    def pad(self, width, side="left", fillchar=" ") -> "ks.Series":
        """
        Pad strings in the Series up to width.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with character defined in `fillchar`.
        side : {‘left’, ‘right’, ‘both’}, default ‘left’
            Side from which to fill resulting string.
        fillchar : str, default ' '
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series of object
            Returns Series with minimum number of char in object.

        Examples
        --------
        >>> s = ks.Series(["caribou", "tiger"])
        >>> s
        0    caribou
        1      tiger
        Name: 0, dtype: object

        >>> s.str.pad(width=10)
        0       caribou
        1         tiger
        Name: 0, dtype: object

        >>> s.str.pad(width=10, side='right', fillchar='-')
        0    caribou---
        1    tiger-----
        Name: 0, dtype: object

        >>> s.str.pad(width=10, side='both', fillchar='-')
        0    -caribou--
        1    --tiger---
        Name: 0, dtype: object
        """

        def pandas_pad(s) -> "ks.Series[str]":
            return s.str.pad(width, side, fillchar)

        return self._data.transform_batch(pandas_pad)

    def partition(self, sep=" ", expand=True) -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def repeat(self, repeats) -> "ks.Series":
        """
        Duplicate each string in the Series.

        Parameters
        ----------
        repeats : int
            Repeat the string given number of times (int). Sequence of int
            is not supported.

        Returns
        -------
        Series of object
            Series or Index of repeated string objects specified by input
            parameter repeats.

        Examples
        --------
        >>> s = ks.Series(['a', 'b', 'c'])
        >>> s
        0    a
        1    b
        2    c
        Name: 0, dtype: object

        Single int repeats string in Series

        >>> s.str.repeat(repeats=2)
        0    aa
        1    bb
        2    cc
        Name: 0, dtype: object
        """
        if not isinstance(repeats, int):
            raise ValueError("repeats expects an int parameter")

        def pandas_repeat(s) -> "ks.Series[str]":
            return s.str.repeat(repeats=repeats)

        return self._data.transform_batch(pandas_repeat)

    def replace(self, pat, repl, n=-1, case=None, flags=0, regex=True) -> "ks.Series":
        """
        Replace occurrences of pattern/regex in the Series with some other
        string. Equivalent to :func:`str.replace` or :func:`re.sub`.

        Parameters
        ----------
        pat : str or compiled regex
            String can be a character sequence or regular expression.
        repl : str or callable
            Replacement string or a callable. The callable is passed the regex
            match object and must return a replacement string to be used. See
            :func:`re.sub`.
        n : int, default -1 (all)
            Number of replacements to make from start.
        case : boolean, default None
            If True, case sensitive (the default if pat is a string).
            Set to False for case insensitive.
            Cannot be set if pat is a compiled regex.
        flags: int, default 0 (no flags)
            re module flags, e.g. re.IGNORECASE.
            Cannot be set if pat is a compiled regex.
        regex : boolean, default True
            If True, assumes the passed-in pattern is a regular expression.
            If False, treats the pattern as a literal string.
            Cannot be set to False if pat is a compile regex or repl is a
            callable.

        Returns
        -------
        Series of object
            A copy of the string with all matching occurrences of pat replaced
            by repl.

        Examples
        --------
        When pat is a string and regex is True (the default), the given pat is
        compiled as a regex. When repl is a string, it replaces matching regex
        patterns as with :func:`re.sub`. NaN value(s) in the Series are changed
        to None:

        >>> ks.Series(['foo', 'fuz', np.nan]).str.replace('f.', 'ba', regex=True)
        0     bao
        1     baz
        2    None
        Name: 0, dtype: object

        When pat is a string and regex is False, every pat is replaced with
        repl as with :func:`str.replace`:

        >>> ks.Series(['f.o', 'fuz', np.nan]).str.replace('f.', 'ba', regex=False)
        0     bao
        1     fuz
        2    None
        Name: 0, dtype: object

        When repl is a callable, it is called on every pat using
        :func:`re.sub`. The callable should expect one positional argument (a
        regex object) and return a string.

        Reverse every lowercase alphabetic word:

        >>> repl = lambda m: m.group(0)[::-1]
        >>> ks.Series(['foo 123', 'bar baz', np.nan]).str.replace(r'[a-z]+', repl)
        0    oof 123
        1    rab zab
        2       None
        Name: 0, dtype: object

        Using regex groups (extract second group and swap case):

        >>> pat = r"(?P<one>\\w+) (?P<two>\\w+) (?P<three>\\w+)"
        >>> repl = lambda m: m.group('two').swapcase()
        >>> ks.Series(['One Two Three', 'Foo Bar Baz']).str.replace(pat, repl)
        0    tWO
        1    bAR
        Name: 0, dtype: object

        Using a compiled regex with flags:

        >>> import re
        >>> regex_pat = re.compile(r'FUZ', flags=re.IGNORECASE)
        >>> ks.Series(['foo', 'fuz', np.nan]).str.replace(regex_pat, 'bar')
        0     foo
        1     bar
        2    None
        Name: 0, dtype: object
        """

        def pandas_replace(s) -> "ks.Series[str]":
            return s.str.replace(pat, repl, n=n, case=case, flags=flags, regex=regex)

        return self._data.transform_batch(pandas_replace)

    def rfind(self, sub, start=0, end=None) -> "ks.Series":
        """
        Return highest indexes in each strings in the Series where the
        substring is fully contained between [start:end].

        Return -1 on failure. Equivalent to standard :func:`str.rfind`.

        Parameters
        ----------
        sub : str
            Substring being searched.
        start : int
            Left edge index.
        end : int
            Right edge index.

        Returns
        -------
        Series of int
            Series of highest matching indexes.

        Examples
        --------
        >>> s = ks.Series(['apple', 'oranges', 'bananas'])

        >>> s.str.rfind('a')
        0    0
        1    2
        2    5
        Name: 0, dtype: int32

        >>> s.str.rfind('a', start=2)
        0   -1
        1    2
        2    5
        Name: 0, dtype: int32

        >>> s.str.rfind('a', end=1)
        0    0
        1   -1
        2   -1
        Name: 0, dtype: int32

        >>> s.str.rfind('a', start=2, end=2)
        0   -1
        1   -1
        2   -1
        Name: 0, dtype: int32
        """

        def pandas_rfind(s) -> "ks.Series[int]":
            return s.str.rfind(sub, start, end)

        return self._data.transform_batch(pandas_rfind)

    def rindex(self, sub, start=0, end=None) -> "ks.Series":
        """
        Return highest indexes in each strings where the substring is fully
        contained between [start:end].

        This is the same as :func:`str.rfind` except instead of returning -1,
        it raises a ValueError when the substring is not found. Equivalent to
        standard :func:`str.rindex`.

        Parameters
        ----------
        sub : str
            Substring being searched.
        start : int
            Left edge index.
        end : int
            Right edge index.

        Returns
        -------
        Series of int
            Series of highest matching indexes.

        Examples
        --------
        >>> s = ks.Series(['apple', 'oranges', 'bananas'])

        >>> s.str.rindex('a')
        0    0
        1    2
        2    5
        Name: 0, dtype: int64

        The following expression throws an exception:

        >>> s.str.rindex('a', start=2) # doctest: +SKIP
        """

        def pandas_rindex(s) -> "ks.Series[np.int64]":
            return s.str.rindex(sub, start, end)

        return self._data.transform_batch(pandas_rindex)

    def rjust(self, width, fillchar=" ") -> "ks.Series":
        """
        Filling left side of strings in the Series with an additional
        character. Equivalent to :func:`str.rjust`.

        Parameters
        ----------
        width : int
            Minimum width of resulting string; additional characters will be
            filled with `fillchar`.
        fillchar : str
            Additional character for filling, default is whitespace.

        Returns
        -------
        Series of object

        Examples
        --------
        >>> s = ks.Series(["caribou", "tiger"])
        >>> s
        0    caribou
        1      tiger
        Name: 0, dtype: object

        >>> s.str.rjust(width=10)
        0       caribou
        1         tiger
        Name: 0, dtype: object

        >>> s.str.rjust(width=10, fillchar='-')
        0    ---caribou
        1    -----tiger
        Name: 0, dtype: object
        """

        def pandas_rjust(s) -> "ks.Series[str]":
            return s.str.rjust(width, fillchar)

        return self._data.transform_batch(pandas_rjust)

    def rpartition(self, sep=" ", expand=True) -> "ks.Series":
        """
        Not supported.
        """
        raise NotImplementedError()

    def slice(self, start=None, stop=None, step=None) -> "ks.Series":
        """
        Slice substrings from each element in the Series.

        Parameters
        ----------
        start : int, optional
            Start position for slice operation.
        stop : int, optional
            Stop position for slice operation.
        step : int, optional
            Step size for slice operation.

        Returns
        -------
        Series of object
            Series from sliced substrings from original string objects.

        Examples
        --------
        >>> s = ks.Series(["koala", "fox", "chameleon"])
        >>> s
        0        koala
        1          fox
        2    chameleon
        Name: 0, dtype: object

        >>> s.str.slice(start=1)
        0        oala
        1          ox
        2    hameleon
        Name: 0, dtype: object

        >>> s.str.slice(stop=2)
        0    ko
        1    fo
        2    ch
        Name: 0, dtype: object

        >>> s.str.slice(step=2)
        0      kaa
        1       fx
        2    caeen
        Name: 0, dtype: object

        >>> s.str.slice(start=0, stop=5, step=3)
        0    kl
        1     f
        2    cm
        Name: 0, dtype: object
        """

        def pandas_slice(s) -> "ks.Series[str]":
            return s.str.slice(start, stop, step)

        return self._data.transform_batch(pandas_slice)

    def slice_replace(self, start=None, stop=None, repl=None) -> "ks.Series":
        """
        Slice substrings from each element in the Series.

        Parameters
        ----------
        start : int, optional
            Start position for slice operation. If not specified (None), the
            slice is unbounded on the left, i.e. slice from the start of the
            string.
        stop : int, optional
            Stop position for slice operation. If not specified (None), the
            slice is unbounded on the right, i.e. slice until the end of the
            string.
        repl : str, optional
            String for replacement. If not specified (None), the sliced region
            is replaced with an empty string.

        Returns
        -------
        Series of object
            Series from sliced substrings from original string objects.

        Examples
        --------
        >>> s = ks.Series(['a', 'ab', 'abc', 'abdc', 'abcde'])
        >>> s
        0        a
        1       ab
        2      abc
        3     abdc
        4    abcde
        Name: 0, dtype: object

        Specify just start, meaning replace start until the end of the string
        with repl.

        >>> s.str.slice_replace(1, repl='X')
        0    aX
        1    aX
        2    aX
        3    aX
        4    aX
        Name: 0, dtype: object

        Specify just stop, meaning the start of the string to stop is replaced
        with repl, and the rest of the string is included.

        >>> s.str.slice_replace(stop=2, repl='X')
        0       X
        1       X
        2      Xc
        3     Xdc
        4    Xcde
        Name: 0, dtype: object

        Specify start and stop, meaning the slice from start to stop is
        replaced with repl. Everything before or after start and stop is
        included as is.

        >>> s.str.slice_replace(start=1, stop=3, repl='X')
        0      aX
        1      aX
        2      aX
        3     aXc
        4    aXde
        Name: 0, dtype: object
        """

        def pandas_slice_replace(s) -> "ks.Series[str]":
            return s.str.slice_replace(start, stop, repl)

        return self._data.transform_batch(pandas_slice_replace)

    def split(self, pat=None, n=-1, expand=False) -> Union["ks.Series", "ks.DataFrame"]:
        """
        Split strings around given separator/delimiter.

        Splits the string in the Series from the beginning, at the specified
        delimiter string. Equivalent to :func:`str.split`.

        Parameters
        ----------
        pat : str, optional
            String or regular expression to split on. If not specified, split
            on whitespace.
        n : int, default -1 (all)
            Limit number of splits in output. None, 0 and -1 will be
            interpreted as return all splits.
        expand : bool, default False
            Expand the splitted strings into separate columns.

            * If ``True``, `n` must be a positive integer, and return DataFrame expanding
              dimensionality.
            * If ``False``, return Series, containing lists of strings.

        Returns
        -------
        Series, DataFrame
            Type matches caller unless `expand=True` (see Notes).

        See Also
        --------
        str.rsplit : Splits string around given separator/delimiter,
            starting from the right.
        str.join : Join lists contained as elements in the Series/Index
            with passed delimiter.

        Notes
        -----
        The handling of the `n` keyword depends on the number of found splits:

        - If found splits > `n`,  make first `n` splits only
        - If found splits <= `n`, make all splits
        - If for a certain row the number of found splits < `n`,
          append `None` for padding up to `n` if ``expand=True``

        If using ``expand=True``, Series callers return DataFrame objects with `n + 1` columns.

        .. note:: Even if `n` is much larger than found splits, the number of columns does NOT
            shrink unlike pandas.

        Examples
        --------
        >>> s = ks.Series(["this is a regular sentence",
        ...                "https://docs.python.org/3/tutorial/index.html",
        ...                np.nan])

        In the default setting, the string is split by whitespace.

        >>> s.str.split()
        0                   [this, is, a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        Name: 0, dtype: object

        Without the n parameter, the outputs of rsplit and split are identical.

        >>> s.str.rsplit()
        0                   [this, is, a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        Name: 0, dtype: object

        The n parameter can be used to limit the number of splits on the
        delimiter. The outputs of split and rsplit are different.

        >>> s.str.split(n=2)
        0                     [this, is, a regular sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        Name: 0, dtype: object

        >>> s.str.rsplit(n=2)
        0                     [this is a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        Name: 0, dtype: object

        The pat parameter can be used to split by other characters.

        >>> s.str.split(pat = "/")
        0                         [this is a regular sentence]
        1    [https:, , docs.python.org, 3, tutorial, index...
        2                                                 None
        Name: 0, dtype: object

        When using ``expand=True``, the split elements will expand out into
        separate columns. If NaN is present, it is propagated throughout
        the columns during the split.

        >>> s.str.split(n=4, expand=True)
                                                       0     1     2        3         4
        0                                           this    is     a  regular  sentence
        1  https://docs.python.org/3/tutorial/index.html  None  None     None      None
        2                                           None  None  None     None      None

        For slightly more complex use cases like splitting the html document name
        from a url, a combination of parameter settings can be used.

        >>> s.str.rsplit("/", n=1, expand=True)
                                            0           1
        0          this is a regular sentence        None
        1  https://docs.python.org/3/tutorial  index.html
        2                                None        None

        Remember to escape special characters when explicitly using regular
        expressions.

        >>> s = ks.Series(["1+1=2"])
        >>> s.str.split(r"\\+|=", n=2, expand=True)
           0  1  2
        0  1  1  2
        """
        from databricks.koalas.frame import DataFrame

        if expand and n <= 0:
            raise NotImplementedError("expand=True is currently only supported with n > 0.")

        # type hint does not support to specify array type yet.
        pudf = pandas_udf(
            lambda s: s.str.split(pat, n),
            returnType=ArrayType(StringType(), containsNull=True),
            functionType=PandasUDFType.SCALAR,
        )
        kser = self._data._with_new_scol(scol=pudf(self._data.spark_column)).rename(self.name)

        if expand:
            kdf = kser.to_frame()
            scol = kdf._internal.data_spark_columns[0]
            spark_columns = [scol[i].alias(str(i)) for i in range(n + 1)]
            column_labels = [(str(i),) for i in range(n + 1)]
            internal = kdf._internal.with_new_columns(spark_columns, column_labels)
            return DataFrame(internal)
        else:
            return kser

    def rsplit(self, pat=None, n=-1, expand=False) -> Union["ks.Series", "ks.DataFrame"]:
        """
        Split strings around given separator/delimiter.

        Splits the string in the Series from the end, at the specified
        delimiter string. Equivalent to :func:`str.rsplit`.

        Parameters
        ----------
        pat : str, optional
            String or regular expression to split on. If not specified, split
            on whitespace.
        n : int, default -1 (all)
            Limit number of splits in output. None, 0 and -1 will be
            interpreted as return all splits.
        expand : bool, default False
            Expand the splitted strings into separate columns.

            * If ``True``, `n` must be a positive integer, and return DataFrame expanding
              dimensionality.
            * If ``False``, return Series, containing lists of strings.

        Returns
        -------
        Series, DataFrame
            Type matches caller unless `expand=True` (see Notes).

        See Also
        --------
        str.split : Split strings around given separator/delimiter.
        str.join : Join lists contained as elements in the Series/Index
            with passed delimiter.

        Notes
        -----
        The handling of the `n` keyword depends on the number of found splits:

        - If found splits > `n`,  make first `n` splits only
        - If found splits <= `n`, make all splits
        - If for a certain row the number of found splits < `n`,
          append `None` for padding up to `n` if ``expand=True``

        If using ``expand=True``, Series callers return DataFrame objects with `n + 1` columns.

        .. note:: Even if `n` is much larger than found splits, the number of columns does NOT
            shrink unlike pandas.

        Examples
        --------
        >>> s = ks.Series(["this is a regular sentence",
        ...                "https://docs.python.org/3/tutorial/index.html",
        ...                np.nan])

        In the default setting, the string is split by whitespace.

        >>> s.str.split()
        0                   [this, is, a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        Name: 0, dtype: object

        Without the n parameter, the outputs of rsplit and split are identical.

        >>> s.str.rsplit()
        0                   [this, is, a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        Name: 0, dtype: object

        The n parameter can be used to limit the number of splits on the
        delimiter. The outputs of split and rsplit are different.

        >>> s.str.split(n=2)
        0                     [this, is, a regular sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        Name: 0, dtype: object

        >>> s.str.rsplit(n=2)
        0                     [this is a, regular, sentence]
        1    [https://docs.python.org/3/tutorial/index.html]
        2                                               None
        Name: 0, dtype: object

        When using ``expand=True``, the split elements will expand out into
        separate columns. If NaN is present, it is propagated throughout
        the columns during the split.

        >>> s.str.split(n=4, expand=True)
                                                       0     1     2        3         4
        0                                           this    is     a  regular  sentence
        1  https://docs.python.org/3/tutorial/index.html  None  None     None      None
        2                                           None  None  None     None      None

        For slightly more complex use cases like splitting the html document name
        from a url, a combination of parameter settings can be used.

        >>> s.str.rsplit("/", n=1, expand=True)
                                            0           1
        0          this is a regular sentence        None
        1  https://docs.python.org/3/tutorial  index.html
        2                                None        None

        Remember to escape special characters when explicitly using regular
        expressions.

        >>> s = ks.Series(["1+1=2"])
        >>> s.str.split(r"\\+|=", n=2, expand=True)
           0  1  2
        0  1  1  2
        """
        from databricks.koalas.frame import DataFrame

        if expand and n <= 0:
            raise NotImplementedError("expand=True is currently only supported with n > 0.")

        # type hint does not support to specify array type yet.
        pudf = pandas_udf(
            lambda s: s.str.rsplit(pat, n),
            returnType=ArrayType(StringType(), containsNull=True),
            functionType=PandasUDFType.SCALAR,
        )
        kser = self._data._with_new_scol(scol=pudf(self._data.spark_column)).rename(self.name)

        if expand:
            kdf = kser.to_frame()
            scol = kdf._internal.data_spark_columns[0]
            spark_columns = [scol[i].alias(str(i)) for i in range(n + 1)]
            column_labels = [(str(i),) for i in range(n + 1)]
            internal = kdf._internal.with_new_columns(spark_columns, column_labels)
            return DataFrame(internal)
        else:
            return kser

    def translate(self, table) -> "ks.Series":
        """
        Map all characters in the string through the given mapping table.
        Equivalent to standard :func:`str.translate`.

        Parameters
        ----------
        table : dict
            Table is a mapping of Unicode ordinals to Unicode ordinals,
            strings, or None. Unmapped characters are left untouched.
            Characters mapped to None are deleted. :func:`str.maketrans` is a
            helper function for making translation tables.

        Returns
        -------
        Series of object
            Series with translated strings.

        Examples
        --------
        >>> s = ks.Series(["dog", "cat", "bird"])
        >>> m = str.maketrans({'a': 'X', 'i': 'Y', 'o': None})
        >>> s.str.translate(m)
        0      dg
        1     cXt
        2    bYrd
        Name: 0, dtype: object
        """

        def pandas_translate(s) -> "ks.Series[str]":
            return s.str.translate(table)

        return self._data.transform_batch(pandas_translate)

    def wrap(self, width, **kwargs) -> "ks.Series":
        """
        Wrap long strings in the Series to be formatted in paragraphs with
        length less than a given width.

        This method has the same keyword parameters and defaults as
        :class:`textwrap.TextWrapper`.

        Parameters
        ----------
        width : int
            Maximum line-width. Lines separated with newline char.
        expand_tabs : bool, optional
            If true, tab characters will be expanded to spaces (default: True).
        replace_whitespace : bool, optional
            If true, each whitespace character remaining after tab expansion
            will be replaced by a single space (default: True).
        drop_whitespace : bool, optional
            If true, whitespace that, after wrapping, happens to end up at the
            beginning or end of a line is dropped (default: True).
        break_long_words : bool, optional
            If true, then words longer than width will be broken in order to
            ensure that no lines are longer than width. If it is false, long
            words will not be broken, and some lines may be longer than width
            (default: True).
        break_on_hyphens : bool, optional
            If true, wrapping will occur preferably on whitespace and right
            after hyphens in compound words, as it is customary in English.
            If false, only whitespaces will be considered as potentially good
            places for line breaks, but you need to set break_long_words to
            false if you want truly insecable words (default: True).

        Returns
        -------
        Series of object
            Series with wrapped strings.

        Examples
        --------
        >>> s = ks.Series(['line to be wrapped', 'another line to be wrapped'])
        >>> s.str.wrap(12)
        0             line to be\\nwrapped
        1    another line\\nto be\\nwrapped
        Name: 0, dtype: object
        """

        def pandas_wrap(s) -> "ks.Series[str]":
            return s.str.wrap(width, **kwargs)

        return self._data.transform_batch(pandas_wrap)

    def zfill(self, width) -> "ks.Series":
        """
        Pad strings in the Series by prepending ‘0’ characters.

        Strings in the Series are padded with ‘0’ characters on the left of the
        string to reach a total string length width. Strings in the Series with
        length greater or equal to width are unchanged.

        Differs from :func:`str.zfill` which has special handling for ‘+’/’-‘
        in the string.

        Parameters
        ----------
        width : int
            Minimum length of resulting string; strings with length less than
            width be prepended with ‘0’ characters.

        Returns
        -------
        Series of object
            Series with '0' left-padded strings.

        Examples
        --------
        >>> s = ks.Series(['-1', '1', '1000', np.nan])
        >>> s
        0      -1
        1       1
        2    1000
        3    None
        Name: 0, dtype: object

        Note that NaN is not a string, therefore it is converted to NaN. The
        minus sign in '-1' is treated as a regular character and the zero is
        added to the left of it (:func:`str.zfill` would have moved it to the
        left). 1000 remains unchanged as it is longer than width.

        >>> s.str.zfill(3)
        0     0-1
        1     001
        2    1000
        3    None
        Name: 0, dtype: object
        """

        def pandas_zfill(s) -> "ks.Series[str]":
            return s.str.zfill(width)

        return self._data.transform_batch(pandas_zfill)

    def get_dummies(self, sep="|"):
        """
        Not supported.
        """
        raise NotImplementedError()
