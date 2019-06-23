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

from pyspark.sql.types import StringType, BinaryType, BooleanType, IntegerType, ArrayType, LongType

from databricks.koalas.base import _wrap_accessor_pandas

if TYPE_CHECKING:
    import databricks.koalas as ks


class StringMethods(object):
    """String methods for Koalas Series"""
    def __init__(self, series: 'ks.Series'):
        if not isinstance(series.spark_type, (StringType, BinaryType, ArrayType)):
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

    def title(self) -> 'ks.Series':
        """
        Convert Strings in the series to be titlecase.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.title(),
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

    def cat(self, others=None, sep=None, na_rep=None, join=None) -> 'ks.Series':
        """
        Not supported.
        """
        raise NotImplementedError()

    def center(self, width, fillchar=' ') -> 'ks.Series':
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
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.center(width, fillchar),
            StringType()
        ).alias(self.name)

    def contains(self, pat, case=True, flags=0, na=np.NaN, regex=True) -> 'ks.Series':
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
        na : default NaN
            Fill value for missing values.
        regex : bool, default True
            If True, assumes the pat is a regular expression.
            If False, treats the pat as a literal string.


        Returns
        -------
        Series of boolean values
            A Series of boolean values indicating whether the given pattern is
            contained within the string of each element of the Series.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.contains(pat, case, flags, na, regex),
            BooleanType()
        ).alias(self.name)

    def count(self, pat, flags=0) -> 'ks.Series':
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
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.count(pat, flags),
            IntegerType()
        ).alias(self.name)

    def decode(self, encoding, errors='strict') -> 'ks.Series':
        """
        Not supported.
        """
        raise NotImplementedError()

    def encode(self, encoding, errors='strict') -> 'ks.Series':
        """
        Not supported.
        """
        raise NotImplementedError()

    def extract(self, pat, flags=0, expand=True) -> 'ks.Series':
        """
        Not supported.
        """
        raise NotImplementedError()

    def extractall(self, pat, flags=0) -> 'ks.Series':
        """
        Not supported.
        """
        raise NotImplementedError()

    def find(self, sub, start=0, end=None) -> 'ks.Series':
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
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.find(sub, start, end),
            IntegerType()
        ).alias(self.name)

    def findall(self, pat, flags=0) -> 'ks.Series':
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
        Series of list of strings
            All non-overlapping matches of pattern or regular expression in
            each string of this Series.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.findall(pat, flags),
            ArrayType(StringType(), containsNull=True)
        ).alias(self.name)

    def index(self, sub, start=0, end=None) -> 'ks.Series':
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
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.index(sub, start, end),
            LongType()
        ).alias(self.name)

    def join(self, sep) -> 'ks.Series':
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
        Series of str
            Series with list entries concatenated by intervening occurrences of
            the delimiter.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.join(sep),
            StringType()
        ).alias(self.name)

    def len(self) -> 'ks.Series':
        """
        Computes the length of each element in the Series.

        The element may be a sequence (such as a string, tuple or list).

        Returns
        -------
        Series of int
            A Series of integer values indicating the length of each element in
            the Series.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.len(),
            LongType()
        ).alias(self.name)

    def ljust(self, width, fillchar=' ') -> 'ks.Series':
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
        Series of str
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.ljust(width, fillchar),
            StringType()
        ).alias(self.name)

    def match(self, pat, case=True, flags=0, na=np.NaN) -> 'ks.Series':
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
        Series of boolean values
            A Series of boolean values indicating whether the given pattern can
            be matched in the string of each element of the Series.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.match(pat, case, flags, na),
            BooleanType()
        ).alias(self.name)

    def normalize(self, form) -> 'ks.Series':
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
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.normalize(form),
            StringType()
        ).alias(self.name)

    def pad(self, width, side='left', fillchar=' ') -> 'ks.Series':
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
        Series of str
            Returns Series with minimum number of char in object.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.pad(width, side, fillchar),
            StringType()
        ).alias(self.name)

    def partition(self, sep=' ', expand=True) -> 'ks.Series':
        """
        Not supported.
        """
        raise NotImplementedError()

    def repeat(self, repeats) -> 'ks.Series':
        """
        Duplicate each string in the Series.

        Parameters
        ----------
        repeats : int
            Repeat the string given number of times (int). Sequence of int
            is not supported.

        Returns
        -------
        Series of str
            Series or Index of repeated string objects specified by input
            parameter repeats.
        """
        if not isinstance(repeats, int):
            raise ValueError("repeats expects an int parameter")

        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.repeat(repeats=repeats),
            StringType()
        ).alias(self.name)

    def replace(self, pat, repl, n=-1, case=None, flags=0, regex=True) -> 'ks.Series':
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
        Series of str
            A copy of the string with all matching occurrences of pat replaced
            by repl.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.replace(
                pat, repl, n=n, case=case, flags=flags, regex=regex
            ),
            StringType()
        ).alias(self.name)

    def rfind(self, sub, start=0, end=None) -> 'ks.Series':
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
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.rfind(sub, start, end),
            IntegerType()
        ).alias(self.name)

    def rindex(self, sub, start=0, end=None) -> 'ks.Series':
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
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.rindex(sub, start, end),
            LongType()
        ).alias(self.name)

    def rjust(self, width, fillchar=' ') -> 'ks.Series':
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
        Series of str
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.rjust(width, fillchar),
            StringType()
        ).alias(self.name)

    def rpartition(self, sep=' ', expand=True) -> 'ks.Series':
        """
        Not supported.
        """
        raise NotImplementedError()

    def slice(self, start=None, stop=None, step=None) -> 'ks.Series':
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
        Series of str
            Series from sliced substrings from original string objects.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.slice(start, stop, step),
            StringType()
        ).alias(self.name)

    def slice_replace(self, start=None, stop=None, repl=None) -> 'ks.Series':
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
        Series of str
            Series from sliced substrings from original string objects.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.slice_replace(start, stop, repl),
            StringType()
        ).alias(self.name)

    def split(self, pat=None, n=-1, expand=False) -> 'ks.Series':
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
        expand : bool, currently only False supported
            Expand the splitted strings into separate columns.

        Returns
        -------
        Series of str arrays
            Series with split strings.
        """
        if expand:
            raise ValueError("expand=True is currently not supported.")

        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.split(pat, n, expand),
            ArrayType(StringType(), containsNull=True)
        ).alias(self.name)

    def rsplit(self, pat=None, n=-1, expand=False) -> 'ks.Series':
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
        expand : bool, currently only False supported
            Expand the splitted strings into separate columns.

        Returns
        -------
        Series of str arrays
            Series with split strings.
        """
        if expand:
            raise ValueError("expand=True is currently not supported.")

        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.rsplit(pat, n, expand),
            ArrayType(StringType(), containsNull=True)
        ).alias(self.name)

    def translate(self, table) -> 'ks.Series':
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
        Series of str
            Series with translated strings.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.translate(table),
            StringType()
        ).alias(self.name)

    def wrap(self, width, **kwargs) -> 'ks.Series':
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
        Series of str
            Series with wrapped strings.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.wrap(width, **kwargs),
            StringType()
        ).alias(self.name)

    def zfill(self, width) -> 'ks.Series':
        """
        Pad strings in the Series by prepending ‘0’ characters.

        Strings in the Series are padded with ‘0’ characters on the left of the
        string to reach a total string length width. Strings in the Series with
        length greater or equal to width are unchanged.

        Parameters
        ----------
        width : int
            Minimum length of resulting string; strings with length less than
            width be prepended with ‘0’ characters.

        Returns
        -------
        Series of str
            Series with '0' left-padded strings.
        """
        return _wrap_accessor_pandas(
            self,
            lambda x: x.str.zfill(width),
            StringType()
        ).alias(self.name)

    def get_dummies(self, sep='|'):
        """
        Not supported.
        """
        raise NotImplementedError()
