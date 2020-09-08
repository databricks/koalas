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

import functools
import shutil
import tempfile
import unittest
import warnings
from contextlib import contextmanager

import pandas as pd

from databricks import koalas as ks
from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes import Index
from databricks.koalas.series import Series
from databricks.koalas.utils import name_like_string, default_session


class SQLTestUtils(object):
    """
    This util assumes the instance of this to have 'spark' attribute, having a spark session.
    It is usually used with 'ReusedSQLTestCase' class but can be used if you feel sure the
    the implementation of this class has 'spark' attribute.
    """

    @contextmanager
    def sql_conf(self, pairs):
        """
        A convenient context manager to test some configuration specific logic. This sets
        `value` to the configuration `key` and then restores it back when it exits.
        """
        assert isinstance(pairs, dict), "pairs should be a dictionary."
        assert hasattr(self, "spark"), "it should have 'spark' attribute, having a spark session."

        keys = pairs.keys()
        new_values = pairs.values()
        old_values = [self.spark.conf.get(key, None) for key in keys]
        for key, new_value in zip(keys, new_values):
            self.spark.conf.set(key, new_value)
        try:
            yield
        finally:
            for key, old_value in zip(keys, old_values):
                if old_value is None:
                    self.spark.conf.unset(key)
                else:
                    self.spark.conf.set(key, old_value)

    @contextmanager
    def database(self, *databases):
        """
        A convenient context manager to test with some specific databases. This drops the given
        databases if it exists and sets current database to "default" when it exits.
        """
        assert hasattr(self, "spark"), "it should have 'spark' attribute, having a spark session."

        try:
            yield
        finally:
            for db in databases:
                self.spark.sql("DROP DATABASE IF EXISTS %s CASCADE" % db)
            self.spark.catalog.setCurrentDatabase("default")

    @contextmanager
    def table(self, *tables):
        """
        A convenient context manager to test with some specific tables. This drops the given tables
        if it exists.
        """
        assert hasattr(self, "spark"), "it should have 'spark' attribute, having a spark session."

        try:
            yield
        finally:
            for t in tables:
                self.spark.sql("DROP TABLE IF EXISTS %s" % t)

    @contextmanager
    def tempView(self, *views):
        """
        A convenient context manager to test with some specific views. This drops the given views
        if it exists.
        """
        assert hasattr(self, "spark"), "it should have 'spark' attribute, having a spark session."

        try:
            yield
        finally:
            for v in views:
                self.spark.catalog.dropTempView(v)

    @contextmanager
    def function(self, *functions):
        """
        A convenient context manager to test with some specific functions. This drops the given
        functions if it exists.
        """
        assert hasattr(self, "spark"), "it should have 'spark' attribute, having a spark session."

        try:
            yield
        finally:
            for f in functions:
                self.spark.sql("DROP FUNCTION IF EXISTS %s" % f)


class ReusedSQLTestCase(unittest.TestCase, SQLTestUtils):
    @classmethod
    def setUpClass(cls):
        cls.spark = default_session()
        cls.spark.conf.set("spark.sql.execution.arrow.enabled", True)

    @classmethod
    def tearDownClass(cls):
        # We don't stop Spark session to reuse across all tests.
        # The Spark session will be started and stopped at PyTest session level.
        # Please see databricks/koalas/conftest.py.
        pass

    def assertPandasEqual(self, left, right, check_exact=True):
        if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
            try:
                pd.util.testing.assert_frame_equal(
                    left,
                    right,
                    check_index_type=("equiv" if len(left.index) > 0 else False),
                    check_column_type=("equiv" if len(left.columns) > 0 else False),
                    check_exact=check_exact,
                )
            except AssertionError as e:
                msg = (
                    str(e)
                    + "\n\nLeft:\n%s\n%s" % (left, left.dtypes)
                    + "\n\nRight:\n%s\n%s" % (right, right.dtypes)
                )
                raise AssertionError(msg) from e
        elif isinstance(left, pd.Series) and isinstance(right, pd.Series):
            try:
                pd.util.testing.assert_series_equal(
                    left,
                    right,
                    check_index_type=("equiv" if len(left.index) > 0 else False),
                    check_exact=check_exact,
                )
            except AssertionError as e:
                msg = (
                    str(e)
                    + "\n\nLeft:\n%s\n%s" % (left, left.dtype)
                    + "\n\nRight:\n%s\n%s" % (right, right.dtype)
                )
                raise AssertionError(msg) from e
        elif isinstance(left, pd.Index) and isinstance(right, pd.Index):
            try:
                pd.util.testing.assert_index_equal(left, right, check_exact=check_exact)
            except AssertionError as e:
                msg = (
                    str(e)
                    + "\n\nLeft:\n%s\n%s" % (left, left.dtype)
                    + "\n\nRight:\n%s\n%s" % (right, right.dtype)
                )
                raise AssertionError(msg) from e
        else:
            raise ValueError("Unexpected values: (%s, %s)" % (left, right))

    def assertPandasAlmostEqual(self, left, right):
        """
        This function checks if given pandas objects approximately same,
        which means the conditions below:
          - Both objects are nullable
          - Compare floats rounding to the number of decimal places, 7 after
            dropping missing values (NaN, NaT, None)
        """
        if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
            msg = (
                "DataFrames are not almost equal: "
                + "\n\nLeft:\n%s\n%s" % (left, left.dtypes)
                + "\n\nRight:\n%s\n%s" % (right, right.dtypes)
            )
            self.assertEqual(left.shape, right.shape, msg=msg)
            for lcol, rcol in zip(left.columns, right.columns):
                self.assertEqual(name_like_string(lcol), name_like_string(rcol), msg=msg)
                for lnull, rnull in zip(left[lcol].isnull(), right[rcol].isnull()):
                    self.assertEqual(lnull, rnull, msg=msg)
                for lval, rval in zip(left[lcol].dropna(), right[rcol].dropna()):
                    self.assertAlmostEqual(lval, rval, msg=msg)
        elif isinstance(left, pd.Series) and isinstance(left, pd.Series):
            msg = (
                "Series are not almost equal: "
                + "\n\nLeft:\n%s\n%s" % (left, left.dtype)
                + "\n\nRight:\n%s\n%s" % (right, right.dtype)
            )
            self.assertEqual(str(left.name), str(right.name), msg=msg)
            self.assertEqual(len(left), len(right), msg=msg)
            for lnull, rnull in zip(left.isnull(), right.isnull()):
                self.assertEqual(lnull, rnull, msg=msg)
            for lval, rval in zip(left.dropna(), right.dropna()):
                self.assertAlmostEqual(lval, rval, msg=msg)
        elif isinstance(left, pd.MultiIndex) and isinstance(left, pd.MultiIndex):
            msg = (
                "MultiIndices are not almost equal: "
                + "\n\nLeft:\n%s\n%s" % (left, left.dtype)
                + "\n\nRight:\n%s\n%s" % (right, right.dtype)
            )
            self.assertEqual(len(left), len(right), msg=msg)
            for lval, rval in zip(left, right):
                self.assertAlmostEqual(lval, rval, msg=msg)
        elif isinstance(left, pd.Index) and isinstance(left, pd.Index):
            msg = (
                "Indices are not almost equal: "
                + "\n\nLeft:\n%s\n%s" % (left, left.dtype)
                + "\n\nRight:\n%s\n%s" % (right, right.dtype)
            )
            self.assertEqual(len(left), len(right), msg=msg)
            for lnull, rnull in zip(left.isnull(), right.isnull()):
                self.assertEqual(lnull, rnull, msg=msg)
            for lval, rval in zip(left.dropna(), right.dropna()):
                self.assertAlmostEqual(lval, rval, msg=msg)
        else:
            raise ValueError("Unexpected values: (%s, %s)" % (left, right))

    def assert_eq(self, left, right, check_exact=True, almost=False):
        """
        Asserts if two arbitrary objects are equal or not. If given objects are Koalas DataFrame
        or Series, they are converted into pandas' and compared.

        :param left: object to compare
        :param right: object to compare
        :param check_exact: if this is False, the comparison is done less precisely.
        :param almost: if this is enabled, the comparison is delegated to `unittest`'s
                       `assertAlmostEqual`. See its documentation for more details.
        """
        lpdf = self._to_pandas(left)
        rpdf = self._to_pandas(right)
        if isinstance(lpdf, (pd.DataFrame, pd.Series, pd.Index)):
            if almost:
                self.assertPandasAlmostEqual(lpdf, rpdf)
            else:
                self.assertPandasEqual(lpdf, rpdf, check_exact=check_exact)
        else:
            if almost:
                self.assertAlmostEqual(lpdf, rpdf)
            else:
                self.assertEqual(lpdf, rpdf)

    def assert_array_eq(self, left, right):
        self.assertTrue((left == right).all())

    def assert_list_eq(self, left, right):
        for litem, ritem in zip(left, right):
            self.assert_eq(litem, ritem)

    @staticmethod
    def _to_pandas(df):
        if isinstance(df, (DataFrame, Series, Index)):
            return df.toPandas()
        else:
            return df


class TestUtils(object):
    @contextmanager
    def temp_dir(self):
        tmp = tempfile.mkdtemp()
        try:
            yield tmp
        finally:
            shutil.rmtree(tmp)

    @contextmanager
    def temp_file(self):
        with self.temp_dir() as tmp:
            yield tempfile.mktemp(dir=tmp)


class ComparisonTestBase(ReusedSQLTestCase):
    @property
    def kdf(self):
        return ks.from_pandas(self.pdf)

    @property
    def pdf(self):
        return self.kdf.toPandas()


def compare_both(f=None, almost=True):

    if f is None:
        return functools.partial(compare_both, almost=almost)
    elif isinstance(f, bool):
        return functools.partial(compare_both, almost=f)

    @functools.wraps(f)
    def wrapped(self):
        if almost:
            compare = self.assertPandasAlmostEqual
        else:
            compare = self.assertPandasEqual

        for result_pandas, result_spark in zip(f(self, self.pdf), f(self, self.kdf)):
            compare(result_pandas, result_spark.toPandas())

    return wrapped


@contextmanager
def assert_produces_warning(
    expected_warning=Warning,
    filter_level="always",
    check_stacklevel=True,
    raise_on_extra_warnings=True,
):
    """
    Context manager for running code expected to either raise a specific
    warning, or not raise any warnings. Verifies that the code raises the
    expected warning, and that it does not raise any other unexpected
    warnings. It is basically a wrapper around ``warnings.catch_warnings``.

    Notes
    -----
    Replicated from pandas._testing.

    Parameters
    ----------
    expected_warning : {Warning, False, None}, default Warning
        The type of Exception raised. ``exception.Warning`` is the base
        class for all warnings. To check that no warning is returned,
        specify ``False`` or ``None``.
    filter_level : str or None, default "always"
        Specifies whether warnings are ignored, displayed, or turned
        into errors.
        Valid values are:
        * "error" - turns matching warnings into exceptions
        * "ignore" - discard the warning
        * "always" - always emit a warning
        * "default" - print the warning the first time it is generated
          from each location
        * "module" - print the warning the first time it is generated
          from each module
        * "once" - print the warning the first time it is generated
    check_stacklevel : bool, default True
        If True, displays the line that called the function containing
        the warning to show were the function is called. Otherwise, the
        line that implements the function is displayed.
    raise_on_extra_warnings : bool, default True
        Whether extra warnings not of the type `expected_warning` should
        cause the test to fail.

    Examples
    --------
    >>> import warnings
    >>> with assert_produces_warning():
    ...     warnings.warn(UserWarning())
    ...
    >>> with assert_produces_warning(False): # doctest: +SKIP
    ...     warnings.warn(RuntimeWarning())
    ...
    Traceback (most recent call last):
        ...
    AssertionError: Caused unexpected warning(s): ['RuntimeWarning'].
    >>> with assert_produces_warning(UserWarning): # doctest: +SKIP
    ...     warnings.warn(RuntimeWarning())
    Traceback (most recent call last):
        ...
    AssertionError: Did not see expected warning of class 'UserWarning'
    ..warn:: This is *not* thread-safe.
    """
    __tracebackhide__ = True

    with warnings.catch_warnings(record=True) as w:

        saw_warning = False
        warnings.simplefilter(filter_level)
        yield w
        extra_warnings = []

        for actual_warning in w:
            if expected_warning and issubclass(actual_warning.category, expected_warning):
                saw_warning = True

                if check_stacklevel and issubclass(
                    actual_warning.category, (FutureWarning, DeprecationWarning)
                ):
                    from inspect import getframeinfo, stack

                    caller = getframeinfo(stack()[2][0])
                    msg = (
                        "Warning not set with correct stacklevel. ",
                        "File where warning is raised: {} != ".format(actual_warning.filename),
                        "{}. Warning message: {}".format(caller.filename, actual_warning.message),
                    )
                    assert actual_warning.filename == caller.filename, msg
            else:
                extra_warnings.append(
                    (
                        actual_warning.category.__name__,
                        actual_warning.message,
                        actual_warning.filename,
                        actual_warning.lineno,
                    )
                )
        if expected_warning:
            msg = "Did not see expected warning of class {}".format(repr(expected_warning.__name__))
            assert saw_warning, msg
        if raise_on_extra_warnings and extra_warnings:
            raise AssertionError("Caused unexpected warning(s): {}".format(repr(extra_warnings)))
