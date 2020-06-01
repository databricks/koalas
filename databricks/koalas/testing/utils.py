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

    def assertPandasEqual(self, left, right):
        if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
            msg = (
                "DataFrames are not equal: "
                + "\n\nLeft:\n%s\n%s" % (left, left.dtypes)
                + "\n\nRight:\n%s\n%s" % (right, right.dtypes)
            )
            self.assertTrue(left.equals(right), msg=msg)
        elif isinstance(left, pd.Series) and isinstance(right, pd.Series):
            msg = (
                "Series are not equal: "
                + "\n\nLeft:\n%s\n%s" % (left, left.dtype)
                + "\n\nRight:\n%s\n%s" % (right, right.dtype)
            )
            self.assertTrue((left == right).all(), msg=msg)
        elif isinstance(left, pd.Index) and isinstance(right, pd.Index):
            msg = (
                "Indices are not equal: "
                + "\n\nLeft:\n%s\n%s" % (left, left.dtype)
                + "\n\nRight:\n%s\n%s" % (right, right.dtype)
            )
            self.assertTrue((left == right).all(), msg=msg)
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
            self.assertEqual(len(left), len(right), msg=msg)
            for lnull, rnull in zip(left.isnull(), right.isnull()):
                self.assertEqual(lnull, rnull, msg=msg)
            for lval, rval in zip(left.dropna(), right.dropna()):
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

    def assert_eq(self, left, right, almost=False):
        """
        Asserts if two arbitrary objects are equal or not. If given objects are Koalas DataFrame
        or Series, they are converted into pandas' and compared.

        :param left: object to compare
        :param right: object to compare
        :param almost: if this is enabled, the comparison is delegated to `unittest`'s
                       `assertAlmostEqual`. See its documentation for more details.
        """
        lpdf = self._to_pandas(left)
        rpdf = self._to_pandas(right)
        if isinstance(lpdf, (pd.DataFrame, pd.Series, pd.Index)):
            if almost:
                self.assertPandasAlmostEqual(lpdf, rpdf)
            else:
                self.assertPandasEqual(lpdf, rpdf)
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
