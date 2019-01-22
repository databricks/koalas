import functools
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession


class PySparkTestCase(unittest.TestCase):

    def setUp(self):
        self._old_sys_path = list(sys.path)
        class_name = self.__class__.__name__
        self.sc = SparkContext('local[4]', class_name)

    def tearDown(self):
        self.sc.stop()
        sys.path = self._old_sys_path


class ReusedPySparkTestCase(unittest.TestCase):

    @classmethod
    def conf(cls):
        """
        Override this in subclasses to supply a more specific conf
        """
        return SparkConf()

    @classmethod
    def setUpClass(cls):
        cls.sc = SparkContext('local[4]', cls.__name__, conf=cls.conf())

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()


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


class ReusedSQLTestCase(ReusedPySparkTestCase, SQLTestUtils):

    @classmethod
    def setUpClass(cls):
        super(ReusedSQLTestCase, cls).setUpClass()
        cls.spark = SparkSession(cls.sc)

        cls.spark.conf.set('spark.sql.execution.arrow.enabled', True)

    @classmethod
    def tearDownClass(cls):
        super(ReusedSQLTestCase, cls).tearDownClass()
        cls.spark.stop()

    def assertPandasEqual(self, expected, result):
        msg = ("DataFrames are not equal: " +
               "\n\nExpected:\n%s\n%s" % (expected, expected.dtypes) +
               "\n\nResult:\n%s\n%s" % (result, result.dtypes))
        self.assertTrue(expected.equals(result), msg=msg)

    def assertPandasAlmostEqual(self, expected, result):
        msg = ("DataFrames are not equal: " +
               "\n\nExpected:\n%s\n%s" % (expected, expected.dtypes) +
               "\n\nResult:\n%s\n%s" % (result, result.dtypes))
        self.assertEqual(expected.shape, result.shape, msg=msg)
        for ecol, rcol in zip(expected.columns, result.columns):
            self.assertEqual(str(ecol), str(rcol), msg=msg)
            for eval, rval in zip(expected[ecol], result[rcol]):
                self.assertAlmostEqual(eval, rval, msg=msg)


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
    def df(self):
        return self.spark.createDataFrame(self.pdf)

    @property
    def pdf(self):
        return self.df.toPandas()


def compare_both(f=None, almost=True):

    if f is None:
        return functools.partial(compare_both, almost=almost)
    elif isinstance(f, bool):
        return functools.partial(compare_both, almost=f)

    def wrapped(self):
        if almost:
            compare = self.assertPandasAlmostEqual
        else:
            compare = self.assertPandasEqual

        for result_pandas, result_spark in zip(f(self, self.pdf), f(self, self.df)):
            compare(result_pandas, result_spark.toPandas())

    wrapped.__doc__ = f.__doc__
    return wrapped
