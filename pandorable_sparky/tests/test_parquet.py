from distutils.version import LooseVersion
import unittest

import numpy as np
import pandas as pd
import pandorable_sparky
import pyspark

from pandorable_sparky.testing.utils import ReusedSQLTestCase, TestUtils


class ParquetTest(ReusedSQLTestCase, TestUtils):

    def test_local(self):
        with self.temp_dir() as tmp:
            data = pd.DataFrame({
                'i32': np.arange(1000, dtype=np.int32),
                'i64': np.arange(1000, dtype=np.int64),
                'f': np.arange(1000, dtype=np.float64),
                'bhello': np.random.choice(['hello', 'yo', 'people'], size=1000).astype("O")})
            data = data[['i32', 'i64', 'f', 'bhello']]
            self.spark.createDataFrame(data, 'i32 int, i64 long, f double, bhello string') \
                .coalesce(1).write.parquet(tmp, mode='overwrite')

            def check(columns, expected):
                if LooseVersion("0.21.1") <= LooseVersion(pd.__version__):
                    expected = pd.read_parquet(tmp, columns=columns)
                actual = pyspark.read_parquet(tmp, columns=columns)
                self.assertPandasEqual(expected, actual.toPandas())

            check(None, data)
            check(['i32', 'i64'], data[['i32', 'i64']])
            check(['i64', 'i32'], data[['i64', 'i32']])
            check(('i32', 'i64'), data[['i32', 'i64']])
            check(['a', 'b', 'i32', 'i64'], data[['i32', 'i64']])
            check([], pd.DataFrame([]))
            check(['a'], pd.DataFrame([]))
            check('i32', pd.DataFrame([]))
            check('float', data[['f']])


if __name__ == "__main__":
    from pandorable_sparky.tests.test_parquet import *

    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
