import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import pandorable_sparky
import pyspark

from pandorable_sparky.testing.utils import ReusedSQLTestCase


class ParquetTest(ReusedSQLTestCase):

    def test_local(self):
        tmp = tempfile.mkdtemp()
        shutil.rmtree(tmp)
        try:
            data = pd.DataFrame({
                'i32': np.arange(1000, dtype=np.int32),
                'i64': np.arange(1000, dtype=np.int64),
                'f': np.arange(1000, dtype=np.float64),
                'bhello': np.random.choice(['hello', 'yo', 'people'], size=1000).astype("O")})
            df = self.spark.createDataFrame(data)
            df.coalesce(1).write.format("parquet").save(tmp)

            out = pyspark.read_parquet(tmp)

            self.assertPandasEqual(data, out.toPandas())
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    from pandorable_sparky.tests.test_parquet import *

    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
