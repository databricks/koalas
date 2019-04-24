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

import unittest

import numpy as np
import pandas as pd

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


class SeriesDatetimeTest(ReusedSQLTestCase, TestUtils):

    @property
    def pdf1(self):
        date1 = pd.Series(pd.date_range('2012-1-1 12:00:00', periods=3, freq='M'))
        date2 = pd.Series(pd.date_range('2013-3-11 21:45:00', periods=3, freq='W'))
        return pd.DataFrame(dict(start_date=date1, end_date=date2))

    @unittest.skip(
        "It fails in certain OSs presumably due to different "
        "timezone behaviours inherited from C library.")
    def test_subtraction(self):
        pdf = self.pdf1
        kdf = koalas.from_pandas(pdf)
        kdf['diff_seconds'] = kdf['end_date'] - kdf['start_date'] - 1

        self.assertEqual(list(kdf['diff_seconds'].toPandas()), [35545499, 33644699, 31571099])

    def test_div(self):
        pdf = self.pdf1
        kdf = koalas.from_pandas(pdf)
        for u in 'D', 's', 'ms':
            duration = np.timedelta64(1, u)
            self.assert_eq(
                (kdf['end_date'] - kdf['start_date']) / duration,
                (pdf['end_date'] - pdf['start_date']) / duration)


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
