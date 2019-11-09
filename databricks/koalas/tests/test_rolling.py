#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pandas as pd

import databricks.koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils
from databricks.koalas.window import Rolling


class RollingTest(ReusedSQLTestCase, TestUtils):

    def test_rolling_error(self):
        with self.assertRaisesRegex(ValueError, "window must be >= 0"):
            ks.range(10).rolling(window=-1)
        with self.assertRaisesRegex(ValueError, "min_periods must be >= 0"):
            ks.range(10).rolling(window=1, min_periods=-1)

        with self.assertRaisesRegex(
                TypeError,
                "kdf_or_kser must be a series or dataframe; however, got:.*int"):
            Rolling(1, 2)

    def _test_rolling_func(self, f):
        kser = ks.Series([1, 2, 3])
        pser = kser.to_pandas()
        self.assert_eq(repr(getattr(kser.rolling(2), f)()), repr(getattr(pser.rolling(2), f)()))

        kdf = ks.DataFrame({'a': [1, 2, 3, 2], 'b': [4.0, 2.0, 3.0, 1.0]})
        pdf = kdf.to_pandas()
        self.assert_eq(repr(getattr(kdf.rolling(2), f)()), repr(getattr(pdf.rolling(2), f)()))

        # Multiindex
        kser = ks.Series(
            [1, 2, 3],
            index=pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'z')]))
        pser = kser.to_pandas()
        self.assert_eq(repr(getattr(kser.rolling(2), f)()), repr(getattr(pser.rolling(2), f)()))

        kdf = ks.DataFrame({'a': [1, 2, 3, 2], 'b': [4.0, 2.0, 3.0, 1.0]})
        pdf = kdf.to_pandas()
        self.assert_eq(repr(getattr(kdf.rolling(2), f)()), repr(getattr(pdf.rolling(2), f)()))

        # Multiindex column
        kdf = ks.DataFrame({'a': [1, 2, 3, 2], 'b': [4.0, 2.0, 3.0, 1.0]})
        kdf.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y')])
        pdf = kdf.to_pandas()
        self.assert_eq(repr(getattr(kdf.expanding(2), f)()), repr(getattr(pdf.expanding(2), f)()))

    def test_rolling_min(self):
        self._test_rolling_func("min")

    def test_rolling_max(self):
        self._test_rolling_func("max")

    def test_rolling_mean(self):
        self._test_rolling_func("mean")

    def test_rolling_sum(self):
        self._test_rolling_func("sum")

    def test_rolling_count(self):
        self._test_rolling_func("count")
