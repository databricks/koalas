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

import pandas as pd
from pyspark import sql as spark

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class NamespaceTest(ReusedSQLTestCase, SQLTestUtils):

    def test_to_datetime(self):
        pdf = pd.DataFrame({'year': [2015, 2016],
                            'month': [2, 3],
                            'day': [4, 5]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pd.to_datetime(pdf), ks.to_datetime(kdf))

    def test_concat(self):
        pdf = pd.DataFrame({'A': [0, 2, 4], 'B': [1, 3, 5]})
        kdf = ks.from_pandas(pdf)

        self.assertRaisesRegex(TypeError, "first argument must be", lambda: ks.concat(kdf))
        self.assertRaisesRegex(
            TypeError, "cannot concatenate object", lambda: ks.concat([kdf, 1]))

        kdf2 = kdf.set_index('B', append=True)
        self.assertRaisesRegex(
            ValueError, "Index type and names should be same", lambda: ks.concat([kdf, kdf2]))
        kdf2 = kdf.reset_index()
        self.assertRaisesRegex(
            ValueError, "Index type and names should be same", lambda: ks.concat([kdf, kdf2]))

        self.assertRaisesRegex(
            ValueError, "All objects passed", lambda: ks.concat([None, None]))

        self.assertRaisesRegex(
            ValueError, 'axis should be either 0 or', lambda: ks.concat([kdf, kdf], axis=1))

    def test_cache(self):
        pdf = pd.DataFrame({'A': [0, 2, 4], 'B': [1, 3, 5]})
        kdf = ks.from_pandas(pdf)
        sdf = kdf.to_spark()

        with ks.cache(kdf) as cached_df:
            assert cached_df.is_cached

        with ks.cache(pdf) as cached_df:
            assert cached_df.is_cached

        with ks.cache(sdf) as cached_df:
            assert cached_df.is_cached
