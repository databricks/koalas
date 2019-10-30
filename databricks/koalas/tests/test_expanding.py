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

import databricks.koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils
from databricks.koalas.window import Expanding


class ExpandingTests(ReusedSQLTestCase, TestUtils):

    def test_expanding_count(self):
        kser = ks.Series(['a', 'b', None, 'd'])
        pser = kser.to_pandas()
        self.assert_eq(kser.expanding(3).count(), pser.expanding(3).count())

        kdf = ks.DataFrame({'a': [1, float('nan'), 3], 'b': [1.0, 2.0, 3.0]})
        pdf = kdf.to_pandas()
        self.assert_eq(kdf.expanding(3).count(), pdf.expanding(3).count())

    def test_expanding_count_error(self):
        with self.assertRaisesRegex(ValueError, "min_periods must be >= 0"):
            ks.range(10).expanding(-1)

        with self.assertRaisesRegex(
                TypeError,
                "kdf_or_kser must be a series or dataframe; however, got:.*int"):
            Expanding(1, 2)
