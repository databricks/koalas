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

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class DataFrameTest(ReusedSQLTestCase, SQLTestUtils):
    def test_isin_series(self):
        s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'], name='animal')

        ds = koalas.from_pandas(s)

        self.assert_eq(ds.isin(['cow', 'lama']), s.isin(['cow', 'lama']))
        self.assert_eq(ds.isin(set(['cow'])), s.isin(set(['cow'])))

        msg = "Values should be list or set"
        with self.assertRaisesRegex(TypeError, msg):
            ds.isin(s)

        # test list sanitizer
        value_list = [s, s]
        msg = "List contains unsupported type <class 'pandas.core.series.Series'>"
        with self.assertRaisesRegex(TypeError, msg):
            ds.isin(value_list)
