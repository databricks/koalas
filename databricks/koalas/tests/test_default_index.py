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

from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


class OneByOneDefaultIndexTest(ReusedSQLTestCase, TestUtils):
    @classmethod
    def setUpClass(cls):
        super(OneByOneDefaultIndexTest, cls).setUpClass()
        set_option("compute.default_index_type", "sequence")

    @classmethod
    def tearDownClass(cls):
        reset_option("compute.default_index_type")
        super(OneByOneDefaultIndexTest, cls).tearDownClass()

    def test_default_index(self):
        sdf = self.spark.range(1000)
        self.assert_eq(ks.DataFrame(sdf).sort_index(), pd.DataFrame({"id": list(range(1000))}))


class DistributedOneByOneDefaultIndexTest(ReusedSQLTestCase, TestUtils):
    @classmethod
    def setUpClass(cls):
        super(DistributedOneByOneDefaultIndexTest, cls).setUpClass()
        set_option("compute.default_index_type", "distributed-sequence")

    @classmethod
    def tearDownClass(cls):
        reset_option("compute.default_index_type")
        super(DistributedOneByOneDefaultIndexTest, cls).tearDownClass()

    def test_default_index(self):
        sdf = self.spark.range(1000)
        self.assert_eq(ks.DataFrame(sdf).sort_index(), pd.DataFrame({"id": list(range(1000))}))


class DistributedDefaultIndexTest(ReusedSQLTestCase, TestUtils):
    @classmethod
    def setUpClass(cls):
        super(DistributedDefaultIndexTest, cls).setUpClass()
        set_option("compute.default_index_type", "distributed")

    @classmethod
    def tearDownClass(cls):
        reset_option("compute.default_index_type")
        super(DistributedDefaultIndexTest, cls).tearDownClass()

    def test_default_index(self):
        sdf = self.spark.range(1000)
        pdf = ks.DataFrame(sdf).to_pandas()
        self.assertEqual(len(set(pdf.index)), len(pdf))
