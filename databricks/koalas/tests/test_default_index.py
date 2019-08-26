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
import os

import pandas as pd

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


class OneByOneDefaultIndexTest(ReusedSQLTestCase, TestUtils):

    @classmethod
    def setUpClass(cls):
        super(OneByOneDefaultIndexTest, cls).setUpClass()
        cls.default_index = os.environ.get('DEFAULT_INDEX', 'sequence')
        os.environ['DEFAULT_INDEX'] = 'sequence'

    @classmethod
    def tearDownClass(cls):
        super(OneByOneDefaultIndexTest, cls).tearDownClass()
        os.environ['DEFAULT_INDEX'] = cls.default_index

    def test_default_index(self):
        sdf = self.spark.range(1000)
        self.assert_eq(ks.DataFrame(sdf).sort_index(), pd.DataFrame({'id': list(range(1000))}))


class DistributedOneByOneDefaultIndexTest(ReusedSQLTestCase, TestUtils):

    @classmethod
    def setUpClass(cls):
        super(DistributedOneByOneDefaultIndexTest, cls).setUpClass()
        cls.default_index = os.environ.get('DEFAULT_INDEX', 'sequence')
        os.environ['DEFAULT_INDEX'] = 'distributed-sequence'

    @classmethod
    def tearDownClass(cls):
        super(DistributedOneByOneDefaultIndexTest, cls).tearDownClass()
        os.environ['DEFAULT_INDEX'] = cls.default_index

    def test_default_index(self):
        sdf = self.spark.range(1000)
        self.assert_eq(ks.DataFrame(sdf).sort_index(), pd.DataFrame({'id': list(range(1000))}))


class DistributedDefaultIndexTest(ReusedSQLTestCase, TestUtils):

    @classmethod
    def setUpClass(cls):
        super(DistributedDefaultIndexTest, cls).setUpClass()
        cls.default_index = os.environ.get('DEFAULT_INDEX', 'sequence')
        os.environ['DEFAULT_INDEX'] = 'distributed'

    @classmethod
    def tearDownClass(cls):
        super(DistributedDefaultIndexTest, cls).tearDownClass()
        os.environ['DEFAULT_INDEX'] = cls.default_index

    def test_default_index(self):
        sdf = self.spark.range(1000)
        pdf = ks.DataFrame(sdf).to_pandas()
        self.assertEqual(len(set(pdf.index)), len(pdf))
