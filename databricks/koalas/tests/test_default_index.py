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
from databricks.koalas.testing.utils import ReusedSQLTestCase


class DefaultIndexTest(ReusedSQLTestCase):
    def test_default_index_sequence(self):
        with ks.option_context("compute.default_index_type", "sequence"):
            sdf = self.spark.range(1000)
            self.assert_eq(ks.DataFrame(sdf), pd.DataFrame({"id": list(range(1000))}))

    def test_default_index_distributed_sequence(self):
        with ks.option_context("compute.default_index_type", "distributed-sequence"):
            sdf = self.spark.range(1000)
            self.assert_eq(ks.DataFrame(sdf), pd.DataFrame({"id": list(range(1000))}))

    def test_default_index_distributed(self):
        with ks.option_context("compute.default_index_type", "distributed"):
            sdf = self.spark.range(1000)
            pdf = ks.DataFrame(sdf).to_pandas()
            self.assertEqual(len(set(pdf.index)), len(pdf))
