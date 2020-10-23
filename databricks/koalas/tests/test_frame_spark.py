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
from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class SparkFrameMethodsTest(ReusedSQLTestCase, SQLTestUtils):
    def test_frame_apply_negative(self):
        with self.assertRaisesRegex(
            ValueError, "The output of the function.* pyspark.sql.DataFrame.*int"
        ):
            ks.range(10).spark.apply(lambda scol: 1)

    def test_repartition(self):
        kdf = ks.DataFrame({"age": [5, 5, 2, 2], "name": ["Bob", "Bob", "Alice", "Alice"]})
        num_partitions = kdf.to_spark().rdd.getNumPartitions() + 1
        new_kdf = kdf.spark.repartition(num_partitions)
        self.assertEqual(new_kdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(kdf.sort_index(), new_kdf.sort_index())

        kdf = kdf.set_index("age")
        num_partitions = kdf.to_spark().rdd.getNumPartitions() + 1
        new_kdf = kdf.spark.repartition(num_partitions)
        self.assertEqual(new_kdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(kdf.sort_index(), new_kdf.sort_index())

        kdf = ks.DataFrame({"a": ["a", "b", "c"]}, index=[[1, 2, 3], [4, 5, 6]])
        num_partitions = kdf.to_spark().rdd.getNumPartitions() + 1
        new_kdf = kdf.spark.repartition(num_partitions)
        self.assertEqual(new_kdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(kdf.sort_index(), new_kdf.sort_index())
