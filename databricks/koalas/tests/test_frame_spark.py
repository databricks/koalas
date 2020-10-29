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
from distutils.version import LooseVersion
import os

import pandas as pd
import pyspark

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils, TestUtils


class SparkFrameMethodsTest(ReusedSQLTestCase, SQLTestUtils, TestUtils):
    def test_frame_apply_negative(self):
        with self.assertRaisesRegex(
            ValueError, "The output of the function.* pyspark.sql.DataFrame.*int"
        ):
            ks.range(10).spark.apply(lambda scol: 1)

    def test_hint(self):
        pdf1 = pd.DataFrame(
            {"lkey": ["foo", "bar", "baz", "foo"], "value": [1, 2, 3, 5]}
        ).set_index("lkey")
        pdf2 = pd.DataFrame(
            {"rkey": ["foo", "bar", "baz", "foo"], "value": [5, 6, 7, 8]}
        ).set_index("rkey")
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        if LooseVersion(pyspark.__version__) >= LooseVersion("3.0"):
            hints = ["broadcast", "merge", "shuffle_hash", "shuffle_replicate_nl"]
        else:
            hints = ["broadcast"]

        for hint in hints:
            self.assert_eq(
                pdf1.merge(pdf2, left_index=True, right_index=True).sort_values(
                    ["value_x", "value_y"]
                ),
                kdf1.merge(kdf2.spark.hint(hint), left_index=True, right_index=True).sort_values(
                    ["value_x", "value_y"]
                ),
                almost=True,
            )
            self.assert_eq(
                pdf1.merge(pdf2 + 1, left_index=True, right_index=True).sort_values(
                    ["value_x", "value_y"]
                ),
                kdf1.merge(
                    (kdf2 + 1).spark.hint(hint), left_index=True, right_index=True
                ).sort_values(["value_x", "value_y"]),
                almost=True,
            )

    def test_repartition(self):
        kdf = ks.DataFrame({"age": [5, 5, 2, 2], "name": ["Bob", "Bob", "Alice", "Alice"]})
        num_partitions = kdf.to_spark().rdd.getNumPartitions() + 1

        num_partitions += 1
        new_kdf = kdf.spark.repartition(num_partitions)
        self.assertEqual(new_kdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(kdf.sort_index(), new_kdf.sort_index())

        # Reserves Index
        kdf = kdf.set_index("age")
        num_partitions += 1
        new_kdf = kdf.spark.repartition(num_partitions)
        self.assertEqual(new_kdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(kdf.sort_index(), new_kdf.sort_index())

        # Reflects internal changes
        kdf = kdf.reset_index()
        kdf = kdf.set_index("name")
        kdf2 = kdf + 1
        num_partitions += 1
        self.assert_eq(kdf2.sort_index(), (kdf + 1).spark.repartition(num_partitions).sort_index())

        # Reserves MultiIndex
        kdf = ks.DataFrame({"a": ["a", "b", "c"]}, index=[[1, 2, 3], [4, 5, 6]])
        num_partitions = kdf.to_spark().rdd.getNumPartitions() + 1
        new_kdf = kdf.spark.repartition(num_partitions)
        self.assertEqual(new_kdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(kdf.sort_index(), new_kdf.sort_index())

    def test_coalesce(self):
        num_partitions = 10
        kdf = ks.DataFrame({"age": [5, 5, 2, 2], "name": ["Bob", "Bob", "Alice", "Alice"]})
        kdf = kdf.spark.repartition(num_partitions)

        num_partitions -= 1
        new_kdf = kdf.spark.coalesce(num_partitions)
        self.assertEqual(new_kdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(kdf.sort_index(), new_kdf.sort_index())

        # Reserves Index
        kdf = kdf.set_index("age")
        num_partitions -= 1
        new_kdf = kdf.spark.coalesce(num_partitions)
        self.assertEqual(new_kdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(kdf.sort_index(), new_kdf.sort_index())

        # Reflects internal changes
        kdf = kdf.reset_index()
        kdf = kdf.set_index("name")
        kdf2 = kdf + 1
        num_partitions -= 1
        self.assert_eq(kdf2.sort_index(), (kdf + 1).spark.coalesce(num_partitions).sort_index())

        # Reserves MultiIndex
        kdf = ks.DataFrame({"a": ["a", "b", "c"]}, index=[[1, 2, 3], [4, 5, 6]])
        num_partitions -= 1
        kdf = kdf.spark.repartition(num_partitions)

        num_partitions -= 1
        new_kdf = kdf.spark.coalesce(num_partitions)
        self.assertEqual(new_kdf.to_spark().rdd.getNumPartitions(), num_partitions)
        self.assert_eq(kdf.sort_index(), new_kdf.sort_index())

    def test_checkpoint(self):
        with self.temp_dir() as tmp:
            self.spark.sparkContext.setCheckpointDir(tmp)
            kdf = ks.DataFrame({"a": ["a", "b", "c"]})
            new_kdf = kdf.spark.checkpoint()
            self.assertIsNotNone(os.listdir(tmp))
            self.assert_eq(kdf, new_kdf)
