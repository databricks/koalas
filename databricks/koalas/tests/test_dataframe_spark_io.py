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

import numpy as np
import pandas as pd

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


class DataFrameSparkIOTest(ReusedSQLTestCase, TestUtils):
    """Test cases for big data I/O using Spark."""

    @property
    def test_column_order(self):
        return ["i32", "i64", "f", "bhello"]

    @property
    def test_pdf(self):
        pdf = pd.DataFrame(
            {
                "i32": np.arange(20, dtype=np.int32) % 3,
                "i64": np.arange(20, dtype=np.int64) % 5,
                "f": np.arange(20, dtype=np.float64),
                "bhello": np.random.choice(["hello", "yo", "people"], size=20).astype("O"),
            },
            columns=self.test_column_order,
        )
        return pdf

    def test_parquet_read(self):
        with self.temp_dir() as tmp:
            data = self.test_pdf
            self.spark.createDataFrame(data, "i32 int, i64 long, f double, bhello string").coalesce(
                1
            ).write.parquet(tmp, mode="overwrite")

            def check(columns, expected):
                if LooseVersion("0.21.1") <= LooseVersion(pd.__version__):
                    expected = pd.read_parquet(tmp, columns=columns)
                actual = ks.read_parquet(tmp, columns=columns)
                self.assertPandasEqual(expected, actual.toPandas())

            check(None, data)
            check(["i32", "i64"], data[["i32", "i64"]])
            check(["i64", "i32"], data[["i64", "i32"]])
            check(("i32", "i64"), data[["i32", "i64"]])
            check(["a", "b", "i32", "i64"], data[["i32", "i64"]])
            check([], pd.DataFrame([]))
            check(["a"], pd.DataFrame([]))
            check("i32", pd.DataFrame([]))
            check("float", data[["f"]])

            # check with pyspark patch.
            if LooseVersion("0.21.1") <= LooseVersion(pd.__version__):
                expected = pd.read_parquet(tmp)
            else:
                expected = data
            actual = ks.read_parquet(tmp)
            self.assertPandasEqual(expected, actual.toPandas())

            # When index columns are known
            pdf = self.test_pdf
            expected = ks.DataFrame(pdf)

            expected_idx = expected.set_index("bhello")[["f", "i32", "i64"]]
            actual_idx = ks.read_parquet(tmp, index_col="bhello")[["f", "i32", "i64"]]
            self.assert_eq(
                actual_idx.sort_values(by="f").to_spark().toPandas(),
                expected_idx.sort_values(by="f").to_spark().toPandas(),
            )

    def test_parquet_write(self):
        with self.temp_dir() as tmp:
            pdf = self.test_pdf
            expected = ks.DataFrame(pdf)

            # Write out partitioned by one column
            expected.to_parquet(tmp, mode="overwrite", partition_cols="i32")
            # Reset column order, as once the data is written out, Spark rearranges partition
            # columns to appear first.
            actual = ks.read_parquet(tmp)[self.test_column_order]
            self.assert_eq(
                actual.sort_values(by="f").to_spark().toPandas(),
                expected.sort_values(by="f").to_spark().toPandas(),
            )

            # Write out partitioned by two columns
            expected.to_parquet(tmp, mode="overwrite", partition_cols=["i32", "bhello"])
            # Reset column order, as once the data is written out, Spark rearranges partition
            # columns to appear first.
            actual = ks.read_parquet(tmp)[self.test_column_order]
            self.assert_eq(
                actual.sort_values(by="f").to_spark().toPandas(),
                expected.sort_values(by="f").to_spark().toPandas(),
            )

    def test_table(self):
        with self.table("test_table"):
            pdf = self.test_pdf
            expected = ks.DataFrame(pdf)

            # Write out partitioned by one column
            expected.spark.to_table("test_table", mode="overwrite", partition_cols="i32")
            # Reset column order, as once the data is written out, Spark rearranges partition
            # columns to appear first.
            actual = ks.read_table("test_table")[self.test_column_order]
            self.assert_eq(
                actual.sort_values(by="f").to_spark().toPandas(),
                expected.sort_values(by="f").to_spark().toPandas(),
            )

            # Write out partitioned by two columns
            expected.to_table("test_table", mode="overwrite", partition_cols=["i32", "bhello"])
            # Reset column order, as once the data is written out, Spark rearranges partition
            # columns to appear first.
            actual = ks.read_table("test_table")[self.test_column_order]
            self.assert_eq(
                actual.sort_values(by="f").to_spark().toPandas(),
                expected.sort_values(by="f").to_spark().toPandas(),
            )

            # When index columns are known
            expected_idx = expected.set_index("bhello")[["f", "i32", "i64"]]
            actual_idx = ks.read_table("test_table", "bhello")[["f", "i32", "i64"]]
            self.assert_eq(
                actual_idx.sort_values(by="f").to_spark().toPandas(),
                expected_idx.sort_values(by="f").to_spark().toPandas(),
            )

            expected_idx = expected.set_index(["bhello"])[["f", "i32", "i64"]]
            actual_idx = ks.read_table("test_table", ["bhello"])[["f", "i32", "i64"]]
            self.assert_eq(
                actual_idx.sort_values(by="f").to_spark().toPandas(),
                expected_idx.sort_values(by="f").to_spark().toPandas(),
            )

            expected_idx = expected.set_index(["i32", "bhello"])[["f", "i64"]]
            actual_idx = ks.read_table("test_table", ["i32", "bhello"])[["f", "i64"]]
            self.assert_eq(
                actual_idx.sort_values(by="f").to_spark().toPandas(),
                expected_idx.sort_values(by="f").to_spark().toPandas(),
            )

    def test_spark_io(self):
        with self.temp_dir() as tmp:
            pdf = self.test_pdf
            expected = ks.DataFrame(pdf)

            # Write out partitioned by one column
            expected.to_spark_io(tmp, format="json", mode="overwrite", partition_cols="i32")
            # Reset column order, as once the data is written out, Spark rearranges partition
            # columns to appear first.
            actual = ks.read_spark_io(tmp, format="json")[self.test_column_order]
            self.assert_eq(
                actual.sort_values(by="f").to_spark().toPandas(),
                expected.sort_values(by="f").to_spark().toPandas(),
            )

            # Write out partitioned by two columns
            expected.to_spark_io(
                tmp, format="json", mode="overwrite", partition_cols=["i32", "bhello"]
            )
            # Reset column order, as once the data is written out, Spark rearranges partition
            # columns to appear first.
            actual = ks.read_spark_io(path=tmp, format="json")[self.test_column_order]
            self.assert_eq(
                actual.sort_values(by="f").to_spark().toPandas(),
                expected.sort_values(by="f").to_spark().toPandas(),
            )

            # When index columns are known
            pdf = self.test_pdf
            expected = ks.DataFrame(pdf)
            col_order = ["f", "i32", "i64"]

            expected_idx = expected.set_index("bhello")[col_order]
            actual_idx = ks.read_spark_io(tmp, format="json", index_col="bhello")[col_order]
            self.assert_eq(
                actual_idx.sort_values(by="f").to_spark().toPandas(),
                expected_idx.sort_values(by="f").to_spark().toPandas(),
            )
