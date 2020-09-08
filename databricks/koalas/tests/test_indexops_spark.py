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
from pyspark.sql.utils import AnalysisException
from pyspark.sql import functions as F

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class SparkIndexOpsMethodsTest(ReusedSQLTestCase, SQLTestUtils):
    @property
    def pser(self):
        return pd.Series([1, 2, 3, 4, 5, 6, 7], name="x")

    @property
    def kser(self):
        return ks.from_pandas(self.pser)

    def test_series_transform_negative(self):
        with self.assertRaisesRegex(
            ValueError, "The output of the function.* pyspark.sql.Column.*int"
        ):
            self.kser.spark.transform(lambda scol: 1)

        with self.assertRaisesRegex(AnalysisException, "cannot resolve.*non-existent.*"):
            self.kser.spark.transform(lambda scol: F.col("non-existent"))

    def test_multiindex_transform_negative(self):
        with self.assertRaisesRegex(
            NotImplementedError, "MultiIndex does not support spark.transform yet"
        ):
            midx = pd.MultiIndex(
                [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
                [[0, 0, 0, 1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 1, 2, 1, 2, 2]],
            )
            s = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
            s.index.spark.transform(lambda scol: scol)

    def test_series_apply_negative(self):
        with self.assertRaisesRegex(
            ValueError, "The output of the function.* pyspark.sql.Column.*int"
        ):
            self.kser.spark.apply(lambda scol: 1)

        with self.assertRaisesRegex(AnalysisException, "cannot resolve.*non-existent.*"):
            self.kser.spark.transform(lambda scol: F.col("non-existent"))

    def test_index_apply_negative(self):
        with self.assertRaisesRegex(NotImplementedError, "Index does not support spark.apply yet"):
            ks.range(10).index.spark.apply(lambda scol: scol)
