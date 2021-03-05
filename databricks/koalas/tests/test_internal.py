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

from databricks.koalas.internal import (
    InternalFrame,
    SPARK_DEFAULT_INDEX_NAME,
    SPARK_INDEX_NAME_FORMAT,
)
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils
from databricks.koalas.utils import spark_column_equals


class InternalFrameTest(ReusedSQLTestCase, SQLTestUtils):
    def test_from_pandas(self):
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        internal = InternalFrame.from_pandas(pdf)
        sdf = internal.spark_frame

        self.assert_eq(internal.index_spark_column_names, [SPARK_DEFAULT_INDEX_NAME])
        self.assert_eq(internal.index_names, [None])
        self.assert_eq(internal.column_labels, [("a",), ("b",)])
        self.assert_eq(internal.data_spark_column_names, ["a", "b"])
        self.assertTrue(spark_column_equals(internal.spark_column_for(("a",)), sdf["a"]))
        self.assertTrue(spark_column_equals(internal.spark_column_for(("b",)), sdf["b"]))

        self.assert_eq(internal.to_pandas_frame, pdf)

        # non-string column name
        pdf1 = pd.DataFrame({0: [1, 2, 3], 1: [4, 5, 6]})

        internal = InternalFrame.from_pandas(pdf1)
        sdf = internal.spark_frame

        self.assert_eq(internal.index_spark_column_names, [SPARK_DEFAULT_INDEX_NAME])
        self.assert_eq(internal.index_names, [None])
        self.assert_eq(internal.column_labels, [(0,), (1,)])
        self.assert_eq(internal.data_spark_column_names, ["0", "1"])
        self.assertTrue(spark_column_equals(internal.spark_column_for((0,)), sdf["0"]))
        self.assertTrue(spark_column_equals(internal.spark_column_for((1,)), sdf["1"]))

        self.assert_eq(internal.to_pandas_frame, pdf1)

        # multi-index
        pdf.set_index("a", append=True, inplace=True)

        internal = InternalFrame.from_pandas(pdf)
        sdf = internal.spark_frame

        self.assert_eq(
            internal.index_spark_column_names,
            [SPARK_INDEX_NAME_FORMAT(0), SPARK_INDEX_NAME_FORMAT(1)],
        )
        self.assert_eq(internal.index_names, [None, ("a",)])
        self.assert_eq(internal.column_labels, [("b",)])
        self.assert_eq(internal.data_spark_column_names, ["b"])
        self.assertTrue(spark_column_equals(internal.spark_column_for(("b",)), sdf["b"]))

        self.assert_eq(internal.to_pandas_frame, pdf)

        # multi-index columns
        pdf.columns = pd.MultiIndex.from_tuples([("x", "b")])

        internal = InternalFrame.from_pandas(pdf)
        sdf = internal.spark_frame

        self.assert_eq(
            internal.index_spark_column_names,
            [SPARK_INDEX_NAME_FORMAT(0), SPARK_INDEX_NAME_FORMAT(1)],
        )
        self.assert_eq(internal.index_names, [None, ("a",)])
        self.assert_eq(internal.column_labels, [("x", "b")])
        self.assert_eq(internal.data_spark_column_names, ["(x, b)"])
        self.assertTrue(spark_column_equals(internal.spark_column_for(("x", "b")), sdf["(x, b)"]))

        self.assert_eq(internal.to_pandas_frame, pdf)
