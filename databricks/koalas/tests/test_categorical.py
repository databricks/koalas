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

import databricks.koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


class CategoricalTest(ReusedSQLTestCase, TestUtils):
    def test_categorical_frame(self):
        pdf = pd.DataFrame(
            {
                "a": pd.Categorical([1, 2, 3, 1, 2, 3]),
                "b": pd.Categorical(["a", "b", "c", "a", "b", "c"], categories=["c", "b", "a"]),
            },
            index=pd.Categorical([10, 20, 30, 20, 30, 10], categories=[30, 10, 20], ordered=True),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf, pdf)
        self.assert_eq(kdf.a, pdf.a)
        self.assert_eq(kdf.b, pdf.b)
        self.assert_eq(kdf.index, pdf.index)

    def test_categorical_series(self):
        pser = pd.Series([1, 2, 3], dtype="category")
        kser = ks.Series([1, 2, 3], dtype="category")

        self.assert_eq(kser, pser)
        self.assert_eq(kser.cat.categories, pser.cat.categories)
        self.assert_eq(kser.cat.codes, pser.cat.codes)
        self.assert_eq(kser.cat.ordered, pser.cat.ordered)

    def test_categorical_index(self):
        pser = pd.Index([1, 2, 3], dtype="category")
        kser = ks.Index([1, 2, 3], dtype="category")

        self.assert_eq(kser, pser)
        self.assert_eq(kser.categories, pser.categories)
        self.assert_eq(kser.codes, pd.Index(pser.codes))
        self.assert_eq(kser.ordered, pser.ordered)
