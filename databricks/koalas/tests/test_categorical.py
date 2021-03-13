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

import pandas as pd
from pandas.api.types import CategoricalDtype

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

        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        self.assert_eq(kdf.sort_values("b"), pdf.sort_values("b"))

    def test_categorical_series(self):
        pser = pd.Series([1, 2, 3], dtype="category")
        kser = ks.Series([1, 2, 3], dtype="category")

        self.assert_eq(kser, pser)
        self.assert_eq(kser.cat.categories, pser.cat.categories)
        self.assert_eq(kser.cat.codes, pser.cat.codes)
        self.assert_eq(kser.cat.ordered, pser.cat.ordered)

    def test_astype(self):
        pser = pd.Series(["a", "b", "c"])
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.astype("category"), pser.astype("category"))
        self.assert_eq(
            kser.astype(CategoricalDtype(["c", "a", "b"])),
            pser.astype(CategoricalDtype(["c", "a", "b"])),
        )

        pcser = pser.astype(CategoricalDtype(["c", "a", "b"]))
        kcser = kser.astype(CategoricalDtype(["c", "a", "b"]))

        self.assert_eq(kcser.astype("category"), pcser.astype("category"))

        if LooseVersion(pd.__version__) >= LooseVersion("1.2"):
            self.assert_eq(
                kcser.astype(CategoricalDtype(["b", "c", "a"])),
                pcser.astype(CategoricalDtype(["b", "c", "a"])),
            )
        else:
            self.assert_eq(
                kcser.astype(CategoricalDtype(["b", "c", "a"])),
                pser.astype(CategoricalDtype(["b", "c", "a"])),
            )

        self.assert_eq(kcser.astype(str), pcser.astype(str))

    def test_factorize(self):
        pser = pd.Series(["a", "b", "c", None], dtype=CategoricalDtype(["c", "a", "d", "b"]))
        kser = ks.from_pandas(pser)

        pcodes, puniques = pser.factorize()
        kcodes, kuniques = kser.factorize()

        self.assert_eq(kcodes.tolist(), pcodes.tolist())
        self.assert_eq(kuniques, puniques)

        pcodes, puniques = pser.factorize(na_sentinel=-2)
        kcodes, kuniques = kser.factorize(na_sentinel=-2)

        self.assert_eq(kcodes.tolist(), pcodes.tolist())
        self.assert_eq(kuniques, puniques)
