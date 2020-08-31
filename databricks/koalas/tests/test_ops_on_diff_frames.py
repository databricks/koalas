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
import numpy as np

import pyspark

from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class OpsOnDiffFramesEnabledTest(ReusedSQLTestCase, SQLTestUtils):
    @classmethod
    def setUpClass(cls):
        super(OpsOnDiffFramesEnabledTest, cls).setUpClass()
        set_option("compute.ops_on_diff_frames", True)

    @classmethod
    def tearDownClass(cls):
        reset_option("compute.ops_on_diff_frames")
        super(OpsOnDiffFramesEnabledTest, cls).tearDownClass()

    @property
    def pdf1(self):
        return pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9], "b": [4, 5, 6, 3, 2, 1, 0, 0, 0]},
            index=[0, 1, 3, 5, 6, 8, 9, 10, 11],
        )

    @property
    def pdf2(self):
        return pd.DataFrame(
            {"a": [9, 8, 7, 6, 5, 4, 3, 2, 1], "b": [0, 0, 0, 4, 5, 6, 1, 2, 3]},
            index=list(range(9)),
        )

    @property
    def pdf3(self):
        return pd.DataFrame(
            {"b": [1, 1, 1, 1, 1, 1, 1, 1, 1], "c": [1, 1, 1, 1, 1, 1, 1, 1, 1]},
            index=list(range(9)),
        )

    @property
    def pdf4(self):
        return pd.DataFrame(
            {"e": [2, 2, 2, 2, 2, 2, 2, 2, 2], "f": [2, 2, 2, 2, 2, 2, 2, 2, 2]},
            index=list(range(9)),
        )

    @property
    def pdf5(self):
        return pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "b": [4, 5, 6, 3, 2, 1, 0, 0, 0],
                "c": [4, 5, 6, 3, 2, 1, 0, 0, 0],
            },
            index=[0, 1, 3, 5, 6, 8, 9, 10, 11],
        ).set_index(["a", "b"])

    @property
    def pdf6(self):
        return pd.DataFrame(
            {
                "a": [9, 8, 7, 6, 5, 4, 3, 2, 1],
                "b": [0, 0, 0, 4, 5, 6, 1, 2, 3],
                "c": [9, 8, 7, 6, 5, 4, 3, 2, 1],
                "e": [4, 5, 6, 3, 2, 1, 0, 0, 0],
            },
            index=list(range(9)),
        ).set_index(["a", "b"])

    @property
    def pser1(self):
        midx = pd.MultiIndex(
            [["lama", "cow", "falcon", "koala"], ["speed", "weight", "length", "power"]],
            [[0, 3, 1, 1, 1, 2, 2, 2], [0, 2, 0, 3, 2, 0, 1, 3]],
        )
        return pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1], index=midx)

    @property
    def pser2(self):
        midx = pd.MultiIndex(
            [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        return pd.Series([-45, 200, -1.2, 30, -250, 1.5, 320, 1, -0.3], index=midx)

    @property
    def pser3(self):
        midx = pd.MultiIndex(
            [["koalas", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [1, 1, 2, 0, 0, 2, 2, 2, 1]],
        )
        return pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)

    @property
    def kdf1(self):
        return ks.from_pandas(self.pdf1)

    @property
    def kdf2(self):
        return ks.from_pandas(self.pdf2)

    @property
    def kdf3(self):
        return ks.from_pandas(self.pdf3)

    @property
    def kdf4(self):
        return ks.from_pandas(self.pdf4)

    @property
    def kdf5(self):
        return ks.from_pandas(self.pdf5)

    @property
    def kdf6(self):
        return ks.from_pandas(self.pdf6)

    @property
    def kser1(self):
        return ks.from_pandas(self.pser1)

    @property
    def kser2(self):
        return ks.from_pandas(self.pser2)

    @property
    def kser3(self):
        return ks.from_pandas(self.pser3)

    def test_ranges(self):
        self.assert_eq(
            (ks.range(10) + ks.range(10)).sort_index(),
            (
                ks.DataFrame({"id": list(range(10))}) + ks.DataFrame({"id": list(range(10))})
            ).sort_index(),
        )

    def test_no_matched_index(self):
        with self.assertRaisesRegex(ValueError, "Index names must be exactly matched"):
            ks.DataFrame({"a": [1, 2, 3]}).set_index("a") + ks.DataFrame(
                {"b": [1, 2, 3]}
            ).set_index("b")

    def test_arithmetic(self):
        kdf1 = self.kdf1
        kdf2 = self.kdf2
        pdf1 = self.pdf1
        pdf2 = self.pdf2
        kser1 = self.kser1
        pser1 = self.pser1
        kser2 = self.kser2
        pser2 = self.pser2

        # Series
        self.assert_eq((kdf1.a - kdf2.b).sort_index(), (pdf1.a - pdf2.b).rename("a").sort_index())

        self.assert_eq((kdf1.a * kdf2.a).sort_index(), (pdf1.a * pdf2.a).rename("a").sort_index())

        self.assert_eq(
            (kdf1["a"] / kdf2["a"]).sort_index(), (pdf1["a"] / pdf2["a"]).rename("a").sort_index()
        )

        # DataFrame
        self.assert_eq((kdf1 + kdf2).sort_index(), (pdf1 + pdf2).sort_index())

        # Multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b")])
        kdf1.columns = columns
        kdf2.columns = columns
        pdf1.columns = columns
        pdf2.columns = columns

        # Series
        self.assert_eq(
            (kdf1[("x", "a")] - kdf2[("x", "b")]).sort_index(),
            (pdf1[("x", "a")] - pdf2[("x", "b")]).rename(("x", "a")).sort_index(),
        )

        self.assert_eq(
            (kdf1[("x", "a")] - kdf2["x"]["b"]).sort_index(),
            (pdf1[("x", "a")] - pdf2["x"]["b"]).rename(("x", "a")).sort_index(),
        )

        self.assert_eq(
            (kdf1["x"]["a"] - kdf2[("x", "b")]).sort_index(),
            (pdf1["x"]["a"] - pdf2[("x", "b")]).rename("a").sort_index(),
        )

        # DataFrame
        self.assert_eq((kdf1 + kdf2).sort_index(), (pdf1 + pdf2).sort_index())

        # MultiIndex Series
        self.assert_eq((kser1 + kser2).sort_index(), (pser1 + pser2).sort_index())

        self.assert_eq((kser1 - kser2).sort_index(), (pser1 - pser2).sort_index())

        self.assert_eq((kser1 * kser2).sort_index(), (pser1 * pser2).sort_index())

        self.assert_eq((kser1 / kser2).sort_index(), (pser1 / pser2).sort_index())

    def test_arithmetic_chain(self):
        kdf1 = self.kdf1
        kdf2 = self.kdf2
        kdf3 = self.kdf3
        pdf1 = self.pdf1
        pdf2 = self.pdf2
        pdf3 = self.pdf3
        kser1 = self.kser1
        pser1 = self.pser1
        kser2 = self.kser2
        pser2 = self.pser2
        kser3 = self.kser3
        pser3 = self.pser3

        # Series
        self.assert_eq(
            (kdf1.a - kdf2.b - kdf3.c).sort_index(),
            (pdf1.a - pdf2.b - pdf3.c).rename("a").sort_index(),
        )

        self.assert_eq(
            (kdf1.a * (kdf2.a * kdf3.c)).sort_index(),
            (pdf1.a * (pdf2.a * pdf3.c)).rename("a").sort_index(),
        )

        self.assert_eq(
            (kdf1["a"] / kdf2["a"] / kdf3["c"]).sort_index(),
            (pdf1["a"] / pdf2["a"] / pdf3["c"]).rename("a").sort_index(),
        )

        # DataFrame
        self.assert_eq(
            (kdf1 + kdf2 - kdf3).sort_index(), (pdf1 + pdf2 - pdf3).sort_index(), almost=True
        )

        # Multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b")])
        kdf1.columns = columns
        kdf2.columns = columns
        pdf1.columns = columns
        pdf2.columns = columns
        columns = pd.MultiIndex.from_tuples([("x", "b"), ("y", "c")])
        kdf3.columns = columns
        pdf3.columns = columns

        # Series
        self.assert_eq(
            (kdf1[("x", "a")] - kdf2[("x", "b")] - kdf3[("y", "c")]).sort_index(),
            (pdf1[("x", "a")] - pdf2[("x", "b")] - pdf3[("y", "c")])
            .rename(("x", "a"))
            .sort_index(),
        )

        self.assert_eq(
            (kdf1[("x", "a")] * (kdf2[("x", "b")] * kdf3[("y", "c")])).sort_index(),
            (pdf1[("x", "a")] * (pdf2[("x", "b")] * pdf3[("y", "c")]))
            .rename(("x", "a"))
            .sort_index(),
        )

        # DataFrame
        self.assert_eq(
            (kdf1 + kdf2 - kdf3).sort_index(), (pdf1 + pdf2 - pdf3).sort_index(), almost=True
        )

        # MultiIndex Series
        self.assert_eq((kser1 + kser2 - kser3).sort_index(), (pser1 + pser2 - pser3).sort_index())

        self.assert_eq((kser1 * kser2 * kser3).sort_index(), (pser1 * pser2 * pser3).sort_index())

        self.assert_eq((kser1 - kser2 / kser3).sort_index(), (pser1 - pser2 / pser3).sort_index())

        self.assert_eq((kser1 + kser2 * kser3).sort_index(), (pser1 + pser2 * pser3).sort_index())

    def test_mod(self):
        pser = pd.Series([100, None, -300, None, 500, -700])
        pser_other = pd.Series([-150] * 6)
        kser = ks.from_pandas(pser)
        kser_other = ks.from_pandas(pser_other)

        self.assert_eq(kser.mod(kser_other).sort_index(), pser.mod(pser_other))
        self.assert_eq(kser.mod(kser_other).sort_index(), pser.mod(pser_other))
        self.assert_eq(kser.mod(kser_other).sort_index(), pser.mod(pser_other))

    def test_rmod(self):
        pser = pd.Series([100, None, -300, None, 500, -700])
        pser_other = pd.Series([-150] * 6)
        kser = ks.from_pandas(pser)
        kser_other = ks.from_pandas(pser_other)

        self.assert_eq(kser.rmod(kser_other).sort_index(), pser.rmod(pser_other))
        self.assert_eq(kser.rmod(kser_other).sort_index(), pser.rmod(pser_other))
        self.assert_eq(kser.rmod(kser_other).sort_index(), pser.rmod(pser_other))

    def test_getitem_boolean_series(self):
        pdf1 = pd.DataFrame(
            {"A": [0, 1, 2, 3, 4], "B": [100, 200, 300, 400, 500]}, index=[20, 10, 30, 0, 50]
        )
        pdf2 = pd.DataFrame(
            {"A": [0, -1, -2, -3, -4], "B": [-100, -200, -300, -400, -500]},
            index=[0, 30, 10, 20, 50],
        )
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(pdf1[pdf2.A > -3].sort_index(), kdf1[kdf2.A > -3].sort_index())

        self.assert_eq(pdf1.A[pdf2.A > -3].sort_index(), kdf1.A[kdf2.A > -3].sort_index())

        self.assert_eq(
            (pdf1.A + 1)[pdf2.A > -3].sort_index(), (kdf1.A + 1)[kdf2.A > -3].sort_index()
        )

    def test_loc_getitem_boolean_series(self):
        pdf1 = pd.DataFrame(
            {"A": [0, 1, 2, 3, 4], "B": [100, 200, 300, 400, 500]}, index=[20, 10, 30, 0, 50]
        )
        pdf2 = pd.DataFrame(
            {"A": [0, -1, -2, -3, -4], "B": [-100, -200, -300, -400, -500]},
            index=[20, 10, 30, 0, 50],
        )
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(pdf1.loc[pdf2.A > -3].sort_index(), kdf1.loc[kdf2.A > -3].sort_index())

        self.assert_eq(pdf1.A.loc[pdf2.A > -3].sort_index(), kdf1.A.loc[kdf2.A > -3].sort_index())

        self.assert_eq(
            (pdf1.A + 1).loc[pdf2.A > -3].sort_index(), (kdf1.A + 1).loc[kdf2.A > -3].sort_index()
        )

    def test_bitwise(self):
        pser1 = pd.Series([True, False, True, False, np.nan, np.nan, True, False, np.nan])
        pser2 = pd.Series([True, False, False, True, True, False, np.nan, np.nan, np.nan])
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)

        self.assert_eq(pser1 | pser2, (kser1 | kser2).sort_index())
        self.assert_eq(pser1 & pser2, (kser1 & kser2).sort_index())

        pser1 = pd.Series([True, False, np.nan], index=list("ABC"))
        pser2 = pd.Series([False, True, np.nan], index=list("DEF"))
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)

        self.assert_eq(pser1 | pser2, (kser1 | kser2).sort_index())
        self.assert_eq(pser1 & pser2, (kser1 & kser2).sort_index())

    def test_different_columns(self):
        kdf1 = self.kdf1
        kdf4 = self.kdf4
        pdf1 = self.pdf1
        pdf4 = self.pdf4

        self.assert_eq((kdf1 + kdf4).sort_index(), (pdf1 + pdf4).sort_index(), almost=True)

        # Multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b")])
        kdf1.columns = columns
        pdf1.columns = columns
        columns = pd.MultiIndex.from_tuples([("z", "e"), ("z", "f")])
        kdf4.columns = columns
        pdf4.columns = columns

        self.assert_eq((kdf1 + kdf4).sort_index(), (pdf1 + pdf4).sort_index(), almost=True)

    def test_assignment_series(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kser = kdf.a
        pser = pdf.a
        kdf["a"] = self.kdf2.a
        pdf["a"] = self.pdf2.a

        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        self.assert_eq(kser, pser)

        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kser = kdf.a
        pser = pdf.a
        kdf["a"] = self.kdf2.b
        pdf["a"] = self.pdf2.b

        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        self.assert_eq(kser, pser)

        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf["c"] = self.kdf2.a
        pdf["c"] = self.pdf2.a

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        # Multi-index columns
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b")])
        kdf.columns = columns
        pdf.columns = columns
        kdf[("y", "c")] = self.kdf2.a
        pdf[("y", "c")] = self.pdf2.a

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        pdf = pd.DataFrame({"a": [1, 2, 3], "Koalas": [0, 1, 2]}).set_index("Koalas", drop=False)
        kdf = ks.from_pandas(pdf)

        kdf.index.name = None
        kdf["NEW"] = ks.Series([100, 200, 300])

        pdf.index.name = None
        pdf["NEW"] = pd.Series([100, 200, 300])

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_assignment_frame(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kser = kdf.a
        pser = pdf.a
        kdf[["a", "b"]] = self.kdf1
        pdf[["a", "b"]] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        self.assert_eq(kser, pser)

        # 'c' does not exist in `kdf`.
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kser = kdf.a
        pser = pdf.a
        kdf[["b", "c"]] = self.kdf1
        pdf[["b", "c"]] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        self.assert_eq(kser, pser)

        # 'c' and 'd' do not exist in `kdf`.
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf[["c", "d"]] = self.kdf1
        pdf[["c", "d"]] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        # Multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b")])
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf.columns = columns
        pdf.columns = columns
        kdf[[("y", "c"), ("z", "d")]] = self.kdf1
        pdf[[("y", "c"), ("z", "d")]] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf1 = ks.from_pandas(self.pdf1)
        pdf1 = self.pdf1
        kdf1.columns = columns
        pdf1.columns = columns
        kdf[["c", "d"]] = kdf1
        pdf[["c", "d"]] = pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_assignment_series_chain(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf["a"] = self.kdf1.a
        pdf["a"] = self.pdf1.a

        kdf["a"] = self.kdf2.b
        pdf["a"] = self.pdf2.b

        kdf["d"] = self.kdf3.c
        pdf["d"] = self.pdf3.c

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_assignment_frame_chain(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf[["a", "b"]] = self.kdf1
        pdf[["a", "b"]] = self.pdf1

        kdf[["e", "f"]] = self.kdf3
        pdf[["e", "f"]] = self.pdf3

        kdf[["b", "c"]] = self.kdf2
        pdf[["b", "c"]] = self.pdf2

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_multi_index_arithmetic(self):
        kdf5 = self.kdf5
        kdf6 = self.kdf6
        pdf5 = self.pdf5
        pdf6 = self.pdf6

        # Series
        self.assert_eq((kdf5.c - kdf6.e).sort_index(), (pdf5.c - pdf6.e).rename("c").sort_index())

        self.assert_eq(
            (kdf5["c"] / kdf6["e"]).sort_index(), (pdf5["c"] / pdf6["e"]).rename("c").sort_index()
        )

        # DataFrame
        self.assert_eq((kdf5 + kdf6).sort_index(), (pdf5 + pdf6).sort_index(), almost=True)

    def test_multi_index_assignment_series(self):
        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5
        kdf["x"] = self.kdf6.e
        pdf["x"] = self.pdf6.e

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5
        kdf["e"] = self.kdf6.e
        pdf["e"] = self.pdf6.e

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5
        kdf["c"] = self.kdf6.e
        pdf["c"] = self.pdf6.e

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_multi_index_assignment_frame(self):
        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5
        kdf[["c"]] = self.kdf5
        pdf[["c"]] = self.pdf5

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5
        kdf[["x"]] = self.kdf5
        pdf[["x"]] = self.pdf5

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf6)
        pdf = self.pdf6
        kdf[["x", "y"]] = self.kdf6
        pdf[["x", "y"]] = self.pdf6

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_frame_loc_setitem(self):
        pdf_orig = pd.DataFrame(
            [[1, 2], [4, 5], [7, 8]],
            index=["cobra", "viper", "sidewinder"],
            columns=["max_speed", "shield"],
        )
        kdf_orig = ks.DataFrame(pdf_orig)

        pdf = pdf_orig.copy()
        kdf = kdf_orig.copy()
        pser1 = pdf.max_speed
        pser2 = pdf.shield
        kser1 = kdf.max_speed
        kser2 = kdf.shield

        another_kdf = ks.DataFrame(pdf_orig)

        kdf.loc[["viper", "sidewinder"], ["shield"]] = -another_kdf.max_speed
        pdf.loc[["viper", "sidewinder"], ["shield"]] = -pdf.max_speed
        self.assert_eq(kdf, pdf)
        self.assert_eq(kser1, pser1)
        self.assert_eq(kser2, pser2)

        pdf = pdf_orig.copy()
        kdf = kdf_orig.copy()
        pser1 = pdf.max_speed
        pser2 = pdf.shield
        kser1 = kdf.max_speed
        kser2 = kdf.shield
        kdf.loc[another_kdf.max_speed < 5, ["shield"]] = -kdf.max_speed
        pdf.loc[pdf.max_speed < 5, ["shield"]] = -pdf.max_speed
        self.assert_eq(kdf, pdf)
        self.assert_eq(kser1, pser1)
        self.assert_eq(kser2, pser2)

        pdf = pdf_orig.copy()
        kdf = kdf_orig.copy()
        pser1 = pdf.max_speed
        pser2 = pdf.shield
        kser1 = kdf.max_speed
        kser2 = kdf.shield
        kdf.loc[another_kdf.max_speed < 5, ["shield"]] = -another_kdf.max_speed
        pdf.loc[pdf.max_speed < 5, ["shield"]] = -pdf.max_speed
        self.assert_eq(kdf, pdf)
        self.assert_eq(kser1, pser1)
        self.assert_eq(kser2, pser2)

    def test_frame_iloc_setitem(self):
        pdf = pd.DataFrame(
            [[1, 2], [4, 5], [7, 8]],
            index=["cobra", "viper", "sidewinder"],
            columns=["max_speed", "shield"],
        )
        kdf = ks.DataFrame(pdf)
        another_kdf = ks.DataFrame(pdf)

        kdf.iloc[[1, 2], [1]] = -another_kdf.max_speed
        pdf.iloc[[1, 2], [1]] = -pdf.max_speed
        self.assert_eq(kdf, pdf)

        kdf.iloc[[0], 1] = 10 * another_kdf.max_speed
        pdf.iloc[[0], 1] = 10 * pdf.max_speed
        self.assert_eq(kdf, pdf)

    def test_series_loc_setitem(self):
        pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=["cobra", "viper", "sidewinder"])
        kdf = ks.from_pandas(pdf)
        pser = pdf.x
        psery = pdf.y
        kser = kdf.x
        ksery = kdf.y

        pser_another = pd.Series([1, 2, 3], index=["cobra", "viper", "sidewinder"])
        kser_another = ks.from_pandas(pser_another)

        kser.loc[kser % 2 == 1] = -kser_another
        pser.loc[pser % 2 == 1] = -pser_another
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)
        self.assert_eq(ksery, psery)

        pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=["cobra", "viper", "sidewinder"])
        kdf = ks.from_pandas(pdf)
        pser = pdf.x
        psery = pdf.y
        kser = kdf.x
        ksery = kdf.y
        kser.loc[kser_another % 2 == 1] = -kser
        pser.loc[pser_another % 2 == 1] = -pser
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)
        self.assert_eq(ksery, psery)

        pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=["cobra", "viper", "sidewinder"])
        kdf = ks.from_pandas(pdf)
        pser = pdf.x
        psery = pdf.y
        kser = kdf.x
        ksery = kdf.y
        kser.loc[kser_another % 2 == 1] = -kser
        pser.loc[pser_another % 2 == 1] = -pser
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)
        self.assert_eq(ksery, psery)

        pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=["cobra", "viper", "sidewinder"])
        kdf = ks.from_pandas(pdf)
        pser = pdf.x
        psery = pdf.y
        kser = kdf.x
        ksery = kdf.y
        kser.loc[kser_another % 2 == 1] = -kser_another
        pser.loc[pser_another % 2 == 1] = -pser_another
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)
        self.assert_eq(ksery, psery)

        pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=["cobra", "viper", "sidewinder"])
        kdf = ks.from_pandas(pdf)
        pser = pdf.x
        psery = pdf.y
        kser = kdf.x
        ksery = kdf.y
        kser.loc[["viper", "sidewinder"]] = -kser_another
        pser.loc[["viper", "sidewinder"]] = -pser_another
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)
        self.assert_eq(ksery, psery)

        pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=["cobra", "viper", "sidewinder"])
        kdf = ks.from_pandas(pdf)
        pser = pdf.x
        psery = pdf.y
        kser = kdf.x
        ksery = kdf.y
        kser.loc[kser_another % 2 == 1] = 10
        pser.loc[pser_another % 2 == 1] = 10
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)
        self.assert_eq(ksery, psery)

    def test_series_iloc_setitem(self):
        pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=["cobra", "viper", "sidewinder"])
        kdf = ks.from_pandas(pdf)

        pser = pdf.x
        psery = pdf.y
        kser = kdf.x
        ksery = kdf.y

        pser1 = pser + 1
        kser1 = kser + 1

        pser_another = pd.Series([1, 2, 3], index=["cobra", "viper", "sidewinder"])
        kser_another = ks.from_pandas(pser_another)

        kser.iloc[[1, 2]] = -kser_another
        pser.iloc[[1, 2]] = -pser_another
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)
        self.assert_eq(ksery, psery)

        kser.iloc[[0]] = 10 * kser_another
        pser.iloc[[0]] = 10 * pser_another
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)
        self.assert_eq(ksery, psery)

        kser1.iloc[[1, 2]] = -kser_another
        pser1.iloc[[1, 2]] = -pser_another
        self.assert_eq(kser1, pser1)
        self.assert_eq(kdf, pdf)
        self.assert_eq(ksery, psery)

        pdf = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=["cobra", "viper", "sidewinder"])
        kdf = ks.from_pandas(pdf)

        pser = pdf.x
        psery = pdf.y
        kser = kdf.x
        ksery = kdf.y

        piloc = pser.iloc
        kiloc = kser.iloc

        kiloc[[1, 2]] = -kser_another
        piloc[[1, 2]] = -pser_another
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)
        self.assert_eq(ksery, psery)

        kiloc[[0]] = 10 * kser_another
        piloc[[0]] = 10 * pser_another
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)
        self.assert_eq(ksery, psery)

    def test_update(self):
        pdf = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
        kdf = ks.from_pandas(pdf)

        pser = pdf.x
        kser = kdf.x
        pser.update(pd.Series([4, 5, 6]))
        kser.update(ks.Series([4, 5, 6]))
        self.assert_eq(kser.sort_index(), pser.sort_index())
        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_where(self):
        pdf1 = pd.DataFrame({"A": [0, 1, 2, 3, 4], "B": [100, 200, 300, 400, 500]})
        pdf2 = pd.DataFrame({"A": [0, -1, -2, -3, -4], "B": [-100, -200, -300, -400, -500]})
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(repr(pdf1.where(pdf2 > 100)), repr(kdf1.where(kdf2 > 100).sort_index()))

        pdf1 = pd.DataFrame({"A": [-1, -2, -3, -4, -5], "B": [-100, -200, -300, -400, -500]})
        pdf2 = pd.DataFrame({"A": [-10, -20, -30, -40, -50], "B": [-5, -4, -3, -2, -1]})
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(repr(pdf1.where(pdf2 < -250)), repr(kdf1.where(kdf2 < -250).sort_index()))

        # multi-index columns
        pdf1 = pd.DataFrame({("X", "A"): [0, 1, 2, 3, 4], ("X", "B"): [100, 200, 300, 400, 500]})
        pdf2 = pd.DataFrame(
            {("X", "A"): [0, -1, -2, -3, -4], ("X", "B"): [-100, -200, -300, -400, -500]}
        )
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(repr(pdf1.where(pdf2 > 100)), repr(kdf1.where(kdf2 > 100).sort_index()))

    def test_mask(self):
        pdf1 = pd.DataFrame({"A": [0, 1, 2, 3, 4], "B": [100, 200, 300, 400, 500]})
        pdf2 = pd.DataFrame({"A": [0, -1, -2, -3, -4], "B": [-100, -200, -300, -400, -500]})
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(repr(pdf1.mask(pdf2 < 100)), repr(kdf1.mask(kdf2 < 100).sort_index()))

        pdf1 = pd.DataFrame({"A": [-1, -2, -3, -4, -5], "B": [-100, -200, -300, -400, -500]})
        pdf2 = pd.DataFrame({"A": [-10, -20, -30, -40, -50], "B": [-5, -4, -3, -2, -1]})
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(repr(pdf1.mask(pdf2 > -250)), repr(kdf1.mask(kdf2 > -250).sort_index()))

        # multi-index columns
        pdf1 = pd.DataFrame({("X", "A"): [0, 1, 2, 3, 4], ("X", "B"): [100, 200, 300, 400, 500]})
        pdf2 = pd.DataFrame(
            {("X", "A"): [0, -1, -2, -3, -4], ("X", "B"): [-100, -200, -300, -400, -500]}
        )
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(repr(pdf1.mask(pdf2 < 100)), repr(kdf1.mask(kdf2 < 100).sort_index()))

    def test_multi_index_column_assignment_frame(self):
        pdf = pd.DataFrame({"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0]})
        pdf.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
        kdf = ks.DataFrame(pdf)

        kdf["c"] = ks.Series([10, 20, 30, 20])
        pdf["c"] = pd.Series([10, 20, 30, 20])

        kdf[("d", "x")] = ks.Series([100, 200, 300, 200], name="1")
        pdf[("d", "x")] = pd.Series([100, 200, 300, 200], name="1")

        kdf[("d", "y")] = ks.Series([1000, 2000, 3000, 2000], name=("1", "2"))
        pdf[("d", "y")] = pd.Series([1000, 2000, 3000, 2000], name=("1", "2"))

        kdf["e"] = ks.Series([10000, 20000, 30000, 20000], name=("1", "2", "3"))
        pdf["e"] = pd.Series([10000, 20000, 30000, 20000], name=("1", "2", "3"))

        kdf[[("f", "x"), ("f", "y")]] = ks.DataFrame(
            {"1": [100000, 200000, 300000, 200000], "2": [1000000, 2000000, 3000000, 2000000]}
        )
        pdf[[("f", "x"), ("f", "y")]] = pd.DataFrame(
            {"1": [100000, 200000, 300000, 200000], "2": [1000000, 2000000, 3000000, 2000000]}
        )

        self.assert_eq(repr(kdf.sort_index()), repr(pdf))

        with self.assertRaisesRegex(KeyError, "Key length \\(3\\) exceeds index depth \\(2\\)"):
            kdf[("1", "2", "3")] = ks.Series([100, 200, 300, 200])

    def test_dot(self):
        pser = pd.Series([90, 91, 85], index=[2, 4, 1])
        kser = ks.from_pandas(pser)
        pser_other = pd.Series([90, 91, 85], index=[2, 4, 1])
        kser_other = ks.from_pandas(pser_other)

        self.assert_eq(kser.dot(kser_other), pser.dot(pser_other))

        kser_other = ks.Series([90, 91, 85], index=[1, 2, 4])
        pser_other = pd.Series([90, 91, 85], index=[1, 2, 4])

        self.assert_eq(kser.dot(kser_other), pser.dot(pser_other))

        # length of index is different
        kser_other = ks.Series([90, 91, 85, 100], index=[2, 4, 1, 0])
        with self.assertRaisesRegex(ValueError, "matrices are not aligned"):
            kser.dot(kser_other)

        # with DataFram is not supported for now since performance issue,
        # now we raise ValueError with proper message instead.
        kdf = ks.DataFrame([[0, 1], [-2, 3], [4, -5]], index=[2, 4, 1])

        with self.assertRaisesRegex(ValueError, r"Series\.dot\(\) is currently not supported*"):
            kser.dot(kdf)

        # for MultiIndex
        midx = pd.MultiIndex(
            [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        pser = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        kser = ks.from_pandas(pser)
        pser_other = pd.Series([-450, 20, 12, -30, -250, 15, -320, 100, 3], index=midx)
        kser_other = ks.from_pandas(pser_other)

        self.assert_eq(kser.dot(kser_other), pser.dot(pser_other))

    def test_to_series_comparison(self):
        kidx1 = ks.Index([1, 2, 3, 4, 5])
        kidx2 = ks.Index([1, 2, 3, 4, 5])

        self.assert_eq((kidx1.to_series() == kidx2.to_series()).all(), True)

        kidx1.name = "koalas"
        kidx2.name = "koalas"

        self.assert_eq((kidx1.to_series() == kidx2.to_series()).all(), True)

    def test_series_repeat(self):
        pser1 = pd.Series(["a", "b", "c"], name="a")
        pser2 = pd.Series([10, 20, 30], name="rep")
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)

        if LooseVersion(pyspark.__version__) < LooseVersion("2.4"):
            self.assertRaises(ValueError, lambda: kser1.repeat(kser2))
        else:
            self.assert_eq(kser1.repeat(kser2).sort_index(), pser1.repeat(pser2).sort_index())


class OpsOnDiffFramesDisabledTest(ReusedSQLTestCase, SQLTestUtils):
    @classmethod
    def setUpClass(cls):
        super(OpsOnDiffFramesDisabledTest, cls).setUpClass()
        set_option("compute.ops_on_diff_frames", False)

    @classmethod
    def tearDownClass(cls):
        reset_option("compute.ops_on_diff_frames")
        super(OpsOnDiffFramesDisabledTest, cls).tearDownClass()

    @property
    def pdf1(self):
        return pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9], "b": [4, 5, 6, 3, 2, 1, 0, 0, 0],},
            index=[0, 1, 3, 5, 6, 8, 9, 9, 9],
        )

    @property
    def pdf2(self):
        return pd.DataFrame(
            {"a": [9, 8, 7, 6, 5, 4, 3, 2, 1], "b": [0, 0, 0, 4, 5, 6, 1, 2, 3],},
            index=list(range(9)),
        )

    @property
    def kdf1(self):
        return ks.from_pandas(self.pdf1)

    @property
    def kdf2(self):
        return ks.from_pandas(self.pdf2)

    def test_arithmetic(self):
        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            self.kdf1.a - self.kdf2.b

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            self.kdf1.a - self.kdf2.a

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            self.kdf1["a"] - self.kdf2["a"]

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            self.kdf1 - self.kdf2

    def test_assignment(self):
        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            kdf = ks.from_pandas(self.pdf1)
            kdf["c"] = self.kdf1.a

    def test_frame_loc_setitem(self):
        pdf = pd.DataFrame(
            [[1, 2], [4, 5], [7, 8]],
            index=["cobra", "viper", "sidewinder"],
            columns=["max_speed", "shield"],
        )
        kdf = ks.DataFrame(pdf)
        another_kdf = ks.DataFrame(pdf)

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            kdf.loc[["viper", "sidewinder"], ["shield"]] = another_kdf.max_speed

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            kdf.loc[another_kdf.max_speed < 5, ["shield"]] = -kdf.max_speed

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            kdf.loc[another_kdf.max_speed < 5, ["shield"]] = -another_kdf.max_speed

    def test_frame_iloc_setitem(self):
        pdf = pd.DataFrame(
            [[1, 2], [4, 5], [7, 8]],
            index=["cobra", "viper", "sidewinder"],
            columns=["max_speed", "shield"],
        )
        kdf = ks.DataFrame(pdf)
        another_kdf = ks.DataFrame(pdf)

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            kdf.iloc[[1, 2], [1]] = another_kdf.max_speed

    def test_series_loc_setitem(self):
        pser = pd.Series([1, 2, 3], index=["cobra", "viper", "sidewinder"])
        kser = ks.from_pandas(pser)

        pser_another = pd.Series([1, 2, 3], index=["cobra", "viper", "sidewinder"])
        kser_another = ks.from_pandas(pser_another)

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            kser.loc[kser % 2 == 1] = -kser_another

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            kser.loc[kser_another % 2 == 1] = -kser

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            kser.loc[kser_another % 2 == 1] = -kser_another

    def test_series_iloc_setitem(self):
        pser = pd.Series([1, 2, 3], index=["cobra", "viper", "sidewinder"])
        kser = ks.from_pandas(pser)

        pser_another = pd.Series([1, 2, 3], index=["cobra", "viper", "sidewinder"])
        kser_another = ks.from_pandas(pser_another)

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            kser.iloc[[1]] = -kser_another

    def test_where(self):
        pdf1 = pd.DataFrame({"A": [0, 1, 2, 3, 4], "B": [100, 200, 300, 400, 500]})
        pdf2 = pd.DataFrame({"A": [0, -1, -2, -3, -4], "B": [-100, -200, -300, -400, -500]})
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            self.assert_eq(repr(pdf1.where(pdf2 > 100)), repr(kdf1.where(kdf2 > 100).sort_index()))

        pdf1 = pd.DataFrame({"A": [-1, -2, -3, -4, -5], "B": [-100, -200, -300, -400, -500]})
        pdf2 = pd.DataFrame({"A": [-10, -20, -30, -40, -50], "B": [-5, -4, -3, -2, -1]})
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            self.assert_eq(
                repr(pdf1.where(pdf2 < -250)), repr(kdf1.where(kdf2 < -250).sort_index())
            )

    def test_mask(self):
        pdf1 = pd.DataFrame({"A": [0, 1, 2, 3, 4], "B": [100, 200, 300, 400, 500]})
        pdf2 = pd.DataFrame({"A": [0, -1, -2, -3, -4], "B": [-100, -200, -300, -400, -500]})
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            self.assert_eq(repr(pdf1.mask(pdf2 < 100)), repr(kdf1.mask(kdf2 < 100).sort_index()))

        pdf1 = pd.DataFrame({"A": [-1, -2, -3, -4, -5], "B": [-100, -200, -300, -400, -500]})
        pdf2 = pd.DataFrame({"A": [-10, -20, -30, -40, -50], "B": [-5, -4, -3, -2, -1]})
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            self.assert_eq(repr(pdf1.mask(pdf2 > -250)), repr(kdf1.mask(kdf2 > -250).sort_index()))
