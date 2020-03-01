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
import numpy as np

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
            {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9], "b": [4, 5, 6, 3, 2, 1, 0, 0, 0],},
            index=[0, 1, 3, 5, 6, 8, 9, 10, 11],
        )

    @property
    def pdf2(self):
        return pd.DataFrame(
            {"a": [9, 8, 7, 6, 5, 4, 3, 2, 1], "b": [0, 0, 0, 4, 5, 6, 1, 2, 3],},
            index=list(range(9)),
        )

    @property
    def pdf3(self):
        return pd.DataFrame(
            {"b": [1, 1, 1, 1, 1, 1, 1, 1, 1], "c": [1, 1, 1, 1, 1, 1, 1, 1, 1],},
            index=list(range(9)),
        )

    @property
    def pdf4(self):
        return pd.DataFrame(
            {"e": [2, 2, 2, 2, 2, 2, 2, 2, 2], "f": [2, 2, 2, 2, 2, 2, 2, 2, 2],},
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
        self.assert_eq(
            (kdf1.a - kdf2.b).sort_index(), (pdf1.a - pdf2.b).rename("a").sort_index(), almost=True
        )

        self.assert_eq(
            (kdf1.a * kdf2.a).sort_index(), (pdf1.a * pdf2.a).rename("a").sort_index(), almost=True
        )

        self.assert_eq(
            (kdf1["a"] / kdf2["a"]).sort_index(),
            (pdf1["a"] / pdf2["a"]).rename("a").sort_index(),
            almost=True,
        )

        # DataFrame
        self.assert_eq((kdf1 + kdf2).sort_index(), (pdf1 + pdf2).sort_index(), almost=True)

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
            almost=True,
        )

        self.assert_eq(
            (kdf1[("x", "a")] - kdf2["x"]["b"]).sort_index(),
            (pdf1[("x", "a")] - pdf2["x"]["b"]).rename(("x", "a")).sort_index(),
            almost=True,
        )

        self.assert_eq(
            (kdf1["x"]["a"] - kdf2[("x", "b")]).sort_index(),
            (pdf1["x"]["a"] - pdf2[("x", "b")]).rename("a").sort_index(),
            almost=True,
        )

        # DataFrame
        self.assert_eq((kdf1 + kdf2).sort_index(), (pdf1 + pdf2).sort_index(), almost=True)

        # MultiIndex Series
        self.assert_eq((kser1 + kser2).sort_index(), (pser1 + pser2).sort_index(), almost=True)

        self.assert_eq((kser1 - kser2).sort_index(), (pser1 - pser2).sort_index(), almost=True)

        self.assert_eq((kser1 * kser2).sort_index(), (pser1 * pser2).sort_index(), almost=True)

        self.assert_eq((kser1 / kser2).sort_index(), (pser1 / pser2).sort_index(), almost=True)

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
            almost=True,
        )

        self.assert_eq(
            (kdf1.a * (kdf2.a * kdf3.c)).sort_index(),
            (pdf1.a * (pdf2.a * pdf3.c)).rename("a").sort_index(),
            almost=True,
        )

        self.assert_eq(
            (kdf1["a"] / kdf2["a"] / kdf3["c"]).sort_index(),
            (pdf1["a"] / pdf2["a"] / pdf3["c"]).rename("a").sort_index(),
            almost=True,
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
            almost=True,
        )

        self.assert_eq(
            (kdf1[("x", "a")] * (kdf2[("x", "b")] * kdf3[("y", "c")])).sort_index(),
            (pdf1[("x", "a")] * (pdf2[("x", "b")] * pdf3[("y", "c")]))
            .rename(("x", "a"))
            .sort_index(),
            almost=True,
        )

        # DataFrame
        self.assert_eq(
            (kdf1 + kdf2 - kdf3).sort_index(), (pdf1 + pdf2 - pdf3).sort_index(), almost=True
        )

        # MultiIndex Series
        self.assert_eq(
            (kser1 + kser2 - kser3).sort_index(), (pser1 + pser2 - pser3).sort_index(), almost=True
        )

        self.assert_eq(
            (kser1 * kser2 * kser3).sort_index(), (pser1 * pser2 * pser3).sort_index(), almost=True
        )

        self.assert_eq(
            (kser1 - kser2 / kser3).sort_index(), (pser1 - pser2 / pser3).sort_index(), almost=True
        )

        self.assert_eq(
            (kser1 + kser2 * kser3).sort_index(), (pser1 + pser2 * pser3).sort_index(), almost=True
        )

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
        kdf["a"] = self.kdf2.a
        pdf["a"] = self.pdf2.a

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf["a"] = self.kdf2.b
        pdf["a"] = self.pdf2.b

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

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

    def test_assignment_frame(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf[["a", "b"]] = self.kdf1
        pdf[["a", "b"]] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        # 'c' does not exist in `kdf`.
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf[["b", "c"]] = self.kdf1
        pdf[["b", "c"]] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

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
        self.assert_eq(
            (kdf5.c - kdf6.e).sort_index(), (pdf5.c - pdf6.e).rename("c").sort_index(), almost=True
        )

        self.assert_eq(
            (kdf5["c"] / kdf6["e"]).sort_index(),
            (pdf5["c"] / pdf6["e"]).rename("c").sort_index(),
            almost=True,
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

    def test_loc_setitem(self):
        pdf = pd.DataFrame(
            [[1, 2], [4, 5], [7, 8]],
            index=["cobra", "viper", "sidewinder"],
            columns=["max_speed", "shield"],
        )
        kdf = ks.DataFrame(pdf)
        another_kdf = ks.DataFrame(pdf)

        kdf.loc[["viper", "sidewinder"], ["shield"]] = another_kdf.max_speed
        pdf.loc[["viper", "sidewinder"], ["shield"]] = pdf.max_speed

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

    def test_combine_first(self):
        # Series.combine_first
        kser1 = ks.Series({'falcon': 330.0, 'eagle': 160.0})
        kser2 = ks.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})
        pser1 = kser1.to_pandas()
        pser2 = kser2.to_pandas()

        self.assert_eq(repr(kser1.combine_first(kser2).sort_index()),
                       repr(pser1.combine_first(pser2).sort_index()))
        with self.assertRaisesRegex(ValueError,
                                    "`combine_first` only allows `Series` for parameter `other`"):
            kser1.combine_first(50)

        # MultiIndex
        midx1 = pd.MultiIndex([['lama', 'cow', 'falcon', 'koala'],
                               ['speed', 'weight', 'length', 'power']],
                              [[0, 3, 1, 1, 1, 2, 2, 2],
                               [0, 2, 0, 3, 2, 0, 1, 3]])
        midx2 = pd.MultiIndex([['lama', 'cow', 'falcon'],
                               ['speed', 'weight', 'length']],
                              [[0, 0, 0, 1, 1, 1, 2, 2, 2],
                               [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        kser1 = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1], index=midx1)
        kser2 = ks.Series([-45, 200, -1.2, 30, -250, 1.5, 320, 1, -0.3], index=midx2)
        pser1 = kser1.to_pandas()
        pser2 = kser2.to_pandas()

        self.assert_eq(repr(kser1.combine_first(kser2).sort_index()),
                       repr(pser1.combine_first(pser2).sort_index()))


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

    def test_loc_setitem(self):
        pdf = pd.DataFrame(
            [[1, 2], [4, 5], [7, 8]],
            index=["cobra", "viper", "sidewinder"],
            columns=["max_speed", "shield"],
        )
        kdf = ks.DataFrame(pdf)
        another_kdf = ks.DataFrame(pdf)

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            kdf.loc[["viper", "sidewinder"], ["shield"]] = another_kdf.max_speed

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
            self.assert_eq(repr(pdf1.mask(pdf2 > -250)),
                           repr(kdf1.mask(kdf2 > -250).sort_index()))

    def test_combine_first(self):
        # Series.combine_first
        kser1 = ks.Series({'falcon': 330.0, 'eagle': 160.0})
        kser2 = ks.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})
        pser1 = kser1.to_pandas()
        pser2 = kser2.to_pandas()

        with self.assertRaisesRegex(ValueError, "Cannot combine the series or dataframe"):
            self.assert_eq(repr(pser1.combine_first(pser2)),
                           repr(kser1.combine_first(kser2).sort_index()))
