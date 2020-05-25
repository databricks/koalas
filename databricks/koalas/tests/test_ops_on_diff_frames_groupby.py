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
import unittest

import pandas as pd

from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class OpsOnDiffFramesGroupByTest(ReusedSQLTestCase, SQLTestUtils):
    @classmethod
    def setUpClass(cls):
        super(OpsOnDiffFramesGroupByTest, cls).setUpClass()
        set_option("compute.ops_on_diff_frames", True)

    @classmethod
    def tearDownClass(cls):
        reset_option("compute.ops_on_diff_frames")
        super(OpsOnDiffFramesGroupByTest, cls).tearDownClass()

    def test_groupby_different_lengths(self):
        pdfs1 = [
            pd.DataFrame({"c": [4, 2, 7, 3, None, 1, 1, 1, 2], "d": list("abcdefght")}),
            pd.DataFrame({"c": [4, 2, 7, None, 1, 1, 2], "d": list("abcdefg")}),
            pd.DataFrame({"c": [4, 2, 7, 3, None, 1, 1, 1, 2, 2], "d": list("abcdefghti")}),
        ]
        pdfs2 = [
            pd.DataFrame({"a": [1, 2, 6, 4, 4, 6, 4, 3, 7], "b": [4, 2, 7, 3, 3, 1, 1, 1, 2]}),
            pd.DataFrame({"a": [1, 2, 6, 4, 4, 6, 4, 7], "b": [4, 2, 7, 3, 3, 1, 1, 2]}),
            pd.DataFrame({"a": [1, 2, 6, 4, 4, 6, 4, 3, 7], "b": [4, 2, 7, 3, 3, 1, 1, 1, 2]}),
        ]

        for pdf1, pdf2 in zip(pdfs1, pdfs2):
            kdf1 = ks.from_pandas(pdf1)
            kdf2 = ks.from_pandas(pdf2)

            for as_index in [True, False]:
                if as_index:
                    sort = lambda df: df.sort_index()
                else:
                    sort = lambda df: df.sort_values("c").reset_index(drop=True)
                self.assert_eq(
                    sort(kdf1.groupby(kdf2.a, as_index=as_index).sum()),
                    sort(pdf1.groupby(pdf2.a, as_index=as_index).sum()),
                )

                self.assert_eq(
                    sort(kdf1.groupby(kdf2.a, as_index=as_index).c.sum()),
                    sort(pdf1.groupby(pdf2.a, as_index=as_index).c.sum()),
                )
                self.assert_eq(
                    sort(kdf1.groupby(kdf2.a, as_index=as_index)["c"].sum()),
                    sort(pdf1.groupby(pdf2.a, as_index=as_index)["c"].sum()),
                )

    def test_groupby_multiindex_columns(self):
        pdf1 = pd.DataFrame(
            {("y", "c"): [4, 2, 7, 3, None, 1, 1, 1, 2], ("z", "d"): list("abcdefght"),}
        )
        pdf2 = pd.DataFrame(
            {("x", "a"): [1, 2, 6, 4, 4, 6, 4, 3, 7], ("x", "b"): [4, 2, 7, 3, 3, 1, 1, 1, 2],}
        )
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(
            kdf1.groupby(kdf2[("x", "a")]).sum().sort_index(),
            pdf1.groupby(pdf2[("x", "a")]).sum().sort_index(),
        )

        self.assert_eq(
            kdf1.groupby(kdf2[("x", "a")], as_index=False)
            .sum()
            .sort_values(("y", "c"))
            .reset_index(drop=True),
            pdf1.groupby(pdf2[("x", "a")], as_index=False)
            .sum()
            .sort_values(("y", "c"))
            .reset_index(drop=True),
        )
        self.assert_eq(
            kdf1.groupby(kdf2[("x", "a")])[[("y", "c")]].sum().sort_index(),
            pdf1.groupby(pdf2[("x", "a")])[[("y", "c")]].sum().sort_index(),
        )

    def test_split_apply_combine_on_series(self):
        pdf1 = pd.DataFrame({"C": [0.362, 0.227, 1.267, -0.562], "B": [1, 2, 3, 4]})
        pdf2 = pd.DataFrame({"A": [1, 1, 2, 2]})
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        for as_index in [True, False]:
            if as_index:
                sort = lambda df: df.sort_index()
            else:
                sort = lambda df: df.sort_values(list(df.columns)).reset_index(drop=True)

            with self.subTest(as_index=as_index):
                self.assert_eq(
                    sort(kdf1.groupby(kdf2.A, as_index=as_index).sum()),
                    sort(pdf1.groupby(pdf2.A, as_index=as_index).sum()),
                )
                self.assert_eq(
                    sort(kdf1.groupby(kdf2.A, as_index=as_index).B.sum()),
                    sort(pdf1.groupby(pdf2.A, as_index=as_index).B.sum()),
                )

        self.assert_eq(
            kdf1.B.groupby(kdf2.A).sum().sort_index(), pdf1.B.groupby(pdf2.A).sum().sort_index(),
        )
        self.assert_eq(
            (kdf1.B + 1).groupby(kdf2.A).sum().sort_index(),
            (pdf1.B + 1).groupby(pdf2.A).sum().sort_index(),
        )

    def test_aggregate(self):
        pdf1 = pd.DataFrame({"C": [0.362, 0.227, 1.267, -0.562], "B": [1, 2, 3, 4]})
        pdf2 = pd.DataFrame({"A": [1, 1, 2, 2]})
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        for as_index in [True, False]:
            if as_index:
                sort = lambda df: df.sort_index()
            else:
                sort = lambda df: df.sort_values(list(df.columns)).reset_index(drop=True)

            with self.subTest(as_index=as_index):
                self.assert_eq(
                    sort(kdf1.groupby(kdf2.A, as_index=as_index).agg("sum")),
                    sort(pdf1.groupby(pdf2.A, as_index=as_index).agg("sum")),
                )
                self.assert_eq(
                    sort(kdf1.groupby(kdf2.A, as_index=as_index).agg({"B": "min", "C": "sum"})),
                    sort(pdf1.groupby(pdf2.A, as_index=as_index).agg({"B": "min", "C": "sum"})),
                )
                self.assert_eq(
                    sort(
                        kdf1.groupby(kdf2.A, as_index=as_index).agg(
                            {"B": ["min", "max"], "C": "sum"}
                        )
                    ),
                    sort(
                        pdf1.groupby(pdf2.A, as_index=as_index).agg(
                            {"B": ["min", "max"], "C": "sum"}
                        )
                    ),
                )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("Y", "C"), ("X", "B")])
        pdf1.columns = columns
        kdf1.columns = columns

        columns = pd.MultiIndex.from_tuples([("X", "A")])
        pdf2.columns = columns
        kdf2.columns = columns

        for as_index in [True, False]:
            stats_kdf = kdf1.groupby(kdf2[("X", "A")], as_index=as_index).agg(
                {("X", "B"): "min", ("Y", "C"): "sum"}
            )
            stats_pdf = pdf1.groupby(pdf2[("X", "A")], as_index=as_index).agg(
                {("X", "B"): "min", ("Y", "C"): "sum"}
            )
            self.assert_eq(
                stats_kdf.sort_values(by=[("X", "B"), ("Y", "C")]).reset_index(drop=True),
                stats_pdf.sort_values(by=[("X", "B"), ("Y", "C")]).reset_index(drop=True),
            )

        stats_kdf = kdf1.groupby(kdf2[("X", "A")]).agg(
            {("X", "B"): ["min", "max"], ("Y", "C"): "sum"}
        )
        stats_pdf = pdf1.groupby(pdf2[("X", "A")]).agg(
            {("X", "B"): ["min", "max"], ("Y", "C"): "sum"}
        )
        self.assert_eq(
            stats_kdf.sort_values(
                by=[("X", "B", "min"), ("X", "B", "max"), ("Y", "C", "sum")]
            ).reset_index(drop=True),
            stats_pdf.sort_values(
                by=[("X", "B", "min"), ("X", "B", "max"), ("Y", "C", "sum")]
            ).reset_index(drop=True),
        )

    def test_duplicated_labels(self):
        pdf1 = pd.DataFrame({"A": [3, 2, 1]})
        pdf2 = pd.DataFrame({"A": [1, 2, 3]})
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(
            kdf1.groupby(kdf2.A).sum().sort_index(), pdf1.groupby(pdf2.A).sum().sort_index()
        )
        self.assert_eq(
            kdf1.groupby(kdf2.A, as_index=False).sum().sort_values("A").reset_index(drop=True),
            pdf1.groupby(pdf2.A, as_index=False).sum().sort_values("A").reset_index(drop=True),
        )

    def test_apply(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": [1, 1, 2, 3, 5, 8], "c": [1, 4, 9, 16, 25, 36]},
            columns=["a", "b", "c"],
        )
        pkey = pd.Series([1, 1, 2, 3, 5, 8])
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            kdf.groupby(kkey).apply(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pkey).apply(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kkey)["a"].apply(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pkey)["a"].apply(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kkey)[["a"]].apply(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pkey)[["a"]].apply(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["a", kkey]).apply(lambda x: x + x.min()).sort_index(),
            pdf.groupby(["a", pkey]).apply(lambda x: x + x.min()).sort_index(),
        )

    def test_transform(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": [1, 1, 2, 3, 5, 8], "c": [1, 4, 9, 16, 25, 36]},
            columns=["a", "b", "c"],
        )
        pkey = pd.Series([1, 1, 2, 3, 5, 8])
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            kdf.groupby(kkey).transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pkey).transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kkey)["a"].transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pkey)["a"].transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kkey)[["a"]].transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pkey)[["a"]].transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["a", kkey]).transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby(["a", pkey]).transform(lambda x: x + x.min()).sort_index(),
        )

    def test_filter(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": [1, 1, 2, 3, 5, 8], "c": [1, 4, 9, 16, 25, 36]},
            columns=["a", "b", "c"],
        )
        pkey = pd.Series([1, 1, 2, 3, 5, 8])
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            kdf.groupby(kkey).filter(lambda x: any(x.a == 2)).sort_index(),
            pdf.groupby(pkey).filter(lambda x: any(x.a == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kkey)["a"].filter(lambda x: any(x == 2)).sort_index(),
            pdf.groupby(pkey)["a"].filter(lambda x: any(x == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kkey)[["a"]].filter(lambda x: any(x.a == 2)).sort_index(),
            pdf.groupby(pkey)[["a"]].filter(lambda x: any(x.a == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["a", kkey]).filter(lambda x: any(x.a == 2)).sort_index(),
            pdf.groupby(["a", pkey]).filter(lambda x: any(x.a == 2)).sort_index(),
        )

    def test_head(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3] * 3,
                "b": [2, 3, 1, 4, 6, 9, 8, 10, 7, 5] * 3,
                "c": [3, 5, 2, 5, 1, 2, 6, 4, 3, 6] * 3,
            },
        )
        pkey = pd.Series([1, 1, 1, 1, 2, 2, 2, 3, 3, 3] * 3)
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            pdf.groupby(pkey).head(2).sort_index(), kdf.groupby(kkey).head(2).sort_index()
        )
        self.assert_eq(
            pdf.groupby("a")["b"].head(2).sort_index(), kdf.groupby("a")["b"].head(2).sort_index()
        )
        self.assert_eq(
            pdf.groupby("a")[["b"]].head(2).sort_index(),
            kdf.groupby("a")[["b"]].head(2).sort_index(),
        )
        self.assert_eq(
            pdf.groupby([pkey, "b"]).head(2).sort_index(),
            kdf.groupby([kkey, "b"]).head(2).sort_index(),
        )

    def test_cummin(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
        )
        pkey = pd.Series([1, 1, 2, 3, 5, 8] * 3)
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            kdf.groupby(kkey).cummin().sort_index(),
            pdf.groupby(pkey).cummin().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)["a"].cummin().sort_index(),
            pdf.groupby(pkey)["a"].cummin().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)[["a"]].cummin().sort_index(),
            pdf.groupby(pkey)[["a"]].cummin().sort_index(),
            almost=True,
        )

    def test_cummax(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
        )
        pkey = pd.Series([1, 1, 2, 3, 5, 8] * 3)
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            kdf.groupby(kkey).cummax().sort_index(),
            pdf.groupby(pkey).cummax().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)["a"].cummax().sort_index(),
            pdf.groupby(pkey)["a"].cummax().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)[["a"]].cummax().sort_index(),
            pdf.groupby(pkey)[["a"]].cummax().sort_index(),
            almost=True,
        )

    def test_cumsum(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
        )
        pkey = pd.Series([1, 1, 2, 3, 5, 8] * 3)
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            kdf.groupby(kkey).cumsum().sort_index(),
            pdf.groupby(pkey).cumsum().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)["a"].cumsum().sort_index(),
            pdf.groupby(pkey)["a"].cumsum().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)[["a"]].cumsum().sort_index(),
            pdf.groupby(pkey)[["a"]].cumsum().sort_index(),
            almost=True,
        )

    def test_cumprod(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
        )
        pkey = pd.Series([1, 1, 2, 3, 5, 8] * 3)
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            kdf.groupby(kkey).cumprod().sort_index(),
            pdf.groupby(pkey).cumprod().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)["a"].cumprod().sort_index(),
            pdf.groupby(pkey)["a"].cumprod().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)[["a"]].cumprod().sort_index(),
            pdf.groupby(pkey)[["a"]].cumprod().sort_index(),
            almost=True,
        )

    def test_diff(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            }
        )
        pkey = pd.Series([1, 1, 2, 3, 5, 8] * 3)
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            kdf.groupby(kkey).diff().sort_index(),
            pdf.groupby(pkey).diff().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)["a"].diff().sort_index(),
            pdf.groupby(pkey)["a"].diff().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)[["a"]].diff().sort_index(),
            pdf.groupby(pkey)[["a"]].diff().sort_index(),
            almost=True,
        )

    def test_rank(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
        )
        pkey = pd.Series([1, 1, 2, 3, 5, 8] * 3)
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            kdf.groupby(kkey).rank().sort_index(),
            pdf.groupby(pkey).rank().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)["a"].rank().sort_index(),
            pdf.groupby(pkey)["a"].rank().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)[["a"]].rank().sort_index(),
            pdf.groupby(pkey)[["a"]].rank().sort_index(),
            almost=True,
        )

    @unittest.skipIf(pd.__version__ < "0.24.0", "not supported before pandas 0.24.0")
    def test_shift(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 1, 2, 2, 3, 3] * 3,
                "b": [1, 1, 2, 2, 3, 4] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
        )
        pkey = pd.Series([1, 1, 2, 2, 3, 4] * 3)
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            kdf.groupby(kkey).shift().sort_index(),
            pdf.groupby(pkey).shift().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)["a"].shift().sort_index(),
            pdf.groupby(pkey)["a"].shift().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)[["a"]].shift().sort_index(),
            pdf.groupby(pkey)[["a"]].shift().sort_index(),
            almost=True,
        )

    def test_fillna(self):
        pdf = pd.DataFrame(
            {
                "A": [1, 1, 2, 2] * 3,
                "B": [2, 4, None, 3] * 3,
                "C": [None, None, None, 1] * 3,
                "D": [0, 1, 5, 4] * 3,
            }
        )
        pkey = pd.Series([1, 1, 2, 2] * 3)
        kdf = ks.from_pandas(pdf)
        kkey = ks.from_pandas(pkey)

        self.assert_eq(
            kdf.groupby(kkey).fillna(0).sort_index(),
            pdf.groupby(pkey).fillna(0).sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)["C"].fillna(0).sort_index(),
            pdf.groupby(pkey)["C"].fillna(0).sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)[["C"]].fillna(0).sort_index(),
            pdf.groupby(pkey)[["C"]].fillna(0).sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey).fillna(method="bfill").sort_index(),
            pdf.groupby(pkey).fillna(method="bfill").sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)["C"].fillna(method="bfill").sort_index(),
            pdf.groupby(pkey)["C"].fillna(method="bfill").sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)[["C"]].fillna(method="bfill").sort_index(),
            pdf.groupby(pkey)[["C"]].fillna(method="bfill").sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey).fillna(method="ffill").sort_index(),
            pdf.groupby(pkey).fillna(method="ffill").sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)["C"].fillna(method="ffill").sort_index(),
            pdf.groupby(pkey)["C"].fillna(method="ffill").sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kkey)[["C"]].fillna(method="ffill").sort_index(),
            pdf.groupby(pkey)[["C"]].fillna(method="ffill").sort_index(),
            almost=True,
        )
