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
import inspect
from distutils.version import LooseVersion
from itertools import product

import numpy as np
import pandas as pd

from databricks import koalas as ks
from databricks.koalas.config import option_context
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.missing.groupby import (
    MissingPandasLikeDataFrameGroupBy,
    MissingPandasLikeSeriesGroupBy,
)
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils
from databricks.koalas.groupby import is_multi_agg_with_relabel


class GroupByTest(ReusedSQLTestCase, TestUtils):
    def test_groupby_simple(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 6, 4, 4, 6, 4, 3, 7],
                "b": [4, 2, 7, 3, 3, 1, 1, 1, 2],
                "c": [4, 2, 7, 3, None, 1, 1, 1, 2],
                "d": list("abcdefght"),
            },
            index=[0, 1, 3, 5, 6, 8, 9, 9, 9],
        )
        kdf = ks.from_pandas(pdf)

        for as_index in [True, False]:
            if as_index:
                sort = lambda df: df.sort_index()
            else:
                sort = lambda df: df.sort_values("a").reset_index(drop=True).sort_index()
            self.assert_eq(
                sort(kdf.groupby("a", as_index=as_index).sum()),
                sort(pdf.groupby("a", as_index=as_index).sum()),
            )
            self.assert_eq(
                sort(kdf.groupby("a", as_index=as_index).b.sum()),
                sort(pdf.groupby("a", as_index=as_index).b.sum()),
            )
            self.assert_eq(
                sort(kdf.groupby("a", as_index=as_index)["b"].sum()),
                sort(pdf.groupby("a", as_index=as_index)["b"].sum()),
            )
            self.assert_eq(
                sort(kdf.groupby("a", as_index=as_index)[["b", "c"]].sum()),
                sort(pdf.groupby("a", as_index=as_index)[["b", "c"]].sum()),
            )
            self.assert_eq(
                sort(kdf.groupby("a", as_index=as_index)[[]].sum()),
                sort(pdf.groupby("a", as_index=as_index)[[]].sum()),
            )
            self.assert_eq(
                sort(kdf.groupby("a", as_index=as_index)["c"].sum()),
                sort(pdf.groupby("a", as_index=as_index)["c"].sum()),
            )

        self.assert_eq(kdf.groupby("a").a.sum().sort_index(), pdf.groupby("a").a.sum().sort_index())
        self.assert_eq(
            kdf.groupby("a")["a"].sum().sort_index(), pdf.groupby("a")["a"].sum().sort_index()
        )
        self.assert_eq(
            kdf.groupby("a")[["a"]].sum().sort_index(), pdf.groupby("a")[["a"]].sum().sort_index()
        )
        self.assert_eq(
            kdf.groupby("a")[["a", "c"]].sum().sort_index(),
            pdf.groupby("a")[["a", "c"]].sum().sort_index(),
        )

        self.assert_eq(
            kdf.a.groupby(kdf.b).sum().sort_index(), pdf.a.groupby(pdf.b).sum().sort_index()
        )

        for axis in [0, "index"]:
            self.assert_eq(
                kdf.groupby("a", axis=axis).a.sum().sort_index(),
                pdf.groupby("a", axis=axis).a.sum().sort_index(),
            )
            self.assert_eq(
                kdf.groupby("a", axis=axis)["a"].sum().sort_index(),
                pdf.groupby("a", axis=axis)["a"].sum().sort_index(),
            )
            self.assert_eq(
                kdf.groupby("a", axis=axis)[["a"]].sum().sort_index(),
                pdf.groupby("a", axis=axis)[["a"]].sum().sort_index(),
            )
            self.assert_eq(
                kdf.groupby("a", axis=axis)[["a", "c"]].sum().sort_index(),
                pdf.groupby("a", axis=axis)[["a", "c"]].sum().sort_index(),
            )

            self.assert_eq(
                kdf.a.groupby(kdf.b, axis=axis).sum().sort_index(),
                pdf.a.groupby(pdf.b, axis=axis).sum().sort_index(),
            )

        self.assertRaises(ValueError, lambda: kdf.groupby("a", as_index=False).a)
        self.assertRaises(ValueError, lambda: kdf.groupby("a", as_index=False)["a"])
        self.assertRaises(ValueError, lambda: kdf.groupby("a", as_index=False)[["a"]])
        self.assertRaises(ValueError, lambda: kdf.groupby("a", as_index=False)[["a", "c"]])
        self.assertRaises(ValueError, lambda: kdf.groupby(0, as_index=False)[["a", "c"]])
        self.assertRaises(KeyError, lambda: kdf.groupby([0], as_index=False)[["a", "c"]])

        self.assertRaises(TypeError, lambda: kdf.a.groupby(kdf.b, as_index=False))

        self.assertRaises(NotImplementedError, lambda: kdf.groupby("a", axis=1))
        self.assertRaises(NotImplementedError, lambda: kdf.groupby("a", axis="columns"))
        self.assertRaises(ValueError, lambda: kdf.groupby("a", "b"))
        self.assertRaises(TypeError, lambda: kdf.a.groupby(kdf.a, kdf.b))

        # we can't use column name/names as a parameter `by` for `SeriesGroupBy`.
        self.assertRaises(KeyError, lambda: kdf.a.groupby(by="a"))
        self.assertRaises(KeyError, lambda: kdf.a.groupby(by=["a", "b"]))
        self.assertRaises(KeyError, lambda: kdf.a.groupby(by=("a", "b")))

        # we can't use DataFrame as a parameter `by` for `DataFrameGroupBy`/`SeriesGroupBy`.
        self.assertRaises(ValueError, lambda: kdf.groupby(kdf))
        self.assertRaises(ValueError, lambda: kdf.a.groupby(kdf))
        self.assertRaises(ValueError, lambda: kdf.a.groupby((kdf,)))

    def test_groupby_multiindex_columns(self):
        pdf = pd.DataFrame(
            {
                ("x", "a"): [1, 2, 6, 4, 4, 6, 4, 3, 7],
                ("x", "b"): [4, 2, 7, 3, 3, 1, 1, 1, 2],
                ("y", "c"): [4, 2, 7, 3, None, 1, 1, 1, 2],
                ("z", "d"): list("abcdefght"),
            },
            index=[0, 1, 3, 5, 6, 8, 9, 9, 9],
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby(("x", "a")).sum().sort_index(), pdf.groupby(("x", "a")).sum().sort_index()
        )
        self.assert_eq(
            kdf.groupby(("x", "a"), as_index=False)
            .sum()
            .sort_values(("x", "a"))
            .reset_index(drop=True)
            .sort_index(),
            pdf.groupby(("x", "a"), as_index=False)
            .sum()
            .sort_values(("x", "a"))
            .reset_index(drop=True)
            .sort_index(),
        )
        self.assert_eq(
            kdf.groupby(("x", "a"))[[("y", "c")]].sum().sort_index(),
            pdf.groupby(("x", "a"))[[("y", "c")]].sum().sort_index(),
        )
        # TODO: seems like a pandas' bug. it works well in Koalas like the below.
        # >>> pdf[('x', 'a')].groupby(pdf[('x', 'b')]).sum().sort_index()
        # Traceback (most recent call last):
        # ...
        # ValueError: Can only tuple-index with a MultiIndex
        # >>> kdf[('x', 'a')].groupby(kdf[('x', 'b')]).sum().sort_index()
        # (x, b)
        # 1    13
        # 2     9
        # 3     8
        # 4     1
        # 7     6
        # Name: (x, a), dtype: int64
        expected_result = ks.Series(
            [13, 9, 8, 1, 6], name=("x", "a"), index=pd.Index([1, 2, 3, 4, 7], name=("x", "b"))
        )
        self.assert_eq(kdf[("x", "a")].groupby(kdf[("x", "b")]).sum().sort_index(), expected_result)

    def test_split_apply_combine_on_series(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 6, 4, 4, 6, 4, 3, 7],
                "b": [4, 2, 7, 3, 3, 1, 1, 1, 2],
                "c": [4, 2, 7, 3, None, 1, 1, 1, 2],
                "d": list("abcdefght"),
            },
            index=[0, 1, 3, 5, 6, 8, 9, 9, 9],
        )
        kdf = ks.from_pandas(pdf)

        funcs = [
            ((True, False), ["sum", "min", "max", "count", "first", "last"]),
            ((True, True), ["mean"]),
            ((False, False), ["var", "std"]),
        ]
        funcs = [(check_exact, almost, f) for (check_exact, almost), fs in funcs for f in fs]

        for as_index in [True, False]:
            if as_index:
                sort = lambda df: df.sort_index()
            else:
                sort = (
                    lambda df: df.sort_values(list(df.columns)).reset_index(drop=True).sort_index()
                )

            for check_exact, almost, func in funcs:
                for kkey, pkey in [("b", "b"), (kdf.b, pdf.b)]:
                    with self.subTest(as_index=as_index, func=func, key=pkey):
                        if as_index is True or func != "std":
                            self.assert_eq(
                                sort(getattr(kdf.groupby(kkey, as_index=as_index).a, func)()),
                                sort(getattr(pdf.groupby(pkey, as_index=as_index).a, func)()),
                                check_exact=check_exact,
                                almost=almost,
                            )
                            self.assert_eq(
                                sort(getattr(kdf.groupby(kkey, as_index=as_index), func)()),
                                sort(getattr(pdf.groupby(pkey, as_index=as_index), func)()),
                                check_exact=check_exact,
                                almost=almost,
                            )
                        else:
                            # seems like a pandas' bug for as_index=False and func == "std"?
                            self.assert_eq(
                                sort(getattr(kdf.groupby(kkey, as_index=as_index).a, func)()),
                                sort(pdf.groupby(pkey, as_index=True).a.std().reset_index()),
                                check_exact=check_exact,
                                almost=almost,
                            )
                            self.assert_eq(
                                sort(getattr(kdf.groupby(kkey, as_index=as_index), func)()),
                                sort(pdf.groupby(pkey, as_index=True).std().reset_index()),
                                check_exact=check_exact,
                                almost=almost,
                            )

                for kkey, pkey in [(kdf.b + 1, pdf.b + 1), (kdf.copy().b, pdf.copy().b)]:
                    with self.subTest(as_index=as_index, func=func, key=pkey):
                        self.assert_eq(
                            sort(getattr(kdf.groupby(kkey, as_index=as_index).a, func)()),
                            sort(getattr(pdf.groupby(pkey, as_index=as_index).a, func)()),
                            check_exact=check_exact,
                            almost=almost,
                        )
                        self.assert_eq(
                            sort(getattr(kdf.groupby(kkey, as_index=as_index), func)()),
                            sort(getattr(pdf.groupby(pkey, as_index=as_index), func)()),
                            check_exact=check_exact,
                            almost=almost,
                        )

            for check_exact, almost, func in funcs:
                for i in [0, 4, 7]:
                    with self.subTest(as_index=as_index, func=func, i=i):
                        self.assert_eq(
                            sort(getattr(kdf.groupby(kdf.b > i, as_index=as_index).a, func)()),
                            sort(getattr(pdf.groupby(pdf.b > i, as_index=as_index).a, func)()),
                            check_exact=check_exact,
                            almost=almost,
                        )
                        self.assert_eq(
                            sort(getattr(kdf.groupby(kdf.b > i, as_index=as_index), func)()),
                            sort(getattr(pdf.groupby(pdf.b > i, as_index=as_index), func)()),
                            check_exact=check_exact,
                            almost=almost,
                        )

        for check_exact, almost, func in funcs:
            for kkey, pkey in [
                (kdf.b, pdf.b),
                (kdf.b + 1, pdf.b + 1),
                (kdf.copy().b, pdf.copy().b),
                (kdf.b.rename(), pdf.b.rename()),
            ]:
                with self.subTest(func=func, key=pkey):
                    self.assert_eq(
                        getattr(kdf.a.groupby(kkey), func)().sort_index(),
                        getattr(pdf.a.groupby(pkey), func)().sort_index(),
                        check_exact=check_exact,
                        almost=almost,
                    )
                    self.assert_eq(
                        getattr((kdf.a + 1).groupby(kkey), func)().sort_index(),
                        getattr((pdf.a + 1).groupby(pkey), func)().sort_index(),
                        check_exact=check_exact,
                        almost=almost,
                    )
                    self.assert_eq(
                        getattr((kdf.b + 1).groupby(kkey), func)().sort_index(),
                        getattr((pdf.b + 1).groupby(pkey), func)().sort_index(),
                        check_exact=check_exact,
                        almost=almost,
                    )
                    self.assert_eq(
                        getattr(kdf.a.rename().groupby(kkey), func)().sort_index(),
                        getattr(pdf.a.rename().groupby(pkey), func)().sort_index(),
                        check_exact=check_exact,
                        almost=almost,
                    )

    def test_aggregate(self):
        pdf = pd.DataFrame(
            {"A": [1, 1, 2, 2], "B": [1, 2, 3, 4], "C": [0.362, 0.227, 1.267, -0.562]}
        )
        kdf = ks.from_pandas(pdf)

        for as_index in [True, False]:
            if as_index:
                sort = lambda df: df.sort_index()
            else:
                sort = lambda df: df.sort_values(list(df.columns)).reset_index(drop=True)

            for kkey, pkey in [("A", "A"), (kdf.A, pdf.A)]:
                with self.subTest(as_index=as_index, key=pkey):
                    self.assert_eq(
                        sort(kdf.groupby(kkey, as_index=as_index).agg("sum")),
                        sort(pdf.groupby(pkey, as_index=as_index).agg("sum")),
                    )
                    self.assert_eq(
                        sort(kdf.groupby(kkey, as_index=as_index).agg({"B": "min", "C": "sum"})),
                        sort(pdf.groupby(pkey, as_index=as_index).agg({"B": "min", "C": "sum"})),
                    )
                    self.assert_eq(
                        sort(
                            kdf.groupby(kkey, as_index=as_index).agg(
                                {"B": ["min", "max"], "C": "sum"}
                            )
                        ),
                        sort(
                            pdf.groupby(pkey, as_index=as_index).agg(
                                {"B": ["min", "max"], "C": "sum"}
                            )
                        ),
                    )

                    if as_index:
                        self.assert_eq(
                            sort(kdf.groupby(kkey, as_index=as_index).agg(["sum"])),
                            sort(pdf.groupby(pkey, as_index=as_index).agg(["sum"])),
                        )
                    else:
                        # seems like a pandas' bug for as_index=False and func_or_funcs is list?
                        self.assert_eq(
                            sort(kdf.groupby(kkey, as_index=as_index).agg(["sum"])),
                            sort(pdf.groupby(pkey, as_index=True).agg(["sum"]).reset_index()),
                        )

            for kkey, pkey in [(kdf.A + 1, pdf.A + 1), (kdf.copy().A, pdf.copy().A)]:
                with self.subTest(as_index=as_index, key=pkey):
                    self.assert_eq(
                        sort(kdf.groupby(kkey, as_index=as_index).agg("sum")),
                        sort(pdf.groupby(pkey, as_index=as_index).agg("sum")),
                    )
                    self.assert_eq(
                        sort(kdf.groupby(kkey, as_index=as_index).agg({"B": "min", "C": "sum"})),
                        sort(pdf.groupby(pkey, as_index=as_index).agg({"B": "min", "C": "sum"})),
                    )
                    self.assert_eq(
                        sort(
                            kdf.groupby(kkey, as_index=as_index).agg(
                                {"B": ["min", "max"], "C": "sum"}
                            )
                        ),
                        sort(
                            pdf.groupby(pkey, as_index=as_index).agg(
                                {"B": ["min", "max"], "C": "sum"}
                            )
                        ),
                    )
                    self.assert_eq(
                        sort(kdf.groupby(kkey, as_index=as_index).agg(["sum"])),
                        sort(pdf.groupby(pkey, as_index=as_index).agg(["sum"])),
                    )

        expected_error_message = (
            r"aggs must be a dict mapping from column name \(string or "
            r"tuple\) to aggregate functions \(string or list of strings\)."
        )
        with self.assertRaisesRegex(ValueError, expected_error_message):
            kdf.groupby("A", as_index=as_index).agg(0)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B"), ("Y", "C")])
        pdf.columns = columns
        kdf.columns = columns

        for as_index in [True, False]:
            stats_kdf = kdf.groupby(("X", "A"), as_index=as_index).agg(
                {("X", "B"): "min", ("Y", "C"): "sum"}
            )
            stats_pdf = pdf.groupby(("X", "A"), as_index=as_index).agg(
                {("X", "B"): "min", ("Y", "C"): "sum"}
            )
            self.assert_eq(
                stats_kdf.sort_values(by=[("X", "B"), ("Y", "C")]).reset_index(drop=True),
                stats_pdf.sort_values(by=[("X", "B"), ("Y", "C")]).reset_index(drop=True),
            )

        stats_kdf = kdf.groupby(("X", "A")).agg({("X", "B"): ["min", "max"], ("Y", "C"): "sum"})
        stats_pdf = pdf.groupby(("X", "A")).agg({("X", "B"): ["min", "max"], ("Y", "C"): "sum"})
        self.assert_eq(
            stats_kdf.sort_values(
                by=[("X", "B", "min"), ("X", "B", "max"), ("Y", "C", "sum")]
            ).reset_index(drop=True),
            stats_pdf.sort_values(
                by=[("X", "B", "min"), ("X", "B", "max"), ("Y", "C", "sum")]
            ).reset_index(drop=True),
        )

    def test_aggregate_func_str_list(self):
        # this is test for cases where only string or list is assigned
        pdf = pd.DataFrame(
            {
                "kind": ["cat", "dog", "cat", "dog"],
                "height": [9.1, 6.0, 9.5, 34.0],
                "weight": [7.9, 7.5, 9.9, 198.0],
            }
        )
        kdf = ks.from_pandas(pdf)

        agg_funcs = ["max", "min", ["min", "max"]]
        for aggfunc in agg_funcs:

            # Since in Koalas groupby, the order of rows might be different
            # so sort on index to ensure they have same output
            sorted_agg_kdf = kdf.groupby("kind").agg(aggfunc).sort_index()
            sorted_agg_pdf = pdf.groupby("kind").agg(aggfunc).sort_index()
            self.assert_eq(sorted_agg_kdf, sorted_agg_pdf)

        # test on multi index column case
        pdf = pd.DataFrame(
            {"A": [1, 1, 2, 2], "B": [1, 2, 3, 4], "C": [0.362, 0.227, 1.267, -0.562]}
        )
        kdf = ks.from_pandas(pdf)

        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B"), ("Y", "C")])
        pdf.columns = columns
        kdf.columns = columns

        for aggfunc in agg_funcs:
            sorted_agg_kdf = kdf.groupby(("X", "A")).agg(aggfunc).sort_index()
            sorted_agg_pdf = pdf.groupby(("X", "A")).agg(aggfunc).sort_index()
            self.assert_eq(sorted_agg_kdf, sorted_agg_pdf)

    @unittest.skipIf(pd.__version__ < "0.25.0", "not supported before pandas 0.25.0")
    def test_aggregate_relabel(self):
        # this is to test named aggregation in groupby
        pdf = pd.DataFrame({"group": ["a", "a", "b", "b"], "A": [0, 1, 2, 3], "B": [5, 6, 7, 8]})
        kdf = ks.from_pandas(pdf)

        # different agg column, same function
        agg_pdf = pdf.groupby("group").agg(a_max=("A", "max"), b_max=("B", "max")).sort_index()
        agg_kdf = kdf.groupby("group").agg(a_max=("A", "max"), b_max=("B", "max")).sort_index()
        self.assert_eq(agg_pdf, agg_kdf)

        # same agg column, different functions
        agg_pdf = pdf.groupby("group").agg(b_max=("B", "max"), b_min=("B", "min")).sort_index()
        agg_kdf = kdf.groupby("group").agg(b_max=("B", "max"), b_min=("B", "min")).sort_index()
        self.assert_eq(agg_pdf, agg_kdf)

        # test on NamedAgg
        agg_pdf = (
            pdf.groupby("group").agg(b_max=pd.NamedAgg(column="B", aggfunc="max")).sort_index()
        )
        agg_kdf = (
            kdf.groupby("group").agg(b_max=ks.NamedAgg(column="B", aggfunc="max")).sort_index()
        )
        self.assert_eq(agg_kdf, agg_pdf)

        # test on NamedAgg multi columns aggregation
        agg_pdf = (
            pdf.groupby("group")
            .agg(
                b_max=pd.NamedAgg(column="B", aggfunc="max"),
                b_min=pd.NamedAgg(column="B", aggfunc="min"),
            )
            .sort_index()
        )
        agg_kdf = (
            kdf.groupby("group")
            .agg(
                b_max=ks.NamedAgg(column="B", aggfunc="max"),
                b_min=ks.NamedAgg(column="B", aggfunc="min"),
            )
            .sort_index()
        )
        self.assert_eq(agg_kdf, agg_pdf)

    def test_describe(self):
        # support for numeric type, not support for string type yet
        datas = []
        datas.append({"a": [1, 1, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        datas.append({"a": [-1, -1, -3], "b": [-4, -5, -6], "c": [-7, -8, -9]})
        datas.append({"a": [0, 0, 0], "b": [0, 0, 0], "c": [0, 8, 0]})
        # it is okay if string type column as a group key
        datas.append({"a": ["a", "a", "c"], "b": [4, 5, 6], "c": [7, 8, 9]})

        for data in datas:
            pdf = pd.DataFrame(data)
            kdf = ks.from_pandas(pdf)

            describe_pdf = pdf.groupby("a").describe().sort_index()
            describe_kdf = kdf.groupby("a").describe().sort_index()

            # since the result of percentile columns are slightly difference from pandas,
            # we should check them separately: non-percentile columns & percentile columns

            # 1. Check that non-percentile columns are equal.
            agg_cols = [col.name for col in kdf.groupby("a")._agg_columns]
            formatted_percentiles = ["25%", "50%", "75%"]
            self.assert_eq(
                describe_kdf.drop(list(product(agg_cols, formatted_percentiles))),
                describe_pdf.drop(columns=formatted_percentiles, level=1),
                check_exact=False,
            )

            # 2. Check that percentile columns are equal.
            percentiles = [0.25, 0.5, 0.75]
            # The interpolation argument is yet to be implemented in Koalas.
            quantile_pdf = pdf.groupby("a").quantile(percentiles, interpolation="nearest")
            quantile_pdf = quantile_pdf.unstack(level=1).astype(float)
            non_percentile_stats = ["count", "mean", "std", "min", "max"]
            self.assert_eq(
                describe_kdf.drop(list(product(agg_cols, non_percentile_stats))),
                quantile_pdf.rename(columns="{:.0%}".format, level=1),
            )

        # not support for string type yet
        datas = []
        datas.append({"a": ["a", "a", "c"], "b": ["d", "e", "f"], "c": ["g", "h", "i"]})
        datas.append({"a": ["a", "a", "c"], "b": [4, 0, 1], "c": ["g", "h", "i"]})
        for data in datas:
            pdf = pd.DataFrame(data)
            kdf = ks.from_pandas(pdf)

            describe_pdf = pdf.groupby("a").describe().sort_index()
            self.assertRaises(NotImplementedError, lambda: kdf.groupby("a").describe().sort_index())

    def test_aggregate_relabel_multiindex(self):
        pdf = pd.DataFrame({"A": [0, 1, 2, 3], "B": [5, 6, 7, 8], "group": ["a", "a", "b", "b"]})
        pdf.columns = pd.MultiIndex.from_tuples([("y", "A"), ("y", "B"), ("x", "group")])
        kdf = ks.from_pandas(pdf)

        if LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            agg_pdf = pd.DataFrame(
                {"a_max": [1, 3]}, index=pd.Index(["a", "b"], name=("x", "group"))
            )
        elif LooseVersion(pd.__version__) >= LooseVersion("1.0.0"):
            agg_pdf = pdf.groupby(("x", "group")).agg(a_max=(("y", "A"), "max")).sort_index()
        agg_kdf = kdf.groupby(("x", "group")).agg(a_max=(("y", "A"), "max")).sort_index()
        self.assert_eq(agg_pdf, agg_kdf)

        # same column, different methods
        if LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            agg_pdf = pd.DataFrame(
                {"a_max": [1, 3], "a_min": [0, 2]}, index=pd.Index(["a", "b"], name=("x", "group"))
            )
        elif LooseVersion(pd.__version__) >= LooseVersion("1.0.0"):
            agg_pdf = (
                pdf.groupby(("x", "group"))
                .agg(a_max=(("y", "A"), "max"), a_min=(("y", "A"), "min"))
                .sort_index()
            )
        agg_kdf = (
            kdf.groupby(("x", "group"))
            .agg(a_max=(("y", "A"), "max"), a_min=(("y", "A"), "min"))
            .sort_index()
        )
        self.assert_eq(agg_pdf, agg_kdf)

        # different column, different methods
        if LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            agg_pdf = pd.DataFrame(
                {"a_max": [6, 8], "a_min": [0, 2]}, index=pd.Index(["a", "b"], name=("x", "group"))
            )
        elif LooseVersion(pd.__version__) >= LooseVersion("1.0.0"):
            agg_pdf = (
                pdf.groupby(("x", "group"))
                .agg(a_max=(("y", "B"), "max"), a_min=(("y", "A"), "min"))
                .sort_index()
            )
        agg_kdf = (
            kdf.groupby(("x", "group"))
            .agg(a_max=(("y", "B"), "max"), a_min=(("y", "A"), "min"))
            .sort_index()
        )
        self.assert_eq(agg_pdf, agg_kdf)

    def test_all_any(self):
        pdf = pd.DataFrame(
            {
                "A": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                "B": [True, True, True, False, False, False, None, True, None, False],
            }
        )
        kdf = ks.from_pandas(pdf)

        for as_index in [True, False]:
            if as_index:
                sort = lambda df: df.sort_index()
            else:
                sort = lambda df: df.sort_values("A").reset_index(drop=True).sort_index()
            self.assert_eq(
                sort(kdf.groupby("A", as_index=as_index).all()),
                sort(pdf.groupby("A", as_index=as_index).all()),
            )
            self.assert_eq(
                sort(kdf.groupby("A", as_index=as_index).any()),
                sort(pdf.groupby("A", as_index=as_index).any()),
            )

            self.assert_eq(
                sort(kdf.groupby("A", as_index=as_index).all()).B,
                sort(pdf.groupby("A", as_index=as_index).all()).B,
            )
            self.assert_eq(
                sort(kdf.groupby("A", as_index=as_index).any()).B,
                sort(pdf.groupby("A", as_index=as_index).any()).B,
            )

        self.assert_eq(
            kdf.B.groupby(kdf.A).all().sort_index(), pdf.B.groupby(pdf.A).all().sort_index()
        )
        self.assert_eq(
            kdf.B.groupby(kdf.A).any().sort_index(), pdf.B.groupby(pdf.A).any().sort_index()
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("X", "A"), ("Y", "B")])
        pdf.columns = columns
        kdf.columns = columns

        for as_index in [True, False]:
            if as_index:
                sort = lambda df: df.sort_index()
            else:
                sort = lambda df: df.sort_values(("X", "A")).reset_index(drop=True).sort_index()
            self.assert_eq(
                sort(kdf.groupby(("X", "A"), as_index=as_index).all()),
                sort(pdf.groupby(("X", "A"), as_index=as_index).all()),
            )
            self.assert_eq(
                sort(kdf.groupby(("X", "A"), as_index=as_index).any()),
                sort(pdf.groupby(("X", "A"), as_index=as_index).any()),
            )

    def test_raises(self):
        kdf = ks.DataFrame(
            {"a": [1, 2, 6, 4, 4, 6, 4, 3, 7], "b": [4, 2, 7, 3, 3, 1, 1, 1, 2]},
            index=[0, 1, 3, 5, 6, 8, 9, 9, 9],
        )
        # test raises with incorrect key
        self.assertRaises(ValueError, lambda: kdf.groupby([]))
        self.assertRaises(KeyError, lambda: kdf.groupby("x"))
        self.assertRaises(KeyError, lambda: kdf.groupby(["a", "x"]))
        self.assertRaises(KeyError, lambda: kdf.groupby("a")["x"])
        self.assertRaises(KeyError, lambda: kdf.groupby("a")["b", "x"])
        self.assertRaises(KeyError, lambda: kdf.groupby("a")[["b", "x"]])

    def test_nunique(self):
        pdf = pd.DataFrame(
            {"a": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], "b": [2, 2, 2, 3, 3, 4, 4, 5, 5, 5]}
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(
            kdf.groupby("a").agg({"b": "nunique"}).sort_index(),
            pdf.groupby("a").agg({"b": "nunique"}).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("a").nunique().sort_index(), pdf.groupby("a").nunique().sort_index()
        )
        self.assert_eq(
            kdf.groupby("a").nunique(dropna=False).sort_index(),
            pdf.groupby("a").nunique(dropna=False).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("a")["b"].nunique().sort_index(),
            pdf.groupby("a")["b"].nunique().sort_index(),
        )
        self.assert_eq(
            kdf.groupby("a")["b"].nunique(dropna=False).sort_index(),
            pdf.groupby("a")["b"].nunique(dropna=False).sort_index(),
        )

        nunique_kdf = kdf.groupby("a", as_index=False).agg({"b": "nunique"})
        nunique_pdf = pdf.groupby("a", as_index=False).agg({"b": "nunique"})
        self.assert_eq(
            nunique_kdf.sort_values(["a", "b"]).reset_index(drop=True),
            nunique_pdf.sort_values(["a", "b"]).reset_index(drop=True),
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("y", "b")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("x", "a")).nunique().sort_index(),
            pdf.groupby(("x", "a")).nunique().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(("x", "a")).nunique(dropna=False).sort_index(),
            pdf.groupby(("x", "a")).nunique(dropna=False).sort_index(),
        )

    def test_unique(self):
        for pdf in [
            pd.DataFrame(
                {"a": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], "b": [2, 2, 2, 3, 3, 4, 4, 5, 5, 5]}
            ),
            pd.DataFrame(
                {
                    "a": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    "b": ["w", "w", "w", "x", "x", "y", "y", "z", "z", "z"],
                }
            ),
        ]:
            with self.subTest(pdf=pdf):
                kdf = ks.from_pandas(pdf)

                actual = kdf.groupby("a")["b"].unique().sort_index().to_pandas()
                expect = pdf.groupby("a")["b"].unique().sort_index()
                self.assert_eq(len(actual), len(expect))
                for act, exp in zip(actual, expect):
                    self.assertTrue(sorted(act) == sorted(exp))

    def test_value_counts(self):
        pdf = pd.DataFrame({"A": [1, 2, 2, 3, 3, 3], "B": [1, 1, 2, 3, 3, 3]}, columns=["A", "B"])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(
            kdf.groupby("A")["B"].value_counts().sort_index(),
            pdf.groupby("A")["B"].value_counts().sort_index(),
        )
        self.assert_eq(
            kdf.groupby("A")["B"].value_counts(sort=True, ascending=False).sort_index(),
            pdf.groupby("A")["B"].value_counts(sort=True, ascending=False).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("A")["B"].value_counts(sort=True, ascending=True).sort_index(),
            pdf.groupby("A")["B"].value_counts(sort=True, ascending=True).sort_index(),
        )
        self.assert_eq(
            kdf.B.rename().groupby(kdf.A).value_counts().sort_index(),
            pdf.B.rename().groupby(pdf.A).value_counts().sort_index(),
        )
        self.assert_eq(
            kdf.B.groupby(kdf.A.rename()).value_counts().sort_index(),
            pdf.B.groupby(pdf.A.rename()).value_counts().sort_index(),
        )
        self.assert_eq(
            kdf.B.rename().groupby(kdf.A.rename()).value_counts().sort_index(),
            pdf.B.rename().groupby(pdf.A.rename()).value_counts().sort_index(),
        )

    def test_size(self):
        pdf = pd.DataFrame({"A": [1, 2, 2, 3, 3, 3], "B": [1, 1, 2, 3, 3, 3]})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.groupby("A").size().sort_index(), pdf.groupby("A").size().sort_index())
        self.assert_eq(
            kdf.groupby("A")["B"].size().sort_index(), pdf.groupby("A")["B"].size().sort_index()
        )
        self.assert_eq(
            kdf.groupby("A")[["B"]].size().sort_index(), pdf.groupby("A")[["B"]].size().sort_index()
        )
        self.assert_eq(
            kdf.groupby(["A", "B"]).size().sort_index(), pdf.groupby(["A", "B"]).size().sort_index()
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("X", "A"), ("Y", "B")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("X", "A")).size().sort_index(), pdf.groupby(("X", "A")).size().sort_index()
        )
        self.assert_eq(
            kdf.groupby([("X", "A"), ("Y", "B")]).size().sort_index(),
            pdf.groupby([("X", "A"), ("Y", "B")]).size().sort_index(),
        )

    def test_diff(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            }
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.groupby("b").diff().sort_index(), pdf.groupby("b").diff().sort_index())
        self.assert_eq(
            kdf.groupby(["a", "b"]).diff().sort_index(), pdf.groupby(["a", "b"]).diff().sort_index()
        )
        self.assert_eq(
            kdf.groupby(["b"])["a"].diff().sort_index(), pdf.groupby(["b"])["a"].diff().sort_index()
        )
        self.assert_eq(
            kdf.groupby(["b"])[["a", "b"]].diff().sort_index(),
            pdf.groupby(["b"])[["a", "b"]].diff().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5).diff().sort_index(), pdf.groupby(pdf.b // 5).diff().sort_index()
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5)["a"].diff().sort_index(),
            pdf.groupby(pdf.b // 5)["a"].diff().sort_index(),
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("x", "b")).diff().sort_index(), pdf.groupby(("x", "b")).diff().sort_index()
        )
        self.assert_eq(
            kdf.groupby([("x", "a"), ("x", "b")]).diff().sort_index(),
            pdf.groupby([("x", "a"), ("x", "b")]).diff().sort_index(),
        )

    def test_rank(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
            index=np.random.rand(6 * 3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.groupby("b").rank().sort_index(), pdf.groupby("b").rank().sort_index())
        self.assert_eq(
            kdf.groupby(["a", "b"]).rank().sort_index(), pdf.groupby(["a", "b"]).rank().sort_index()
        )
        self.assert_eq(
            kdf.groupby(["b"])["a"].rank().sort_index(), pdf.groupby(["b"])["a"].rank().sort_index()
        )
        self.assert_eq(
            kdf.groupby(["b"])[["a", "c"]].rank().sort_index(),
            pdf.groupby(["b"])[["a", "c"]].rank().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5).rank().sort_index(), pdf.groupby(pdf.b // 5).rank().sort_index()
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5)["a"].rank().sort_index(),
            pdf.groupby(pdf.b // 5)["a"].rank().sort_index(),
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("x", "b")).rank().sort_index(), pdf.groupby(("x", "b")).rank().sort_index()
        )
        self.assert_eq(
            kdf.groupby([("x", "a"), ("x", "b")]).rank().sort_index(),
            pdf.groupby([("x", "a"), ("x", "b")]).rank().sort_index(),
        )

    def test_cumcount(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
            index=np.random.rand(6 * 3),
        )
        kdf = ks.from_pandas(pdf)

        for ascending in [True, False]:
            self.assert_eq(
                kdf.groupby("b").cumcount(ascending=ascending).sort_index(),
                pdf.groupby("b").cumcount(ascending=ascending).sort_index(),
            )
            self.assert_eq(
                kdf.groupby(["a", "b"]).cumcount(ascending=ascending).sort_index(),
                pdf.groupby(["a", "b"]).cumcount(ascending=ascending).sort_index(),
            )
            self.assert_eq(
                kdf.groupby(["b"])["a"].cumcount(ascending=ascending).sort_index(),
                pdf.groupby(["b"])["a"].cumcount(ascending=ascending).sort_index(),
            )
            self.assert_eq(
                kdf.groupby(["b"])[["a", "c"]].cumcount(ascending=ascending).sort_index(),
                pdf.groupby(["b"])[["a", "c"]].cumcount(ascending=ascending).sort_index(),
            )
            self.assert_eq(
                kdf.groupby(kdf.b // 5).cumcount(ascending=ascending).sort_index(),
                pdf.groupby(pdf.b // 5).cumcount(ascending=ascending).sort_index(),
            )
            self.assert_eq(
                kdf.groupby(kdf.b // 5)["a"].cumcount(ascending=ascending).sort_index(),
                pdf.groupby(pdf.b // 5)["a"].cumcount(ascending=ascending).sort_index(),
            )
            self.assert_eq(
                kdf.groupby("b").cumcount(ascending=ascending).sum(),
                pdf.groupby("b").cumcount(ascending=ascending).sum(),
            )
            self.assert_eq(
                kdf.a.rename().groupby(kdf.b).cumcount(ascending=ascending).sort_index(),
                pdf.a.rename().groupby(pdf.b).cumcount(ascending=ascending).sort_index(),
            )
            self.assert_eq(
                kdf.a.groupby(kdf.b.rename()).cumcount(ascending=ascending).sort_index(),
                pdf.a.groupby(pdf.b.rename()).cumcount(ascending=ascending).sort_index(),
            )
            self.assert_eq(
                kdf.a.rename().groupby(kdf.b.rename()).cumcount(ascending=ascending).sort_index(),
                pdf.a.rename().groupby(pdf.b.rename()).cumcount(ascending=ascending).sort_index(),
            )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        for ascending in [True, False]:
            self.assert_eq(
                kdf.groupby(("x", "b")).cumcount(ascending=ascending).sort_index(),
                pdf.groupby(("x", "b")).cumcount(ascending=ascending).sort_index(),
            )
            self.assert_eq(
                kdf.groupby([("x", "a"), ("x", "b")]).cumcount(ascending=ascending).sort_index(),
                pdf.groupby([("x", "a"), ("x", "b")]).cumcount(ascending=ascending).sort_index(),
            )

    def test_cummin(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
            index=np.random.rand(6 * 3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby("b").cummin().sort_index(), pdf.groupby("b").cummin().sort_index()
        )
        self.assert_eq(
            kdf.groupby(["a", "b"]).cummin().sort_index(),
            pdf.groupby(["a", "b"]).cummin().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["b"])["a"].cummin().sort_index(),
            pdf.groupby(["b"])["a"].cummin().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["b"])[["a", "c"]].cummin().sort_index(),
            pdf.groupby(["b"])[["a", "c"]].cummin().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5).cummin().sort_index(),
            pdf.groupby(pdf.b // 5).cummin().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5)["a"].cummin().sort_index(),
            pdf.groupby(pdf.b // 5)["a"].cummin().sort_index(),
        )
        self.assert_eq(
            kdf.groupby("b").cummin().sum().sort_index(),
            pdf.groupby("b").cummin().sum().sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b).cummin().sort_index(),
            pdf.a.rename().groupby(pdf.b).cummin().sort_index(),
        )
        self.assert_eq(
            kdf.a.groupby(kdf.b.rename()).cummin().sort_index(),
            pdf.a.groupby(pdf.b.rename()).cummin().sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b.rename()).cummin().sort_index(),
            pdf.a.rename().groupby(pdf.b.rename()).cummin().sort_index(),
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("x", "b")).cummin().sort_index(),
            pdf.groupby(("x", "b")).cummin().sort_index(),
        )
        self.assert_eq(
            kdf.groupby([("x", "a"), ("x", "b")]).cummin().sort_index(),
            pdf.groupby([("x", "a"), ("x", "b")]).cummin().sort_index(),
        )

    def test_cummax(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
            index=np.random.rand(6 * 3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby("b").cummax().sort_index(), pdf.groupby("b").cummax().sort_index()
        )
        self.assert_eq(
            kdf.groupby(["a", "b"]).cummax().sort_index(),
            pdf.groupby(["a", "b"]).cummax().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["b"])["a"].cummax().sort_index(),
            pdf.groupby(["b"])["a"].cummax().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["b"])[["a", "c"]].cummax().sort_index(),
            pdf.groupby(["b"])[["a", "c"]].cummax().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5).cummax().sort_index(),
            pdf.groupby(pdf.b // 5).cummax().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5)["a"].cummax().sort_index(),
            pdf.groupby(pdf.b // 5)["a"].cummax().sort_index(),
        )
        self.assert_eq(
            kdf.groupby("b").cummax().sum().sort_index(),
            pdf.groupby("b").cummax().sum().sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b).cummax().sort_index(),
            pdf.a.rename().groupby(pdf.b).cummax().sort_index(),
        )
        self.assert_eq(
            kdf.a.groupby(kdf.b.rename()).cummax().sort_index(),
            pdf.a.groupby(pdf.b.rename()).cummax().sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b.rename()).cummax().sort_index(),
            pdf.a.rename().groupby(pdf.b.rename()).cummax().sort_index(),
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("x", "b")).cummax().sort_index(),
            pdf.groupby(("x", "b")).cummax().sort_index(),
        )
        self.assert_eq(
            kdf.groupby([("x", "a"), ("x", "b")]).cummax().sort_index(),
            pdf.groupby([("x", "a"), ("x", "b")]).cummax().sort_index(),
        )

    def test_cumsum(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
            index=np.random.rand(6 * 3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby("b").cumsum().sort_index(), pdf.groupby("b").cumsum().sort_index()
        )
        self.assert_eq(
            kdf.groupby(["a", "b"]).cumsum().sort_index(),
            pdf.groupby(["a", "b"]).cumsum().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["b"])["a"].cumsum().sort_index(),
            pdf.groupby(["b"])["a"].cumsum().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["b"])[["a", "c"]].cumsum().sort_index(),
            pdf.groupby(["b"])[["a", "c"]].cumsum().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5).cumsum().sort_index(),
            pdf.groupby(pdf.b // 5).cumsum().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5)["a"].cumsum().sort_index(),
            pdf.groupby(pdf.b // 5)["a"].cumsum().sort_index(),
        )
        self.assert_eq(
            kdf.groupby("b").cumsum().sum().sort_index(),
            pdf.groupby("b").cumsum().sum().sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b).cumsum().sort_index(),
            pdf.a.rename().groupby(pdf.b).cumsum().sort_index(),
        )
        self.assert_eq(
            kdf.a.groupby(kdf.b.rename()).cumsum().sort_index(),
            pdf.a.groupby(pdf.b.rename()).cumsum().sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b.rename()).cumsum().sort_index(),
            pdf.a.rename().groupby(pdf.b.rename()).cumsum().sort_index(),
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("x", "b")).cumsum().sort_index(),
            pdf.groupby(("x", "b")).cumsum().sort_index(),
        )
        self.assert_eq(
            kdf.groupby([("x", "a"), ("x", "b")]).cumsum().sort_index(),
            pdf.groupby([("x", "a"), ("x", "b")]).cumsum().sort_index(),
        )

    def test_cumprod(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 3,
                "b": [1, 1, 2, 3, 5, 8] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
            index=np.random.rand(6 * 3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby("b").cumprod().sort_index(),
            pdf.groupby("b").cumprod().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(["a", "b"]).cumprod().sort_index(),
            pdf.groupby(["a", "b"]).cumprod().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(["b"])["a"].cumprod().sort_index(),
            pdf.groupby(["b"])["a"].cumprod().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(["b"])[["a", "c"]].cumprod().sort_index(),
            pdf.groupby(["b"])[["a", "c"]].cumprod().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 3).cumprod().sort_index(),
            pdf.groupby(pdf.b // 3).cumprod().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 3)["a"].cumprod().sort_index(),
            pdf.groupby(pdf.b // 3)["a"].cumprod().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby("b").cumprod().sum().sort_index(),
            pdf.groupby("b").cumprod().sum().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b).cumprod().sort_index(),
            pdf.a.rename().groupby(pdf.b).cumprod().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.a.groupby(kdf.b.rename()).cumprod().sort_index(),
            pdf.a.groupby(pdf.b.rename()).cumprod().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b.rename()).cumprod().sort_index(),
            pdf.a.rename().groupby(pdf.b.rename()).cumprod().sort_index(),
            almost=True,
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("x", "b")).cumprod().sort_index(),
            pdf.groupby(("x", "b")).cumprod().sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.groupby([("x", "a"), ("x", "b")]).cumprod().sort_index(),
            pdf.groupby([("x", "a"), ("x", "b")]).cumprod().sort_index(),
            almost=True,
        )

    def test_nsmallest(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 1, 1, 2, 2, 2, 3, 3, 3] * 3,
                "b": [1, 2, 2, 2, 3, 3, 3, 4, 4] * 3,
                "c": [1, 2, 2, 2, 3, 3, 3, 4, 4] * 3,
                "d": [1, 2, 2, 2, 3, 3, 3, 4, 4] * 3,
            },
            index=np.random.rand(9 * 3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby(["a"])["b"].nsmallest(1).sort_values(),
            pdf.groupby(["a"])["b"].nsmallest(1).sort_values(),
        )
        self.assert_eq(
            kdf.groupby(["a"])["b"].nsmallest(2).sort_index(),
            pdf.groupby(["a"])["b"].nsmallest(2).sort_index(),
        )
        self.assert_eq(
            (kdf.b * 10).groupby(kdf.a).nsmallest(2).sort_index(),
            (pdf.b * 10).groupby(pdf.a).nsmallest(2).sort_index(),
        )
        self.assert_eq(
            kdf.b.rename().groupby(kdf.a).nsmallest(2).sort_index(),
            pdf.b.rename().groupby(pdf.a).nsmallest(2).sort_index(),
        )
        self.assert_eq(
            kdf.b.groupby(kdf.a.rename()).nsmallest(2).sort_index(),
            pdf.b.groupby(pdf.a.rename()).nsmallest(2).sort_index(),
        )
        self.assert_eq(
            kdf.b.rename().groupby(kdf.a.rename()).nsmallest(2).sort_index(),
            pdf.b.rename().groupby(pdf.a.rename()).nsmallest(2).sort_index(),
        )
        with self.assertRaisesRegex(ValueError, "nsmallest do not support multi-index now"):
            kdf.set_index(["a", "b"]).groupby(["c"])["d"].nsmallest(1)

    def test_nlargest(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 1, 1, 2, 2, 2, 3, 3, 3] * 3,
                "b": [1, 2, 2, 2, 3, 3, 3, 4, 4] * 3,
                "c": [1, 2, 2, 2, 3, 3, 3, 4, 4] * 3,
                "d": [1, 2, 2, 2, 3, 3, 3, 4, 4] * 3,
            },
            index=np.random.rand(9 * 3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby(["a"])["b"].nlargest(1).sort_values(),
            pdf.groupby(["a"])["b"].nlargest(1).sort_values(),
        )
        self.assert_eq(
            kdf.groupby(["a"])["b"].nlargest(2).sort_index(),
            pdf.groupby(["a"])["b"].nlargest(2).sort_index(),
        )
        self.assert_eq(
            (kdf.b * 10).groupby(kdf.a).nlargest(2).sort_index(),
            (pdf.b * 10).groupby(pdf.a).nlargest(2).sort_index(),
        )
        self.assert_eq(
            kdf.b.rename().groupby(kdf.a).nlargest(2).sort_index(),
            pdf.b.rename().groupby(pdf.a).nlargest(2).sort_index(),
        )
        self.assert_eq(
            kdf.b.groupby(kdf.a.rename()).nlargest(2).sort_index(),
            pdf.b.groupby(pdf.a.rename()).nlargest(2).sort_index(),
        )
        self.assert_eq(
            kdf.b.rename().groupby(kdf.a.rename()).nlargest(2).sort_index(),
            pdf.b.rename().groupby(pdf.a.rename()).nlargest(2).sort_index(),
        )
        with self.assertRaisesRegex(ValueError, "nlargest do not support multi-index now"):
            kdf.set_index(["a", "b"]).groupby(["c"])["d"].nlargest(1)

    def test_fillna(self):
        pdf = pd.DataFrame(
            {
                "A": [1, 1, 2, 2] * 3,
                "B": [2, 4, None, 3] * 3,
                "C": [None, None, None, 1] * 3,
                "D": [0, 1, 5, 4] * 3,
            }
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby("A").fillna(0).sort_index(), pdf.groupby("A").fillna(0).sort_index()
        )
        self.assert_eq(
            kdf.groupby("A")["C"].fillna(0).sort_index(),
            pdf.groupby("A")["C"].fillna(0).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("A")[["C"]].fillna(0).sort_index(),
            pdf.groupby("A")[["C"]].fillna(0).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("A").fillna(method="bfill").sort_index(),
            pdf.groupby("A").fillna(method="bfill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby("A")["C"].fillna(method="bfill").sort_index(),
            pdf.groupby("A")["C"].fillna(method="bfill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby("A")[["C"]].fillna(method="bfill").sort_index(),
            pdf.groupby("A")[["C"]].fillna(method="bfill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby("A").fillna(method="ffill").sort_index(),
            pdf.groupby("A").fillna(method="ffill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby("A")["C"].fillna(method="ffill").sort_index(),
            pdf.groupby("A")["C"].fillna(method="ffill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby("A")[["C"]].fillna(method="ffill").sort_index(),
            pdf.groupby("A")[["C"]].fillna(method="ffill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.A // 5).fillna(method="bfill").sort_index(),
            pdf.groupby(pdf.A // 5).fillna(method="bfill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.A // 5)["C"].fillna(method="bfill").sort_index(),
            pdf.groupby(pdf.A // 5)["C"].fillna(method="bfill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.A // 5)[["C"]].fillna(method="bfill").sort_index(),
            pdf.groupby(pdf.A // 5)[["C"]].fillna(method="bfill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.A // 5).fillna(method="ffill").sort_index(),
            pdf.groupby(pdf.A // 5).fillna(method="ffill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.A // 5)["C"].fillna(method="ffill").sort_index(),
            pdf.groupby(pdf.A // 5)["C"].fillna(method="ffill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.A // 5)[["C"]].fillna(method="ffill").sort_index(),
            pdf.groupby(pdf.A // 5)[["C"]].fillna(method="ffill").sort_index(),
        )
        self.assert_eq(
            kdf.C.rename().groupby(kdf.A).fillna(0).sort_index(),
            pdf.C.rename().groupby(pdf.A).fillna(0).sort_index(),
        )
        self.assert_eq(
            kdf.C.groupby(kdf.A.rename()).fillna(0).sort_index(),
            pdf.C.groupby(pdf.A.rename()).fillna(0).sort_index(),
        )
        self.assert_eq(
            kdf.C.rename().groupby(kdf.A.rename()).fillna(0).sort_index(),
            pdf.C.rename().groupby(pdf.A.rename()).fillna(0).sort_index(),
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B"), ("Y", "C"), ("Z", "D")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("X", "A")).fillna(0).sort_index(),
            pdf.groupby(("X", "A")).fillna(0).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(("X", "A")).fillna(method="bfill").sort_index(),
            pdf.groupby(("X", "A")).fillna(method="bfill").sort_index(),
        )
        self.assert_eq(
            kdf.groupby(("X", "A")).fillna(method="ffill").sort_index(),
            pdf.groupby(("X", "A")).fillna(method="ffill").sort_index(),
        )

    def test_ffill(self):
        idx = np.random.rand(4 * 3)
        pdf = pd.DataFrame(
            {
                "A": [1, 1, 2, 2] * 3,
                "B": [2, 4, None, 3] * 3,
                "C": [None, None, None, 1] * 3,
                "D": [0, 1, 5, 4] * 3,
            },
            index=idx,
        )
        kdf = ks.from_pandas(pdf)

        if LooseVersion(pd.__version__) <= LooseVersion("0.24.2"):
            self.assert_eq(
                kdf.groupby("A").ffill().sort_index(),
                pdf.groupby("A").ffill().sort_index().drop("A", 1),
            )
            self.assert_eq(
                kdf.groupby("A")[["B"]].ffill().sort_index(),
                pdf.groupby("A")[["B"]].ffill().sort_index().drop("A", 1),
            )
        else:
            self.assert_eq(
                kdf.groupby("A").ffill().sort_index(), pdf.groupby("A").ffill().sort_index()
            )
            self.assert_eq(
                kdf.groupby("A")[["B"]].ffill().sort_index(),
                pdf.groupby("A")[["B"]].ffill().sort_index(),
            )
        self.assert_eq(
            kdf.groupby("A")["B"].ffill().sort_index(), pdf.groupby("A")["B"].ffill().sort_index()
        )
        self.assert_eq(kdf.groupby("A")["B"].ffill()[idx[6]], pdf.groupby("A")["B"].ffill()[idx[6]])

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B"), ("Y", "C"), ("Z", "D")])
        pdf.columns = columns
        kdf.columns = columns

        if LooseVersion(pd.__version__) <= LooseVersion("0.24.2"):
            self.assert_eq(
                kdf.groupby(("X", "A")).ffill().sort_index(),
                pdf.groupby(("X", "A")).ffill().sort_index().drop(("X", "A"), 1),
            )
        else:
            self.assert_eq(
                kdf.groupby(("X", "A")).ffill().sort_index(),
                pdf.groupby(("X", "A")).ffill().sort_index(),
            )

    def test_bfill(self):
        idx = np.random.rand(4 * 3)
        pdf = pd.DataFrame(
            {
                "A": [1, 1, 2, 2] * 3,
                "B": [2, 4, None, 3] * 3,
                "C": [None, None, None, 1] * 3,
                "D": [0, 1, 5, 4] * 3,
            },
            index=idx,
        )
        kdf = ks.from_pandas(pdf)

        if LooseVersion(pd.__version__) <= LooseVersion("0.24.2"):
            self.assert_eq(
                kdf.groupby("A").bfill().sort_index(),
                pdf.groupby("A").bfill().sort_index().drop("A", 1),
            )
            self.assert_eq(
                kdf.groupby("A")[["B"]].bfill().sort_index(),
                pdf.groupby("A")[["B"]].bfill().sort_index().drop("A", 1),
            )
        else:
            self.assert_eq(
                kdf.groupby("A").bfill().sort_index(), pdf.groupby("A").bfill().sort_index()
            )
            self.assert_eq(
                kdf.groupby("A")[["B"]].bfill().sort_index(),
                pdf.groupby("A")[["B"]].bfill().sort_index(),
            )
        self.assert_eq(
            kdf.groupby("A")["B"].bfill().sort_index(), pdf.groupby("A")["B"].bfill().sort_index(),
        )
        self.assert_eq(kdf.groupby("A")["B"].bfill()[idx[6]], pdf.groupby("A")["B"].bfill()[idx[6]])

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B"), ("Y", "C"), ("Z", "D")])
        pdf.columns = columns
        kdf.columns = columns

        if LooseVersion(pd.__version__) <= LooseVersion("0.24.2"):
            self.assert_eq(
                kdf.groupby(("X", "A")).bfill().sort_index(),
                pdf.groupby(("X", "A")).bfill().sort_index().drop(("X", "A"), 1),
            )
        else:
            self.assert_eq(
                kdf.groupby(("X", "A")).bfill().sort_index(),
                pdf.groupby(("X", "A")).bfill().sort_index(),
            )

    @unittest.skipIf(pd.__version__ < "0.24.0", "not supported before pandas 0.24.0")
    def test_shift(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 1, 2, 2, 3, 3] * 3,
                "b": [1, 1, 2, 2, 3, 4] * 3,
                "c": [1, 4, 9, 16, 25, 36] * 3,
            },
            index=np.random.rand(6 * 3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.groupby("a").shift().sort_index(), pdf.groupby("a").shift().sort_index())
        # TODO: seems like a pandas' bug when fill_value is not None?
        # self.assert_eq(kdf.groupby(['a', 'b']).shift(periods=-1, fill_value=0).sort_index(),
        #                pdf.groupby(['a', 'b']).shift(periods=-1, fill_value=0).sort_index())
        self.assert_eq(
            kdf.groupby(["b"])["a"].shift().sort_index(),
            pdf.groupby(["b"])["a"].shift().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["a", "b"])["c"].shift().sort_index(),
            pdf.groupby(["a", "b"])["c"].shift().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5).shift().sort_index(),
            pdf.groupby(pdf.b // 5).shift().sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5)["a"].shift().sort_index(),
            pdf.groupby(pdf.b // 5)["a"].shift().sort_index(),
        )
        # TODO: known pandas' bug when fill_value is not None pandas>=1.0.0
        # https://github.com/pandas-dev/pandas/issues/31971#issue-565171762
        if LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            self.assert_eq(
                kdf.groupby(["b"])[["a", "c"]].shift(periods=-1, fill_value=0).sort_index(),
                pdf.groupby(["b"])[["a", "c"]].shift(periods=-1, fill_value=0).sort_index(),
            )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b).shift().sort_index(),
            pdf.a.rename().groupby(pdf.b).shift().sort_index(),
        )
        self.assert_eq(
            kdf.a.groupby(kdf.b.rename()).shift().sort_index(),
            pdf.a.groupby(pdf.b.rename()).shift().sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b.rename()).shift().sort_index(),
            pdf.a.rename().groupby(pdf.b.rename()).shift().sort_index(),
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("x", "a")).shift().sort_index(),
            pdf.groupby(("x", "a")).shift().sort_index(),
        )
        # TODO: seems like a pandas' bug when fill_value is not None?
        # self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')]).shift(periods=-1,
        #                                                            fill_value=0).sort_index(),
        #                pdf.groupby([('x', 'a'), ('x', 'b')]).shift(periods=-1,
        #                                                            fill_value=0).sort_index())

    def test_apply(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": [1, 1, 2, 3, 5, 8], "c": [1, 4, 9, 16, 25, 36]},
            columns=["a", "b", "c"],
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(
            kdf.groupby("b").apply(lambda x: x + x.min()).sort_index(),
            pdf.groupby("b").apply(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("b").apply(len).sort_index(), pdf.groupby("b").apply(len).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("b")["a"].apply(lambda x, y, z: x + x.min() + y * z, 10, z=20).sort_index(),
            pdf.groupby("b")["a"].apply(lambda x, y, z: x + x.min() + y * z, 10, z=20).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("b")[["a"]].apply(lambda x: x + x.min()).sort_index(),
            pdf.groupby("b")[["a"]].apply(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["a", "b"]).apply(lambda x, y, z: x + x.min() + y + z, 1, z=2).sort_index(),
            pdf.groupby(["a", "b"]).apply(lambda x, y, z: x + x.min() + y + z, 1, z=2).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["b"])["c"].apply(lambda x: 1).sort_index(),
            pdf.groupby(["b"])["c"].apply(lambda x: 1).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["b"])["c"].apply(len).sort_index(),
            pdf.groupby(["b"])["c"].apply(len).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5).apply(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pdf.b // 5).apply(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5)["a"].apply(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pdf.b // 5)["a"].apply(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5)[["a"]].apply(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pdf.b // 5)[["a"]].apply(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5)[["a"]].apply(len).sort_index(),
            pdf.groupby(pdf.b // 5)[["a"]].apply(len).sort_index(),
            almost=True,
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b).apply(lambda x: x + x.min()).sort_index(),
            pdf.a.rename().groupby(pdf.b).apply(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.a.groupby(kdf.b.rename()).apply(lambda x: x + x.min()).sort_index(),
            pdf.a.groupby(pdf.b.rename()).apply(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b.rename()).apply(lambda x: x + x.min()).sort_index(),
            pdf.a.rename().groupby(pdf.b.rename()).apply(lambda x: x + x.min()).sort_index(),
        )

        with self.assertRaisesRegex(TypeError, "<class 'int'> object is not callable"):
            kdf.groupby("b").apply(1)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("x", "b")).apply(lambda x: 1).sort_index(),
            pdf.groupby(("x", "b")).apply(lambda x: 1).sort_index(),
        )
        self.assert_eq(
            kdf.groupby([("x", "a"), ("x", "b")]).apply(lambda x: x + x.min()).sort_index(),
            pdf.groupby([("x", "a"), ("x", "b")]).apply(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(("x", "b")).apply(len).sort_index(),
            pdf.groupby(("x", "b")).apply(len).sort_index(),
        )
        self.assert_eq(
            kdf.groupby([("x", "a"), ("x", "b")]).apply(len).sort_index(),
            pdf.groupby([("x", "a"), ("x", "b")]).apply(len).sort_index(),
        )

    def test_apply_without_shortcut(self):
        with option_context("compute.shortcut_limit", 0):
            self.test_apply()

    def test_apply_negative(self):
        def func(_) -> ks.Series[int]:
            return pd.Series([1])

        with self.assertRaisesRegex(TypeError, "Series as a return type hint at frame groupby"):
            ks.range(10).groupby("id").apply(func)

    def test_apply_with_new_dataframe(self):
        pdf = pd.DataFrame(
            {"timestamp": [0.0, 0.5, 1.0, 0.0, 0.5], "car_id": ["A", "A", "A", "B", "B"]}
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby("car_id").apply(lambda _: pd.DataFrame({"column": [0.0]})).sort_index(),
            pdf.groupby("car_id").apply(lambda _: pd.DataFrame({"column": [0.0]})).sort_index(),
        )

        self.assert_eq(
            kdf.groupby("car_id")
            .apply(lambda df: pd.DataFrame({"mean": [df["timestamp"].mean()]}))
            .sort_index(),
            pdf.groupby("car_id")
            .apply(lambda df: pd.DataFrame({"mean": [df["timestamp"].mean()]}))
            .sort_index(),
        )

        # dataframe with 1000+ records
        pdf = pd.DataFrame(
            {
                "timestamp": [0.0, 0.5, 1.0, 0.0, 0.5] * 300,
                "car_id": ["A", "A", "A", "B", "B"] * 300,
            }
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby("car_id").apply(lambda _: pd.DataFrame({"column": [0.0]})).sort_index(),
            pdf.groupby("car_id").apply(lambda _: pd.DataFrame({"column": [0.0]})).sort_index(),
        )

        self.assert_eq(
            kdf.groupby("car_id")
            .apply(lambda df: pd.DataFrame({"mean": [df["timestamp"].mean()]}))
            .sort_index(),
            pdf.groupby("car_id")
            .apply(lambda df: pd.DataFrame({"mean": [df["timestamp"].mean()]}))
            .sort_index(),
        )

    def test_apply_with_new_dataframe_without_shortcut(self):
        with option_context("compute.shortcut_limit", 0):
            self.test_apply_with_new_dataframe()

    def test_apply_key_handling(self):
        pdf = pd.DataFrame(
            {"d": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby("d").apply(sum).sort_index(), pdf.groupby("d").apply(sum).sort_index()
        )

        with ks.option_context("compute.shortcut_limit", 1):
            self.assert_eq(
                kdf.groupby("d").apply(sum).sort_index(), pdf.groupby("d").apply(sum).sort_index()
            )

    def test_apply_with_side_effect(self):
        pdf = pd.DataFrame(
            {"d": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0], "v": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        )
        kdf = ks.from_pandas(pdf)

        acc = ks.utils.default_session().sparkContext.accumulator(0)

        def sum_with_acc_frame(x) -> ks.DataFrame[np.float64, np.float64]:
            nonlocal acc
            acc += 1
            return np.sum(x)

        actual = kdf.groupby("d").apply(sum_with_acc_frame).sort_index()
        actual.columns = ["d", "v"]
        self.assert_eq(actual, pdf.groupby("d").apply(sum).sort_index().reset_index(drop=True))
        self.assert_eq(acc.value, 2)

        def sum_with_acc_series(x) -> np.float64:
            nonlocal acc
            acc += 1
            return np.sum(x)

        self.assert_eq(
            kdf.groupby("d")["v"].apply(sum_with_acc_series).sort_index(),
            pdf.groupby("d")["v"].apply(sum).sort_index().reset_index(drop=True),
        )
        self.assert_eq(acc.value, 4)

    def test_transform(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": [1, 1, 2, 3, 5, 8], "c": [1, 4, 9, 16, 25, 36]},
            columns=["a", "b", "c"],
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(
            kdf.groupby("b").transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby("b").transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("b")["a"].transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby("b")["a"].transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("b")[["a"]].transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby("b")[["a"]].transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["a", "b"]).transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby(["a", "b"]).transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["b"])["c"].transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby(["b"])["c"].transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5).transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pdf.b // 5).transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5)["a"].transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pdf.b // 5)["a"].transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf.b // 5)[["a"]].transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby(pdf.b // 5)[["a"]].transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b).transform(lambda x: x + x.min()).sort_index(),
            pdf.a.rename().groupby(pdf.b).transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.a.groupby(kdf.b.rename()).transform(lambda x: x + x.min()).sort_index(),
            pdf.a.groupby(pdf.b.rename()).transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b.rename()).transform(lambda x: x + x.min()).sort_index(),
            pdf.a.rename().groupby(pdf.b.rename()).transform(lambda x: x + x.min()).sort_index(),
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("x", "b")).transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby(("x", "b")).transform(lambda x: x + x.min()).sort_index(),
        )
        self.assert_eq(
            kdf.groupby([("x", "a"), ("x", "b")]).transform(lambda x: x + x.min()).sort_index(),
            pdf.groupby([("x", "a"), ("x", "b")]).transform(lambda x: x + x.min()).sort_index(),
        )

    def test_transform_without_shortcut(self):
        with option_context("compute.shortcut_limit", 0):
            self.test_transform()

    def test_filter(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": [1, 1, 2, 3, 5, 8], "c": [1, 4, 9, 16, 25, 36]},
            columns=["a", "b", "c"],
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby("b").filter(lambda x: any(x.a == 2)).sort_index(),
            pdf.groupby("b").filter(lambda x: any(x.a == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("b")["a"].filter(lambda x: any(x == 2)).sort_index(),
            pdf.groupby("b")["a"].filter(lambda x: any(x == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.groupby("b")[["a"]].filter(lambda x: any(x.a == 2)).sort_index(),
            pdf.groupby("b")[["a"]].filter(lambda x: any(x.a == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(["a", "b"]).filter(lambda x: any(x.a == 2)).sort_index(),
            pdf.groupby(["a", "b"]).filter(lambda x: any(x.a == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf["b"] // 5).filter(lambda x: any(x.a == 2)).sort_index(),
            pdf.groupby(pdf["b"] // 5).filter(lambda x: any(x.a == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf["b"] // 5)["a"].filter(lambda x: any(x == 2)).sort_index(),
            pdf.groupby(pdf["b"] // 5)["a"].filter(lambda x: any(x == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.groupby(kdf["b"] // 5)[["a"]].filter(lambda x: any(x.a == 2)).sort_index(),
            pdf.groupby(pdf["b"] // 5)[["a"]].filter(lambda x: any(x.a == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b).filter(lambda x: any(x == 2)).sort_index(),
            pdf.a.rename().groupby(pdf.b).filter(lambda x: any(x == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.a.groupby(kdf.b.rename()).filter(lambda x: any(x == 2)).sort_index(),
            pdf.a.groupby(pdf.b.rename()).filter(lambda x: any(x == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.a.rename().groupby(kdf.b.rename()).filter(lambda x: any(x == 2)).sort_index(),
            pdf.a.rename().groupby(pdf.b.rename()).filter(lambda x: any(x == 2)).sort_index(),
        )

        with self.assertRaisesRegex(TypeError, "<class 'int'> object is not callable"):
            kdf.groupby("b").filter(1)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.groupby(("x", "b")).filter(lambda x: any(x[("x", "a")] == 2)).sort_index(),
            pdf.groupby(("x", "b")).filter(lambda x: any(x[("x", "a")] == 2)).sort_index(),
        )
        self.assert_eq(
            kdf.groupby([("x", "a"), ("x", "b")])
            .filter(lambda x: any(x[("x", "a")] == 2))
            .sort_index(),
            pdf.groupby([("x", "a"), ("x", "b")])
            .filter(lambda x: any(x[("x", "a")] == 2))
            .sort_index(),
        )

    def test_idxmax(self):
        pdf = pd.DataFrame(
            {"a": [1, 1, 2, 2, 3] * 3, "b": [1, 2, 3, 4, 5] * 3, "c": [5, 4, 3, 2, 1] * 3}
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            pdf.groupby(["a"]).idxmax().sort_index(), kdf.groupby(["a"]).idxmax().sort_index()
        )
        self.assert_eq(
            pdf.groupby(["a"]).idxmax(skipna=False).sort_index(),
            kdf.groupby(["a"]).idxmax(skipna=False).sort_index(),
        )
        self.assert_eq(
            pdf.groupby(["a"])["b"].idxmax().sort_index(),
            kdf.groupby(["a"])["b"].idxmax().sort_index(),
        )
        self.assert_eq(
            pdf.b.rename().groupby(pdf.a).idxmax().sort_index(),
            kdf.b.rename().groupby(kdf.a).idxmax().sort_index(),
        )
        self.assert_eq(
            pdf.b.groupby(pdf.a.rename()).idxmax().sort_index(),
            kdf.b.groupby(kdf.a.rename()).idxmax().sort_index(),
        )
        self.assert_eq(
            pdf.b.rename().groupby(pdf.a.rename()).idxmax().sort_index(),
            kdf.b.rename().groupby(kdf.a.rename()).idxmax().sort_index(),
        )

        with self.assertRaisesRegex(ValueError, "idxmax only support one-level index now"):
            kdf.set_index(["a", "b"]).groupby(["c"]).idxmax()

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            pdf.groupby(("x", "a")).idxmax().sort_index(),
            kdf.groupby(("x", "a")).idxmax().sort_index(),
        )
        self.assert_eq(
            pdf.groupby(("x", "a")).idxmax(skipna=False).sort_index(),
            kdf.groupby(("x", "a")).idxmax(skipna=False).sort_index(),
        )

    def test_idxmin(self):
        pdf = pd.DataFrame(
            {"a": [1, 1, 2, 2, 3] * 3, "b": [1, 2, 3, 4, 5] * 3, "c": [5, 4, 3, 2, 1] * 3}
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            pdf.groupby(["a"]).idxmin().sort_index(), kdf.groupby(["a"]).idxmin().sort_index()
        )
        self.assert_eq(
            pdf.groupby(["a"]).idxmin(skipna=False).sort_index(),
            kdf.groupby(["a"]).idxmin(skipna=False).sort_index(),
        )
        self.assert_eq(
            pdf.groupby(["a"])["b"].idxmin().sort_index(),
            kdf.groupby(["a"])["b"].idxmin().sort_index(),
        )
        self.assert_eq(
            pdf.b.rename().groupby(pdf.a).idxmin().sort_index(),
            kdf.b.rename().groupby(kdf.a).idxmin().sort_index(),
        )
        self.assert_eq(
            pdf.b.groupby(pdf.a.rename()).idxmin().sort_index(),
            kdf.b.groupby(kdf.a.rename()).idxmin().sort_index(),
        )
        self.assert_eq(
            pdf.b.rename().groupby(pdf.a.rename()).idxmin().sort_index(),
            kdf.b.rename().groupby(kdf.a.rename()).idxmin().sort_index(),
        )

        with self.assertRaisesRegex(ValueError, "idxmin only support one-level index now"):
            kdf.set_index(["a", "b"]).groupby(["c"]).idxmin()

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            pdf.groupby(("x", "a")).idxmin().sort_index(),
            kdf.groupby(("x", "a")).idxmin().sort_index(),
        )
        self.assert_eq(
            pdf.groupby(("x", "a")).idxmin(skipna=False).sort_index(),
            kdf.groupby(("x", "a")).idxmin(skipna=False).sort_index(),
        )

    def test_head(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3] * 3,
                "b": [2, 3, 1, 4, 6, 9, 8, 10, 7, 5] * 3,
                "c": [3, 5, 2, 5, 1, 2, 6, 4, 3, 6] * 3,
            },
            index=np.random.rand(10 * 3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.groupby("a").head(2).sort_index(), kdf.groupby("a").head(2).sort_index())
        self.assert_eq(
            pdf.groupby("a").head(-2).sort_index(), kdf.groupby("a").head(-2).sort_index()
        )
        self.assert_eq(
            pdf.groupby("a").head(100000).sort_index(), kdf.groupby("a").head(100000).sort_index()
        )

        self.assert_eq(
            pdf.groupby("a")["b"].head(2).sort_index(), kdf.groupby("a")["b"].head(2).sort_index()
        )
        self.assert_eq(
            pdf.groupby("a")["b"].head(-2).sort_index(), kdf.groupby("a")["b"].head(-2).sort_index()
        )
        self.assert_eq(
            pdf.groupby("a")["b"].head(100000).sort_index(),
            kdf.groupby("a")["b"].head(100000).sort_index(),
        )

        self.assert_eq(
            pdf.groupby("a")[["b"]].head(2).sort_index(),
            kdf.groupby("a")[["b"]].head(2).sort_index(),
        )
        self.assert_eq(
            pdf.groupby("a")[["b"]].head(-2).sort_index(),
            kdf.groupby("a")[["b"]].head(-2).sort_index(),
        )
        self.assert_eq(
            pdf.groupby("a")[["b"]].head(100000).sort_index(),
            kdf.groupby("a")[["b"]].head(100000).sort_index(),
        )

        self.assert_eq(
            pdf.groupby(pdf.a // 2).head(2).sort_index(),
            kdf.groupby(kdf.a // 2).head(2).sort_index(),
        )
        self.assert_eq(
            pdf.groupby(pdf.a // 2)["b"].head(2).sort_index(),
            kdf.groupby(kdf.a // 2)["b"].head(2).sort_index(),
        )
        self.assert_eq(
            pdf.groupby(pdf.a // 2)[["b"]].head(2).sort_index(),
            kdf.groupby(kdf.a // 2)[["b"]].head(2).sort_index(),
        )

        self.assert_eq(
            pdf.b.rename().groupby(pdf.a).head(2).sort_index(),
            kdf.b.rename().groupby(kdf.a).head(2).sort_index(),
        )
        self.assert_eq(
            pdf.b.groupby(pdf.a.rename()).head(2).sort_index(),
            kdf.b.groupby(kdf.a.rename()).head(2).sort_index(),
        )
        self.assert_eq(
            pdf.b.rename().groupby(pdf.a.rename()).head(2).sort_index(),
            kdf.b.rename().groupby(kdf.a.rename()).head(2).sort_index(),
        )

        # multi-index
        midx = pd.MultiIndex(
            [["x", "y"], ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]],
            [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
        )
        pdf = pd.DataFrame(
            {
                "a": [1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                "b": [2, 3, 1, 4, 6, 9, 8, 10, 7, 5],
                "c": [3, 5, 2, 5, 1, 2, 6, 4, 3, 6],
            },
            columns=["a", "b", "c"],
            index=midx,
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.groupby("a").head(2).sort_index(), kdf.groupby("a").head(2).sort_index())
        self.assert_eq(
            pdf.groupby("a").head(-2).sort_index(), kdf.groupby("a").head(-2).sort_index()
        )
        self.assert_eq(
            pdf.groupby("a").head(100000).sort_index(), kdf.groupby("a").head(100000).sort_index()
        )

        self.assert_eq(
            pdf.groupby("a")["b"].head(2).sort_index(), kdf.groupby("a")["b"].head(2).sort_index()
        )
        self.assert_eq(
            pdf.groupby("a")["b"].head(-2).sort_index(), kdf.groupby("a")["b"].head(-2).sort_index()
        )
        self.assert_eq(
            pdf.groupby("a")["b"].head(100000).sort_index(),
            kdf.groupby("a")["b"].head(100000).sort_index(),
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            pdf.groupby(("x", "a")).head(2).sort_index(),
            kdf.groupby(("x", "a")).head(2).sort_index(),
        )
        self.assert_eq(
            pdf.groupby(("x", "a")).head(-2).sort_index(),
            kdf.groupby(("x", "a")).head(-2).sort_index(),
        )
        self.assert_eq(
            pdf.groupby(("x", "a")).head(100000).sort_index(),
            kdf.groupby(("x", "a")).head(100000).sort_index(),
        )

    def test_missing(self):
        kdf = ks.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9]})

        # DataFrameGroupBy functions
        missing_functions = inspect.getmembers(
            MissingPandasLikeDataFrameGroupBy, inspect.isfunction
        )
        unsupported_functions = [
            name for (name, type_) in missing_functions if type_.__name__ == "unsupported_function"
        ]
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                PandasNotImplementedError,
                "method.*GroupBy.*{}.*not implemented( yet\\.|\\. .+)".format(name),
            ):
                getattr(kdf.groupby("a"), name)()

        deprecated_functions = [
            name for (name, type_) in missing_functions if type_.__name__ == "deprecated_function"
        ]
        for name in deprecated_functions:
            with self.assertRaisesRegex(
                PandasNotImplementedError, "method.*GroupBy.*{}.*is deprecated".format(name)
            ):
                getattr(kdf.groupby("a"), name)()

        # SeriesGroupBy functions
        missing_functions = inspect.getmembers(MissingPandasLikeSeriesGroupBy, inspect.isfunction)
        unsupported_functions = [
            name for (name, type_) in missing_functions if type_.__name__ == "unsupported_function"
        ]
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                PandasNotImplementedError,
                "method.*GroupBy.*{}.*not implemented( yet\\.|\\. .+)".format(name),
            ):
                getattr(kdf.a.groupby(kdf.a), name)()

        deprecated_functions = [
            name for (name, type_) in missing_functions if type_.__name__ == "deprecated_function"
        ]
        for name in deprecated_functions:
            with self.assertRaisesRegex(
                PandasNotImplementedError, "method.*GroupBy.*{}.*is deprecated".format(name)
            ):
                getattr(kdf.a.groupby(kdf.a), name)()

        # DataFrameGroupBy properties
        missing_properties = inspect.getmembers(
            MissingPandasLikeDataFrameGroupBy, lambda o: isinstance(o, property)
        )
        unsupported_properties = [
            name
            for (name, type_) in missing_properties
            if type_.fget.__name__ == "unsupported_property"
        ]
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                PandasNotImplementedError,
                "property.*GroupBy.*{}.*not implemented( yet\\.|\\. .+)".format(name),
            ):
                getattr(kdf.groupby("a"), name)
        deprecated_properties = [
            name
            for (name, type_) in missing_properties
            if type_.fget.__name__ == "deprecated_property"
        ]
        for name in deprecated_properties:
            with self.assertRaisesRegex(
                PandasNotImplementedError, "property.*GroupBy.*{}.*is deprecated".format(name)
            ):
                getattr(kdf.groupby("a"), name)

        # SeriesGroupBy properties
        missing_properties = inspect.getmembers(
            MissingPandasLikeSeriesGroupBy, lambda o: isinstance(o, property)
        )
        unsupported_properties = [
            name
            for (name, type_) in missing_properties
            if type_.fget.__name__ == "unsupported_property"
        ]
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                PandasNotImplementedError,
                "property.*GroupBy.*{}.*not implemented( yet\\.|\\. .+)".format(name),
            ):
                getattr(kdf.a.groupby(kdf.a), name)
        deprecated_properties = [
            name
            for (name, type_) in missing_properties
            if type_.fget.__name__ == "deprecated_property"
        ]
        for name in deprecated_properties:
            with self.assertRaisesRegex(
                PandasNotImplementedError, "property.*GroupBy.*{}.*is deprecated".format(name)
            ):
                getattr(kdf.a.groupby(kdf.a), name)

    @staticmethod
    def test_is_multi_agg_with_relabel():

        assert is_multi_agg_with_relabel(a="max") is False
        assert is_multi_agg_with_relabel(a_min=("a", "max"), a_max=("a", "min")) is True
