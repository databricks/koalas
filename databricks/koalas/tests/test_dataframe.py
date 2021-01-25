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
from datetime import datetime
from distutils.version import LooseVersion
import inspect
import sys
import unittest
from io import StringIO

import numpy as np
import pandas as pd
import pyspark
from pyspark import StorageLevel
from pyspark.ml.linalg import SparseVector

from databricks import koalas as ks
from databricks.koalas.config import option_context
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.frame import CachedDataFrame
from databricks.koalas.missing.frame import _MissingPandasLikeDataFrame
from databricks.koalas.testing.utils import (
    ReusedSQLTestCase,
    SQLTestUtils,
    SPARK_CONF_ARROW_ENABLED,
)
from databricks.koalas.utils import name_like_string


class DataFrameTest(ReusedSQLTestCase, SQLTestUtils):
    @property
    def pdf(self):
        return pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9], "b": [4, 5, 6, 3, 2, 1, 0, 0, 0],},
            index=np.random.rand(9),
        )

    @property
    def kdf(self):
        return ks.from_pandas(self.pdf)

    @property
    def df_pair(self):
        pdf = self.pdf
        kdf = ks.from_pandas(pdf)
        return pdf, kdf

    def test_dataframe(self):
        pdf, kdf = self.df_pair

        self.assert_eq(kdf["a"] + 1, pdf["a"] + 1)

        self.assert_eq(kdf.columns, pd.Index(["a", "b"]))

        self.assert_eq(kdf[kdf["b"] > 2], pdf[pdf["b"] > 2])
        self.assert_eq(-kdf[kdf["b"] > 2], -pdf[pdf["b"] > 2])
        self.assert_eq(kdf[["a", "b"]], pdf[["a", "b"]])
        self.assert_eq(kdf.a, pdf.a)
        self.assert_eq(kdf.b.mean(), pdf.b.mean())
        self.assert_eq(kdf.b.var(), pdf.b.var())
        self.assert_eq(kdf.b.std(), pdf.b.std())

        pdf, kdf = self.df_pair
        self.assert_eq(kdf[["a", "b"]], pdf[["a", "b"]])

        self.assertEqual(kdf.a.notnull().rename("x").name, "x")

        # check ks.DataFrame(ks.Series)
        pser = pd.Series([1, 2, 3], name="x", index=np.random.rand(3))
        kser = ks.from_pandas(pser)
        self.assert_eq(pd.DataFrame(pser), ks.DataFrame(kser))

        # check kdf[pd.Index]
        pdf, kdf = self.df_pair
        column_mask = pdf.columns.isin(["a", "b"])
        index_cols = pdf.columns[column_mask]
        self.assert_eq(kdf[index_cols], pdf[index_cols])

    def test_inplace(self):
        pdf, kdf = self.df_pair

        pser = pdf.a
        kser = kdf.a

        pdf["a"] = pdf["a"] + 10
        kdf["a"] = kdf["a"] + 10

        self.assert_eq(kdf, pdf)
        self.assert_eq(kser, pser)

    def test_assign_list(self):
        pdf, kdf = self.df_pair

        pser = pdf.a
        kser = kdf.a

        pdf["x"] = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        kdf["x"] = [10, 20, 30, 40, 50, 60, 70, 80, 90]

        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        self.assert_eq(kser, pser)

        with self.assertRaisesRegex(ValueError, "Length of values does not match length of index"):
            kdf["z"] = [10, 20, 30, 40, 50, 60, 70, 80]

    def test_dataframe_multiindex_columns(self):
        pdf = pd.DataFrame(
            {
                ("x", "a", "1"): [1, 2, 3],
                ("x", "b", "2"): [4, 5, 6],
                ("y.z", "c.d", "3"): [7, 8, 9],
                ("x", "b", "4"): [10, 11, 12],
            },
            index=np.random.rand(3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf, pdf)
        self.assert_eq(kdf["x"], pdf["x"])
        self.assert_eq(kdf["y.z"], pdf["y.z"])
        self.assert_eq(kdf["x"]["b"], pdf["x"]["b"])
        self.assert_eq(kdf["x"]["b"]["2"], pdf["x"]["b"]["2"])

        self.assert_eq(kdf.x, pdf.x)
        self.assert_eq(kdf.x.b, pdf.x.b)
        self.assert_eq(kdf.x.b["2"], pdf.x.b["2"])

        self.assertRaises(KeyError, lambda: kdf["z"])
        self.assertRaises(AttributeError, lambda: kdf.z)

        self.assert_eq(kdf[("x",)], pdf[("x",)])
        self.assert_eq(kdf[("x", "a")], pdf[("x", "a")])
        self.assert_eq(kdf[("x", "a", "1")], pdf[("x", "a", "1")])

    def test_dataframe_column_level_name(self):
        column = pd.Index(["A", "B", "C"], name="X")
        pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=column, index=np.random.rand(2))
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf, pdf)
        self.assert_eq(kdf.columns.names, pdf.columns.names)
        self.assert_eq(kdf.to_pandas().columns.names, pdf.columns.names)

    def test_dataframe_multiindex_names_level(self):
        columns = pd.MultiIndex.from_tuples(
            [("X", "A", "Z"), ("X", "B", "Z"), ("Y", "C", "Z"), ("Y", "D", "Z")],
            names=["lvl_1", "lvl_2", "lv_3"],
        )
        pdf = pd.DataFrame(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]],
            columns=columns,
            index=np.random.rand(5),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.columns.names, pdf.columns.names)
        self.assert_eq(kdf.to_pandas().columns.names, pdf.columns.names)

        kdf1 = ks.from_pandas(pdf)
        self.assert_eq(kdf1.columns.names, pdf.columns.names)

        self.assertRaises(
            AssertionError, lambda: ks.DataFrame(kdf1._internal.copy(column_label_names=("level",)))
        )

        self.assert_eq(kdf["X"], pdf["X"])
        self.assert_eq(kdf["X"].columns.names, pdf["X"].columns.names)
        self.assert_eq(kdf["X"].to_pandas().columns.names, pdf["X"].columns.names)
        self.assert_eq(kdf["X"]["A"], pdf["X"]["A"])
        self.assert_eq(kdf["X"]["A"].columns.names, pdf["X"]["A"].columns.names)
        self.assert_eq(kdf["X"]["A"].to_pandas().columns.names, pdf["X"]["A"].columns.names)
        self.assert_eq(kdf[("X", "A")], pdf[("X", "A")])
        self.assert_eq(kdf[("X", "A")].columns.names, pdf[("X", "A")].columns.names)
        self.assert_eq(kdf[("X", "A")].to_pandas().columns.names, pdf[("X", "A")].columns.names)
        self.assert_eq(kdf[("X", "A", "Z")], pdf[("X", "A", "Z")])

    def test_iterrows(self):
        pdf = pd.DataFrame(
            {
                ("x", "a", "1"): [1, 2, 3],
                ("x", "b", "2"): [4, 5, 6],
                ("y.z", "c.d", "3"): [7, 8, 9],
                ("x", "b", "4"): [10, 11, 12],
            },
            index=np.random.rand(3),
        )
        kdf = ks.from_pandas(pdf)

        for (pdf_k, pdf_v), (kdf_k, kdf_v) in zip(pdf.iterrows(), kdf.iterrows()):
            self.assert_eq(pdf_k, kdf_k)
            self.assert_eq(pdf_v, kdf_v)

    def test_reset_index(self):
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=np.random.rand(3))
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.reset_index(), pdf.reset_index())
        self.assert_eq(kdf.reset_index(drop=True), pdf.reset_index(drop=True))

        pdf.index.name = "a"
        kdf.index.name = "a"

        with self.assertRaisesRegex(ValueError, "cannot insert a, already exists"):
            kdf.reset_index()

        self.assert_eq(kdf.reset_index(drop=True), pdf.reset_index(drop=True))

        # inplace
        pser = pdf.a
        kser = kdf.a
        pdf.reset_index(drop=True, inplace=True)
        kdf.reset_index(drop=True, inplace=True)
        self.assert_eq(kdf, pdf)
        self.assert_eq(kser, pser)

    def test_reset_index_with_default_index_types(self):
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, index=np.random.rand(3))
        kdf = ks.from_pandas(pdf)

        with ks.option_context("compute.default_index_type", "sequence"):
            self.assert_eq(kdf.reset_index(), pdf.reset_index())

        with ks.option_context("compute.default_index_type", "distributed-sequence"):
            self.assert_eq(kdf.reset_index(), pdf.reset_index())

        with ks.option_context("compute.default_index_type", "distributed"):
            # the index is different.
            self.assert_eq(kdf.reset_index().to_pandas().reset_index(drop=True), pdf.reset_index())

    def test_reset_index_with_multiindex_columns(self):
        index = pd.MultiIndex.from_tuples(
            [("bird", "falcon"), ("bird", "parrot"), ("mammal", "lion"), ("mammal", "monkey")],
            names=["class", "name"],
        )
        columns = pd.MultiIndex.from_tuples([("speed", "max"), ("species", "type")])
        pdf = pd.DataFrame(
            [(389.0, "fly"), (24.0, "fly"), (80.5, "run"), (np.nan, "jump")],
            index=index,
            columns=columns,
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf, pdf)
        self.assert_eq(kdf.reset_index(), pdf.reset_index())
        self.assert_eq(kdf.reset_index(level="class"), pdf.reset_index(level="class"))
        self.assert_eq(
            kdf.reset_index(level="class", col_level=1), pdf.reset_index(level="class", col_level=1)
        )
        self.assert_eq(
            kdf.reset_index(level="class", col_level=1, col_fill="species"),
            pdf.reset_index(level="class", col_level=1, col_fill="species"),
        )
        self.assert_eq(
            kdf.reset_index(level="class", col_level=1, col_fill="genus"),
            pdf.reset_index(level="class", col_level=1, col_fill="genus"),
        )

        with self.assertRaisesRegex(IndexError, "Index has only 2 levels, not 3"):
            kdf.reset_index(col_level=2)

        pdf.index.names = [("x", "class"), ("y", "name")]
        kdf.index.names = [("x", "class"), ("y", "name")]

        self.assert_eq(kdf.reset_index(), pdf.reset_index())

        with self.assertRaisesRegex(ValueError, "Item must have length equal to number of levels."):
            kdf.reset_index(col_level=1)

    def test_multiindex_column_access(self):
        columns = pd.MultiIndex.from_tuples(
            [
                ("a", "", "", "b"),
                ("c", "", "d", ""),
                ("e", "", "f", ""),
                ("e", "g", "", ""),
                ("", "", "", "h"),
                ("i", "", "", ""),
            ]
        )

        pdf = pd.DataFrame(
            [
                (1, "a", "x", 10, 100, 1000),
                (2, "b", "y", 20, 200, 2000),
                (3, "c", "z", 30, 300, 3000),
            ],
            columns=columns,
            index=np.random.rand(3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf, pdf)
        self.assert_eq(kdf["a"], pdf["a"])
        self.assert_eq(kdf["a"]["b"], pdf["a"]["b"])
        self.assert_eq(kdf["c"], pdf["c"])
        self.assert_eq(kdf["c"]["d"], pdf["c"]["d"])
        self.assert_eq(kdf["e"], pdf["e"])
        self.assert_eq(kdf["e"][""]["f"], pdf["e"][""]["f"])
        self.assert_eq(kdf["e"]["g"], pdf["e"]["g"])
        self.assert_eq(kdf[""], pdf[""])
        self.assert_eq(kdf[""]["h"], pdf[""]["h"])
        self.assert_eq(kdf["i"], pdf["i"])

        self.assert_eq(kdf[["a", "e"]], pdf[["a", "e"]])
        self.assert_eq(kdf[["e", "a"]], pdf[["e", "a"]])

        self.assert_eq(kdf[("a",)], pdf[("a",)])
        self.assert_eq(kdf[("e", "g")], pdf[("e", "g")])
        # self.assert_eq(kdf[("i",)], pdf[("i",)])
        self.assert_eq(kdf[("i", "")], pdf[("i", "")])

        self.assertRaises(KeyError, lambda: kdf[("a", "b")])

    def test_repr_cache_invalidation(self):
        # If there is any cache, inplace operations should invalidate it.
        df = ks.range(10)
        df.__repr__()
        df["a"] = df["id"]
        self.assertEqual(df.__repr__(), df.to_pandas().__repr__())

    def test_repr_html_cache_invalidation(self):
        # If there is any cache, inplace operations should invalidate it.
        df = ks.range(10)
        df._repr_html_()
        df["a"] = df["id"]
        self.assertEqual(df._repr_html_(), df.to_pandas()._repr_html_())

    def test_empty_dataframe(self):
        pdf = pd.DataFrame({"a": pd.Series([], dtype="i1"), "b": pd.Series([], dtype="str")})

        self.assertRaises(ValueError, lambda: ks.from_pandas(pdf))

        with self.sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
            self.assertRaises(ValueError, lambda: ks.from_pandas(pdf))

    def test_all_null_dataframe(self):

        pdf = pd.DataFrame(
            {
                "a": pd.Series([None, None, None], dtype="float64"),
                "b": pd.Series([None, None, None], dtype="str"),
            },
            index=np.random.rand(3),
        )

        self.assertRaises(ValueError, lambda: ks.from_pandas(pdf))

        with self.sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
            self.assertRaises(ValueError, lambda: ks.from_pandas(pdf))

    def test_nullable_object(self):
        pdf = pd.DataFrame(
            {
                "a": list("abc") + [np.nan],
                "b": list(range(1, 4)) + [np.nan],
                "c": list(np.arange(3, 6).astype("i1")) + [np.nan],
                "d": list(np.arange(4.0, 7.0, dtype="float64")) + [np.nan],
                "e": [True, False, True, np.nan],
                "f": list(pd.date_range("20130101", periods=3)) + [np.nan],
            },
            index=np.random.rand(4),
        )

        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf, pdf)

        with self.sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
            kdf = ks.from_pandas(pdf)
            self.assert_eq(kdf, pdf)

    def test_assign(self):
        pdf, kdf = self.df_pair

        kdf["w"] = 1.0
        pdf["w"] = 1.0

        self.assert_eq(kdf, pdf)

        kdf[1] = 1.0
        pdf[1] = 1.0

        self.assert_eq(kdf, pdf)

        kdf = kdf.assign(a=kdf["a"] * 2)
        pdf = pdf.assign(a=pdf["a"] * 2)

        self.assert_eq(kdf, pdf)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "w"), ("y", "v")])
        pdf.columns = columns
        kdf.columns = columns

        kdf[("a", "c")] = "def"
        pdf[("a", "c")] = "def"

        self.assert_eq(kdf, pdf)

        kdf = kdf.assign(Z="ZZ")
        pdf = pdf.assign(Z="ZZ")

        self.assert_eq(kdf, pdf)

        kdf["x"] = "ghi"
        pdf["x"] = "ghi"

        self.assert_eq(kdf, pdf)

    def test_head(self):
        pdf, kdf = self.df_pair

        self.assert_eq(kdf.head(2), pdf.head(2))
        self.assert_eq(kdf.head(3), pdf.head(3))
        self.assert_eq(kdf.head(0), pdf.head(0))
        self.assert_eq(kdf.head(-3), pdf.head(-3))
        self.assert_eq(kdf.head(-10), pdf.head(-10))

    def test_attributes(self):
        kdf = self.kdf

        self.assertIn("a", dir(kdf))
        self.assertNotIn("foo", dir(kdf))
        self.assertRaises(AttributeError, lambda: kdf.foo)

        kdf = ks.DataFrame({"a b c": [1, 2, 3]})
        self.assertNotIn("a b c", dir(kdf))
        kdf = ks.DataFrame({"a": [1, 2], 5: [1, 2]})
        self.assertIn("a", dir(kdf))
        self.assertNotIn(5, dir(kdf))

    def test_column_names(self):
        pdf, kdf = self.df_pair

        self.assert_eq(kdf.columns, pdf.columns)
        self.assert_eq(kdf[["b", "a"]].columns, pdf[["b", "a"]].columns)
        self.assert_eq(kdf["a"].name, pdf["a"].name)
        self.assert_eq((kdf["a"] + 1).name, (pdf["a"] + 1).name)

        self.assert_eq((kdf.a + kdf.b).name, (pdf.a + pdf.b).name)
        self.assert_eq((kdf.a + kdf.b.rename("a")).name, (pdf.a + pdf.b.rename("a")).name)
        self.assert_eq((kdf.a + kdf.b.rename()).name, (pdf.a + pdf.b.rename()).name)
        self.assert_eq((kdf.a.rename() + kdf.b).name, (pdf.a.rename() + pdf.b).name)
        self.assert_eq(
            (kdf.a.rename() + kdf.b.rename()).name, (pdf.a.rename() + pdf.b.rename()).name
        )

    def test_rename_columns(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7], "b": [7, 6, 5, 4, 3, 2, 1]}, index=np.random.rand(7)
        )
        kdf = ks.from_pandas(pdf)

        kdf.columns = ["x", "y"]
        pdf.columns = ["x", "y"]
        self.assert_eq(kdf.columns, pd.Index(["x", "y"]))
        self.assert_eq(kdf, pdf)
        self.assert_eq(kdf._internal.data_spark_column_names, ["x", "y"])
        self.assert_eq(kdf.to_spark().columns, ["x", "y"])
        self.assert_eq(kdf.to_spark(index_col="index").columns, ["index", "x", "y"])

        columns = pdf.columns
        columns.name = "lvl_1"

        kdf.columns = columns
        self.assert_eq(kdf.columns.names, ["lvl_1"])
        self.assert_eq(kdf, pdf)

        msg = "Length mismatch: Expected axis has 2 elements, new values have 4 elements"
        with self.assertRaisesRegex(ValueError, msg):
            kdf.columns = [1, 2, 3, 4]

        # Multi-index columns
        pdf = pd.DataFrame(
            {("A", "0"): [1, 2, 2, 3], ("B", "1"): [1, 2, 3, 4]}, index=np.random.rand(4)
        )
        kdf = ks.from_pandas(pdf)

        columns = pdf.columns
        self.assert_eq(kdf.columns, columns)
        self.assert_eq(kdf, pdf)

        pdf.columns = ["x", "y"]
        kdf.columns = ["x", "y"]
        self.assert_eq(kdf.columns, pd.Index(["x", "y"]))
        self.assert_eq(kdf, pdf)
        self.assert_eq(kdf._internal.data_spark_column_names, ["x", "y"])
        self.assert_eq(kdf.to_spark().columns, ["x", "y"])
        self.assert_eq(kdf.to_spark(index_col="index").columns, ["index", "x", "y"])

        pdf.columns = columns
        kdf.columns = columns
        self.assert_eq(kdf.columns, columns)
        self.assert_eq(kdf, pdf)
        self.assert_eq(kdf._internal.data_spark_column_names, ["(A, 0)", "(B, 1)"])
        self.assert_eq(kdf.to_spark().columns, ["(A, 0)", "(B, 1)"])
        self.assert_eq(kdf.to_spark(index_col="index").columns, ["index", "(A, 0)", "(B, 1)"])

        columns.names = ["lvl_1", "lvl_2"]

        kdf.columns = columns
        self.assert_eq(kdf.columns.names, ["lvl_1", "lvl_2"])
        self.assert_eq(kdf, pdf)
        self.assert_eq(kdf._internal.data_spark_column_names, ["(A, 0)", "(B, 1)"])
        self.assert_eq(kdf.to_spark().columns, ["(A, 0)", "(B, 1)"])
        self.assert_eq(kdf.to_spark(index_col="index").columns, ["index", "(A, 0)", "(B, 1)"])

    def test_rename_dataframe(self):
        pdf1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        kdf1 = ks.from_pandas(pdf1)

        self.assert_eq(
            kdf1.rename(columns={"A": "a", "B": "b"}), pdf1.rename(columns={"A": "a", "B": "b"})
        )

        result_kdf = kdf1.rename(index={1: 10, 2: 20})
        result_pdf = pdf1.rename(index={1: 10, 2: 20})
        self.assert_eq(result_kdf, result_pdf)

        # inplace
        pser = result_pdf.A
        kser = result_kdf.A
        result_kdf.rename(index={10: 100, 20: 200}, inplace=True)
        result_pdf.rename(index={10: 100, 20: 200}, inplace=True)
        self.assert_eq(result_kdf, result_pdf)
        self.assert_eq(kser, pser)

        def str_lower(s) -> str:
            return str.lower(s)

        self.assert_eq(
            kdf1.rename(str_lower, axis="columns"), pdf1.rename(str_lower, axis="columns")
        )

        def mul10(x) -> int:
            return x * 10

        self.assert_eq(kdf1.rename(mul10, axis="index"), pdf1.rename(mul10, axis="index"))

        self.assert_eq(
            kdf1.rename(columns=str_lower, index={1: 10, 2: 20}),
            pdf1.rename(columns=str_lower, index={1: 10, 2: 20}),
        )

        idx = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B"), ("Y", "C"), ("Y", "D")])
        pdf2 = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=idx)
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(kdf2.rename(columns=str_lower), pdf2.rename(columns=str_lower))

        self.assert_eq(
            kdf2.rename(columns=str_lower, level=0), pdf2.rename(columns=str_lower, level=0)
        )
        self.assert_eq(
            kdf2.rename(columns=str_lower, level=1), pdf2.rename(columns=str_lower, level=1)
        )

        pdf3 = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=idx, columns=list("ab"))
        kdf3 = ks.from_pandas(pdf3)

        self.assert_eq(kdf3.rename(index=str_lower), pdf3.rename(index=str_lower))
        self.assert_eq(kdf3.rename(index=str_lower, level=0), pdf3.rename(index=str_lower, level=0))
        self.assert_eq(kdf3.rename(index=str_lower, level=1), pdf3.rename(index=str_lower, level=1))

        pdf4 = pdf2 + 1
        kdf4 = kdf2 + 1
        self.assert_eq(kdf4.rename(columns=str_lower), pdf4.rename(columns=str_lower))

        pdf5 = pdf3 + 1
        kdf5 = kdf3 + 1
        self.assert_eq(kdf5.rename(index=str_lower), pdf5.rename(index=str_lower))

    def test_dot_in_column_name(self):
        self.assert_eq(
            ks.DataFrame(ks.range(1)._internal.spark_frame.selectExpr("1L as `a.b`"))["a.b"],
            ks.Series([1], name="a.b"),
        )

    def test_aggregate(self):
        pdf = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=["A", "B", "C"]
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.agg(["sum", "min"])[["A", "B", "C"]].sort_index(),  # TODO?: fix column order
            pdf.agg(["sum", "min"])[["A", "B", "C"]].sort_index(),
        )
        self.assert_eq(
            kdf.agg({"A": ["sum", "min"], "B": ["min", "max"]})[["A", "B"]].sort_index(),
            pdf.agg({"A": ["sum", "min"], "B": ["min", "max"]})[["A", "B"]].sort_index(),
        )

        self.assertRaises(KeyError, lambda: kdf.agg({"A": ["sum", "min"], "X": ["min", "max"]}))

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B"), ("Y", "C")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.agg(["sum", "min"])[[("X", "A"), ("X", "B"), ("Y", "C")]].sort_index(),
            pdf.agg(["sum", "min"])[[("X", "A"), ("X", "B"), ("Y", "C")]].sort_index(),
        )
        self.assert_eq(
            kdf.agg({("X", "A"): ["sum", "min"], ("X", "B"): ["min", "max"]})[
                [("X", "A"), ("X", "B")]
            ].sort_index(),
            pdf.agg({("X", "A"): ["sum", "min"], ("X", "B"): ["min", "max"]})[
                [("X", "A"), ("X", "B")]
            ].sort_index(),
        )

        self.assertRaises(TypeError, lambda: kdf.agg({"X": ["sum", "min"], "Y": ["min", "max"]}))

        # non-string names
        pdf = pd.DataFrame(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [np.nan, np.nan, np.nan]], columns=[10, 20, 30]
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.agg(["sum", "min"])[[10, 20, 30]].sort_index(),
            pdf.agg(["sum", "min"])[[10, 20, 30]].sort_index(),
        )
        self.assert_eq(
            kdf.agg({10: ["sum", "min"], 20: ["min", "max"]})[[10, 20]].sort_index(),
            pdf.agg({10: ["sum", "min"], 20: ["min", "max"]})[[10, 20]].sort_index(),
        )

        columns = pd.MultiIndex.from_tuples([("X", 10), ("X", 20), ("Y", 30)])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.agg(["sum", "min"])[[("X", 10), ("X", 20), ("Y", 30)]].sort_index(),
            pdf.agg(["sum", "min"])[[("X", 10), ("X", 20), ("Y", 30)]].sort_index(),
        )
        self.assert_eq(
            kdf.agg({("X", 10): ["sum", "min"], ("X", 20): ["min", "max"]})[
                [("X", 10), ("X", 20)]
            ].sort_index(),
            pdf.agg({("X", 10): ["sum", "min"], ("X", 20): ["min", "max"]})[
                [("X", 10), ("X", 20)]
            ].sort_index(),
        )

    def test_droplevel(self):
        pdf = (
            pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
            .set_index([0, 1])
            .rename_axis(["a", "b"])
        )
        pdf.columns = pd.MultiIndex.from_tuples(
            [("c", "e"), ("d", "f")], names=["level_1", "level_2"]
        )
        kdf = ks.from_pandas(pdf)

        self.assertRaises(ValueError, lambda: kdf.droplevel(["a", "b"]))
        self.assertRaises(ValueError, lambda: kdf.droplevel([1, 1, 1, 1, 1]))
        self.assertRaises(IndexError, lambda: kdf.droplevel(2))
        self.assertRaises(IndexError, lambda: kdf.droplevel(-3))
        self.assertRaises(KeyError, lambda: kdf.droplevel({"a"}))
        self.assertRaises(KeyError, lambda: kdf.droplevel({"a": 1}))

        self.assertRaises(ValueError, lambda: kdf.droplevel(["level_1", "level_2"], axis=1))
        self.assertRaises(IndexError, lambda: kdf.droplevel(2, axis=1))
        self.assertRaises(IndexError, lambda: kdf.droplevel(-3, axis=1))
        self.assertRaises(KeyError, lambda: kdf.droplevel({"level_1"}, axis=1))
        self.assertRaises(KeyError, lambda: kdf.droplevel({"level_1": 1}, axis=1))

        # droplevel is new in pandas 0.24.0
        if LooseVersion(pd.__version__) >= LooseVersion("0.24.0"):
            self.assert_eq(pdf.droplevel("a"), kdf.droplevel("a"))
            self.assert_eq(pdf.droplevel(["a"]), kdf.droplevel(["a"]))
            self.assert_eq(pdf.droplevel(("a",)), kdf.droplevel(("a",)))
            self.assert_eq(pdf.droplevel(0), kdf.droplevel(0))
            self.assert_eq(pdf.droplevel(-1), kdf.droplevel(-1))

            self.assert_eq(pdf.droplevel("level_1", axis=1), kdf.droplevel("level_1", axis=1))
            self.assert_eq(pdf.droplevel(["level_1"], axis=1), kdf.droplevel(["level_1"], axis=1))
            self.assert_eq(pdf.droplevel(("level_1",), axis=1), kdf.droplevel(("level_1",), axis=1))
            self.assert_eq(pdf.droplevel(0, axis=1), kdf.droplevel(0, axis=1))
            self.assert_eq(pdf.droplevel(-1, axis=1), kdf.droplevel(-1, axis=1))
        else:
            expected = pdf.copy()
            expected.index = expected.index.droplevel("a")

            self.assert_eq(expected, kdf.droplevel("a"))
            self.assert_eq(expected, kdf.droplevel(["a"]))
            self.assert_eq(expected, kdf.droplevel(("a",)))
            self.assert_eq(expected, kdf.droplevel(0))

            expected = pdf.copy()
            expected.index = expected.index.droplevel(-1)

            self.assert_eq(expected, kdf.droplevel(-1))

            expected = pdf.copy()
            expected.columns = expected.columns.droplevel("level_1")

            self.assert_eq(expected, kdf.droplevel("level_1", axis=1))
            self.assert_eq(expected, kdf.droplevel(["level_1"], axis=1))
            self.assert_eq(expected, kdf.droplevel(("level_1",), axis=1))
            self.assert_eq(expected, kdf.droplevel(0, axis=1))

            expected = pdf.copy()
            expected.columns = expected.columns.droplevel(-1)

            self.assert_eq(expected, kdf.droplevel(-1, axis=1))

        # Tupled names
        pdf.columns.names = [("level", 1), ("level", 2)]
        pdf.index.names = [("a", 10), ("x", 20)]
        kdf = ks.from_pandas(pdf)

        self.assertRaises(KeyError, lambda: kdf.droplevel("a"))
        self.assertRaises(KeyError, lambda: kdf.droplevel(("a", 10)))

        # droplevel is new in pandas 0.24.0
        if LooseVersion(pd.__version__) >= LooseVersion("0.24.0"):
            self.assert_eq(pdf.droplevel([("a", 10)]), kdf.droplevel([("a", 10)]))
            self.assert_eq(
                pdf.droplevel([("level", 1)], axis=1), kdf.droplevel([("level", 1)], axis=1)
            )
        else:
            expected = pdf.copy()
            expected.index = expected.index.droplevel([("a", 10)])

            self.assert_eq(expected, kdf.droplevel([("a", 10)]))

            expected = pdf.copy()
            expected.columns = expected.columns.droplevel([("level", 1)])

            self.assert_eq(expected, kdf.droplevel([("level", 1)], axis=1))

        # non-string names
        pdf = (
            pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
            .set_index([0, 1])
            .rename_axis([10.0, 20.0])
        )
        pdf.columns = pd.MultiIndex.from_tuples([("c", "e"), ("d", "f")], names=[100.0, 200.0])
        kdf = ks.from_pandas(pdf)

        # droplevel is new in pandas 0.24.0
        if LooseVersion(pd.__version__) >= LooseVersion("0.24.0"):
            self.assert_eq(pdf.droplevel(10.0), kdf.droplevel(10.0))
            self.assert_eq(pdf.droplevel([10.0]), kdf.droplevel([10.0]))
            self.assert_eq(pdf.droplevel((10.0,)), kdf.droplevel((10.0,)))
            self.assert_eq(pdf.droplevel(0), kdf.droplevel(0))
            self.assert_eq(pdf.droplevel(-1), kdf.droplevel(-1))
            self.assert_eq(pdf.droplevel(100.0, axis=1), kdf.droplevel(100.0, axis=1))
            self.assert_eq(pdf.droplevel(0, axis=1), kdf.droplevel(0, axis=1))
        else:
            expected = pdf.copy()
            expected.index = expected.index.droplevel(10.0)

            self.assert_eq(expected, kdf.droplevel(10.0))
            self.assert_eq(expected, kdf.droplevel([10.0]))
            self.assert_eq(expected, kdf.droplevel((10.0,)))
            self.assert_eq(expected, kdf.droplevel(0))

            expected = pdf.copy()
            expected.index = expected.index.droplevel(-1)
            self.assert_eq(expected, kdf.droplevel(-1))

            expected = pdf.copy()
            expected.columns = expected.columns.droplevel(100.0)

            self.assert_eq(expected, kdf.droplevel(100.0, axis=1))
            self.assert_eq(expected, kdf.droplevel(0, axis=1))

    def test_drop(self):
        pdf = pd.DataFrame({"x": [1, 2], "y": [3, 4], "z": [5, 6]}, index=np.random.rand(2))
        kdf = ks.from_pandas(pdf)

        # Assert 'labels' or 'columns' parameter is set
        expected_error_message = "Need to specify at least one of 'labels' or 'columns'"
        with self.assertRaisesRegex(ValueError, expected_error_message):
            kdf.drop()
        # Assert axis cannot be 0
        with self.assertRaisesRegex(NotImplementedError, "Drop currently only works for axis=1"):
            kdf.drop("x", axis=0)
        # Assert using a str for 'labels' works
        self.assert_eq(kdf.drop("x", axis=1), pdf.drop("x", axis=1))
        # Assert axis is 1 by default
        self.assert_eq(kdf.drop("x"), pdf.drop("x", axis=1))
        # Assert using a list for 'labels' works
        self.assert_eq(kdf.drop(["y", "z"], axis=1), pdf.drop(["y", "z"], axis=1))
        # Assert using 'columns' instead of 'labels' produces the same results
        self.assert_eq(kdf.drop(columns="x"), pdf.drop(columns="x"))
        self.assert_eq(kdf.drop(columns=["y", "z"]), pdf.drop(columns=["y", "z"]))

        # Assert 'labels' being used when both 'labels' and 'columns' are specified
        # TODO: should throw an error?
        expected_output = pd.DataFrame({"y": [3, 4], "z": [5, 6]}, index=kdf.index.to_pandas())
        self.assert_eq(kdf.drop(labels=["x"], columns=["y"]), expected_output)

        columns = pd.MultiIndex.from_tuples([(1, "x"), (1, "y"), (2, "z")])
        pdf.columns = columns
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.drop(columns=1), pdf.drop(columns=1))
        self.assert_eq(kdf.drop(columns=(1, "x")), pdf.drop(columns=(1, "x")))
        self.assert_eq(kdf.drop(columns=[(1, "x"), 2]), pdf.drop(columns=[(1, "x"), 2]))

        self.assertRaises(KeyError, lambda: kdf.drop(columns=3))
        self.assertRaises(KeyError, lambda: kdf.drop(columns=(1, "z")))

        # non-string names
        pdf = pd.DataFrame({10: [1, 2], 20: [3, 4], 30: [5, 6]}, index=np.random.rand(2))
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.drop(10), pdf.drop(10, axis=1))
        self.assert_eq(kdf.drop([20, 30]), pdf.drop([20, 30], axis=1))

    def _test_dropna(self, pdf, axis):
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.dropna(axis=axis), pdf.dropna(axis=axis))
        self.assert_eq(kdf.dropna(axis=axis, how="all"), pdf.dropna(axis=axis, how="all"))
        self.assert_eq(kdf.dropna(axis=axis, subset=["x"]), pdf.dropna(axis=axis, subset=["x"]))
        self.assert_eq(kdf.dropna(axis=axis, subset="x"), pdf.dropna(axis=axis, subset=["x"]))
        self.assert_eq(
            kdf.dropna(axis=axis, subset=["y", "z"]), pdf.dropna(axis=axis, subset=["y", "z"])
        )
        self.assert_eq(
            kdf.dropna(axis=axis, subset=["y", "z"], how="all"),
            pdf.dropna(axis=axis, subset=["y", "z"], how="all"),
        )

        self.assert_eq(kdf.dropna(axis=axis, thresh=2), pdf.dropna(axis=axis, thresh=2))
        self.assert_eq(
            kdf.dropna(axis=axis, thresh=1, subset=["y", "z"]),
            pdf.dropna(axis=axis, thresh=1, subset=["y", "z"]),
        )

        pdf2 = pdf.copy()
        kdf2 = kdf.copy()
        pser = pdf2[pdf2.columns[0]]
        kser = kdf2[kdf2.columns[0]]
        pdf2.dropna(inplace=True)
        kdf2.dropna(inplace=True)
        self.assert_eq(kdf2, pdf2)
        self.assert_eq(kser, pser)

        # multi-index
        columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y"), ("b", "z")])
        if axis == 0:
            pdf.columns = columns
        else:
            pdf.index = columns
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.dropna(axis=axis), pdf.dropna(axis=axis))
        self.assert_eq(kdf.dropna(axis=axis, how="all"), pdf.dropna(axis=axis, how="all"))
        self.assert_eq(
            kdf.dropna(axis=axis, subset=[("a", "x")]), pdf.dropna(axis=axis, subset=[("a", "x")])
        )
        self.assert_eq(
            kdf.dropna(axis=axis, subset=("a", "x")), pdf.dropna(axis=axis, subset=[("a", "x")])
        )
        self.assert_eq(
            kdf.dropna(axis=axis, subset=[("a", "y"), ("b", "z")]),
            pdf.dropna(axis=axis, subset=[("a", "y"), ("b", "z")]),
        )
        self.assert_eq(
            kdf.dropna(axis=axis, subset=[("a", "y"), ("b", "z")], how="all"),
            pdf.dropna(axis=axis, subset=[("a", "y"), ("b", "z")], how="all"),
        )

        self.assert_eq(kdf.dropna(axis=axis, thresh=2), pdf.dropna(axis=axis, thresh=2))
        self.assert_eq(
            kdf.dropna(axis=axis, thresh=1, subset=[("a", "y"), ("b", "z")]),
            pdf.dropna(axis=axis, thresh=1, subset=[("a", "y"), ("b", "z")]),
        )

    def test_dropna_axis_index(self):
        pdf = pd.DataFrame(
            {
                "x": [np.nan, 2, 3, 4, np.nan, 6],
                "y": [1, 2, np.nan, 4, np.nan, np.nan],
                "z": [1, 2, 3, 4, np.nan, np.nan],
            },
            index=np.random.rand(6),
        )
        kdf = ks.from_pandas(pdf)

        self._test_dropna(pdf, axis=0)

        # empty
        pdf = pd.DataFrame(index=np.random.rand(6))
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.dropna(), pdf.dropna())
        self.assert_eq(kdf.dropna(how="all"), pdf.dropna(how="all"))
        self.assert_eq(kdf.dropna(thresh=0), pdf.dropna(thresh=0))
        self.assert_eq(kdf.dropna(thresh=1), pdf.dropna(thresh=1))

        with self.assertRaisesRegex(ValueError, "No axis named foo"):
            kdf.dropna(axis="foo")

        self.assertRaises(KeyError, lambda: kdf.dropna(subset="1"))
        with self.assertRaisesRegex(ValueError, "invalid how option: 1"):
            kdf.dropna(how=1)
        with self.assertRaisesRegex(TypeError, "must specify how or thresh"):
            kdf.dropna(how=None)

    def test_dropna_axis_column(self):
        pdf = pd.DataFrame(
            {
                "x": [np.nan, 2, 3, 4, np.nan, 6],
                "y": [1, 2, np.nan, 4, np.nan, np.nan],
                "z": [1, 2, 3, 4, np.nan, np.nan],
            },
            index=[str(r) for r in np.random.rand(6)],
        ).T

        self._test_dropna(pdf, axis=1)

        # empty
        pdf = pd.DataFrame({"x": [], "y": [], "z": []})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.dropna(axis=1), pdf.dropna(axis=1))
        self.assert_eq(kdf.dropna(axis=1, how="all"), pdf.dropna(axis=1, how="all"))
        self.assert_eq(kdf.dropna(axis=1, thresh=0), pdf.dropna(axis=1, thresh=0))
        self.assert_eq(kdf.dropna(axis=1, thresh=1), pdf.dropna(axis=1, thresh=1))

    def test_dtype(self):
        pdf = pd.DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("i1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("20130101", periods=3),
            },
            index=np.random.rand(3),
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf, pdf)
        self.assertTrue((kdf.dtypes == pdf.dtypes).all())

        # multi-index columns
        columns = pd.MultiIndex.from_tuples(zip(list("xxxyyz"), list("abcdef")))
        pdf.columns = columns
        kdf.columns = columns
        self.assertTrue((kdf.dtypes == pdf.dtypes).all())

    def test_fillna(self):
        pdf = pd.DataFrame(
            {
                "x": [np.nan, 2, 3, 4, np.nan, 6],
                "y": [1, 2, np.nan, 4, np.nan, np.nan],
                "z": [1, 2, 3, 4, np.nan, np.nan],
            },
            index=np.random.rand(6),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf, pdf)
        self.assert_eq(kdf.fillna(-1), pdf.fillna(-1))
        self.assert_eq(
            kdf.fillna({"x": -1, "y": -2, "z": -5}), pdf.fillna({"x": -1, "y": -2, "z": -5})
        )
        self.assert_eq(pdf.fillna(method="ffill"), kdf.fillna(method="ffill"))
        self.assert_eq(pdf.fillna(method="ffill", limit=2), kdf.fillna(method="ffill", limit=2))
        self.assert_eq(pdf.fillna(method="bfill"), kdf.fillna(method="bfill"))
        self.assert_eq(pdf.fillna(method="bfill", limit=2), kdf.fillna(method="bfill", limit=2))

        pdf = pdf.set_index(["x", "y"])
        kdf = ks.from_pandas(pdf)
        # check multi index
        self.assert_eq(kdf.fillna(-1), pdf.fillna(-1))
        self.assert_eq(pdf.fillna(method="bfill"), kdf.fillna(method="bfill"))
        self.assert_eq(pdf.fillna(method="ffill"), kdf.fillna(method="ffill"))

        pser = pdf.z
        kser = kdf.z
        pdf.fillna({"x": -1, "y": -2, "z": -5}, inplace=True)
        kdf.fillna({"x": -1, "y": -2, "z": -5}, inplace=True)
        self.assert_eq(kdf, pdf)
        self.assert_eq(kser, pser)

        s_nan = pd.Series([-1, -2, -5], index=["x", "y", "z"], dtype=int)
        self.assert_eq(kdf.fillna(s_nan), pdf.fillna(s_nan))

        with self.assertRaisesRegex(NotImplementedError, "fillna currently only"):
            kdf.fillna(-1, axis=1)
        with self.assertRaisesRegex(NotImplementedError, "fillna currently only"):
            kdf.fillna(-1, axis="columns")
        with self.assertRaisesRegex(ValueError, "limit parameter for value is not support now"):
            kdf.fillna(-1, limit=1)
        with self.assertRaisesRegex(TypeError, "Unsupported.*DataFrame"):
            kdf.fillna(pd.DataFrame({"x": [-1], "y": [-1], "z": [-1]}))
        with self.assertRaisesRegex(TypeError, "Unsupported.*numpy.int64"):
            kdf.fillna({"x": np.int64(-6), "y": np.int64(-4), "z": -5})
        with self.assertRaisesRegex(ValueError, "Expecting 'pad', 'ffill', 'backfill' or 'bfill'."):
            kdf.fillna(method="xxx")
        with self.assertRaisesRegex(
            ValueError, "Must specify a fillna 'value' or 'method' parameter."
        ):
            kdf.fillna()

        # multi-index columns
        pdf = pd.DataFrame(
            {
                ("x", "a"): [np.nan, 2, 3, 4, np.nan, 6],
                ("x", "b"): [1, 2, np.nan, 4, np.nan, np.nan],
                ("y", "c"): [1, 2, 3, 4, np.nan, np.nan],
            },
            index=np.random.rand(6),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.fillna(-1), pdf.fillna(-1))
        self.assert_eq(
            kdf.fillna({("x", "a"): -1, ("x", "b"): -2, ("y", "c"): -5}),
            pdf.fillna({("x", "a"): -1, ("x", "b"): -2, ("y", "c"): -5}),
        )
        self.assert_eq(pdf.fillna(method="ffill"), kdf.fillna(method="ffill"))
        self.assert_eq(pdf.fillna(method="ffill", limit=2), kdf.fillna(method="ffill", limit=2))
        self.assert_eq(pdf.fillna(method="bfill"), kdf.fillna(method="bfill"))
        self.assert_eq(pdf.fillna(method="bfill", limit=2), kdf.fillna(method="bfill", limit=2))

        self.assert_eq(kdf.fillna({"x": -1}), pdf.fillna({"x": -1}))

        if sys.version_info >= (3, 6):
            # flaky in Python 3.5.
            self.assert_eq(
                kdf.fillna({"x": -1, ("x", "b"): -2}), pdf.fillna({"x": -1, ("x", "b"): -2})
            )
            self.assert_eq(
                kdf.fillna({("x", "b"): -2, "x": -1}), pdf.fillna({("x", "b"): -2, "x": -1})
            )

        # check multi index
        pdf = pdf.set_index([("x", "a"), ("x", "b")])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.fillna(-1), pdf.fillna(-1))
        self.assert_eq(
            kdf.fillna({("x", "a"): -1, ("x", "b"): -2, ("y", "c"): -5}),
            pdf.fillna({("x", "a"): -1, ("x", "b"): -2, ("y", "c"): -5}),
        )

    def test_isnull(self):
        pdf = pd.DataFrame(
            {"x": [1, 2, 3, 4, None, 6], "y": list("abdabd")}, index=np.random.rand(6)
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.notnull(), pdf.notnull())
        self.assert_eq(kdf.isnull(), pdf.isnull())

    def test_to_datetime(self):
        pdf = pd.DataFrame(
            {"year": [2015, 2016], "month": [2, 3], "day": [4, 5]}, index=np.random.rand(2)
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pd.to_datetime(pdf), ks.to_datetime(kdf))

    def test_nunique(self):
        pdf = pd.DataFrame({"A": [1, 2, 3], "B": [np.nan, 3, np.nan]}, index=np.random.rand(3))
        kdf = ks.from_pandas(pdf)

        # Assert NaNs are dropped by default
        self.assert_eq(kdf.nunique(), pdf.nunique())

        # Assert including NaN values
        self.assert_eq(kdf.nunique(dropna=False), pdf.nunique(dropna=False))

        # Assert approximate counts
        self.assert_eq(
            ks.DataFrame({"A": range(100)}).nunique(approx=True), pd.Series([103], index=["A"]),
        )
        self.assert_eq(
            ks.DataFrame({"A": range(100)}).nunique(approx=True, rsd=0.01),
            pd.Series([100], index=["A"]),
        )

        # Assert unsupported axis value yet
        msg = 'axis should be either 0 or "index" currently.'
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.nunique(axis=1)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("X", "A"), ("Y", "B")], names=["1", "2"])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.nunique(), pdf.nunique())
        self.assert_eq(kdf.nunique(dropna=False), pdf.nunique(dropna=False))

    def test_sort_values(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, None, 7], "b": [7, 6, 5, 4, 3, 2, 1]}, index=np.random.rand(7)
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.sort_values("b"), pdf.sort_values("b"))
        self.assert_eq(kdf.sort_values(["b", "a"]), pdf.sort_values(["b", "a"]))
        self.assert_eq(
            kdf.sort_values(["b", "a"], ascending=[False, True]),
            pdf.sort_values(["b", "a"], ascending=[False, True]),
        )

        self.assertRaises(ValueError, lambda: kdf.sort_values(["b", "a"], ascending=[False]))

        self.assert_eq(
            kdf.sort_values(["b", "a"], na_position="first"),
            pdf.sort_values(["b", "a"], na_position="first"),
        )

        self.assertRaises(ValueError, lambda: kdf.sort_values(["b", "a"], na_position="invalid"))

        pserA = pdf.a
        kserA = kdf.a
        self.assert_eq(kdf.sort_values("b", inplace=True), pdf.sort_values("b", inplace=True))
        self.assert_eq(kdf, pdf)
        self.assert_eq(kserA, pserA)

        # multi-index columns
        pdf = pd.DataFrame(
            {("X", 10): [1, 2, 3, 4, 5, None, 7], ("X", 20): [7, 6, 5, 4, 3, 2, 1]},
            index=np.random.rand(7),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.sort_values(("X", 20)), pdf.sort_values(("X", 20)))
        self.assert_eq(
            kdf.sort_values([("X", 20), ("X", 10)]), pdf.sort_values([("X", 20), ("X", 10)])
        )

        self.assertRaisesRegex(
            ValueError,
            "For a multi-index, the label must be a tuple with elements",
            lambda: kdf.sort_values(["X"]),
        )

        # non-string names
        pdf = pd.DataFrame(
            {10: [1, 2, 3, 4, 5, None, 7], 20: [7, 6, 5, 4, 3, 2, 1]}, index=np.random.rand(7)
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.sort_values(20), pdf.sort_values(20))
        self.assert_eq(kdf.sort_values([20, 10]), pdf.sort_values([20, 10]))

    def test_sort_index(self):
        pdf = pd.DataFrame(
            {"A": [2, 1, np.nan], "B": [np.nan, 0, np.nan]}, index=["b", "a", np.nan]
        )
        kdf = ks.from_pandas(pdf)

        # Assert invalid parameters
        self.assertRaises(NotImplementedError, lambda: kdf.sort_index(axis=1))
        self.assertRaises(NotImplementedError, lambda: kdf.sort_index(kind="mergesort"))
        self.assertRaises(ValueError, lambda: kdf.sort_index(na_position="invalid"))

        # Assert default behavior without parameters
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        # Assert sorting descending
        self.assert_eq(kdf.sort_index(ascending=False), pdf.sort_index(ascending=False))
        # Assert sorting NA indices first
        self.assert_eq(kdf.sort_index(na_position="first"), pdf.sort_index(na_position="first"))

        # Assert sorting inplace
        pserA = pdf.A
        kserA = kdf.A
        self.assertEqual(kdf.sort_index(inplace=True), pdf.sort_index(inplace=True))
        self.assert_eq(kdf, pdf)
        self.assert_eq(kserA, pserA)

        # Assert multi-indices
        pdf = pd.DataFrame(
            {"A": range(4), "B": range(4)[::-1]}, index=[["b", "b", "a", "a"], [1, 0, 1, 0]]
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        self.assert_eq(kdf.sort_index(level=[1, 0]), pdf.sort_index(level=[1, 0]))
        self.assert_eq(kdf.reset_index().sort_index(), pdf.reset_index().sort_index())

        # Assert with multi-index columns
        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_nlargest(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, None, 7], "b": [7, 6, 5, 4, 3, 2, 1]}, index=np.random.rand(7)
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.nlargest(n=5, columns="a"), pdf.nlargest(5, columns="a"))
        self.assert_eq(kdf.nlargest(n=5, columns=["a", "b"]), pdf.nlargest(5, columns=["a", "b"]))

    def test_nsmallest(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, None, 7], "b": [7, 6, 5, 4, 3, 2, 1]}, index=np.random.rand(7)
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.nsmallest(n=5, columns="a"), pdf.nsmallest(5, columns="a"))
        self.assert_eq(kdf.nsmallest(n=5, columns=["a", "b"]), pdf.nsmallest(5, columns=["a", "b"]))

    def test_xs(self):
        d = {
            "num_legs": [4, 4, 2, 2],
            "num_wings": [0, 0, 2, 2],
            "class": ["mammal", "mammal", "mammal", "bird"],
            "animal": ["cat", "dog", "bat", "penguin"],
            "locomotion": ["walks", "walks", "flies", "walks"],
        }
        pdf = pd.DataFrame(data=d)
        pdf = pdf.set_index(["class", "animal", "locomotion"])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.xs("mammal"), pdf.xs("mammal"))
        self.assert_eq(kdf.xs(("mammal",)), pdf.xs(("mammal",)))
        self.assert_eq(kdf.xs(("mammal", "dog", "walks")), pdf.xs(("mammal", "dog", "walks")))
        self.assert_eq(kdf.xs("cat", level=1), pdf.xs("cat", level=1))

        msg = 'axis should be either 0 or "index" currently.'
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.xs("num_wings", axis=1)
        msg = r"'Key length \(4\) exceeds index depth \(3\)'"
        with self.assertRaisesRegex(KeyError, msg):
            kdf.xs(("mammal", "dog", "walks", "foo"))

        self.assertRaises(KeyError, lambda: kdf.xs(("dog", "walks"), level=1))

        # non-string names
        pdf = pd.DataFrame(data=d)
        pdf = pdf.set_index(["class", "animal", "num_legs", "num_wings"])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.xs(("mammal", "dog", 4)), pdf.xs(("mammal", "dog", 4)))
        self.assert_eq(kdf.xs(2, level=2), pdf.xs(2, level=2))

    def test_missing(self):
        kdf = self.kdf

        missing_functions = inspect.getmembers(_MissingPandasLikeDataFrame, inspect.isfunction)
        unsupported_functions = [
            name for (name, type_) in missing_functions if type_.__name__ == "unsupported_function"
        ]
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                PandasNotImplementedError,
                "method.*DataFrame.*{}.*not implemented( yet\\.|\\. .+)".format(name),
            ):
                getattr(kdf, name)()

        deprecated_functions = [
            name for (name, type_) in missing_functions if type_.__name__ == "deprecated_function"
        ]
        for name in deprecated_functions:
            with self.assertRaisesRegex(
                PandasNotImplementedError, "method.*DataFrame.*{}.*is deprecated".format(name)
            ):
                getattr(kdf, name)()

        missing_properties = inspect.getmembers(
            _MissingPandasLikeDataFrame, lambda o: isinstance(o, property)
        )
        unsupported_properties = [
            name
            for (name, type_) in missing_properties
            if type_.fget.__name__ == "unsupported_property"
        ]
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                PandasNotImplementedError,
                "property.*DataFrame.*{}.*not implemented( yet\\.|\\. .+)".format(name),
            ):
                getattr(kdf, name)
        deprecated_properties = [
            name
            for (name, type_) in missing_properties
            if type_.fget.__name__ == "deprecated_property"
        ]
        for name in deprecated_properties:
            with self.assertRaisesRegex(
                PandasNotImplementedError, "property.*DataFrame.*{}.*is deprecated".format(name)
            ):
                getattr(kdf, name)

    def test_to_numpy(self):
        pdf = pd.DataFrame(
            {
                "a": [4, 2, 3, 4, 8, 6],
                "b": [1, 2, 9, 4, 2, 4],
                "c": ["one", "three", "six", "seven", "one", "5"],
            },
            index=np.random.rand(6),
        )

        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.to_numpy(), pdf.values)

    def test_to_pandas(self):
        pdf, kdf = self.df_pair
        self.assert_eq(kdf.toPandas(), pdf)
        self.assert_eq(kdf.to_pandas(), pdf)

    def test_isin(self):
        pdf = pd.DataFrame(
            {
                "a": [4, 2, 3, 4, 8, 6],
                "b": [1, 2, 9, 4, 2, 4],
                "c": ["one", "three", "six", "seven", "one", "5"],
            },
            index=np.random.rand(6),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.isin([4, "six"]), pdf.isin([4, "six"]))
        self.assert_eq(
            kdf.isin({"a": [2, 8], "c": ["three", "one"]}),
            pdf.isin({"a": [2, 8], "c": ["three", "one"]}),
        )

        msg = "'DataFrame' object has no attribute {'e'}"
        with self.assertRaisesRegex(AttributeError, msg):
            kdf.isin({"e": [5, 7], "a": [1, 6]})

        msg = "DataFrame and Series are not supported"
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.isin(pdf)

        msg = "Values should be iterable, Series, DataFrame or dict."
        with self.assertRaisesRegex(TypeError, msg):
            kdf.isin(1)

    def test_merge(self):
        left_pdf = pd.DataFrame(
            {
                "lkey": ["foo", "bar", "baz", "foo", "bar", "l"],
                "value": [1, 2, 3, 5, 6, 7],
                "x": list("abcdef"),
            },
            columns=["lkey", "value", "x"],
        )
        right_pdf = pd.DataFrame(
            {
                "rkey": ["baz", "foo", "bar", "baz", "foo", "r"],
                "value": [4, 5, 6, 7, 8, 9],
                "y": list("efghij"),
            },
            columns=["rkey", "value", "y"],
        )
        right_ps = pd.Series(list("defghi"), name="x", index=[5, 6, 7, 8, 9, 10])

        left_kdf = ks.from_pandas(left_pdf)
        right_kdf = ks.from_pandas(right_pdf)
        right_kser = ks.from_pandas(right_ps)

        def check(op, right_kdf=right_kdf, right_pdf=right_pdf):
            k_res = op(left_kdf, right_kdf)
            k_res = k_res.to_pandas()
            k_res = k_res.sort_values(by=list(k_res.columns))
            k_res = k_res.reset_index(drop=True)
            p_res = op(left_pdf, right_pdf)
            p_res = p_res.sort_values(by=list(p_res.columns))
            p_res = p_res.reset_index(drop=True)
            self.assert_eq(k_res, p_res)

        check(lambda left, right: left.merge(right))
        check(lambda left, right: left.merge(right, on="value"))
        check(lambda left, right: left.merge(right, left_on="lkey", right_on="rkey"))
        check(lambda left, right: left.set_index("lkey").merge(right.set_index("rkey")))
        check(
            lambda left, right: left.set_index("lkey").merge(
                right, left_index=True, right_on="rkey"
            )
        )
        check(
            lambda left, right: left.merge(
                right.set_index("rkey"), left_on="lkey", right_index=True
            )
        )
        check(
            lambda left, right: left.set_index("lkey").merge(
                right.set_index("rkey"), left_index=True, right_index=True
            )
        )

        # MultiIndex
        check(
            lambda left, right: left.merge(
                right, left_on=["lkey", "value"], right_on=["rkey", "value"]
            )
        )
        check(
            lambda left, right: left.set_index(["lkey", "value"]).merge(
                right, left_index=True, right_on=["rkey", "value"]
            )
        )
        check(
            lambda left, right: left.merge(
                right.set_index(["rkey", "value"]), left_on=["lkey", "value"], right_index=True
            )
        )
        # TODO: when both left_index=True and right_index=True with multi-index
        # check(lambda left, right: left.set_index(['lkey', 'value']).merge(
        #     right.set_index(['rkey', 'value']), left_index=True, right_index=True))

        # join types
        for how in ["inner", "left", "right", "outer"]:
            check(lambda left, right: left.merge(right, on="value", how=how))
            check(lambda left, right: left.merge(right, left_on="lkey", right_on="rkey", how=how))

        # suffix
        check(
            lambda left, right: left.merge(
                right, left_on="lkey", right_on="rkey", suffixes=["_left", "_right"]
            )
        )

        # Test Series on the right
        # pd.DataFrame.merge with Series is implemented since version 0.24.0
        if LooseVersion(pd.__version__) >= LooseVersion("0.24.0"):
            check(lambda left, right: left.merge(right), right_kser, right_ps)
            check(
                lambda left, right: left.merge(right, left_on="x", right_on="x"),
                right_kser,
                right_ps,
            )
            check(
                lambda left, right: left.set_index("x").merge(right, left_index=True, right_on="x"),
                right_kser,
                right_ps,
            )

            # Test join types with Series
            for how in ["inner", "left", "right", "outer"]:
                check(lambda left, right: left.merge(right, how=how), right_kser, right_ps)
                check(
                    lambda left, right: left.merge(right, left_on="x", right_on="x", how=how),
                    right_kser,
                    right_ps,
                )

            # suffix with Series
            check(
                lambda left, right: left.merge(
                    right,
                    suffixes=["_left", "_right"],
                    how="outer",
                    left_index=True,
                    right_index=True,
                ),
                right_kser,
                right_ps,
            )

        # multi-index columns
        left_columns = pd.MultiIndex.from_tuples([(10, "lkey"), (10, "value"), (20, "x")])
        left_pdf.columns = left_columns
        left_kdf.columns = left_columns

        right_columns = pd.MultiIndex.from_tuples([(10, "rkey"), (10, "value"), (30, "y")])
        right_pdf.columns = right_columns
        right_kdf.columns = right_columns

        check(lambda left, right: left.merge(right))
        check(lambda left, right: left.merge(right, on=[(10, "value")]))
        check(
            lambda left, right: (left.set_index((10, "lkey")).merge(right.set_index((10, "rkey"))))
        )
        check(
            lambda left, right: (
                left.set_index((10, "lkey")).merge(
                    right.set_index((10, "rkey")), left_index=True, right_index=True
                )
            )
        )
        # TODO: when both left_index=True and right_index=True with multi-index columns
        # check(lambda left, right: left.merge(right,
        #                                      left_on=[('a', 'lkey')], right_on=[('a', 'rkey')]))
        # check(lambda left, right: (left.set_index(('a', 'lkey'))
        #                            .merge(right, left_index=True, right_on=[('a', 'rkey')])))

        # non-string names
        left_pdf.columns = [10, 100, 1000]
        left_kdf.columns = [10, 100, 1000]

        right_pdf.columns = [20, 100, 2000]
        right_kdf.columns = [20, 100, 2000]

        check(lambda left, right: left.merge(right))
        check(lambda left, right: left.merge(right, on=[100]))
        check(lambda left, right: (left.set_index(10).merge(right.set_index(20))))
        check(
            lambda left, right: (
                left.set_index(10).merge(right.set_index(20), left_index=True, right_index=True)
            )
        )

    def test_merge_retains_indices(self):
        left_pdf = pd.DataFrame({"A": [0, 1]})
        right_pdf = pd.DataFrame({"B": [1, 2]}, index=[1, 2])
        left_kdf = ks.from_pandas(left_pdf)
        right_kdf = ks.from_pandas(right_pdf)

        self.assert_eq(
            left_kdf.merge(right_kdf, left_index=True, right_index=True),
            left_pdf.merge(right_pdf, left_index=True, right_index=True),
        )
        self.assert_eq(
            left_kdf.merge(right_kdf, left_on="A", right_index=True),
            left_pdf.merge(right_pdf, left_on="A", right_index=True),
        )
        self.assert_eq(
            left_kdf.merge(right_kdf, left_index=True, right_on="B"),
            left_pdf.merge(right_pdf, left_index=True, right_on="B"),
        )
        self.assert_eq(
            left_kdf.merge(right_kdf, left_on="A", right_on="B"),
            left_pdf.merge(right_pdf, left_on="A", right_on="B"),
        )

    def test_merge_how_parameter(self):
        left_pdf = pd.DataFrame({"A": [1, 2]})
        right_pdf = pd.DataFrame({"B": ["x", "y"]}, index=[1, 2])
        left_kdf = ks.from_pandas(left_pdf)
        right_kdf = ks.from_pandas(right_pdf)

        kdf = left_kdf.merge(right_kdf, left_index=True, right_index=True)
        pdf = left_pdf.merge(right_pdf, left_index=True, right_index=True)
        self.assert_eq(
            kdf.sort_values(by=list(kdf.columns)).reset_index(drop=True),
            pdf.sort_values(by=list(pdf.columns)).reset_index(drop=True),
        )

        kdf = left_kdf.merge(right_kdf, left_index=True, right_index=True, how="left")
        pdf = left_pdf.merge(right_pdf, left_index=True, right_index=True, how="left")
        self.assert_eq(
            kdf.sort_values(by=list(kdf.columns)).reset_index(drop=True),
            pdf.sort_values(by=list(pdf.columns)).reset_index(drop=True),
        )

        kdf = left_kdf.merge(right_kdf, left_index=True, right_index=True, how="right")
        pdf = left_pdf.merge(right_pdf, left_index=True, right_index=True, how="right")
        self.assert_eq(
            kdf.sort_values(by=list(kdf.columns)).reset_index(drop=True),
            pdf.sort_values(by=list(pdf.columns)).reset_index(drop=True),
        )

        kdf = left_kdf.merge(right_kdf, left_index=True, right_index=True, how="outer")
        pdf = left_pdf.merge(right_pdf, left_index=True, right_index=True, how="outer")
        self.assert_eq(
            kdf.sort_values(by=list(kdf.columns)).reset_index(drop=True),
            pdf.sort_values(by=list(pdf.columns)).reset_index(drop=True),
        )

    def test_merge_raises(self):
        left = ks.DataFrame(
            {"value": [1, 2, 3, 5, 6], "x": list("abcde")},
            columns=["value", "x"],
            index=["foo", "bar", "baz", "foo", "bar"],
        )
        right = ks.DataFrame(
            {"value": [4, 5, 6, 7, 8], "y": list("fghij")},
            columns=["value", "y"],
            index=["baz", "foo", "bar", "baz", "foo"],
        )

        with self.assertRaisesRegex(ValueError, "No common columns to perform merge on"):
            left[["x"]].merge(right[["y"]])

        with self.assertRaisesRegex(ValueError, "not a combination of both"):
            left.merge(right, on="value", left_on="x")

        with self.assertRaisesRegex(ValueError, "Must pass right_on or right_index=True"):
            left.merge(right, left_on="x")

        with self.assertRaisesRegex(ValueError, "Must pass right_on or right_index=True"):
            left.merge(right, left_index=True)

        with self.assertRaisesRegex(ValueError, "Must pass left_on or left_index=True"):
            left.merge(right, right_on="y")

        with self.assertRaisesRegex(ValueError, "Must pass left_on or left_index=True"):
            left.merge(right, right_index=True)

        with self.assertRaisesRegex(
            ValueError, "len\\(left_keys\\) must equal len\\(right_keys\\)"
        ):
            left.merge(right, left_on="value", right_on=["value", "y"])

        with self.assertRaisesRegex(
            ValueError, "len\\(left_keys\\) must equal len\\(right_keys\\)"
        ):
            left.merge(right, left_on=["value", "x"], right_on="value")

        with self.assertRaisesRegex(ValueError, "['inner', 'left', 'right', 'full', 'outer']"):
            left.merge(right, left_index=True, right_index=True, how="foo")

        with self.assertRaisesRegex(KeyError, "id"):
            left.merge(right, on="id")

    def test_append(self):
        pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"))
        kdf = ks.from_pandas(pdf)
        other_pdf = pd.DataFrame([[3, 4], [5, 6]], columns=list("BC"), index=[2, 3])
        other_kdf = ks.from_pandas(other_pdf)

        self.assert_eq(kdf.append(kdf), pdf.append(pdf))
        self.assert_eq(kdf.append(kdf, ignore_index=True), pdf.append(pdf, ignore_index=True))

        # Assert DataFrames with non-matching columns
        self.assert_eq(kdf.append(other_kdf), pdf.append(other_pdf))

        # Assert appending a Series fails
        msg = "DataFrames.append() does not support appending Series to DataFrames"
        with self.assertRaises(ValueError, msg=msg):
            kdf.append(kdf["A"])

        # Assert using the sort parameter raises an exception
        msg = "The 'sort' parameter is currently not supported"
        with self.assertRaises(NotImplementedError, msg=msg):
            kdf.append(kdf, sort=True)

        # Assert using 'verify_integrity' only raises an exception for overlapping indices
        self.assert_eq(
            kdf.append(other_kdf, verify_integrity=True),
            pdf.append(other_pdf, verify_integrity=True),
        )
        msg = "Indices have overlapping values"
        with self.assertRaises(ValueError, msg=msg):
            kdf.append(kdf, verify_integrity=True)

        # Skip integrity verification when ignore_index=True
        self.assert_eq(
            kdf.append(kdf, ignore_index=True, verify_integrity=True),
            pdf.append(pdf, ignore_index=True, verify_integrity=True),
        )

        # Assert appending multi-index DataFrames
        multi_index_pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list("AB"), index=[[2, 3], [4, 5]])
        multi_index_kdf = ks.from_pandas(multi_index_pdf)
        other_multi_index_pdf = pd.DataFrame(
            [[5, 6], [7, 8]], columns=list("AB"), index=[[2, 3], [6, 7]]
        )
        other_multi_index_kdf = ks.from_pandas(other_multi_index_pdf)

        self.assert_eq(
            multi_index_kdf.append(multi_index_kdf), multi_index_pdf.append(multi_index_pdf)
        )

        # Assert DataFrames with non-matching columns
        self.assert_eq(
            multi_index_kdf.append(other_multi_index_kdf),
            multi_index_pdf.append(other_multi_index_pdf),
        )

        # Assert using 'verify_integrity' only raises an exception for overlapping indices
        self.assert_eq(
            multi_index_kdf.append(other_multi_index_kdf, verify_integrity=True),
            multi_index_pdf.append(other_multi_index_pdf, verify_integrity=True),
        )
        with self.assertRaises(ValueError, msg=msg):
            multi_index_kdf.append(multi_index_kdf, verify_integrity=True)

        # Skip integrity verification when ignore_index=True
        self.assert_eq(
            multi_index_kdf.append(multi_index_kdf, ignore_index=True, verify_integrity=True),
            multi_index_pdf.append(multi_index_pdf, ignore_index=True, verify_integrity=True),
        )

        # Assert trying to append DataFrames with different index levels
        msg = "Both DataFrames have to have the same number of index levels"
        with self.assertRaises(ValueError, msg=msg):
            kdf.append(multi_index_kdf)

        # Skip index level check when ignore_index=True
        self.assert_eq(
            kdf.append(multi_index_kdf, ignore_index=True),
            pdf.append(multi_index_pdf, ignore_index=True),
        )

        columns = pd.MultiIndex.from_tuples([("A", "X"), ("A", "Y")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.append(kdf), pdf.append(pdf))

    def test_clip(self):
        pdf = pd.DataFrame(
            {"A": [0, 2, 4], "B": [4, 2, 0], "X": [-1, 10, 0]}, index=np.random.rand(3)
        )
        kdf = ks.from_pandas(pdf)

        # Assert list-like values are not accepted for 'lower' and 'upper'
        msg = "List-like value are not supported for 'lower' and 'upper' at the moment"
        with self.assertRaises(ValueError, msg=msg):
            kdf.clip(lower=[1])
        with self.assertRaises(ValueError, msg=msg):
            kdf.clip(upper=[1])

        # Assert no lower or upper
        self.assert_eq(kdf.clip(), pdf.clip())
        # Assert lower only
        self.assert_eq(kdf.clip(1), pdf.clip(1))
        # Assert upper only
        self.assert_eq(kdf.clip(upper=3), pdf.clip(upper=3))
        # Assert lower and upper
        self.assert_eq(kdf.clip(1, 3), pdf.clip(1, 3))

        pdf["clip"] = pdf.A.clip(lower=1, upper=3)
        kdf["clip"] = kdf.A.clip(lower=1, upper=3)
        self.assert_eq(kdf, pdf)

        # Assert behavior on string values
        str_kdf = ks.DataFrame({"A": ["a", "b", "c"]}, index=np.random.rand(3))
        self.assert_eq(str_kdf.clip(1, 3), str_kdf)

    def test_binary_operators(self):
        pdf = pd.DataFrame(
            {"A": [0, 2, 4], "B": [4, 2, 0], "X": [-1, 10, 0]}, index=np.random.rand(3)
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf + kdf.copy(), pdf + pdf.copy())

        self.assertRaisesRegex(
            ValueError,
            "it comes from a different dataframe",
            lambda: ks.range(10).add(ks.range(10)),
        )

        self.assertRaisesRegex(
            ValueError,
            "add with a sequence is currently not supported",
            lambda: ks.range(10).add(ks.range(10).id),
        )

    def test_binary_operator_add(self):
        # Positive
        pdf = pd.DataFrame({"a": ["x"], "b": ["y"], "c": [1], "d": [2]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf["a"] + kdf["b"], pdf["a"] + pdf["b"])
        self.assert_eq(kdf["c"] + kdf["d"], pdf["c"] + pdf["d"])

        # Negative
        ks_err_msg = "string addition can only be applied to string series or literals"

        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["a"] + kdf["c"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["c"] + kdf["a"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["c"] + "literal")
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: "literal" + kdf["c"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: 1 + kdf["a"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["a"] + 1)

    def test_binary_operator_sub(self):
        # Positive
        pdf = pd.DataFrame({"a": [2], "b": [1]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf["a"] - kdf["b"], pdf["a"] - pdf["b"])

        # Negative
        kdf = ks.DataFrame({"a": ["x"], "b": [1]})
        ks_err_msg = "substraction can not be applied to string series or literals"

        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["a"] - kdf["b"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["b"] - kdf["a"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["b"] - "literal")
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: "literal" - kdf["b"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: 1 - kdf["a"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["a"] - 1)

        kdf = ks.DataFrame({"a": ["x"], "b": ["y"]})
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["a"] - kdf["b"])

    def test_binary_operator_truediv(self):
        # Positive
        pdf = pd.DataFrame({"a": [3], "b": [2]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf["a"] / kdf["b"], pdf["a"] / pdf["b"])

        # Negative
        kdf = ks.DataFrame({"a": ["x"], "b": [1]})
        ks_err_msg = "division can not be applied on string series or literals"

        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["a"] / kdf["b"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["b"] / kdf["a"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["b"] / "literal")
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: "literal" / kdf["b"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: 1 / kdf["a"])

    def test_binary_operator_floordiv(self):
        kdf = ks.DataFrame({"a": ["x"], "b": [1]})
        ks_err_msg = "division can not be applied on string series or literals"

        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["a"] // kdf["b"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["b"] // kdf["a"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["b"] // "literal")
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: "literal" // kdf["b"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: 1 // kdf["a"])

    def test_binary_operator_mod(self):
        # Positive
        pdf = pd.DataFrame({"a": [3], "b": [2]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf["a"] % kdf["b"], pdf["a"] % pdf["b"])

        # Negative
        kdf = ks.DataFrame({"a": ["x"], "b": [1]})
        ks_err_msg = "modulo can not be applied on string series or literals"

        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["a"] % kdf["b"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["b"] % kdf["a"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["b"] % "literal")
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: 1 % kdf["a"])

    def test_binary_operator_multiply(self):
        # Positive
        pdf = pd.DataFrame({"a": ["x", "y"], "b": [1, 2], "c": [3, 4]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf["b"] * kdf["c"], pdf["b"] * pdf["c"])
        self.assert_eq(kdf["c"] * kdf["b"], pdf["c"] * pdf["b"])
        self.assert_eq(kdf["a"] * kdf["b"], pdf["a"] * pdf["b"])
        self.assert_eq(kdf["b"] * kdf["a"], pdf["b"] * pdf["a"])
        self.assert_eq(kdf["a"] * 2, pdf["a"] * 2)
        self.assert_eq(kdf["b"] * 2, pdf["b"] * 2)
        self.assert_eq(2 * kdf["a"], 2 * pdf["a"])
        self.assert_eq(2 * kdf["b"], 2 * pdf["b"])

        # Negative
        kdf = ks.DataFrame({"a": ["x"], "b": [2]})
        ks_err_msg = "multiplication can not be applied to a string literal"
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["b"] * "literal")
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: "literal" * kdf["b"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["a"] * "literal")
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: "literal" * kdf["a"])

        ks_err_msg = "a string series can only be multiplied to an int series or literal"
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["a"] * kdf["a"])
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: kdf["a"] * 0.1)
        self.assertRaisesRegex(TypeError, ks_err_msg, lambda: 0.1 * kdf["a"])

    def test_sample(self):
        pdf = pd.DataFrame({"A": [0, 2, 4]})
        kdf = ks.from_pandas(pdf)

        # Make sure the tests run, but we can't check the result because they are non-deterministic.
        kdf.sample(frac=0.1)
        kdf.sample(frac=0.2, replace=True)
        kdf.sample(frac=0.2, random_state=5)
        kdf["A"].sample(frac=0.2)
        kdf["A"].sample(frac=0.2, replace=True)
        kdf["A"].sample(frac=0.2, random_state=5)

        with self.assertRaises(ValueError):
            kdf.sample()
        with self.assertRaises(NotImplementedError):
            kdf.sample(n=1)

    def test_add_prefix(self):
        pdf = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]}, index=np.random.rand(4))
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.add_prefix("col_"), kdf.add_prefix("col_"))

        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B")])
        pdf.columns = columns
        kdf.columns = columns
        self.assert_eq(pdf.add_prefix("col_"), kdf.add_prefix("col_"))

    def test_add_suffix(self):
        pdf = pd.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]}, index=np.random.rand(4))
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.add_suffix("first_series"), kdf.add_suffix("first_series"))

        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B")])
        pdf.columns = columns
        kdf.columns = columns
        self.assert_eq(pdf.add_suffix("first_series"), kdf.add_suffix("first_series"))

    def test_join(self):
        # check basic function
        pdf1 = pd.DataFrame(
            {"key": ["K0", "K1", "K2", "K3"], "A": ["A0", "A1", "A2", "A3"]}, columns=["key", "A"]
        )
        pdf2 = pd.DataFrame(
            {"key": ["K0", "K1", "K2"], "B": ["B0", "B1", "B2"]}, columns=["key", "B"]
        )
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        join_pdf = pdf1.join(pdf2, lsuffix="_left", rsuffix="_right")
        join_pdf.sort_values(by=list(join_pdf.columns), inplace=True)

        join_kdf = kdf1.join(kdf2, lsuffix="_left", rsuffix="_right")
        join_kdf.sort_values(by=list(join_kdf.columns), inplace=True)

        self.assert_eq(join_pdf, join_kdf)

        # join with duplicated columns in Series
        with self.assertRaisesRegex(ValueError, "columns overlap but no suffix specified"):
            ks1 = ks.Series(["A1", "A5"], index=[1, 2], name="A")
            kdf1.join(ks1, how="outer")
        # join with duplicated columns in DataFrame
        with self.assertRaisesRegex(ValueError, "columns overlap but no suffix specified"):
            kdf1.join(kdf2, how="outer")

        # check `on` parameter
        join_pdf = pdf1.join(pdf2.set_index("key"), on="key", lsuffix="_left", rsuffix="_right")
        join_pdf.sort_values(by=list(join_pdf.columns), inplace=True)

        join_kdf = kdf1.join(kdf2.set_index("key"), on="key", lsuffix="_left", rsuffix="_right")
        join_kdf.sort_values(by=list(join_kdf.columns), inplace=True)
        self.assert_eq(join_pdf.reset_index(drop=True), join_kdf.reset_index(drop=True))

        join_pdf = pdf1.set_index("key").join(
            pdf2.set_index("key"), on="key", lsuffix="_left", rsuffix="_right"
        )
        join_pdf.sort_values(by=list(join_pdf.columns), inplace=True)

        join_kdf = kdf1.set_index("key").join(
            kdf2.set_index("key"), on="key", lsuffix="_left", rsuffix="_right"
        )
        join_kdf.sort_values(by=list(join_kdf.columns), inplace=True)
        self.assert_eq(join_pdf.reset_index(drop=True), join_kdf.reset_index(drop=True))

        # multi-index columns
        columns1 = pd.MultiIndex.from_tuples([("x", "key"), ("Y", "A")])
        columns2 = pd.MultiIndex.from_tuples([("x", "key"), ("Y", "B")])
        pdf1.columns = columns1
        pdf2.columns = columns2
        kdf1.columns = columns1
        kdf2.columns = columns2

        join_pdf = pdf1.join(pdf2, lsuffix="_left", rsuffix="_right")
        join_pdf.sort_values(by=list(join_pdf.columns), inplace=True)

        join_kdf = kdf1.join(kdf2, lsuffix="_left", rsuffix="_right")
        join_kdf.sort_values(by=list(join_kdf.columns), inplace=True)

        self.assert_eq(join_pdf, join_kdf)

        # check `on` parameter
        join_pdf = pdf1.join(
            pdf2.set_index(("x", "key")), on=[("x", "key")], lsuffix="_left", rsuffix="_right"
        )
        join_pdf.sort_values(by=list(join_pdf.columns), inplace=True)

        join_kdf = kdf1.join(
            kdf2.set_index(("x", "key")), on=[("x", "key")], lsuffix="_left", rsuffix="_right"
        )
        join_kdf.sort_values(by=list(join_kdf.columns), inplace=True)

        self.assert_eq(join_pdf.reset_index(drop=True), join_kdf.reset_index(drop=True))

        join_pdf = pdf1.set_index(("x", "key")).join(
            pdf2.set_index(("x", "key")), on=[("x", "key")], lsuffix="_left", rsuffix="_right"
        )
        join_pdf.sort_values(by=list(join_pdf.columns), inplace=True)

        join_kdf = kdf1.set_index(("x", "key")).join(
            kdf2.set_index(("x", "key")), on=[("x", "key")], lsuffix="_left", rsuffix="_right"
        )
        join_kdf.sort_values(by=list(join_kdf.columns), inplace=True)

        self.assert_eq(join_pdf.reset_index(drop=True), join_kdf.reset_index(drop=True))

        # multi-index
        midx1 = pd.MultiIndex.from_tuples(
            [("w", "a"), ("x", "b"), ("y", "c"), ("z", "d")], names=["index1", "index2"]
        )
        midx2 = pd.MultiIndex.from_tuples(
            [("w", "a"), ("x", "b"), ("y", "c")], names=["index1", "index2"]
        )
        pdf1.index = midx1
        pdf2.index = midx2
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        join_pdf = pdf1.join(pdf2, on=["index1", "index2"], rsuffix="_right")
        join_pdf.sort_values(by=list(join_pdf.columns), inplace=True)

        join_kdf = kdf1.join(kdf2, on=["index1", "index2"], rsuffix="_right")
        join_kdf.sort_values(by=list(join_kdf.columns), inplace=True)

        self.assert_eq(join_pdf, join_kdf)

        with self.assertRaisesRegex(
            ValueError, r'len\(left_on\) must equal the number of levels in the index of "right"'
        ):
            kdf1.join(kdf2, on=["index1"], rsuffix="_right")

    def test_replace(self):
        pdf = pd.DataFrame(
            {
                "name": ["Ironman", "Captain America", "Thor", "Hulk"],
                "weapon": ["Mark-45", "Shield", "Mjolnir", "Smash"],
            },
            index=np.random.rand(4),
        )
        kdf = ks.from_pandas(pdf)

        with self.assertRaisesRegex(
            NotImplementedError, "replace currently works only for method='pad"
        ):
            kdf.replace(method="bfill")
        with self.assertRaisesRegex(
            NotImplementedError, "replace currently works only when limit=None"
        ):
            kdf.replace(limit=10)
        with self.assertRaisesRegex(
            NotImplementedError, "replace currently doesn't supports regex"
        ):
            kdf.replace(regex="")

        with self.assertRaisesRegex(TypeError, "Unsupported type <class 'tuple'>"):
            kdf.replace(value=(1, 2, 3))
        with self.assertRaisesRegex(TypeError, "Unsupported type <class 'tuple'>"):
            kdf.replace(to_replace=(1, 2, 3))

        with self.assertRaisesRegex(ValueError, "Length of to_replace and value must be same"):
            kdf.replace(to_replace=["Ironman"], value=["Spiderman", "Doctor Strange"])

        self.assert_eq(kdf.replace("Ironman", "Spiderman"), pdf.replace("Ironman", "Spiderman"))
        self.assert_eq(
            kdf.replace(["Ironman", "Captain America"], ["Rescue", "Hawkeye"]),
            pdf.replace(["Ironman", "Captain America"], ["Rescue", "Hawkeye"]),
        )

        # inplace
        pser = pdf.name
        kser = kdf.name
        pdf.replace("Ironman", "Spiderman", inplace=True)
        kdf.replace("Ironman", "Spiderman", inplace=True)
        self.assert_eq(kdf, pdf)
        self.assert_eq(kser, pser)

        pdf = pd.DataFrame(
            {"A": [0, 1, 2, 3, 4], "B": [5, 6, 7, 8, 9], "C": ["a", "b", "c", "d", "e"]},
            index=np.random.rand(5),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.replace([0, 1, 2, 3, 5, 6], 4), pdf.replace([0, 1, 2, 3, 5, 6], 4))

        self.assert_eq(
            kdf.replace([0, 1, 2, 3, 5, 6], [6, 5, 4, 3, 2, 1]),
            pdf.replace([0, 1, 2, 3, 5, 6], [6, 5, 4, 3, 2, 1]),
        )

        self.assert_eq(kdf.replace({0: 10, 1: 100, 7: 200}), pdf.replace({0: 10, 1: 100, 7: 200}))

        self.assert_eq(kdf.replace({"A": 0, "B": 5}, 100), pdf.replace({"A": 0, "B": 5}, 100))

        self.assert_eq(kdf.replace({"A": {0: 100, 4: 400}}), pdf.replace({"A": {0: 100, 4: 400}}))
        self.assert_eq(kdf.replace({"X": {0: 100, 4: 400}}), pdf.replace({"X": {0: 100, 4: 400}}))

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B"), ("Y", "C")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.replace([0, 1, 2, 3, 5, 6], 4), pdf.replace([0, 1, 2, 3, 5, 6], 4))

        self.assert_eq(
            kdf.replace([0, 1, 2, 3, 5, 6], [6, 5, 4, 3, 2, 1]),
            pdf.replace([0, 1, 2, 3, 5, 6], [6, 5, 4, 3, 2, 1]),
        )

        self.assert_eq(kdf.replace({0: 10, 1: 100, 7: 200}), pdf.replace({0: 10, 1: 100, 7: 200}))

        self.assert_eq(
            kdf.replace({("X", "A"): 0, ("X", "B"): 5}, 100),
            pdf.replace({("X", "A"): 0, ("X", "B"): 5}, 100),
        )

        self.assert_eq(
            kdf.replace({("X", "A"): {0: 100, 4: 400}}), pdf.replace({("X", "A"): {0: 100, 4: 400}})
        )
        self.assert_eq(
            kdf.replace({("X", "B"): {0: 100, 4: 400}}), pdf.replace({("X", "B"): {0: 100, 4: 400}})
        )

    def test_update(self):
        # check base function
        def get_data(left_columns=None, right_columns=None):
            left_pdf = pd.DataFrame(
                {"A": ["1", "2", "3", "4"], "B": ["100", "200", np.nan, np.nan]}, columns=["A", "B"]
            )
            right_pdf = pd.DataFrame(
                {"B": ["x", np.nan, "y", np.nan], "C": ["100", "200", "300", "400"]},
                columns=["B", "C"],
            )

            left_kdf = ks.DataFrame(
                {"A": ["1", "2", "3", "4"], "B": ["100", "200", None, None]}, columns=["A", "B"]
            )
            right_kdf = ks.DataFrame(
                {"B": ["x", None, "y", None], "C": ["100", "200", "300", "400"]}, columns=["B", "C"]
            )
            if left_columns is not None:
                left_pdf.columns = left_columns
                left_kdf.columns = left_columns
            if right_columns is not None:
                right_pdf.columns = right_columns
                right_kdf.columns = right_columns
            return left_kdf, left_pdf, right_kdf, right_pdf

        left_kdf, left_pdf, right_kdf, right_pdf = get_data()
        pser = left_pdf.B
        kser = left_kdf.B
        left_pdf.update(right_pdf)
        left_kdf.update(right_kdf)
        self.assert_eq(left_pdf.sort_values(by=["A", "B"]), left_kdf.sort_values(by=["A", "B"]))
        self.assert_eq(kser.sort_index(), pser.sort_index())

        left_kdf, left_pdf, right_kdf, right_pdf = get_data()
        left_pdf.update(right_pdf, overwrite=False)
        left_kdf.update(right_kdf, overwrite=False)
        self.assert_eq(left_pdf.sort_values(by=["A", "B"]), left_kdf.sort_values(by=["A", "B"]))

        with self.assertRaises(NotImplementedError):
            left_kdf.update(right_kdf, join="right")

        # multi-index columns
        left_columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B")])
        right_columns = pd.MultiIndex.from_tuples([("X", "B"), ("Y", "C")])

        left_kdf, left_pdf, right_kdf, right_pdf = get_data(
            left_columns=left_columns, right_columns=right_columns
        )
        left_pdf.update(right_pdf)
        left_kdf.update(right_kdf)
        self.assert_eq(
            left_pdf.sort_values(by=[("X", "A"), ("X", "B")]),
            left_kdf.sort_values(by=[("X", "A"), ("X", "B")]),
        )

        left_kdf, left_pdf, right_kdf, right_pdf = get_data(
            left_columns=left_columns, right_columns=right_columns
        )
        left_pdf.update(right_pdf, overwrite=False)
        left_kdf.update(right_kdf, overwrite=False)
        self.assert_eq(
            left_pdf.sort_values(by=[("X", "A"), ("X", "B")]),
            left_kdf.sort_values(by=[("X", "A"), ("X", "B")]),
        )

        right_columns = pd.MultiIndex.from_tuples([("Y", "B"), ("Y", "C")])
        left_kdf, left_pdf, right_kdf, right_pdf = get_data(
            left_columns=left_columns, right_columns=right_columns
        )
        left_pdf.update(right_pdf)
        left_kdf.update(right_kdf)
        self.assert_eq(
            left_pdf.sort_values(by=[("X", "A"), ("X", "B")]),
            left_kdf.sort_values(by=[("X", "A"), ("X", "B")]),
        )

    def test_pivot_table_dtypes(self):
        pdf = pd.DataFrame(
            {
                "a": [4, 2, 3, 4, 8, 6],
                "b": [1, 2, 2, 4, 2, 4],
                "e": [1, 2, 2, 4, 2, 4],
                "c": [1, 2, 9, 4, 7, 4],
            },
            index=np.random.rand(6),
        )
        kdf = ks.from_pandas(pdf)

        # Skip columns comparison by reset_index
        res_df = kdf.pivot_table(
            index=["c"], columns="a", values=["b"], aggfunc={"b": "mean"}
        ).dtypes.reset_index(drop=True)
        exp_df = pdf.pivot_table(
            index=["c"], columns="a", values=["b"], aggfunc={"b": "mean"}
        ).dtypes.reset_index(drop=True)
        self.assert_eq(res_df, exp_df)

        # Results don't have the same column's name

        # Todo: self.assert_eq(kdf.pivot_table(columns="a", values="b").dtypes,
        #  pdf.pivot_table(columns="a", values="b").dtypes)

        # Todo: self.assert_eq(kdf.pivot_table(index=['c'], columns="a", values="b").dtypes,
        #  pdf.pivot_table(index=['c'], columns="a", values="b").dtypes)

        # Todo: self.assert_eq(kdf.pivot_table(index=['e', 'c'], columns="a", values="b").dtypes,
        #  pdf.pivot_table(index=['e', 'c'], columns="a", values="b").dtypes)

        # Todo: self.assert_eq(kdf.pivot_table(index=['e', 'c'],
        #  columns="a", values="b", fill_value=999).dtypes, pdf.pivot_table(index=['e', 'c'],
        #  columns="a", values="b", fill_value=999).dtypes)

    def test_pivot_table(self):
        pdf = pd.DataFrame(
            {
                "a": [4, 2, 3, 4, 8, 6],
                "b": [1, 2, 2, 4, 2, 4],
                "e": [10, 20, 20, 40, 20, 40],
                "c": [1, 2, 9, 4, 7, 4],
                "d": [-1, -2, -3, -4, -5, -6],
            },
            index=np.random.rand(6),
        )
        kdf = ks.from_pandas(pdf)

        # Checking if both DataFrames have the same results
        self.assert_eq(
            kdf.pivot_table(columns="a", values="b").sort_index(),
            pdf.pivot_table(columns="a", values="b").sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(index=["c"], columns="a", values="b").sort_index(),
            pdf.pivot_table(index=["c"], columns="a", values="b").sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(index=["c"], columns="a", values="b", aggfunc="sum").sort_index(),
            pdf.pivot_table(index=["c"], columns="a", values="b", aggfunc="sum").sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(index=["c"], columns="a", values=["b"], aggfunc="sum").sort_index(),
            pdf.pivot_table(index=["c"], columns="a", values=["b"], aggfunc="sum").sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(
                index=["c"], columns="a", values=["b", "e"], aggfunc="sum"
            ).sort_index(),
            pdf.pivot_table(
                index=["c"], columns="a", values=["b", "e"], aggfunc="sum"
            ).sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(
                index=["c"], columns="a", values=["b", "e", "d"], aggfunc="sum"
            ).sort_index(),
            pdf.pivot_table(
                index=["c"], columns="a", values=["b", "e", "d"], aggfunc="sum"
            ).sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(
                index=["c"], columns="a", values=["b", "e"], aggfunc={"b": "mean", "e": "sum"}
            ).sort_index(),
            pdf.pivot_table(
                index=["c"], columns="a", values=["b", "e"], aggfunc={"b": "mean", "e": "sum"}
            ).sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(index=["e", "c"], columns="a", values="b").sort_index(),
            pdf.pivot_table(index=["e", "c"], columns="a", values="b").sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(index=["e", "c"], columns="a", values="b", fill_value=999).sort_index(),
            pdf.pivot_table(index=["e", "c"], columns="a", values="b", fill_value=999).sort_index(),
            almost=True,
        )

        # multi-index columns
        columns = pd.MultiIndex.from_tuples(
            [("x", "a"), ("x", "b"), ("y", "e"), ("z", "c"), ("w", "d")]
        )
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.pivot_table(columns=("x", "a"), values=("x", "b")).sort_index(),
            pdf.pivot_table(columns=[("x", "a")], values=[("x", "b")]).sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(
                index=[("z", "c")], columns=("x", "a"), values=[("x", "b")]
            ).sort_index(),
            pdf.pivot_table(
                index=[("z", "c")], columns=[("x", "a")], values=[("x", "b")]
            ).sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(
                index=[("z", "c")], columns=("x", "a"), values=[("x", "b"), ("y", "e")]
            ).sort_index(),
            pdf.pivot_table(
                index=[("z", "c")], columns=[("x", "a")], values=[("x", "b"), ("y", "e")]
            ).sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(
                index=[("z", "c")], columns=("x", "a"), values=[("x", "b"), ("y", "e"), ("w", "d")]
            ).sort_index(),
            pdf.pivot_table(
                index=[("z", "c")],
                columns=[("x", "a")],
                values=[("x", "b"), ("y", "e"), ("w", "d")],
            ).sort_index(),
            almost=True,
        )

        self.assert_eq(
            kdf.pivot_table(
                index=[("z", "c")],
                columns=("x", "a"),
                values=[("x", "b"), ("y", "e")],
                aggfunc={("x", "b"): "mean", ("y", "e"): "sum"},
            ).sort_index(),
            pdf.pivot_table(
                index=[("z", "c")],
                columns=[("x", "a")],
                values=[("x", "b"), ("y", "e")],
                aggfunc={("x", "b"): "mean", ("y", "e"): "sum"},
            ).sort_index(),
            almost=True,
        )

    def test_pivot_table_and_index(self):
        # https://github.com/databricks/koalas/issues/805
        pdf = pd.DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            },
            columns=["A", "B", "C", "D", "E"],
            index=np.random.rand(9),
        )
        kdf = ks.from_pandas(pdf)

        ptable = pdf.pivot_table(
            values="D", index=["A", "B"], columns="C", aggfunc="sum", fill_value=0
        ).sort_index()
        ktable = kdf.pivot_table(
            values="D", index=["A", "B"], columns="C", aggfunc="sum", fill_value=0
        ).sort_index()

        self.assert_eq(ktable, ptable)
        self.assert_eq(ktable.index, ptable.index)
        self.assert_eq(repr(ktable.index), repr(ptable.index))

    @unittest.skipIf(
        LooseVersion(pyspark.__version__) < LooseVersion("2.4"),
        "stack won't work properly with PySpark<2.4",
    )
    def test_stack(self):
        pdf_single_level_cols = pd.DataFrame(
            [[0, 1], [2, 3]], index=["cat", "dog"], columns=["weight", "height"]
        )
        kdf_single_level_cols = ks.from_pandas(pdf_single_level_cols)

        self.assert_eq(
            kdf_single_level_cols.stack().sort_index(), pdf_single_level_cols.stack().sort_index()
        )

        multicol1 = pd.MultiIndex.from_tuples(
            [("weight", "kg"), ("weight", "pounds")], names=["x", "y"]
        )
        pdf_multi_level_cols1 = pd.DataFrame(
            [[1, 2], [2, 4]], index=["cat", "dog"], columns=multicol1
        )
        kdf_multi_level_cols1 = ks.from_pandas(pdf_multi_level_cols1)

        self.assert_eq(
            kdf_multi_level_cols1.stack().sort_index(), pdf_multi_level_cols1.stack().sort_index()
        )

        multicol2 = pd.MultiIndex.from_tuples([("weight", "kg"), ("height", "m")])
        pdf_multi_level_cols2 = pd.DataFrame(
            [[1.0, 2.0], [3.0, 4.0]], index=["cat", "dog"], columns=multicol2
        )
        kdf_multi_level_cols2 = ks.from_pandas(pdf_multi_level_cols2)

        self.assert_eq(
            kdf_multi_level_cols2.stack().sort_index(), pdf_multi_level_cols2.stack().sort_index()
        )

        pdf = pd.DataFrame(
            {
                ("y", "c"): [True, True],
                ("x", "b"): [False, False],
                ("x", "c"): [True, False],
                ("y", "a"): [False, True],
            }
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.stack().sort_index(), pdf.stack().sort_index())
        self.assert_eq(kdf[[]].stack().sort_index(), pdf[[]].stack().sort_index(), almost=True)

    def test_unstack(self):
        pdf = pd.DataFrame(
            np.random.randn(3, 3),
            index=pd.MultiIndex.from_tuples([("rg1", "x"), ("rg1", "y"), ("rg2", "z")]),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.unstack().sort_index(), pdf.unstack().sort_index(), almost=True)

    def test_pivot_errors(self):
        kdf = ks.range(10)

        with self.assertRaisesRegex(ValueError, "columns should be set"):
            kdf.pivot(index="id")

        with self.assertRaisesRegex(ValueError, "values should be set"):
            kdf.pivot(index="id", columns="id")

    def test_pivot_table_errors(self):
        pdf = pd.DataFrame(
            {
                "a": [4, 2, 3, 4, 8, 6],
                "b": [1, 2, 2, 4, 2, 4],
                "e": [1, 2, 2, 4, 2, 4],
                "c": [1, 2, 9, 4, 7, 4],
            },
            index=np.random.rand(6),
        )
        kdf = ks.from_pandas(pdf)

        self.assertRaises(KeyError, lambda: kdf.pivot_table(index=["c"], columns="a", values=5))

        msg = "index should be a None or a list of columns."
        with self.assertRaisesRegex(ValueError, msg):
            kdf.pivot_table(index="c", columns="a", values="b")

        msg = "pivot_table doesn't support aggfunc as dict and without index."
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.pivot_table(columns="a", values=["b", "e"], aggfunc={"b": "mean", "e": "sum"})

        msg = "columns should be one column name."
        with self.assertRaisesRegex(ValueError, msg):
            kdf.pivot_table(columns=["a"], values=["b"], aggfunc={"b": "mean", "e": "sum"})

        msg = "Columns in aggfunc must be the same as values."
        with self.assertRaisesRegex(ValueError, msg):
            kdf.pivot_table(
                index=["e", "c"], columns="a", values="b", aggfunc={"b": "mean", "e": "sum"}
            )

        msg = "values can't be a list without index."
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.pivot_table(columns="a", values=["b", "e"])

        msg = "Wrong columns A."
        with self.assertRaisesRegex(ValueError, msg):
            kdf.pivot_table(
                index=["c"], columns="A", values=["b", "e"], aggfunc={"b": "mean", "e": "sum"}
            )

        kdf = ks.DataFrame(
            {
                "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                "C": [
                    "small",
                    "large",
                    "large",
                    "small",
                    "small",
                    "large",
                    "small",
                    "small",
                    "large",
                ],
                "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
            },
            columns=["A", "B", "C", "D", "E"],
            index=np.random.rand(9),
        )

        msg = "values should be a numeric type."
        with self.assertRaisesRegex(TypeError, msg):
            kdf.pivot_table(
                index=["C"], columns="A", values=["B", "E"], aggfunc={"B": "mean", "E": "sum"}
            )

        msg = "values should be a numeric type."
        with self.assertRaisesRegex(TypeError, msg):
            kdf.pivot_table(index=["C"], columns="A", values="B", aggfunc={"B": "mean"})

    def test_transpose(self):
        # TODO: what if with random index?
        pdf1 = pd.DataFrame(data={"col1": [1, 2], "col2": [3, 4]}, columns=["col1", "col2"])
        kdf1 = ks.from_pandas(pdf1)

        pdf2 = pd.DataFrame(
            data={"score": [9, 8], "kids": [0, 0], "age": [12, 22]},
            columns=["score", "kids", "age"],
        )
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(pdf1.transpose().sort_index(), kdf1.transpose().sort_index())
        self.assert_eq(pdf2.transpose().sort_index(), kdf2.transpose().sort_index())

        with option_context("compute.max_rows", None):
            self.assert_eq(pdf1.transpose().sort_index(), kdf1.transpose().sort_index())

            self.assert_eq(pdf2.transpose().sort_index(), kdf2.transpose().sort_index())

        pdf3 = pd.DataFrame(
            {
                ("cg1", "a"): [1, 2, 3],
                ("cg1", "b"): [4, 5, 6],
                ("cg2", "c"): [7, 8, 9],
                ("cg3", "d"): [9, 9, 9],
            },
            index=pd.MultiIndex.from_tuples([("rg1", "x"), ("rg1", "y"), ("rg2", "z")]),
        )
        kdf3 = ks.from_pandas(pdf3)

        self.assert_eq(pdf3.transpose().sort_index(), kdf3.transpose().sort_index())

        with option_context("compute.max_rows", None):
            self.assert_eq(pdf3.transpose().sort_index(), kdf3.transpose().sort_index())

    def _test_cummin(self, pdf, kdf):
        self.assert_eq(pdf.cummin(), kdf.cummin())
        self.assert_eq(pdf.cummin(skipna=False), kdf.cummin(skipna=False))
        self.assert_eq(pdf.cummin().sum(), kdf.cummin().sum())

    def test_cummin(self):
        pdf = pd.DataFrame(
            [[2.0, 1.0], [5, None], [1.0, 0.0], [2.0, 4.0], [4.0, 9.0]],
            columns=list("AB"),
            index=np.random.rand(5),
        )
        kdf = ks.from_pandas(pdf)
        self._test_cummin(pdf, kdf)

    def test_cummin_multiindex_columns(self):
        arrays = [np.array(["A", "A", "B", "B"]), np.array(["one", "two", "one", "two"])]
        pdf = pd.DataFrame(np.random.randn(3, 4), index=["A", "C", "B"], columns=arrays)
        pdf.at["C", ("A", "two")] = None
        kdf = ks.from_pandas(pdf)
        self._test_cummin(pdf, kdf)

    def _test_cummax(self, pdf, kdf):
        self.assert_eq(pdf.cummax(), kdf.cummax())
        self.assert_eq(pdf.cummax(skipna=False), kdf.cummax(skipna=False))
        self.assert_eq(pdf.cummax().sum(), kdf.cummax().sum())

    def test_cummax(self):
        pdf = pd.DataFrame(
            [[2.0, 1.0], [5, None], [1.0, 0.0], [2.0, 4.0], [4.0, 9.0]],
            columns=list("AB"),
            index=np.random.rand(5),
        )
        kdf = ks.from_pandas(pdf)
        self._test_cummax(pdf, kdf)

    def test_cummax_multiindex_columns(self):
        arrays = [np.array(["A", "A", "B", "B"]), np.array(["one", "two", "one", "two"])]
        pdf = pd.DataFrame(np.random.randn(3, 4), index=["A", "C", "B"], columns=arrays)
        pdf.at["C", ("A", "two")] = None
        kdf = ks.from_pandas(pdf)
        self._test_cummax(pdf, kdf)

    def _test_cumsum(self, pdf, kdf):
        self.assert_eq(pdf.cumsum(), kdf.cumsum())
        self.assert_eq(pdf.cumsum(skipna=False), kdf.cumsum(skipna=False))
        self.assert_eq(pdf.cumsum().sum(), kdf.cumsum().sum())

    def test_cumsum(self):
        pdf = pd.DataFrame(
            [[2.0, 1.0], [5, None], [1.0, 0.0], [2.0, 4.0], [4.0, 9.0]],
            columns=list("AB"),
            index=np.random.rand(5),
        )
        kdf = ks.from_pandas(pdf)
        self._test_cumsum(pdf, kdf)

    def test_cumsum_multiindex_columns(self):
        arrays = [np.array(["A", "A", "B", "B"]), np.array(["one", "two", "one", "two"])]
        pdf = pd.DataFrame(np.random.randn(3, 4), index=["A", "C", "B"], columns=arrays)
        pdf.at["C", ("A", "two")] = None
        kdf = ks.from_pandas(pdf)
        self._test_cumsum(pdf, kdf)

    def _test_cumprod(self, pdf, kdf):
        self.assert_eq(pdf.cumprod(), kdf.cumprod(), almost=True)
        self.assert_eq(pdf.cumprod(skipna=False), kdf.cumprod(skipna=False), almost=True)
        self.assert_eq(pdf.cumprod().sum(), kdf.cumprod().sum(), almost=True)

    def test_cumprod(self):
        if LooseVersion(pyspark.__version__) >= LooseVersion("2.4"):
            pdf = pd.DataFrame(
                [[2.0, 1.0, 1], [5, None, 2], [1.0, 1.0, 3], [2.0, 4.0, 4], [4.0, 9.0, 5]],
                columns=list("ABC"),
                index=np.random.rand(5),
            )
            kdf = ks.from_pandas(pdf)
            self._test_cumprod(pdf, kdf)
        else:
            pdf = pd.DataFrame(
                [[2, 1, 1], [5, 1, 2], [1, 1, 3], [2, 4, 4], [4, 9, 5]],
                columns=list("ABC"),
                index=np.random.rand(5),
            )
            kdf = ks.from_pandas(pdf)
            self._test_cumprod(pdf, kdf)

    def test_cumprod_multiindex_columns(self):
        arrays = [np.array(["A", "A", "B", "B"]), np.array(["one", "two", "one", "two"])]
        pdf = pd.DataFrame(np.random.rand(3, 4), index=["A", "C", "B"], columns=arrays)
        pdf.at["C", ("A", "two")] = None
        kdf = ks.from_pandas(pdf)
        self._test_cumprod(pdf, kdf)

    def test_drop_duplicates(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 2, 2, 3], "b": ["a", "a", "a", "c", "d"]}, index=np.random.rand(5)
        )
        kdf = ks.from_pandas(pdf)

        # inplace is False
        for keep in ["first", "last", False]:
            with self.subTest(keep=keep):
                self.assert_eq(
                    pdf.drop_duplicates(keep=keep).sort_index(),
                    kdf.drop_duplicates(keep=keep).sort_index(),
                )
                self.assert_eq(
                    pdf.drop_duplicates("a", keep=keep).sort_index(),
                    kdf.drop_duplicates("a", keep=keep).sort_index(),
                )
                self.assert_eq(
                    pdf.drop_duplicates(["a", "b"], keep=keep).sort_index(),
                    kdf.drop_duplicates(["a", "b"], keep=keep).sort_index(),
                )
                self.assert_eq(
                    pdf.set_index("a", append=True).drop_duplicates(keep=keep).sort_index(),
                    kdf.set_index("a", append=True).drop_duplicates(keep=keep).sort_index(),
                )
                self.assert_eq(
                    pdf.set_index("a", append=True).drop_duplicates("b", keep=keep).sort_index(),
                    kdf.set_index("a", append=True).drop_duplicates("b", keep=keep).sort_index(),
                )

        columns = pd.MultiIndex.from_tuples([("x", "a"), ("y", "b")])
        pdf.columns = columns
        kdf.columns = columns

        # inplace is False
        for keep in ["first", "last", False]:
            with self.subTest("multi-index columns", keep=keep):
                self.assert_eq(
                    pdf.drop_duplicates(keep=keep).sort_index(),
                    kdf.drop_duplicates(keep=keep).sort_index(),
                )
                self.assert_eq(
                    pdf.drop_duplicates(("x", "a"), keep=keep).sort_index(),
                    kdf.drop_duplicates(("x", "a"), keep=keep).sort_index(),
                )
                self.assert_eq(
                    pdf.drop_duplicates([("x", "a"), ("y", "b")], keep=keep).sort_index(),
                    kdf.drop_duplicates([("x", "a"), ("y", "b")], keep=keep).sort_index(),
                )

        # inplace is True
        subset_list = [None, "a", ["a", "b"]]
        for subset in subset_list:
            pdf = pd.DataFrame(
                {"a": [1, 2, 2, 2, 3], "b": ["a", "a", "a", "c", "d"]}, index=np.random.rand(5)
            )
            kdf = ks.from_pandas(pdf)
            pser = pdf.a
            kser = kdf.a
            pdf.drop_duplicates(subset=subset, inplace=True)
            kdf.drop_duplicates(subset=subset, inplace=True)
            self.assert_eq(kdf.sort_index(), pdf.sort_index())
            self.assert_eq(kser.sort_index(), pser.sort_index())

        # multi-index columns, inplace is True
        subset_list = [None, ("x", "a"), [("x", "a"), ("y", "b")]]
        for subset in subset_list:
            pdf = pd.DataFrame(
                {"a": [1, 2, 2, 2, 3], "b": ["a", "a", "a", "c", "d"]}, index=np.random.rand(5)
            )
            kdf = ks.from_pandas(pdf)
            columns = pd.MultiIndex.from_tuples([("x", "a"), ("y", "b")])
            pdf.columns = columns
            kdf.columns = columns
            pser = pdf[("x", "a")]
            kser = kdf[("x", "a")]
            pdf.drop_duplicates(subset=subset, inplace=True)
            kdf.drop_duplicates(subset=subset, inplace=True)
            self.assert_eq(kdf.sort_index(), pdf.sort_index())
            self.assert_eq(kser.sort_index(), pser.sort_index())

        # non-string names
        pdf = pd.DataFrame(
            {10: [1, 2, 2, 2, 3], 20: ["a", "a", "a", "c", "d"]}, index=np.random.rand(5)
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            pdf.drop_duplicates(10, keep=keep).sort_index(),
            kdf.drop_duplicates(10, keep=keep).sort_index(),
        )
        self.assert_eq(
            pdf.drop_duplicates([10, 20], keep=keep).sort_index(),
            kdf.drop_duplicates([10, 20], keep=keep).sort_index(),
        )

    def test_reindex(self):
        index = pd.Index(["A", "B", "C", "D", "E"])

        pdf = pd.DataFrame({"numbers": [1.0, 2.0, 3.0, 4.0, None]}, index=index)
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            pdf.reindex(["A", "B", "C"], columns=["numbers", "2", "3"]).sort_index(),
            kdf.reindex(["A", "B", "C"], columns=["numbers", "2", "3"]).sort_index(),
        )

        self.assert_eq(
            pdf.reindex(["A", "B", "C"], index=["numbers", "2", "3"]).sort_index(),
            kdf.reindex(["A", "B", "C"], index=["numbers", "2", "3"]).sort_index(),
        )

        self.assert_eq(
            pdf.reindex(index=["A", "B"]).sort_index(), kdf.reindex(index=["A", "B"]).sort_index()
        )

        self.assert_eq(
            pdf.reindex(index=["A", "B", "2", "3"]).sort_index(),
            kdf.reindex(index=["A", "B", "2", "3"]).sort_index(),
        )

        self.assert_eq(
            pdf.reindex(index=["A", "E", "2", "3"], fill_value=0).sort_index(),
            kdf.reindex(index=["A", "E", "2", "3"], fill_value=0).sort_index(),
        )

        self.assert_eq(
            pdf.reindex(columns=["numbers"]).sort_index(),
            kdf.reindex(columns=["numbers"]).sort_index(),
        )

        # Using float as fill_value to avoid int64/32 clash
        self.assert_eq(
            pdf.reindex(columns=["numbers", "2", "3"], fill_value=0.0).sort_index(),
            kdf.reindex(columns=["numbers", "2", "3"], fill_value=0.0).sort_index(),
        )

        # Reindexing single Index on single Index
        pindex2 = pd.Index(["A", "C", "D", "E", "0"], name="index2")
        kindex2 = ks.from_pandas(pindex2)

        for fill_value in [None, 0]:
            self.assert_eq(
                pdf.reindex(index=pindex2, fill_value=fill_value).sort_index(),
                kdf.reindex(index=kindex2, fill_value=fill_value).sort_index(),
            )

        pindex2 = pd.DataFrame({"index2": ["A", "C", "D", "E", "0"]}).set_index("index2").index
        kindex2 = ks.from_pandas(pindex2)

        for fill_value in [None, 0]:
            self.assert_eq(
                pdf.reindex(index=pindex2, fill_value=fill_value).sort_index(),
                kdf.reindex(index=kindex2, fill_value=fill_value).sort_index(),
            )

        # Reindexing MultiIndex on single Index
        pindex = pd.MultiIndex.from_tuples(
            [("A", "B"), ("C", "D"), ("F", "G")], names=["name1", "name2"]
        )
        kindex = ks.from_pandas(pindex)

        self.assert_eq(
            pdf.reindex(index=pindex, fill_value=0.0).sort_index(),
            kdf.reindex(index=kindex, fill_value=0.0).sort_index(),
        )

        self.assertRaises(TypeError, lambda: kdf.reindex(columns=["numbers", "2", "3"], axis=1))
        self.assertRaises(TypeError, lambda: kdf.reindex(columns=["numbers", "2", "3"], axis=2))
        self.assertRaises(TypeError, lambda: kdf.reindex(index=["A", "B", "C"], axis=1))
        self.assertRaises(TypeError, lambda: kdf.reindex(index=123))

        # Reindexing MultiIndex on MultiIndex
        pdf = pd.DataFrame({"numbers": [1.0, 2.0, None]}, index=pindex)
        kdf = ks.from_pandas(pdf)
        pindex2 = pd.MultiIndex.from_tuples(
            [("A", "G"), ("C", "D"), ("I", "J")], names=["name1", "name2"]
        )
        kindex2 = ks.from_pandas(pindex2)

        for fill_value in [None, 0.0]:
            self.assert_eq(
                pdf.reindex(index=pindex2, fill_value=fill_value).sort_index(),
                kdf.reindex(index=kindex2, fill_value=fill_value).sort_index(),
            )

        pindex2 = (
            pd.DataFrame({"index_level_1": ["A", "C", "I"], "index_level_2": ["G", "D", "J"]})
            .set_index(["index_level_1", "index_level_2"])
            .index
        )
        kindex2 = ks.from_pandas(pindex2)

        for fill_value in [None, 0.0]:
            self.assert_eq(
                pdf.reindex(index=pindex2, fill_value=fill_value).sort_index(),
                kdf.reindex(index=kindex2, fill_value=fill_value).sort_index(),
            )

        columns = pd.MultiIndex.from_tuples([("X", "numbers")])
        pdf.columns = columns
        kdf.columns = columns

        # Reindexing MultiIndex index on MultiIndex columns and MultiIndex index
        for fill_value in [None, 0.0]:
            self.assert_eq(
                pdf.reindex(index=pindex2, fill_value=fill_value).sort_index(),
                kdf.reindex(index=kindex2, fill_value=fill_value).sort_index(),
            )

        index = pd.Index(["A", "B", "C", "D", "E"])
        pdf = pd.DataFrame(data=[1.0, 2.0, 3.0, 4.0, None], index=index, columns=columns)
        kdf = ks.from_pandas(pdf)
        pindex2 = pd.Index(["A", "C", "D", "E", "0"], name="index2")
        kindex2 = ks.from_pandas(pindex2)

        # Reindexing single Index on MultiIndex columns and single Index
        for fill_value in [None, 0.0]:
            self.assert_eq(
                pdf.reindex(index=pindex2, fill_value=fill_value).sort_index(),
                kdf.reindex(index=kindex2, fill_value=fill_value).sort_index(),
            )

        for fill_value in [None, 0.0]:
            self.assert_eq(
                pdf.reindex(
                    columns=[("X", "numbers"), ("Y", "2"), ("Y", "3")], fill_value=fill_value
                ).sort_index(),
                kdf.reindex(
                    columns=[("X", "numbers"), ("Y", "2"), ("Y", "3")], fill_value=fill_value
                ).sort_index(),
            )

        self.assertRaises(TypeError, lambda: kdf.reindex(columns=["X"]))
        self.assertRaises(ValueError, lambda: kdf.reindex(columns=[("X",)]))

    def test_melt(self):
        pdf = pd.DataFrame(
            {"A": [1, 3, 5], "B": [2, 4, 6], "C": [7, 8, 9]}, index=np.random.rand(3)
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.melt().sort_values(["variable", "value"]).reset_index(drop=True),
            pdf.melt().sort_values(["variable", "value"]),
        )
        self.assert_eq(
            kdf.melt(id_vars="A").sort_values(["variable", "value"]).reset_index(drop=True),
            pdf.melt(id_vars="A").sort_values(["variable", "value"]),
        )
        self.assert_eq(
            kdf.melt(id_vars=["A", "B"]).sort_values(["variable", "value"]).reset_index(drop=True),
            pdf.melt(id_vars=["A", "B"]).sort_values(["variable", "value"]),
        )
        self.assert_eq(
            kdf.melt(id_vars=("A", "B")).sort_values(["variable", "value"]).reset_index(drop=True),
            pdf.melt(id_vars=("A", "B")).sort_values(["variable", "value"]),
        )
        self.assert_eq(
            kdf.melt(id_vars=["A"], value_vars=["C"])
            .sort_values(["variable", "value"])
            .reset_index(drop=True),
            pdf.melt(id_vars=["A"], value_vars=["C"]).sort_values(["variable", "value"]),
        )
        self.assert_eq(
            kdf.melt(id_vars=["A"], value_vars=["B"], var_name="myVarname", value_name="myValname")
            .sort_values(["myVarname", "myValname"])
            .reset_index(drop=True),
            pdf.melt(
                id_vars=["A"], value_vars=["B"], var_name="myVarname", value_name="myValname"
            ).sort_values(["myVarname", "myValname"]),
        )
        self.assert_eq(
            kdf.melt(value_vars=("A", "B"))
            .sort_values(["variable", "value"])
            .reset_index(drop=True),
            pdf.melt(value_vars=("A", "B")).sort_values(["variable", "value"]),
        )

        self.assertRaises(KeyError, lambda: kdf.melt(id_vars="Z"))
        self.assertRaises(KeyError, lambda: kdf.melt(value_vars="Z"))

        # multi-index columns
        if LooseVersion("0.24") <= LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            # pandas >=0.24,<1.0 doesn't support mixed int/str columns in melt.
            # see: https://github.com/pandas-dev/pandas/pull/29792
            TEN = "10"
            TWELVE = "20"
        else:
            TEN = 10.0
            TWELVE = 20.0

        columns = pd.MultiIndex.from_tuples([(TEN, "A"), (TEN, "B"), (TWELVE, "C")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.melt().sort_values(["variable_0", "variable_1", "value"]).reset_index(drop=True),
            pdf.melt().sort_values(["variable_0", "variable_1", "value"]),
        )
        self.assert_eq(
            kdf.melt(id_vars=[(TEN, "A")])
            .sort_values(["variable_0", "variable_1", "value"])
            .reset_index(drop=True),
            pdf.melt(id_vars=[(TEN, "A")])
            .sort_values(["variable_0", "variable_1", "value"])
            .rename(columns=name_like_string),
        )
        self.assert_eq(
            kdf.melt(id_vars=[(TEN, "A")], value_vars=[(TWELVE, "C")])
            .sort_values(["variable_0", "variable_1", "value"])
            .reset_index(drop=True),
            pdf.melt(id_vars=[(TEN, "A")], value_vars=[(TWELVE, "C")])
            .sort_values(["variable_0", "variable_1", "value"])
            .rename(columns=name_like_string),
        )
        self.assert_eq(
            kdf.melt(
                id_vars=[(TEN, "A")],
                value_vars=[(TEN, "B")],
                var_name=["myV1", "myV2"],
                value_name="myValname",
            )
            .sort_values(["myV1", "myV2", "myValname"])
            .reset_index(drop=True),
            pdf.melt(
                id_vars=[(TEN, "A")],
                value_vars=[(TEN, "B")],
                var_name=["myV1", "myV2"],
                value_name="myValname",
            )
            .sort_values(["myV1", "myV2", "myValname"])
            .rename(columns=name_like_string),
        )

        columns.names = ["v0", "v1"]
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.melt().sort_values(["v0", "v1", "value"]).reset_index(drop=True),
            pdf.melt().sort_values(["v0", "v1", "value"]),
        )

        self.assertRaises(ValueError, lambda: kdf.melt(id_vars=(TEN, "A")))
        self.assertRaises(ValueError, lambda: kdf.melt(value_vars=(TEN, "A")))
        self.assertRaises(KeyError, lambda: kdf.melt(id_vars=[TEN]))
        self.assertRaises(KeyError, lambda: kdf.melt(id_vars=[(TWELVE, "A")]))
        self.assertRaises(KeyError, lambda: kdf.melt(value_vars=[TWELVE]))
        self.assertRaises(KeyError, lambda: kdf.melt(value_vars=[(TWELVE, "A")]))

        # non-string names
        pdf.columns = [10.0, 20.0, 30.0]
        kdf.columns = [10.0, 20.0, 30.0]

        self.assert_eq(
            kdf.melt().sort_values(["variable", "value"]).reset_index(drop=True),
            pdf.melt().sort_values(["variable", "value"]),
        )
        self.assert_eq(
            kdf.melt(id_vars=10.0).sort_values(["variable", "value"]).reset_index(drop=True),
            pdf.melt(id_vars=10.0).sort_values(["variable", "value"]),
        )
        self.assert_eq(
            kdf.melt(id_vars=[10.0, 20.0])
            .sort_values(["variable", "value"])
            .reset_index(drop=True),
            pdf.melt(id_vars=[10.0, 20.0]).sort_values(["variable", "value"]),
        )
        self.assert_eq(
            kdf.melt(id_vars=(10.0, 20.0))
            .sort_values(["variable", "value"])
            .reset_index(drop=True),
            pdf.melt(id_vars=(10.0, 20.0)).sort_values(["variable", "value"]),
        )
        self.assert_eq(
            kdf.melt(id_vars=[10.0], value_vars=[30.0])
            .sort_values(["variable", "value"])
            .reset_index(drop=True),
            pdf.melt(id_vars=[10.0], value_vars=[30.0]).sort_values(["variable", "value"]),
        )
        self.assert_eq(
            kdf.melt(value_vars=(10.0, 20.0))
            .sort_values(["variable", "value"])
            .reset_index(drop=True),
            pdf.melt(value_vars=(10.0, 20.0)).sort_values(["variable", "value"]),
        )

    def test_all(self):
        pdf = pd.DataFrame(
            {
                "col1": [False, False, False],
                "col2": [True, False, False],
                "col3": [0, 0, 1],
                "col4": [0, 1, 2],
                "col5": [False, False, None],
                "col6": [True, False, None],
            },
            index=np.random.rand(3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.all(), pdf.all())

        columns = pd.MultiIndex.from_tuples(
            [
                ("a", "col1"),
                ("a", "col2"),
                ("a", "col3"),
                ("b", "col4"),
                ("b", "col5"),
                ("c", "col6"),
            ]
        )
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.all(), pdf.all())

        columns.names = ["X", "Y"]
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.all(), pdf.all())

        with self.assertRaisesRegex(
            NotImplementedError, 'axis should be either 0 or "index" currently.'
        ):
            kdf.all(axis=1)

    def test_any(self):
        pdf = pd.DataFrame(
            {
                "col1": [False, False, False],
                "col2": [True, False, False],
                "col3": [0, 0, 1],
                "col4": [0, 1, 2],
                "col5": [False, False, None],
                "col6": [True, False, None],
            },
            index=np.random.rand(3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.any(), pdf.any())

        columns = pd.MultiIndex.from_tuples(
            [
                ("a", "col1"),
                ("a", "col2"),
                ("a", "col3"),
                ("b", "col4"),
                ("b", "col5"),
                ("c", "col6"),
            ]
        )
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.any(), pdf.any())

        columns.names = ["X", "Y"]
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.any(), pdf.any())

        with self.assertRaisesRegex(
            NotImplementedError, 'axis should be either 0 or "index" currently.'
        ):
            kdf.any(axis=1)

    def test_rank(self):
        pdf = pd.DataFrame(
            data={"col1": [1, 2, 3, 1], "col2": [3, 4, 3, 1]},
            columns=["col1", "col2"],
            index=np.random.rand(4),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.rank().sort_index(), kdf.rank().sort_index())
        self.assert_eq(
            pdf.rank(ascending=False).sort_index(), kdf.rank(ascending=False).sort_index()
        )
        self.assert_eq(pdf.rank(method="min").sort_index(), kdf.rank(method="min").sort_index())
        self.assert_eq(pdf.rank(method="max").sort_index(), kdf.rank(method="max").sort_index())
        self.assert_eq(pdf.rank(method="first").sort_index(), kdf.rank(method="first").sort_index())
        self.assert_eq(pdf.rank(method="dense").sort_index(), kdf.rank(method="dense").sort_index())

        msg = "method must be one of 'average', 'min', 'max', 'first', 'dense'"
        with self.assertRaisesRegex(ValueError, msg):
            kdf.rank(method="nothing")

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "col1"), ("y", "col2")])
        pdf.columns = columns
        kdf.columns = columns
        self.assert_eq(pdf.rank().sort_index(), kdf.rank().sort_index())

    def test_round(self):
        pdf = pd.DataFrame(
            {
                "A": [0.028208, 0.038683, 0.877076],
                "B": [0.992815, 0.645646, 0.149370],
                "C": [0.173891, 0.577595, 0.491027],
            },
            columns=["A", "B", "C"],
            index=np.random.rand(3),
        )
        kdf = ks.from_pandas(pdf)

        pser = pd.Series([1, 0, 2], index=["A", "B", "C"])
        kser = ks.Series([1, 0, 2], index=["A", "B", "C"])
        self.assert_eq(pdf.round(2), kdf.round(2))
        self.assert_eq(pdf.round({"A": 1, "C": 2}), kdf.round({"A": 1, "C": 2}))
        self.assert_eq(pdf.round({"A": 1, "D": 2}), kdf.round({"A": 1, "D": 2}))
        self.assert_eq(pdf.round(pser), kdf.round(kser))
        msg = "decimals must be an integer, a dict-like or a Series"
        with self.assertRaisesRegex(ValueError, msg):
            kdf.round(1.5)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B"), ("Y", "C")])
        pdf.columns = columns
        kdf.columns = columns
        pser = pd.Series([1, 0, 2], index=columns)
        kser = ks.Series([1, 0, 2], index=columns)
        self.assert_eq(pdf.round(2), kdf.round(2))
        self.assert_eq(
            pdf.round({("X", "A"): 1, ("Y", "C"): 2}), kdf.round({("X", "A"): 1, ("Y", "C"): 2})
        )
        self.assert_eq(pdf.round({("X", "A"): 1, "Y": 2}), kdf.round({("X", "A"): 1, "Y": 2}))
        self.assert_eq(pdf.round(pser), kdf.round(kser))

        # non-string names
        pdf = pd.DataFrame(
            {
                10: [0.028208, 0.038683, 0.877076],
                20: [0.992815, 0.645646, 0.149370],
                30: [0.173891, 0.577595, 0.491027],
            },
            index=np.random.rand(3),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.round({10: 1, 30: 2}), kdf.round({10: 1, 30: 2}))

    def test_shift(self):
        pdf = pd.DataFrame(
            {
                "Col1": [10, 20, 15, 30, 45],
                "Col2": [13, 23, 18, 33, 48],
                "Col3": [17, 27, 22, 37, 52],
            },
            index=np.random.rand(5),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.shift(3), kdf.shift(3))

        # Need the expected result since pandas 0.23 does not support `fill_value` argument.
        pdf1 = pd.DataFrame(
            {"Col1": [0, 0, 0, 10, 20], "Col2": [0, 0, 0, 13, 23], "Col3": [0, 0, 0, 17, 27]},
            index=pdf.index,
        )
        self.assert_eq(pdf1, kdf.shift(periods=3, fill_value=0))
        msg = "should be an int"
        with self.assertRaisesRegex(ValueError, msg):
            kdf.shift(1.5)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "Col1"), ("x", "Col2"), ("y", "Col3")])
        pdf.columns = columns
        kdf.columns = columns
        self.assert_eq(pdf.shift(3), kdf.shift(3))

    def test_diff(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6], "b": [1, 1, 2, 3, 5, 8], "c": [1, 4, 9, 16, 25, 36]},
            index=np.random.rand(6),
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.diff(), kdf.diff())

        msg = "should be an int"
        with self.assertRaisesRegex(ValueError, msg):
            kdf.diff(1.5)
        msg = 'axis should be either 0 or "index" currently.'
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.diff(axis=1)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "Col1"), ("x", "Col2"), ("y", "Col3")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(pdf.diff(), kdf.diff())

    def test_duplicated(self):
        pdf = pd.DataFrame(
            {"a": [1, 1, 2, 3], "b": [1, 1, 1, 4], "c": [1, 1, 1, 5]}, index=np.random.rand(4)
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.duplicated().sort_index(), kdf.duplicated().sort_index())
        self.assert_eq(
            pdf.duplicated(keep="last").sort_index(), kdf.duplicated(keep="last").sort_index(),
        )
        self.assert_eq(
            pdf.duplicated(keep=False).sort_index(), kdf.duplicated(keep=False).sort_index(),
        )
        self.assert_eq(
            pdf.duplicated(subset="b").sort_index(), kdf.duplicated(subset="b").sort_index(),
        )
        self.assert_eq(
            pdf.duplicated(subset=["b"]).sort_index(), kdf.duplicated(subset=["b"]).sort_index(),
        )
        with self.assertRaisesRegex(ValueError, "'keep' only supports 'first', 'last' and False"):
            kdf.duplicated(keep="false")
        with self.assertRaisesRegex(KeyError, "'d'"):
            kdf.duplicated(subset=["d"])

        pdf.index.name = "x"
        kdf.index.name = "x"
        self.assert_eq(pdf.duplicated().sort_index(), kdf.duplicated().sort_index())

        # multi-index
        self.assert_eq(
            pdf.set_index("a", append=True).duplicated().sort_index(),
            kdf.set_index("a", append=True).duplicated().sort_index(),
        )
        self.assert_eq(
            pdf.set_index("a", append=True).duplicated(keep=False).sort_index(),
            kdf.set_index("a", append=True).duplicated(keep=False).sort_index(),
        )
        self.assert_eq(
            pdf.set_index("a", append=True).duplicated(subset=["b"]).sort_index(),
            kdf.set_index("a", append=True).duplicated(subset=["b"]).sort_index(),
        )

        # mutli-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns
        self.assert_eq(pdf.duplicated().sort_index(), kdf.duplicated().sort_index())
        self.assert_eq(
            pdf.duplicated(subset=("x", "b")).sort_index(),
            kdf.duplicated(subset=("x", "b")).sort_index(),
        )
        self.assert_eq(
            pdf.duplicated(subset=[("x", "b")]).sort_index(),
            kdf.duplicated(subset=[("x", "b")]).sort_index(),
        )

        # non-string names
        pdf = pd.DataFrame(
            {10: [1, 1, 2, 3], 20: [1, 1, 1, 4], 30: [1, 1, 1, 5]}, index=np.random.rand(4)
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.duplicated().sort_index(), kdf.duplicated().sort_index())
        self.assert_eq(
            pdf.duplicated(subset=10).sort_index(), kdf.duplicated(subset=10).sort_index(),
        )

    def test_ffill(self):
        idx = np.random.rand(6)
        pdf = pd.DataFrame(
            {
                "x": [np.nan, 2, 3, 4, np.nan, 6],
                "y": [1, 2, np.nan, 4, np.nan, np.nan],
                "z": [1, 2, 3, 4, np.nan, np.nan],
            },
            index=idx,
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.ffill(), pdf.ffill())
        self.assert_eq(kdf.ffill(limit=1), pdf.ffill(limit=1))

        pser = pdf.y
        kser = kdf.y

        kdf.ffill(inplace=True)
        pdf.ffill(inplace=True)

        self.assert_eq(kdf, pdf)
        self.assert_eq(kser, pser)
        self.assert_eq(kser[idx[2]], pser[idx[2]])

    def test_bfill(self):
        idx = np.random.rand(6)
        pdf = pd.DataFrame(
            {
                "x": [np.nan, 2, 3, 4, np.nan, 6],
                "y": [1, 2, np.nan, 4, np.nan, np.nan],
                "z": [1, 2, 3, 4, np.nan, np.nan],
            },
            index=idx,
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.bfill(), pdf.bfill())
        self.assert_eq(kdf.bfill(limit=1), pdf.bfill(limit=1))

        pser = pdf.x
        kser = kdf.x

        kdf.bfill(inplace=True)
        pdf.bfill(inplace=True)

        self.assert_eq(kdf, pdf)
        self.assert_eq(kser, pser)
        self.assert_eq(kser[idx[0]], pser[idx[0]])

    def test_filter(self):
        pdf = pd.DataFrame(
            {
                "aa": ["aa", "bd", "bc", "ab", "ce"],
                "ba": [1, 2, 3, 4, 5],
                "cb": [1.0, 2.0, 3.0, 4.0, 5.0],
                "db": [1.0, np.nan, 3.0, np.nan, 5.0],
            }
        )
        pdf = pdf.set_index("aa")
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.filter(items=["ab", "aa"], axis=0).sort_index(),
            pdf.filter(items=["ab", "aa"], axis=0).sort_index(),
        )
        self.assert_eq(
            kdf.filter(items=["ba", "db"], axis=1).sort_index(),
            pdf.filter(items=["ba", "db"], axis=1).sort_index(),
        )

        self.assert_eq(kdf.filter(like="b", axis="index"), pdf.filter(like="b", axis="index"))
        self.assert_eq(kdf.filter(like="c", axis="columns"), pdf.filter(like="c", axis="columns"))

        self.assert_eq(kdf.filter(regex="b.*", axis="index"), pdf.filter(regex="b.*", axis="index"))
        self.assert_eq(
            kdf.filter(regex="b.*", axis="columns"), pdf.filter(regex="b.*", axis="columns")
        )

        pdf = pdf.set_index("ba", append=True)
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.filter(items=[("aa", 1), ("bd", 2)], axis=0).sort_index(),
            pdf.filter(items=[("aa", 1), ("bd", 2)], axis=0).sort_index(),
        )

        with self.assertRaisesRegex(TypeError, "Unsupported type <class 'list'>"):
            kdf.filter(items=[["aa", 1], ("bd", 2)], axis=0)

        with self.assertRaisesRegex(ValueError, "The item should not be empty."):
            kdf.filter(items=[(), ("bd", 2)], axis=0)

        self.assert_eq(kdf.filter(like="b", axis=0), pdf.filter(like="b", axis=0))

        self.assert_eq(kdf.filter(regex="b.*", axis=0), pdf.filter(regex="b.*", axis=0))

        with self.assertRaisesRegex(ValueError, "items should be a list-like object"):
            kdf.filter(items="b")

        with self.assertRaisesRegex(ValueError, "No axis named"):
            kdf.filter(regex="b.*", axis=123)

        with self.assertRaisesRegex(TypeError, "Must pass either `items`, `like`"):
            kdf.filter()

        with self.assertRaisesRegex(TypeError, "mutually exclusive"):
            kdf.filter(regex="b.*", like="aaa")

        # multi-index columns
        pdf = pd.DataFrame(
            {
                ("x", "aa"): ["aa", "ab", "bc", "bd", "ce"],
                ("x", "ba"): [1, 2, 3, 4, 5],
                ("y", "cb"): [1.0, 2.0, 3.0, 4.0, 5.0],
                ("z", "db"): [1.0, np.nan, 3.0, np.nan, 5.0],
            }
        )
        pdf = pdf.set_index(("x", "aa"))
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.filter(items=["ab", "aa"], axis=0).sort_index(),
            pdf.filter(items=["ab", "aa"], axis=0).sort_index(),
        )
        self.assert_eq(
            kdf.filter(items=[("x", "ba"), ("z", "db")], axis=1).sort_index(),
            pdf.filter(items=[("x", "ba"), ("z", "db")], axis=1).sort_index(),
        )

        self.assert_eq(kdf.filter(like="b", axis="index"), pdf.filter(like="b", axis="index"))
        self.assert_eq(kdf.filter(like="c", axis="columns"), pdf.filter(like="c", axis="columns"))

        self.assert_eq(kdf.filter(regex="b.*", axis="index"), pdf.filter(regex="b.*", axis="index"))
        self.assert_eq(
            kdf.filter(regex="b.*", axis="columns"), pdf.filter(regex="b.*", axis="columns")
        )

    def test_pipe(self):
        kdf = ks.DataFrame(
            {"category": ["A", "A", "B"], "col1": [1, 2, 3], "col2": [4, 5, 6]},
            columns=["category", "col1", "col2"],
        )

        self.assertRaisesRegex(
            ValueError,
            "arg is both the pipe target and a keyword argument",
            lambda: kdf.pipe((lambda x: x, "arg"), arg="1"),
        )

    def test_transform(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 100,
                "b": [1.0, 1.0, 2.0, 3.0, 5.0, 8.0] * 100,
                "c": [1, 4, 9, 16, 25, 36] * 100,
            },
            columns=["a", "b", "c"],
            index=np.random.rand(600),
        )
        kdf = ks.DataFrame(pdf)
        self.assert_eq(
            kdf.transform(lambda x: x + 1).sort_index(), pdf.transform(lambda x: x + 1).sort_index()
        )
        self.assert_eq(
            kdf.transform(lambda x, y: x + y, y=2).sort_index(),
            pdf.transform(lambda x, y: x + y, y=2).sort_index(),
        )
        with option_context("compute.shortcut_limit", 500):
            self.assert_eq(
                kdf.transform(lambda x: x + 1).sort_index(),
                pdf.transform(lambda x: x + 1).sort_index(),
            )
            self.assert_eq(
                kdf.transform(lambda x, y: x + y, y=1).sort_index(),
                pdf.transform(lambda x, y: x + y, y=1).sort_index(),
            )

        with self.assertRaisesRegex(AssertionError, "the first argument should be a callable"):
            kdf.transform(1)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.transform(lambda x: x + 1).sort_index(), pdf.transform(lambda x: x + 1).sort_index()
        )
        with option_context("compute.shortcut_limit", 500):
            self.assert_eq(
                kdf.transform(lambda x: x + 1).sort_index(),
                pdf.transform(lambda x: x + 1).sort_index(),
            )

    def test_apply(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 100,
                "b": [1.0, 1.0, 2.0, 3.0, 5.0, 8.0] * 100,
                "c": [1, 4, 9, 16, 25, 36] * 100,
            },
            columns=["a", "b", "c"],
            index=np.random.rand(600),
        )
        kdf = ks.DataFrame(pdf)

        self.assert_eq(
            kdf.apply(lambda x: x + 1).sort_index(), pdf.apply(lambda x: x + 1).sort_index()
        )
        self.assert_eq(
            kdf.apply(lambda x, b: x + b, args=(1,)).sort_index(),
            pdf.apply(lambda x, b: x + b, args=(1,)).sort_index(),
        )
        self.assert_eq(
            kdf.apply(lambda x, b: x + b, b=1).sort_index(),
            pdf.apply(lambda x, b: x + b, b=1).sort_index(),
        )

        with option_context("compute.shortcut_limit", 500):
            self.assert_eq(
                kdf.apply(lambda x: x + 1).sort_index(), pdf.apply(lambda x: x + 1).sort_index()
            )
            self.assert_eq(
                kdf.apply(lambda x, b: x + b, args=(1,)).sort_index(),
                pdf.apply(lambda x, b: x + b, args=(1,)).sort_index(),
            )
            self.assert_eq(
                kdf.apply(lambda x, b: x + b, b=1).sort_index(),
                pdf.apply(lambda x, b: x + b, b=1).sort_index(),
            )

        # returning a Series
        self.assert_eq(
            kdf.apply(lambda x: len(x), axis=1).sort_index(),
            pdf.apply(lambda x: len(x), axis=1).sort_index(),
        )
        self.assert_eq(
            kdf.apply(lambda x, c: len(x) + c, axis=1, c=100).sort_index(),
            pdf.apply(lambda x, c: len(x) + c, axis=1, c=100).sort_index(),
        )
        with option_context("compute.shortcut_limit", 500):
            self.assert_eq(
                kdf.apply(lambda x: len(x), axis=1).sort_index(),
                pdf.apply(lambda x: len(x), axis=1).sort_index(),
            )
            self.assert_eq(
                kdf.apply(lambda x, c: len(x) + c, axis=1, c=100).sort_index(),
                pdf.apply(lambda x, c: len(x) + c, axis=1, c=100).sort_index(),
            )

        with self.assertRaisesRegex(AssertionError, "the first argument should be a callable"):
            kdf.apply(1)

        with self.assertRaisesRegex(TypeError, "The given function.*1 or 'column'; however"):

            def f1(_) -> ks.DataFrame[int]:
                pass

            kdf.apply(f1, axis=0)

        with self.assertRaisesRegex(TypeError, "The given function.*0 or 'index'; however"):

            def f2(_) -> ks.Series[int]:
                pass

            kdf.apply(f2, axis=1)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.apply(lambda x: x + 1).sort_index(), pdf.apply(lambda x: x + 1).sort_index()
        )
        with option_context("compute.shortcut_limit", 500):
            self.assert_eq(
                kdf.apply(lambda x: x + 1).sort_index(), pdf.apply(lambda x: x + 1).sort_index()
            )

        # returning a Series
        self.assert_eq(
            kdf.apply(lambda x: len(x), axis=1).sort_index(),
            pdf.apply(lambda x: len(x), axis=1).sort_index(),
        )
        with option_context("compute.shortcut_limit", 500):
            self.assert_eq(
                kdf.apply(lambda x: len(x), axis=1).sort_index(),
                pdf.apply(lambda x: len(x), axis=1).sort_index(),
            )

    def test_apply_batch(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 100,
                "b": [1.0, 1.0, 2.0, 3.0, 5.0, 8.0] * 100,
                "c": [1, 4, 9, 16, 25, 36] * 100,
            },
            columns=["a", "b", "c"],
            index=np.random.rand(600),
        )
        kdf = ks.DataFrame(pdf)

        # One to test alias.
        self.assert_eq(kdf.apply_batch(lambda pdf: pdf + 1).sort_index(), (pdf + 1).sort_index())
        self.assert_eq(
            kdf.koalas.apply_batch(lambda pdf, a: pdf + a, args=(1,)).sort_index(),
            (pdf + 1).sort_index(),
        )
        with option_context("compute.shortcut_limit", 500):
            self.assert_eq(
                kdf.koalas.apply_batch(lambda pdf: pdf + 1).sort_index(), (pdf + 1).sort_index()
            )
            self.assert_eq(
                kdf.koalas.apply_batch(lambda pdf, b: pdf + b, b=1).sort_index(),
                (pdf + 1).sort_index(),
            )

        with self.assertRaisesRegex(AssertionError, "the first argument should be a callable"):
            kdf.koalas.apply_batch(1)

        with self.assertRaisesRegex(TypeError, "The given function.*frame as its type hints"):

            def f2(_) -> ks.Series[int]:
                pass

            kdf.koalas.apply_batch(f2)

        with self.assertRaisesRegex(ValueError, "The given function should return a frame"):
            kdf.koalas.apply_batch(lambda pdf: 1)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.koalas.apply_batch(lambda x: x + 1).sort_index(), (pdf + 1).sort_index())
        with option_context("compute.shortcut_limit", 500):
            self.assert_eq(
                kdf.koalas.apply_batch(lambda x: x + 1).sort_index(), (pdf + 1).sort_index()
            )

    def test_transform_batch(self):
        pdf = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6] * 100,
                "b": [1.0, 1.0, 2.0, 3.0, 5.0, 8.0] * 100,
                "c": [1, 4, 9, 16, 25, 36] * 100,
            },
            columns=["a", "b", "c"],
            index=np.random.rand(600),
        )
        kdf = ks.DataFrame(pdf)

        # One to test alias.
        self.assert_eq(
            kdf.transform_batch(lambda pdf: pdf + 1).sort_index(), (pdf + 1).sort_index()
        )
        self.assert_eq(
            kdf.koalas.transform_batch(lambda pdf: pdf.c + 1).sort_index(), (pdf.c + 1).sort_index()
        )
        self.assert_eq(
            kdf.koalas.transform_batch(lambda pdf, a: pdf + a, 1).sort_index(),
            (pdf + 1).sort_index(),
        )
        self.assert_eq(
            kdf.koalas.transform_batch(lambda pdf, a: pdf.c + a, a=1).sort_index(),
            (pdf.c + 1).sort_index(),
        )

        with option_context("compute.shortcut_limit", 500):
            self.assert_eq(
                kdf.koalas.transform_batch(lambda pdf: pdf + 1).sort_index(), (pdf + 1).sort_index()
            )
            self.assert_eq(
                kdf.koalas.transform_batch(lambda pdf: pdf.b + 1).sort_index(),
                (pdf.b + 1).sort_index(),
            )
            self.assert_eq(
                kdf.koalas.transform_batch(lambda pdf, a: pdf + a, 1).sort_index(),
                (pdf + 1).sort_index(),
            )
            self.assert_eq(
                kdf.koalas.transform_batch(lambda pdf, a: pdf.c + a, a=1).sort_index(),
                (pdf.c + 1).sort_index(),
            )

        with self.assertRaisesRegex(AssertionError, "the first argument should be a callable"):
            kdf.koalas.transform_batch(1)

        with self.assertRaisesRegex(ValueError, "The given function should return a frame"):
            kdf.koalas.transform_batch(lambda pdf: 1)

        with self.assertRaisesRegex(
            ValueError, "transform_batch cannot produce aggregated results"
        ):
            kdf.koalas.transform_batch(lambda pdf: pd.Series(1))

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(
            kdf.koalas.transform_batch(lambda x: x + 1).sort_index(), (pdf + 1).sort_index()
        )
        with option_context("compute.shortcut_limit", 500):
            self.assert_eq(
                kdf.koalas.transform_batch(lambda x: x + 1).sort_index(), (pdf + 1).sort_index()
            )

    def test_transform_batch_same_anchor(self):
        kdf = ks.range(10)
        kdf["d"] = kdf.koalas.transform_batch(lambda pdf: pdf.id + 1)
        self.assert_eq(
            kdf, pd.DataFrame({"id": list(range(10)), "d": list(range(1, 11))}, columns=["id", "d"])
        )

        kdf = ks.range(10)
        # One to test alias.
        kdf["d"] = kdf.id.transform_batch(lambda ser: ser + 1)
        self.assert_eq(
            kdf, pd.DataFrame({"id": list(range(10)), "d": list(range(1, 11))}, columns=["id", "d"])
        )

        kdf = ks.range(10)

        def plus_one(pdf) -> ks.Series[np.int64]:
            return pdf.id + 1

        kdf["d"] = kdf.koalas.transform_batch(plus_one)
        self.assert_eq(
            kdf, pd.DataFrame({"id": list(range(10)), "d": list(range(1, 11))}, columns=["id", "d"])
        )

        kdf = ks.range(10)

        def plus_one(ser) -> ks.Series[np.int64]:
            return ser + 1

        kdf["d"] = kdf.id.koalas.transform_batch(plus_one)
        self.assert_eq(
            kdf, pd.DataFrame({"id": list(range(10)), "d": list(range(1, 11))}, columns=["id", "d"])
        )

    def test_empty_timestamp(self):
        pdf = pd.DataFrame(
            {
                "t": [
                    datetime(2019, 1, 1, 0, 0, 0),
                    datetime(2019, 1, 2, 0, 0, 0),
                    datetime(2019, 1, 3, 0, 0, 0),
                ]
            },
            index=np.random.rand(3),
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf[kdf["t"] != kdf["t"]], pdf[pdf["t"] != pdf["t"]])
        self.assert_eq(kdf[kdf["t"] != kdf["t"]].dtypes, pdf[pdf["t"] != pdf["t"]].dtypes)

    def test_to_spark(self):
        kdf = ks.from_pandas(self.pdf)

        with self.assertRaisesRegex(ValueError, "'index_col' cannot be overlapped"):
            kdf.to_spark(index_col="a")

        with self.assertRaisesRegex(ValueError, "length of index columns.*1.*3"):
            kdf.to_spark(index_col=["x", "y", "z"])

    def test_keys(self):
        pdf = pd.DataFrame(
            [[1, 2], [4, 5], [7, 8]],
            index=["cobra", "viper", "sidewinder"],
            columns=["max_speed", "shield"],
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.keys(), pdf.keys())

    def test_quantile(self):
        kdf = ks.from_pandas(self.pdf)

        with self.assertRaisesRegex(
            NotImplementedError, 'axis should be either 0 or "index" currently.'
        ):
            kdf.quantile(0.5, axis=1)

        with self.assertRaisesRegex(
            NotImplementedError, "quantile currently doesn't supports numeric_only"
        ):
            kdf.quantile(0.5, numeric_only=False)

    def test_pct_change(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0], "c": [300, 200, 400, 200]},
            index=np.random.rand(4),
        )
        pdf.columns = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.pct_change(2), pdf.pct_change(2), check_exact=False)

    def test_where(self):
        kdf = ks.from_pandas(self.pdf)

        with self.assertRaisesRegex(ValueError, "type of cond must be a DataFrame or Series"):
            kdf.where(1)

    def test_mask(self):
        kdf = ks.from_pandas(self.pdf)

        with self.assertRaisesRegex(ValueError, "type of cond must be a DataFrame or Series"):
            kdf.mask(1)

    def test_query(self):
        pdf = pd.DataFrame({"A": range(1, 6), "B": range(10, 0, -2), "C": range(10, 5, -1)})
        kdf = ks.from_pandas(pdf)

        exprs = ("A > B", "A < C", "C == B")
        for expr in exprs:
            self.assert_eq(kdf.query(expr), pdf.query(expr))

        # test `inplace=True`
        for expr in exprs:
            dummy_kdf = kdf.copy()
            dummy_pdf = pdf.copy()

            pser = dummy_pdf.A
            kser = dummy_kdf.A
            dummy_pdf.query(expr, inplace=True)
            dummy_kdf.query(expr, inplace=True)

            self.assert_eq(dummy_kdf, dummy_pdf)
            self.assert_eq(kser, pser)

        # invalid values for `expr`
        invalid_exprs = (1, 1.0, (exprs[0],), [exprs[0]])
        for expr in invalid_exprs:
            with self.assertRaisesRegex(
                ValueError, "expr must be a string to be evaluated, {} given".format(type(expr))
            ):
                kdf.query(expr)

        # invalid values for `inplace`
        invalid_inplaces = (1, 0, "True", "False")
        for inplace in invalid_inplaces:
            with self.assertRaisesRegex(
                ValueError,
                'For argument "inplace" expected type bool, received type {}.'.format(
                    type(inplace).__name__
                ),
            ):
                kdf.query("a < b", inplace=inplace)

        # doesn't support for MultiIndex columns
        columns = pd.MultiIndex.from_tuples([("A", "Z"), ("B", "X"), ("C", "C")])
        kdf.columns = columns
        with self.assertRaisesRegex(ValueError, "Doesn't support for MultiIndex columns"):
            kdf.query("('A', 'Z') > ('B', 'X')")

    def test_take(self):
        pdf = pd.DataFrame(
            {"A": range(0, 50000), "B": range(100000, 0, -2), "C": range(100000, 50000, -1)}
        )
        kdf = ks.from_pandas(pdf)

        # axis=0 (default)
        self.assert_eq(kdf.take([1, 2]).sort_index(), pdf.take([1, 2]).sort_index())
        self.assert_eq(kdf.take([-1, -2]).sort_index(), pdf.take([-1, -2]).sort_index())
        self.assert_eq(
            kdf.take(range(100, 110)).sort_index(), pdf.take(range(100, 110)).sort_index()
        )
        self.assert_eq(
            kdf.take(range(-110, -100)).sort_index(), pdf.take(range(-110, -100)).sort_index()
        )
        self.assert_eq(
            kdf.take([10, 100, 1000, 10000]).sort_index(),
            pdf.take([10, 100, 1000, 10000]).sort_index(),
        )
        self.assert_eq(
            kdf.take([-10, -100, -1000, -10000]).sort_index(),
            pdf.take([-10, -100, -1000, -10000]).sort_index(),
        )

        # axis=1
        self.assert_eq(kdf.take([1, 2], axis=1).sort_index(), pdf.take([1, 2], axis=1).sort_index())
        self.assert_eq(
            kdf.take([-1, -2], axis=1).sort_index(), pdf.take([-1, -2], axis=1).sort_index()
        )
        self.assert_eq(
            kdf.take(range(1, 3), axis=1).sort_index(), pdf.take(range(1, 3), axis=1).sort_index(),
        )
        self.assert_eq(
            kdf.take(range(-1, -3), axis=1).sort_index(),
            pdf.take(range(-1, -3), axis=1).sort_index(),
        )
        self.assert_eq(
            kdf.take([2, 1], axis=1).sort_index(), pdf.take([2, 1], axis=1).sort_index(),
        )
        self.assert_eq(
            kdf.take([-1, -2], axis=1).sort_index(), pdf.take([-1, -2], axis=1).sort_index(),
        )

        # MultiIndex columns
        columns = pd.MultiIndex.from_tuples([("A", "Z"), ("B", "X"), ("C", "C")])
        kdf.columns = columns
        pdf.columns = columns

        # MultiIndex columns with axis=0 (default)
        self.assert_eq(kdf.take([1, 2]).sort_index(), pdf.take([1, 2]).sort_index())
        self.assert_eq(kdf.take([-1, -2]).sort_index(), pdf.take([-1, -2]).sort_index())
        self.assert_eq(
            kdf.take(range(100, 110)).sort_index(), pdf.take(range(100, 110)).sort_index()
        )
        self.assert_eq(
            kdf.take(range(-110, -100)).sort_index(), pdf.take(range(-110, -100)).sort_index()
        )
        self.assert_eq(
            kdf.take([10, 100, 1000, 10000]).sort_index(),
            pdf.take([10, 100, 1000, 10000]).sort_index(),
        )
        self.assert_eq(
            kdf.take([-10, -100, -1000, -10000]).sort_index(),
            pdf.take([-10, -100, -1000, -10000]).sort_index(),
        )

        # axis=1
        self.assert_eq(kdf.take([1, 2], axis=1).sort_index(), pdf.take([1, 2], axis=1).sort_index())
        self.assert_eq(
            kdf.take([-1, -2], axis=1).sort_index(), pdf.take([-1, -2], axis=1).sort_index()
        )
        self.assert_eq(
            kdf.take(range(1, 3), axis=1).sort_index(), pdf.take(range(1, 3), axis=1).sort_index(),
        )
        self.assert_eq(
            kdf.take(range(-1, -3), axis=1).sort_index(),
            pdf.take(range(-1, -3), axis=1).sort_index(),
        )
        self.assert_eq(
            kdf.take([2, 1], axis=1).sort_index(), pdf.take([2, 1], axis=1).sort_index(),
        )
        self.assert_eq(
            kdf.take([-1, -2], axis=1).sort_index(), pdf.take([-1, -2], axis=1).sort_index(),
        )

        # Checking the type of indices.
        self.assertRaises(ValueError, lambda: kdf.take(1))
        self.assertRaises(ValueError, lambda: kdf.take("1"))
        self.assertRaises(ValueError, lambda: kdf.take({1, 2}))
        self.assertRaises(ValueError, lambda: kdf.take({1: None, 2: None}))

    def test_axes(self):
        pdf = self.pdf
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.axes, kdf.axes)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("y", "b")])
        pdf.columns = columns
        kdf.columns = columns
        self.assert_eq(pdf.axes, kdf.axes)

    def test_udt(self):
        sparse_values = {0: 0.1, 1: 1.1}
        sparse_vector = SparseVector(len(sparse_values), sparse_values)
        pdf = pd.DataFrame({"a": [sparse_vector], "b": [10]})

        if LooseVersion(pyspark.__version__) < LooseVersion("2.4"):
            with self.sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
                kdf = ks.from_pandas(pdf)
                self.assert_eq(kdf, pdf)
        else:
            kdf = ks.from_pandas(pdf)
            self.assert_eq(kdf, pdf)

    def test_eval(self):
        pdf = pd.DataFrame({"A": range(1, 6), "B": range(10, 0, -2)})
        kdf = ks.from_pandas(pdf)

        # operation between columns (returns Series)
        self.assert_eq(pdf.eval("A + B"), kdf.eval("A + B"))
        self.assert_eq(pdf.eval("A + A"), kdf.eval("A + A"))
        # assignment (returns DataFrame)
        self.assert_eq(pdf.eval("C = A + B"), kdf.eval("C = A + B"))
        self.assert_eq(pdf.eval("A = A + A"), kdf.eval("A = A + A"))
        # operation between scalars (returns scalar)
        self.assert_eq(pdf.eval("1 + 1"), kdf.eval("1 + 1"))
        # complicated operations with assignment
        self.assert_eq(
            pdf.eval("B = A + B // (100 + 200) * (500 - B) - 10.5"),
            kdf.eval("B = A + B // (100 + 200) * (500 - B) - 10.5"),
        )

        # inplace=True (only support for assignment)
        pdf.eval("C = A + B", inplace=True)
        kdf.eval("C = A + B", inplace=True)
        self.assert_eq(pdf, kdf)
        pser = pdf.A
        kser = kdf.A
        pdf.eval("A = B + C", inplace=True)
        kdf.eval("A = B + C", inplace=True)
        self.assert_eq(pdf, kdf)
        self.assert_eq(pser, kser)

        # doesn't support for multi-index columns
        columns = pd.MultiIndex.from_tuples([("x", "a"), ("y", "b"), ("z", "c")])
        kdf.columns = columns
        self.assertRaises(ValueError, lambda: kdf.eval("x.a + y.b"))

    def test_to_markdown(self):
        pdf = pd.DataFrame(data={"animal_1": ["elk", "pig"], "animal_2": ["dog", "quetzal"]})
        kdf = ks.from_pandas(pdf)

        # `to_markdown()` is supported in pandas >= 1.0.0 since it's newly added in pandas 1.0.0.
        if LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            self.assertRaises(NotImplementedError, lambda: kdf.to_markdown())
        else:
            self.assert_eq(pdf.to_markdown(), kdf.to_markdown())

    def test_cache(self):
        pdf = pd.DataFrame(
            [(0.2, 0.3), (0.0, 0.6), (0.6, 0.0), (0.2, 0.1)], columns=["dogs", "cats"]
        )
        kdf = ks.from_pandas(pdf)

        with kdf.cache() as cached_df:
            self.assert_eq(isinstance(cached_df, CachedDataFrame), True)
            self.assert_eq(
                repr(cached_df.storage_level), repr(StorageLevel(True, True, False, True))
            )

    def test_persist(self):
        pdf = pd.DataFrame(
            [(0.2, 0.3), (0.0, 0.6), (0.6, 0.0), (0.2, 0.1)], columns=["dogs", "cats"]
        )
        kdf = ks.from_pandas(pdf)
        storage_levels = [
            StorageLevel.DISK_ONLY,
            StorageLevel.MEMORY_AND_DISK,
            StorageLevel.MEMORY_ONLY,
            StorageLevel.OFF_HEAP,
        ]

        for storage_level in storage_levels:
            with kdf.persist(storage_level) as cached_df:
                self.assert_eq(isinstance(cached_df, CachedDataFrame), True)
                self.assert_eq(repr(cached_df.storage_level), repr(storage_level))

        self.assertRaises(TypeError, lambda: kdf.persist("DISK_ONLY"))

    def test_squeeze(self):
        axises = [None, 0, 1, "rows", "index", "columns"]

        # Multiple columns
        pdf = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"], index=["x", "y"])
        kdf = ks.from_pandas(pdf)
        for axis in axises:
            self.assert_eq(pdf.squeeze(axis), kdf.squeeze(axis))
        # Multiple columns with MultiIndex columns
        columns = pd.MultiIndex.from_tuples([("A", "Z"), ("B", "X")])
        pdf.columns = columns
        kdf.columns = columns
        for axis in axises:
            self.assert_eq(pdf.squeeze(axis), kdf.squeeze(axis))

        # Single column with single value
        pdf = pd.DataFrame([[1]], columns=["a"], index=["x"])
        kdf = ks.from_pandas(pdf)
        for axis in axises:
            self.assert_eq(pdf.squeeze(axis), kdf.squeeze(axis))
        # Single column with single value with MultiIndex column
        columns = pd.MultiIndex.from_tuples([("A", "Z")])
        pdf.columns = columns
        kdf.columns = columns
        for axis in axises:
            self.assert_eq(pdf.squeeze(axis), kdf.squeeze(axis))

        # Single column with multiple values
        pdf = pd.DataFrame([1, 2, 3, 4], columns=["a"])
        kdf = ks.from_pandas(pdf)
        for axis in axises:
            self.assert_eq(pdf.squeeze(axis), kdf.squeeze(axis))
        # Single column with multiple values with MultiIndex column
        pdf.columns = columns
        kdf.columns = columns
        for axis in axises:
            self.assert_eq(pdf.squeeze(axis), kdf.squeeze(axis))

    def test_rfloordiv(self):
        pdf = pd.DataFrame(
            {"angles": [0, 3, 4], "degrees": [360, 180, 360]},
            index=["circle", "triangle", "rectangle"],
            columns=["angles", "degrees"],
        )
        kdf = ks.from_pandas(pdf)

        if LooseVersion(pd.__version__) < LooseVersion("1.0.0") and LooseVersion(
            pd.__version__
        ) >= LooseVersion("0.24.0"):
            expected_result = pd.DataFrame(
                {"angles": [np.inf, 3.0, 2.0], "degrees": [0.0, 0.0, 0.0]},
                index=["circle", "triangle", "rectangle"],
                columns=["angles", "degrees"],
            )
        else:
            expected_result = pdf.rfloordiv(10)

        self.assert_eq(kdf.rfloordiv(10), expected_result)

    def test_truncate(self):
        pdf1 = pd.DataFrame(
            {
                "A": ["a", "b", "c", "d", "e", "f", "g"],
                "B": ["h", "i", "j", "k", "l", "m", "n"],
                "C": ["o", "p", "q", "r", "s", "t", "u"],
            },
            index=[-500, -20, -1, 0, 400, 550, 1000],
        )
        kdf1 = ks.from_pandas(pdf1)
        pdf2 = pd.DataFrame(
            {
                "A": ["a", "b", "c", "d", "e", "f", "g"],
                "B": ["h", "i", "j", "k", "l", "m", "n"],
                "C": ["o", "p", "q", "r", "s", "t", "u"],
            },
            index=[1000, 550, 400, 0, -1, -20, -500],
        )
        kdf2 = ks.from_pandas(pdf2)

        self.assert_eq(kdf1.truncate(), pdf1.truncate())
        self.assert_eq(kdf1.truncate(before=-20), pdf1.truncate(before=-20))
        self.assert_eq(kdf1.truncate(after=400), pdf1.truncate(after=400))
        self.assert_eq(kdf1.truncate(copy=False), pdf1.truncate(copy=False))
        self.assert_eq(kdf1.truncate(-20, 400, copy=False), pdf1.truncate(-20, 400, copy=False))
        # The bug for these tests has been fixed in pandas 1.1.0.
        if LooseVersion(pd.__version__) >= LooseVersion("1.1.0"):
            self.assert_eq(kdf2.truncate(0, 550), pdf2.truncate(0, 550))
            self.assert_eq(kdf2.truncate(0, 550, copy=False), pdf2.truncate(0, 550, copy=False))
        else:
            expected_kdf = ks.DataFrame(
                {"A": ["b", "c", "d"], "B": ["i", "j", "k"], "C": ["p", "q", "r"],},
                index=[550, 400, 0],
            )
            self.assert_eq(kdf2.truncate(0, 550), expected_kdf)
            self.assert_eq(kdf2.truncate(0, 550, copy=False), expected_kdf)

        # axis = 1
        self.assert_eq(kdf1.truncate(axis=1), pdf1.truncate(axis=1))
        self.assert_eq(kdf1.truncate(before="B", axis=1), pdf1.truncate(before="B", axis=1))
        self.assert_eq(kdf1.truncate(after="A", axis=1), pdf1.truncate(after="A", axis=1))
        self.assert_eq(kdf1.truncate(copy=False, axis=1), pdf1.truncate(copy=False, axis=1))
        self.assert_eq(kdf2.truncate("B", "C", axis=1), pdf2.truncate("B", "C", axis=1))
        self.assert_eq(
            kdf1.truncate("B", "C", copy=False, axis=1),
            pdf1.truncate("B", "C", copy=False, axis=1),
        )

        # MultiIndex columns
        columns = pd.MultiIndex.from_tuples([("A", "Z"), ("B", "X"), ("C", "Z")])
        pdf1.columns = columns
        kdf1.columns = columns
        pdf2.columns = columns
        kdf2.columns = columns

        self.assert_eq(kdf1.truncate(), pdf1.truncate())
        self.assert_eq(kdf1.truncate(before=-20), pdf1.truncate(before=-20))
        self.assert_eq(kdf1.truncate(after=400), pdf1.truncate(after=400))
        self.assert_eq(kdf1.truncate(copy=False), pdf1.truncate(copy=False))
        self.assert_eq(kdf1.truncate(-20, 400, copy=False), pdf1.truncate(-20, 400, copy=False))
        # The bug for these tests has been fixed in pandas 1.1.0.
        if LooseVersion(pd.__version__) >= LooseVersion("1.1.0"):
            self.assert_eq(kdf2.truncate(0, 550), pdf2.truncate(0, 550))
            self.assert_eq(kdf2.truncate(0, 550, copy=False), pdf2.truncate(0, 550, copy=False))
        else:
            expected_kdf.columns = columns
            self.assert_eq(kdf2.truncate(0, 550), expected_kdf)
            self.assert_eq(kdf2.truncate(0, 550, copy=False), expected_kdf)
        # axis = 1
        self.assert_eq(kdf1.truncate(axis=1), pdf1.truncate(axis=1))
        self.assert_eq(kdf1.truncate(before="B", axis=1), pdf1.truncate(before="B", axis=1))
        self.assert_eq(kdf1.truncate(after="A", axis=1), pdf1.truncate(after="A", axis=1))
        self.assert_eq(kdf1.truncate(copy=False, axis=1), pdf1.truncate(copy=False, axis=1))
        self.assert_eq(kdf2.truncate("B", "C", axis=1), pdf2.truncate("B", "C", axis=1))
        self.assert_eq(
            kdf1.truncate("B", "C", copy=False, axis=1),
            pdf1.truncate("B", "C", copy=False, axis=1),
        )

        # Exceptions
        kdf = ks.DataFrame(
            {
                "A": ["a", "b", "c", "d", "e", "f", "g"],
                "B": ["h", "i", "j", "k", "l", "m", "n"],
                "C": ["o", "p", "q", "r", "s", "t", "u"],
            },
            index=[-500, 100, 400, 0, -1, 550, -20],
        )
        msg = "truncate requires a sorted index"
        with self.assertRaisesRegex(ValueError, msg):
            kdf.truncate()

        kdf = ks.DataFrame(
            {
                "A": ["a", "b", "c", "d", "e", "f", "g"],
                "B": ["h", "i", "j", "k", "l", "m", "n"],
                "C": ["o", "p", "q", "r", "s", "t", "u"],
            },
            index=[-500, -20, -1, 0, 400, 550, 1000],
        )
        msg = "Truncate: -20 must be after 400"
        with self.assertRaisesRegex(ValueError, msg):
            kdf.truncate(400, -20)
        msg = "Truncate: B must be after C"
        with self.assertRaisesRegex(ValueError, msg):
            kdf.truncate("C", "B", axis=1)

    def test_explode(self):
        pdf = pd.DataFrame({"A": [[-1.0, np.nan], [0.0, np.inf], [1.0, -np.inf]], "B": 1})
        pdf.index.name = "index"
        pdf.columns.name = "columns"
        kdf = ks.from_pandas(pdf)

        if LooseVersion(pd.__version__) >= LooseVersion("0.25.0"):
            expected_result1 = pdf.explode("A")
            expected_result2 = pdf.explode("B")
        else:
            expected_result1 = pd.DataFrame(
                {"A": [-1, np.nan, 0, np.inf, 1, -np.inf], "B": [1, 1, 1, 1, 1, 1]},
                index=pd.Index([0, 0, 1, 1, 2, 2]),
            )
            expected_result1.index.name = "index"
            expected_result1.columns.name = "columns"
            expected_result2 = pdf

        self.assert_eq(kdf.explode("A"), expected_result1, almost=True)
        self.assert_eq(repr(kdf.explode("B")), repr(expected_result2))
        self.assert_eq(kdf.explode("A").index.name, expected_result1.index.name)
        self.assert_eq(kdf.explode("A").columns.name, expected_result1.columns.name)

        self.assertRaises(ValueError, lambda: kdf.explode(["A", "B"]))

        # MultiIndex
        midx = pd.MultiIndex.from_tuples(
            [("x", "a"), ("x", "b"), ("y", "c")], names=["index1", "index2"]
        )
        pdf.index = midx
        kdf = ks.from_pandas(pdf)

        if LooseVersion(pd.__version__) >= LooseVersion("0.25.0"):
            expected_result1 = pdf.explode("A")
            expected_result2 = pdf.explode("B")
        else:
            midx = pd.MultiIndex.from_tuples(
                [("x", "a"), ("x", "a"), ("x", "b"), ("x", "b"), ("y", "c"), ("y", "c")],
                names=["index1", "index2"],
            )
            expected_result1.index = midx
            expected_result2 = pdf

        self.assert_eq(kdf.explode("A"), expected_result1, almost=True)
        self.assert_eq(repr(kdf.explode("B")), repr(expected_result2))
        self.assert_eq(kdf.explode("A").index.names, expected_result1.index.names)
        self.assert_eq(kdf.explode("A").columns.name, expected_result1.columns.name)

        self.assertRaises(ValueError, lambda: kdf.explode(["A", "B"]))

        # MultiIndex columns
        columns = pd.MultiIndex.from_tuples([("A", "Z"), ("B", "X")], names=["column1", "column2"])
        pdf.columns = columns
        kdf.columns = columns

        if LooseVersion(pd.__version__) >= LooseVersion("0.25.0"):
            expected_result1 = pdf.explode(("A", "Z"))
            expected_result2 = pdf.explode(("B", "X"))
            expected_result3 = pdf.A.explode("Z")
        else:
            expected_result1.columns = columns
            expected_result2 = pdf
            expected_result3 = pd.DataFrame({"Z": [-1, np.nan, 0, np.inf, 1, -np.inf]}, index=midx)
            expected_result3.index.name = "index"
            expected_result3.columns.name = "column2"

        self.assert_eq(kdf.explode(("A", "Z")), expected_result1, almost=True)
        self.assert_eq(repr(kdf.explode(("B", "X"))), repr(expected_result2))
        self.assert_eq(kdf.explode(("A", "Z")).index.names, expected_result1.index.names)
        self.assert_eq(kdf.explode(("A", "Z")).columns.names, expected_result1.columns.names)

        self.assert_eq(kdf.A.explode("Z"), expected_result3, almost=True)

        self.assertRaises(ValueError, lambda: kdf.explode(["A", "B"]))
        self.assertRaises(ValueError, lambda: kdf.explode("A"))

    def test_spark_schema(self):
        kdf = ks.DataFrame(
            {
                "a": list("abc"),
                "b": list(range(1, 4)),
                "c": np.arange(3, 6).astype("i1"),
                "d": np.arange(4.0, 7.0, dtype="float64"),
                "e": [True, False, True],
                "f": pd.date_range("20130101", periods=3),
            },
            columns=["a", "b", "c", "d", "e", "f"],
        )
        self.assertEqual(kdf.spark_schema(), kdf.spark.schema())
        self.assertEqual(kdf.spark_schema("index"), kdf.spark.schema("index"))

    def test_print_schema(self):
        kdf = ks.DataFrame(
            {"a": list("abc"), "b": list(range(1, 4)), "c": np.arange(3, 6).astype("i1")},
            columns=["a", "b", "c"],
        )

        prev = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            kdf.print_schema()
            actual = out.getvalue().strip()

            out = StringIO()
            sys.stdout = out
            kdf.spark.print_schema()
            expected = out.getvalue().strip()

            self.assertEqual(actual, expected)
        finally:
            sys.stdout = prev

    def test_explain_hint(self):
        kdf1 = ks.DataFrame(
            {"lkey": ["foo", "bar", "baz", "foo"], "value": [1, 2, 3, 5]}, columns=["lkey", "value"]
        )
        kdf2 = ks.DataFrame(
            {"rkey": ["foo", "bar", "baz", "foo"], "value": [5, 6, 7, 8]}, columns=["rkey", "value"]
        )
        merged = kdf1.merge(kdf2.hint("broadcast"), left_on="lkey", right_on="rkey")
        prev = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            merged.explain()
            actual = out.getvalue().strip()

            out = StringIO()
            sys.stdout = out
            merged.spark.explain()
            expected = out.getvalue().strip()

            self.assertEqual(actual, expected)
        finally:
            sys.stdout = prev

    def test_mad(self):
        pdf = pd.DataFrame(
            {
                "A": [1, 2, None, 4, np.nan],
                "B": [-0.1, 0.2, -0.3, np.nan, 0.5],
                "C": ["a", "b", "c", "d", "e"],
            }
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.mad(), pdf.mad())
        self.assert_eq(kdf.mad(axis=1), pdf.mad(axis=1))

        with self.assertRaises(ValueError):
            kdf.mad(axis=2)

        # MultiIndex columns
        columns = pd.MultiIndex.from_tuples([("A", "X"), ("A", "Y"), ("A", "Z")])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.mad(), pdf.mad())
        self.assert_eq(kdf.mad(axis=1), pdf.mad(axis=1))

        pdf = pd.DataFrame({"A": [True, True, False, False], "B": [True, False, False, True]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.mad(), pdf.mad())
        self.assert_eq(kdf.mad(axis=1), pdf.mad(axis=1))

    def test_abs(self):
        pdf = pd.DataFrame({"a": [-2, -1, 0, 1]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(abs(kdf), abs(pdf))
        self.assert_eq(np.abs(kdf), np.abs(pdf))

    def test_iteritems(self):
        pdf = pd.DataFrame(
            {"species": ["bear", "bear", "marsupial"], "population": [1864, 22000, 80000]},
            index=["panda", "polar", "koala"],
            columns=["species", "population"],
        )
        kdf = ks.from_pandas(pdf)

        for (p_name, p_items), (k_name, k_items) in zip(pdf.iteritems(), kdf.iteritems()):
            self.assert_eq(p_name, k_name)
            self.assert_eq(p_items, k_items)

    @unittest.skipIf(
        LooseVersion(pyspark.__version__) < LooseVersion("3.0"),
        "tail won't work properly with PySpark<3.0",
    )
    def test_tail(self):
        pdf = pd.DataFrame({"x": range(1000)})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.tail(), kdf.tail())
        self.assert_eq(pdf.tail(10), kdf.tail(10))
        self.assert_eq(pdf.tail(-990), kdf.tail(-990))
        self.assert_eq(pdf.tail(0), kdf.tail(0))
        self.assert_eq(pdf.tail(-1001), kdf.tail(-1001))
        self.assert_eq(pdf.tail(1001), kdf.tail(1001))
        with self.assertRaisesRegex(TypeError, "bad operand type for unary -: 'str'"):
            kdf.tail("10")

    def test_last_valid_index(self):
        # `pyspark.sql.dataframe.DataFrame.tail` is new in pyspark >= 3.0.
        if LooseVersion(pyspark.__version__) >= LooseVersion("3.0"):
            pdf = pd.DataFrame(
                {"a": [1, 2, 3, None], "b": [1.0, 2.0, 3.0, None], "c": [100, 200, 400, None]},
                index=["Q", "W", "E", "R"],
            )
            kdf = ks.from_pandas(pdf)
            self.assert_eq(pdf.last_valid_index(), kdf.last_valid_index())

            # MultiIndex columns
            pdf.columns = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
            kdf = ks.from_pandas(pdf)
            self.assert_eq(pdf.last_valid_index(), kdf.last_valid_index())

            # Empty DataFrame
            pdf = pd.Series([]).to_frame()
            kdf = ks.Series([]).to_frame()
            self.assert_eq(pdf.last_valid_index(), kdf.last_valid_index())

    def test_first_valid_index(self):
        # Empty DataFrame
        pdf = pd.Series([]).to_frame()
        kdf = ks.Series([]).to_frame()
        self.assert_eq(pdf.first_valid_index(), kdf.first_valid_index())

    def test_product(self):
        pdf = pd.DataFrame(
            {"A": [1, 2, 3, 4, 5], "B": [10, 20, 30, 40, 50], "C": ["a", "b", "c", "d", "e"]}
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index())

        # Named columns
        pdf.columns.name = "Koalas"
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index())

        # MultiIndex columns
        pdf.columns = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index())

        # Named MultiIndex columns
        pdf.columns.names = ["Hello", "Koalas"]
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index())

        # No numeric columns
        pdf = pd.DataFrame({"key": ["a", "b", "c"], "val": ["x", "y", "z"]})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index())

        # No numeric named columns
        pdf.columns.name = "Koalas"
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index(), almost=True)

        # No numeric MultiIndex columns
        pdf.columns = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index(), almost=True)

        # No numeric named MultiIndex columns
        pdf.columns.names = ["Hello", "Koalas"]
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index(), almost=True)

        # All NaN columns
        pdf = pd.DataFrame(
            {
                "A": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "B": [10, 20, 30, 40, 50],
                "C": ["a", "b", "c", "d", "e"],
            }
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index(), check_exact=False)

        # All NaN named columns
        pdf.columns.name = "Koalas"
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index(), check_exact=False)

        # All NaN MultiIndex columns
        pdf.columns = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index(), check_exact=False)

        # All NaN named MultiIndex columns
        pdf.columns.names = ["Hello", "Koalas"]
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.prod(), kdf.prod().sort_index(), check_exact=False)

    def test_from_dict(self):
        data = {"row_1": [3, 2, 1, 0], "row_2": [10, 20, 30, 40]}
        pdf = pd.DataFrame.from_dict(data)
        kdf = ks.DataFrame.from_dict(data)
        self.assert_eq(pdf, kdf)

        pdf = pd.DataFrame.from_dict(data, dtype="int8")
        kdf = ks.DataFrame.from_dict(data, dtype="int8")
        self.assert_eq(pdf, kdf)

        pdf = pd.DataFrame.from_dict(data, orient="index", columns=["A", "B", "C", "D"])
        kdf = ks.DataFrame.from_dict(data, orient="index", columns=["A", "B", "C", "D"])
        self.assert_eq(pdf, kdf)

    def test_pad(self):
        pdf = pd.DataFrame(
            {
                "A": [None, 3, None, None],
                "B": [2, 4, None, 3],
                "C": [None, None, None, 1],
                "D": [0, 1, 5, 4],
            },
            columns=["A", "B", "C", "D"],
        )
        kdf = ks.from_pandas(pdf)

        if LooseVersion(pd.__version__) >= LooseVersion("1.1"):
            self.assert_eq(pdf.pad(), kdf.pad())

            # Test `inplace=True`
            pdf.pad(inplace=True)
            kdf.pad(inplace=True)
            self.assert_eq(pdf, kdf)
        else:
            expected = ks.DataFrame(
                {
                    "A": [None, 3, 3, 3],
                    "B": [2.0, 4.0, 4.0, 3.0],
                    "C": [None, None, None, 1],
                    "D": [0, 1, 5, 4],
                },
                columns=["A", "B", "C", "D"],
            )
            self.assert_eq(expected, kdf.pad())

            # Test `inplace=True`
            kdf.pad(inplace=True)
            self.assert_eq(expected, kdf)

    def test_backfill(self):
        pdf = pd.DataFrame(
            {
                "A": [None, 3, None, None],
                "B": [2, 4, None, 3],
                "C": [None, None, None, 1],
                "D": [0, 1, 5, 4],
            },
            columns=["A", "B", "C", "D"],
        )
        kdf = ks.from_pandas(pdf)

        if LooseVersion(pd.__version__) >= LooseVersion("1.1"):
            self.assert_eq(pdf.backfill(), kdf.backfill())

            # Test `inplace=True`
            pdf.backfill(inplace=True)
            kdf.backfill(inplace=True)
            self.assert_eq(pdf, kdf)
        else:
            expected = ks.DataFrame(
                {
                    "A": [3.0, 3.0, None, None],
                    "B": [2.0, 4.0, 3.0, 3.0],
                    "C": [1.0, 1.0, 1.0, 1.0],
                    "D": [0, 1, 5, 4],
                },
                columns=["A", "B", "C", "D"],
            )
            self.assert_eq(expected, kdf.backfill())

            # Test `inplace=True`
            kdf.backfill(inplace=True)
            self.assert_eq(expected, kdf)
