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

import base64
import unittest
from collections import defaultdict
from distutils.version import LooseVersion
import inspect
from io import BytesIO
from itertools import product
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyspark
from pyspark.ml.linalg import SparseVector

from databricks import koalas as ks
from databricks.koalas import Series
from databricks.koalas.testing.utils import (
    ReusedSQLTestCase,
    SQLTestUtils,
    SPARK_CONF_ARROW_ENABLED,
)
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.missing.series import MissingPandasLikeSeries


class SeriesTest(ReusedSQLTestCase, SQLTestUtils):
    @property
    def pser(self):
        return pd.Series([1, 2, 3, 4, 5, 6, 7], name="x")

    @property
    def kser(self):
        return ks.from_pandas(self.pser)

    def test_series(self):
        kser = self.kser

        self.assertTrue(isinstance(kser, Series))

        self.assert_eq(kser + 1, self.pser + 1)

    def test_series_tuple_name(self):
        pser = self.pser
        pser.name = ("x", "a")

        kser = ks.from_pandas(pser)

        self.assert_eq(kser, pser)
        self.assert_eq(kser.name, pser.name)

        pser.name = ("y", "z")
        kser.name = ("y", "z")

        self.assert_eq(kser, pser)
        self.assert_eq(kser.name, pser.name)

    def test_repr_cache_invalidation(self):
        # If there is any cache, inplace operations should invalidate it.
        s = ks.range(10)["id"]
        s.__repr__()
        s.rename("a", inplace=True)
        self.assertEqual(s.__repr__(), s.rename("a").__repr__())

    def test_empty_series(self):
        a = pd.Series([], dtype="i1")
        b = pd.Series([], dtype="str")

        self.assert_eq(ks.from_pandas(a), a)
        self.assertRaises(ValueError, lambda: ks.from_pandas(b))

        with self.sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
            self.assert_eq(ks.from_pandas(a), a)
            self.assertRaises(ValueError, lambda: ks.from_pandas(b))

    def test_all_null_series(self):
        a = pd.Series([None, None, None], dtype="float64")
        b = pd.Series([None, None, None], dtype="str")

        self.assert_eq(ks.from_pandas(a).dtype, a.dtype)
        self.assertTrue(ks.from_pandas(a).to_pandas().isnull().all())
        self.assertRaises(ValueError, lambda: ks.from_pandas(b))

        with self.sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
            self.assert_eq(ks.from_pandas(a).dtype, a.dtype)
            self.assertTrue(ks.from_pandas(a).to_pandas().isnull().all())
            self.assertRaises(ValueError, lambda: ks.from_pandas(b))

    def test_head(self):
        kser = self.kser
        pser = self.pser

        self.assert_eq(kser.head(3), pser.head(3))
        self.assert_eq(kser.head(0), pser.head(0))
        self.assert_eq(kser.head(-3), pser.head(-3))
        self.assert_eq(kser.head(-10), pser.head(-10))

    def test_rename(self):
        pser = pd.Series([1, 2, 3, 4, 5, 6, 7], name="x")
        kser = ks.from_pandas(pser)

        pser.name = "renamed"
        kser.name = "renamed"
        self.assertEqual(kser.name, "renamed")
        self.assert_eq(kser, pser)

        # pser.name = None
        # kser.name = None
        # self.assertEqual(kser.name, None)
        # self.assert_eq(kser, pser)

        pidx = pser.index
        kidx = kser.index
        pidx.name = "renamed"
        kidx.name = "renamed"
        self.assertEqual(kidx.name, "renamed")
        self.assert_eq(kidx, pidx)

    def test_rename_method(self):
        # Series name
        pser = pd.Series([1, 2, 3, 4, 5, 6, 7], name="x")
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.rename("y"), pser.rename("y"))
        self.assertEqual(kser.name, "x")  # no mutation
        self.assert_eq(kser.rename(), pser.rename())

        self.assert_eq((kser.rename("y") + 1).head(), (pser.rename("y") + 1).head())

        kser.rename("z", inplace=True)
        pser.rename("z", inplace=True)
        self.assertEqual(kser.name, "z")
        self.assert_eq(kser, pser)

        # Series index
        # pser = pd.Series(['a', 'b', 'c', 'd', 'e', 'f', 'g'], name='x')
        # kser = ks.from_pandas(s)

        # TODO: index
        # res = kser.rename(lambda x: x ** 2)
        # self.assert_eq(res, pser.rename(lambda x: x ** 2))

        # res = kser.rename(pser)
        # self.assert_eq(res, pser.rename(pser))

        # res = kser.rename(kser)
        # self.assert_eq(res, pser.rename(pser))

        # res = kser.rename(lambda x: x**2, inplace=True)
        # self.assertis(res, kser)
        # s.rename(lambda x: x**2, inplace=True)
        # self.assert_eq(kser, pser)

    def test_rename_axis(self):
        index = pd.Index(["A", "B", "C"], name="index")
        pser = pd.Series([1.0, 2.0, 3.0], index=index, name="name")
        kser = ks.from_pandas(pser)

        self.assert_eq(
            pser.rename_axis("index2").sort_index(), kser.rename_axis("index2").sort_index(),
        )

        self.assert_eq(
            (pser + 1).rename_axis("index2").sort_index(),
            (kser + 1).rename_axis("index2").sort_index(),
        )

        pser2 = pser.copy()
        kser2 = kser.copy()
        pser2.rename_axis("index2", inplace=True)
        kser2.rename_axis("index2", inplace=True)
        self.assert_eq(pser2.sort_index(), kser2.sort_index())

        self.assertRaises(ValueError, lambda: kser.rename_axis(["index2", "index3"]))
        self.assertRaises(TypeError, lambda: kser.rename_axis(mapper=["index2"], index=["index3"]))

        # index/columns parameters and dict_like/functions mappers introduced in pandas 0.24.0
        if LooseVersion(pd.__version__) >= LooseVersion("0.24.0"):
            self.assert_eq(
                pser.rename_axis(index={"index": "index2", "missing": "index4"}).sort_index(),
                kser.rename_axis(index={"index": "index2", "missing": "index4"}).sort_index(),
            )

            self.assert_eq(
                pser.rename_axis(index=str.upper).sort_index(),
                kser.rename_axis(index=str.upper).sort_index(),
            )
        else:
            expected = kser
            expected.index.name = "index2"
            result = kser.rename_axis(index={"index": "index2", "missing": "index4"}).sort_index()
            self.assert_eq(expected, result)

            expected = kser
            expected.index.name = "INDEX"
            result = kser.rename_axis(index=str.upper).sort_index()
            self.assert_eq(expected, result)

        index = pd.MultiIndex.from_tuples(
            [("A", "B"), ("C", "D"), ("E", "F")], names=["index1", "index2"]
        )
        pser = pd.Series([1.0, 2.0, 3.0], index=index, name="name")
        kser = ks.from_pandas(pser)

        self.assert_eq(
            pser.rename_axis(["index3", "index4"]).sort_index(),
            kser.rename_axis(["index3", "index4"]).sort_index(),
        )

        self.assertRaises(ValueError, lambda: kser.rename_axis(["index3", "index4", "index5"]))

        # index/columns parameters and dict_like/functions mappers introduced in pandas 0.24.0
        if LooseVersion(pd.__version__) >= LooseVersion("0.24.0"):
            self.assert_eq(
                pser.rename_axis(
                    index={"index1": "index3", "index2": "index4", "missing": "index5"}
                ).sort_index(),
                kser.rename_axis(
                    index={"index1": "index3", "index2": "index4", "missing": "index5"}
                ).sort_index(),
            )

            self.assert_eq(
                pser.rename_axis(index=str.upper).sort_index(),
                kser.rename_axis(index=str.upper).sort_index(),
            )
        else:
            expected = kser
            expected.index.names = ["index3", "index4"]
            result = kser.rename_axis(
                index={"index1": "index3", "index2": "index4", "missing": "index5"}
            ).sort_index()
            self.assert_eq(expected, result)

            expected.index.names = ["INDEX1", "INDEX2"]
            result = kser.rename_axis(index=str.upper).sort_index()
            self.assert_eq(expected, result)

    def test_or(self):
        pdf = pd.DataFrame(
            {
                "left": [True, False, True, False, np.nan, np.nan, True, False, np.nan],
                "right": [True, False, False, True, True, False, np.nan, np.nan, np.nan],
            }
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf["left"] | pdf["right"], kdf["left"] | kdf["right"])

    def test_and(self):
        pdf = pd.DataFrame(
            {
                "left": [True, False, True, False, np.nan, np.nan, True, False, np.nan],
                "right": [True, False, False, True, True, False, np.nan, np.nan, np.nan],
            }
        )
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            pdf["left"] & pdf["right"], kdf["left"] & kdf["right"],
        )

    def test_to_numpy(self):
        pser = pd.Series([1, 2, 3, 4, 5, 6, 7], name="x")

        kser = ks.from_pandas(pser)
        self.assert_eq(kser.to_numpy(), pser.values)

    def test_isin(self):
        pser = pd.Series(["lama", "cow", "lama", "beetle", "lama", "hippo"], name="animal")

        kser = ks.from_pandas(pser)

        self.assert_eq(kser.isin(["cow", "lama"]), pser.isin(["cow", "lama"]))
        self.assert_eq(kser.isin({"cow"}), pser.isin({"cow"}))

        msg = "only list-like objects are allowed to be passed to isin()"
        with self.assertRaisesRegex(TypeError, msg):
            kser.isin(1)

    def test_drop_duplicates(self):
        pdf = pd.DataFrame({"animal": ["lama", "cow", "lama", "beetle", "lama", "hippo"]})
        kdf = ks.from_pandas(pdf)

        pser = pdf.animal
        kser = kdf.animal

        self.assert_eq(kser.drop_duplicates().sort_index(), pser.drop_duplicates().sort_index())
        self.assert_eq(
            kser.drop_duplicates(keep="last").sort_index(),
            pser.drop_duplicates(keep="last").sort_index(),
        )

        # inplace
        kser.drop_duplicates(keep=False, inplace=True)
        pser.drop_duplicates(keep=False, inplace=True)
        self.assert_eq(kser.sort_index(), pser.sort_index())
        self.assert_eq(kdf, pdf)

    def test_reindex(self):
        index = ["A", "B", "C", "D", "E"]
        pser = pd.Series([1.0, 2.0, 3.0, 4.0, None], index=index, name="x")
        kser = ks.from_pandas(pser)

        self.assert_eq(pser, kser)

        self.assert_eq(
            pser.reindex(["A", "B"]).sort_index(), kser.reindex(["A", "B"]).sort_index(),
        )

        self.assert_eq(
            pser.reindex(["A", "B", "2", "3"]).sort_index(),
            kser.reindex(["A", "B", "2", "3"]).sort_index(),
        )

        self.assert_eq(
            pser.reindex(["A", "E", "2"], fill_value=0).sort_index(),
            kser.reindex(["A", "E", "2"], fill_value=0).sort_index(),
        )

        self.assertRaises(TypeError, lambda: kser.reindex(index=123))

    def test_fillna(self):
        pdf = pd.DataFrame({"x": [np.nan, 2, 3, 4, np.nan, 6], "y": [np.nan, 2, 3, 4, np.nan, 6]})
        kdf = ks.from_pandas(pdf)

        pser = pdf.x
        kser = kdf.x

        self.assert_eq(kser.fillna(0), pser.fillna(0))
        self.assert_eq(kser.fillna(np.nan).fillna(0), pser.fillna(np.nan).fillna(0))

        kser.fillna(0, inplace=True)
        pser.fillna(0, inplace=True)
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)

        # test considering series does not have NA/NaN values
        kser.fillna(0, inplace=True)
        pser.fillna(0, inplace=True)
        self.assert_eq(kser, pser)

        kser = kdf.x.rename("y")
        pser = pdf.x.rename("y")
        kser.fillna(0, inplace=True)
        pser.fillna(0, inplace=True)
        self.assert_eq(kser.head(), pser.head())

        pser = pd.Series([1, 2, 3, 4, 5, 6], name="x")
        kser = ks.from_pandas(pser)

        pser.loc[3] = np.nan
        kser.loc[3] = np.nan

        self.assert_eq(kser.fillna(0), pser.fillna(0))
        self.assert_eq(kser.fillna(method="ffill"), pser.fillna(method="ffill"))
        self.assert_eq(kser.fillna(method="bfill"), pser.fillna(method="bfill"))

        # inplace fillna on non-nullable column
        pdf = pd.DataFrame({"a": [1, 2, None], "b": [1, 2, 3]})
        kdf = ks.from_pandas(pdf)

        pser = pdf.b
        kser = kdf.b

        self.assert_eq(kser.fillna(0), pser.fillna(0))
        self.assert_eq(kser.fillna(np.nan).fillna(0), pser.fillna(np.nan).fillna(0))

        kser.fillna(0, inplace=True)
        pser.fillna(0, inplace=True)
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)

    def test_dropna(self):
        pdf = pd.DataFrame({"x": [np.nan, 2, 3, 4, np.nan, 6]})
        kdf = ks.from_pandas(pdf)

        pser = pdf.x
        kser = kdf.x

        self.assert_eq(kser.dropna(), pser.dropna())

        pser.dropna(inplace=True)
        kser.dropna(inplace=True)
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)

    def test_nunique(self):
        pser = pd.Series([1, 2, 1, np.nan])
        kser = ks.from_pandas(pser)

        # Assert NaNs are dropped by default
        nunique_result = kser.nunique()
        self.assertEqual(nunique_result, 2)
        self.assert_eq(nunique_result, pser.nunique())

        # Assert including NaN values
        nunique_result = kser.nunique(dropna=False)
        self.assertEqual(nunique_result, 3)
        self.assert_eq(nunique_result, pser.nunique(dropna=False))

        # Assert approximate counts
        self.assertEqual(ks.Series(range(100)).nunique(approx=True), 103)
        self.assertEqual(ks.Series(range(100)).nunique(approx=True, rsd=0.01), 100)

    def _test_value_counts(self):
        # this is also containing test for Index & MultiIndex
        pser = pd.Series(
            [1, 2, 1, 3, 3, np.nan, 1, 4, 2, np.nan, 3, np.nan, 3, 1, 3],
            index=[1, 2, 1, 3, 3, np.nan, 1, 4, 2, np.nan, 3, np.nan, 3, 1, 3],
            name="x",
        )
        kser = ks.from_pandas(pser)

        exp = pser.value_counts()
        res = kser.value_counts()
        self.assertEqual(res.name, exp.name)
        self.assert_eq(res, exp)

        self.assert_eq(kser.value_counts(normalize=True), pser.value_counts(normalize=True))
        self.assert_eq(kser.value_counts(ascending=True), pser.value_counts(ascending=True))
        self.assert_eq(
            kser.value_counts(normalize=True, dropna=False),
            pser.value_counts(normalize=True, dropna=False),
        )
        self.assert_eq(
            kser.value_counts(ascending=True, dropna=False),
            pser.value_counts(ascending=True, dropna=False),
        )

        self.assert_eq(
            kser.index.value_counts(normalize=True), pser.index.value_counts(normalize=True)
        )
        self.assert_eq(
            kser.index.value_counts(ascending=True), pser.index.value_counts(ascending=True)
        )
        self.assert_eq(
            kser.index.value_counts(normalize=True, dropna=False),
            pser.index.value_counts(normalize=True, dropna=False),
        )
        self.assert_eq(
            kser.index.value_counts(ascending=True, dropna=False),
            pser.index.value_counts(ascending=True, dropna=False),
        )

        with self.assertRaisesRegex(
            NotImplementedError, "value_counts currently does not support bins"
        ):
            kser.value_counts(bins=3)

        pser.name = "index"
        kser.name = "index"
        self.assert_eq(kser.value_counts(), pser.value_counts())

        # Series from DataFrame
        pdf = pd.DataFrame({"a": [2, 2, 3], "b": [None, 1, None]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.a.value_counts(normalize=True), pdf.a.value_counts(normalize=True))
        self.assert_eq(kdf.a.value_counts(ascending=True), pdf.a.value_counts(ascending=True))
        self.assert_eq(
            kdf.a.value_counts(normalize=True, dropna=False),
            pdf.a.value_counts(normalize=True, dropna=False),
        )
        self.assert_eq(
            kdf.a.value_counts(ascending=True, dropna=False),
            pdf.a.value_counts(ascending=True, dropna=False),
        )

        self.assert_eq(
            kser.index.value_counts(normalize=True), pser.index.value_counts(normalize=True)
        )
        self.assert_eq(
            kser.index.value_counts(ascending=True), pser.index.value_counts(ascending=True)
        )
        self.assert_eq(
            kser.index.value_counts(normalize=True, dropna=False),
            pser.index.value_counts(normalize=True, dropna=False),
        )
        self.assert_eq(
            kser.index.value_counts(ascending=True, dropna=False),
            pser.index.value_counts(ascending=True, dropna=False),
        )

        # Series with NaN index
        pser = pd.Series([3, 2, 3, 1, 2, 3], index=[2.0, None, 5.0, 5.0, None, 5.0])
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.value_counts(normalize=True), pser.value_counts(normalize=True))
        self.assert_eq(kser.value_counts(ascending=True), pser.value_counts(ascending=True))
        self.assert_eq(
            kser.value_counts(normalize=True, dropna=False),
            pser.value_counts(normalize=True, dropna=False),
        )
        self.assert_eq(
            kser.value_counts(ascending=True, dropna=False),
            pser.value_counts(ascending=True, dropna=False),
        )

        self.assert_eq(
            kser.index.value_counts(normalize=True), pser.index.value_counts(normalize=True)
        )
        self.assert_eq(
            kser.index.value_counts(ascending=True), pser.index.value_counts(ascending=True)
        )
        self.assert_eq(
            kser.index.value_counts(normalize=True, dropna=False),
            pser.index.value_counts(normalize=True, dropna=False),
        )
        self.assert_eq(
            kser.index.value_counts(ascending=True, dropna=False),
            pser.index.value_counts(ascending=True, dropna=False),
        )

        # Series with MultiIndex
        pser.index = pd.MultiIndex.from_tuples(
            [("x", "a"), ("x", "b"), ("y", "c"), ("x", "a"), ("y", "c"), ("x", "a")]
        )
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.value_counts(normalize=True), pser.value_counts(normalize=True))
        self.assert_eq(kser.value_counts(ascending=True), pser.value_counts(ascending=True))
        self.assert_eq(
            kser.value_counts(normalize=True, dropna=False),
            pser.value_counts(normalize=True, dropna=False),
        )
        self.assert_eq(
            kser.value_counts(ascending=True, dropna=False),
            pser.value_counts(ascending=True, dropna=False),
        )

        # FIXME: MultiIndex.value_counts returns wrong indices.
        self.assert_eq(
            kser.index.value_counts(normalize=True),
            pser.index.value_counts(normalize=True),
            almost=True,
        )
        self.assert_eq(
            kser.index.value_counts(ascending=True),
            pser.index.value_counts(ascending=True),
            almost=True,
        )
        self.assert_eq(
            kser.index.value_counts(normalize=True, dropna=False),
            pser.index.value_counts(normalize=True, dropna=False),
            almost=True,
        )
        self.assert_eq(
            kser.index.value_counts(ascending=True, dropna=False),
            pser.index.value_counts(ascending=True, dropna=False),
            almost=True,
        )

        # Series with MultiIndex some of index has NaN
        pser.index = pd.MultiIndex.from_tuples(
            [("x", "a"), ("x", None), ("y", "c"), ("x", "a"), ("y", "c"), ("x", "a")]
        )
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.value_counts(normalize=True), pser.value_counts(normalize=True))
        self.assert_eq(kser.value_counts(ascending=True), pser.value_counts(ascending=True))
        self.assert_eq(
            kser.value_counts(normalize=True, dropna=False),
            pser.value_counts(normalize=True, dropna=False),
        )
        self.assert_eq(
            kser.value_counts(ascending=True, dropna=False),
            pser.value_counts(ascending=True, dropna=False),
        )

        # FIXME: MultiIndex.value_counts returns wrong indices.
        self.assert_eq(
            kser.index.value_counts(normalize=True),
            pser.index.value_counts(normalize=True),
            almost=True,
        )
        self.assert_eq(
            kser.index.value_counts(ascending=True),
            pser.index.value_counts(ascending=True),
            almost=True,
        )
        self.assert_eq(
            kser.index.value_counts(normalize=True, dropna=False),
            pser.index.value_counts(normalize=True, dropna=False),
            almost=True,
        )
        self.assert_eq(
            kser.index.value_counts(ascending=True, dropna=False),
            pser.index.value_counts(ascending=True, dropna=False),
            almost=True,
        )

        # Series with MultiIndex some of index is NaN.
        # This test only available for pandas >= 0.24.
        if LooseVersion(pd.__version__) >= LooseVersion("0.24"):
            pser.index = pd.MultiIndex.from_tuples(
                [("x", "a"), None, ("y", "c"), ("x", "a"), ("y", "c"), ("x", "a")]
            )
            kser = ks.from_pandas(pser)

            self.assert_eq(kser.value_counts(normalize=True), pser.value_counts(normalize=True))
            self.assert_eq(kser.value_counts(ascending=True), pser.value_counts(ascending=True))
            self.assert_eq(
                kser.value_counts(normalize=True, dropna=False),
                pser.value_counts(normalize=True, dropna=False),
            )
            self.assert_eq(
                kser.value_counts(ascending=True, dropna=False),
                pser.value_counts(ascending=True, dropna=False),
            )

            # FIXME: MultiIndex.value_counts returns wrong indices.
            self.assert_eq(
                kser.index.value_counts(normalize=True),
                pser.index.value_counts(normalize=True),
                almost=True,
            )
            self.assert_eq(
                kser.index.value_counts(ascending=True),
                pser.index.value_counts(ascending=True),
                almost=True,
            )
            self.assert_eq(
                kser.index.value_counts(normalize=True, dropna=False),
                pser.index.value_counts(normalize=True, dropna=False),
                almost=True,
            )
            self.assert_eq(
                kser.index.value_counts(ascending=True, dropna=False),
                pser.index.value_counts(ascending=True, dropna=False),
                almost=True,
            )

    def test_value_counts(self):
        if LooseVersion(pyspark.__version__) < LooseVersion("2.4"):
            with self.sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
                self._test_value_counts()
            self.assertRaises(
                RuntimeError,
                lambda: ks.MultiIndex.from_tuples([("x", "a"), ("x", "b")]).value_counts(),
            )
        else:
            self._test_value_counts()

    def test_nsmallest(self):
        sample_lst = [1, 2, 3, 4, np.nan, 6]
        pser = pd.Series(sample_lst, name="x")
        kser = ks.Series(sample_lst, name="x")
        self.assert_eq(kser.nsmallest(n=3), pser.nsmallest(n=3))
        self.assert_eq(kser.nsmallest(), pser.nsmallest())
        self.assert_eq((kser + 1).nsmallest(), (pser + 1).nsmallest())

    def test_nlargest(self):
        sample_lst = [1, 2, 3, 4, np.nan, 6]
        pser = pd.Series(sample_lst, name="x")
        kser = ks.Series(sample_lst, name="x")
        self.assert_eq(kser.nlargest(n=3), pser.nlargest(n=3))
        self.assert_eq(kser.nlargest(), pser.nlargest())
        self.assert_eq((kser + 1).nlargest(), (pser + 1).nlargest())

    def test_isnull(self):
        pser = pd.Series([1, 2, 3, 4, np.nan, 6], name="x")
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.notnull(), pser.notnull())
        self.assert_eq(kser.isnull(), pser.isnull())

        pser = self.pser
        kser = self.kser

        self.assert_eq(kser.notnull(), pser.notnull())
        self.assert_eq(kser.isnull(), pser.isnull())

    def test_all(self):
        for pser in [
            pd.Series([True, True], name="x"),
            pd.Series([True, False], name="x"),
            pd.Series([0, 1], name="x"),
            pd.Series([1, 2, 3], name="x"),
            pd.Series([True, True, None], name="x"),
            pd.Series([True, False, None], name="x"),
            pd.Series([], name="x"),
            pd.Series([np.nan], name="x"),
        ]:
            kser = ks.from_pandas(pser)
            self.assert_eq(kser.all(), pser.all())

        pser = pd.Series([1, 2, 3, 4], name="x")
        kser = ks.from_pandas(pser)

        self.assert_eq((kser % 2 == 0).all(), (pser % 2 == 0).all())

        with self.assertRaisesRegex(
            NotImplementedError, 'axis should be either 0 or "index" currently.'
        ):
            kser.all(axis=1)

    def test_any(self):
        for pser in [
            pd.Series([False, False], name="x"),
            pd.Series([True, False], name="x"),
            pd.Series([0, 1], name="x"),
            pd.Series([1, 2, 3], name="x"),
            pd.Series([True, True, None], name="x"),
            pd.Series([True, False, None], name="x"),
            pd.Series([], name="x"),
            pd.Series([np.nan], name="x"),
        ]:
            kser = ks.from_pandas(pser)
            self.assert_eq(kser.any(), pser.any())

        pser = pd.Series([1, 2, 3, 4], name="x")
        kser = ks.from_pandas(pser)

        self.assert_eq((kser % 2 == 0).any(), (pser % 2 == 0).any())

        with self.assertRaisesRegex(
            NotImplementedError, 'axis should be either 0 or "index" currently.'
        ):
            kser.any(axis=1)

    def test_reset_index(self):
        pdf = pd.DataFrame({"foo": [1, 2, 3, 4]}, index=pd.Index(["a", "b", "c", "d"], name="idx"))
        kdf = ks.from_pandas(pdf)

        pser = pdf.foo
        kser = kdf.foo

        self.assert_eq(kser.reset_index(), pser.reset_index())
        self.assert_eq(kser.reset_index(name="values"), pser.reset_index(name="values"))
        self.assert_eq(kser.reset_index(drop=True), pser.reset_index(drop=True))

        # inplace
        kser.reset_index(drop=True, inplace=True)
        pser.reset_index(drop=True, inplace=True)
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)

    def test_reset_index_with_default_index_types(self):
        pser = pd.Series([1, 2, 3], name="0", index=np.random.rand(3))
        kser = ks.from_pandas(pser)

        with ks.option_context("compute.default_index_type", "sequence"):
            self.assert_eq(kser.reset_index(), pser.reset_index())

        with ks.option_context("compute.default_index_type", "distributed-sequence"):
            # the order might be changed.
            self.assert_eq(kser.reset_index().sort_index(), pser.reset_index())

        with ks.option_context("compute.default_index_type", "distributed"):
            # the index is different.
            self.assert_eq(
                kser.reset_index().to_pandas().reset_index(drop=True), pser.reset_index()
            )

    def test_sort_values(self):
        pdf = pd.DataFrame({"x": [1, 2, 3, 4, 5, None, 7]})
        kdf = ks.from_pandas(pdf)

        pser = pdf.x
        kser = kdf.x

        self.assert_eq(kser.sort_values(), pser.sort_values())
        self.assert_eq(kser.sort_values(ascending=False), pser.sort_values(ascending=False))
        self.assert_eq(kser.sort_values(na_position="first"), pser.sort_values(na_position="first"))

        self.assertRaises(ValueError, lambda: kser.sort_values(na_position="invalid"))

        # inplace
        # pandas raises an exception when the Series is derived from DataFrame
        kser.sort_values(inplace=True)
        self.assert_eq(kser, pser.sort_values())
        self.assert_eq(kdf, pdf)

        pser = pdf.x.copy()
        kser = kdf.x.copy()

        kser.sort_values(inplace=True)
        pser.sort_values(inplace=True)
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)

    def test_sort_index(self):
        pdf = pd.DataFrame({"x": [2, 1, np.nan]}, index=["b", "a", np.nan])
        kdf = ks.from_pandas(pdf)

        pser = pdf.x
        kser = kdf.x

        # Assert invalid parameters
        self.assertRaises(NotImplementedError, lambda: kser.sort_index(axis=1))
        self.assertRaises(NotImplementedError, lambda: kser.sort_index(kind="mergesort"))
        self.assertRaises(ValueError, lambda: kser.sort_index(na_position="invalid"))

        # Assert default behavior without parameters
        self.assert_eq(kser.sort_index(), pser.sort_index())
        # Assert sorting descending
        self.assert_eq(kser.sort_index(ascending=False), pser.sort_index(ascending=False))
        # Assert sorting NA indices first
        self.assert_eq(kser.sort_index(na_position="first"), pser.sort_index(na_position="first"))

        # Assert sorting inplace
        # pandas sorts pdf.x by the index and update the column only
        # when the Series is derived from DataFrame.
        kser.sort_index(inplace=True)
        self.assert_eq(kser, pser.sort_index())
        self.assert_eq(kdf, pdf)

        pser = pdf.x.copy()
        kser = kdf.x.copy()

        kser.sort_index(inplace=True)
        pser.sort_index(inplace=True)
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)

        # Assert multi-indices
        pser = pd.Series(range(4), index=[["b", "b", "a", "a"], [1, 0, 1, 0]], name="0")
        kser = ks.from_pandas(pser)
        self.assert_eq(kser.sort_index(), pser.sort_index())
        self.assert_eq(kser.sort_index(level=[1, 0]), pser.sort_index(level=[1, 0]))

        self.assert_eq(kser.reset_index().sort_index(), pser.reset_index().sort_index())

    def test_to_datetime(self):
        pser = pd.Series(["3/11/2000", "3/12/2000", "3/13/2000"] * 100)
        kser = ks.from_pandas(pser)

        self.assert_eq(
            pd.to_datetime(pser, infer_datetime_format=True),
            ks.to_datetime(kser, infer_datetime_format=True),
        )

    def test_missing(self):
        kser = self.kser

        missing_functions = inspect.getmembers(MissingPandasLikeSeries, inspect.isfunction)
        unsupported_functions = [
            name for (name, type_) in missing_functions if type_.__name__ == "unsupported_function"
        ]
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                PandasNotImplementedError,
                "method.*Series.*{}.*not implemented( yet\\.|\\. .+)".format(name),
            ):
                getattr(kser, name)()

        deprecated_functions = [
            name for (name, type_) in missing_functions if type_.__name__ == "deprecated_function"
        ]
        for name in deprecated_functions:
            with self.assertRaisesRegex(
                PandasNotImplementedError, "method.*Series.*{}.*is deprecated".format(name)
            ):
                getattr(kser, name)()

        missing_properties = inspect.getmembers(
            MissingPandasLikeSeries, lambda o: isinstance(o, property)
        )
        unsupported_properties = [
            name
            for (name, type_) in missing_properties
            if type_.fget.__name__ == "unsupported_property"
        ]
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                PandasNotImplementedError,
                "property.*Series.*{}.*not implemented( yet\\.|\\. .+)".format(name),
            ):
                getattr(kser, name)
        deprecated_properties = [
            name
            for (name, type_) in missing_properties
            if type_.fget.__name__ == "deprecated_property"
        ]
        for name in deprecated_properties:
            with self.assertRaisesRegex(
                PandasNotImplementedError, "property.*Series.*{}.*is deprecated".format(name)
            ):
                getattr(kser, name)

    def test_clip(self):
        pser = pd.Series([0, 2, 4], index=np.random.rand(3))
        kser = ks.from_pandas(pser)

        # Assert list-like values are not accepted for 'lower' and 'upper'
        msg = "List-like value are not supported for 'lower' and 'upper' at the moment"
        with self.assertRaises(ValueError, msg=msg):
            kser.clip(lower=[1])
        with self.assertRaises(ValueError, msg=msg):
            kser.clip(upper=[1])

        # Assert no lower or upper
        self.assert_eq(kser.clip(), pser.clip())
        # Assert lower only
        self.assert_eq(kser.clip(1), pser.clip(1))
        # Assert upper only
        self.assert_eq(kser.clip(upper=3), pser.clip(upper=3))
        # Assert lower and upper
        self.assert_eq(kser.clip(1, 3), pser.clip(1, 3))

        # Assert behavior on string values
        str_kser = ks.Series(["a", "b", "c"])
        self.assert_eq(str_kser.clip(1, 3), str_kser)

    def test_is_unique(self):
        # We can't use pandas' is_unique for comparison. pandas 0.23 ignores None
        pser = pd.Series([1, 2, 2, None, None])
        kser = ks.from_pandas(pser)
        self.assertEqual(False, kser.is_unique)
        self.assertEqual(False, (kser + 1).is_unique)

        pser = pd.Series([1, None, None])
        kser = ks.from_pandas(pser)
        self.assertEqual(False, kser.is_unique)
        self.assertEqual(False, (kser + 1).is_unique)

        pser = pd.Series([1])
        kser = ks.from_pandas(pser)
        self.assertEqual(pser.is_unique, kser.is_unique)
        self.assertEqual((pser + 1).is_unique, (kser + 1).is_unique)

        pser = pd.Series([1, 1, 1])
        kser = ks.from_pandas(pser)
        self.assertEqual(pser.is_unique, kser.is_unique)
        self.assertEqual((pser + 1).is_unique, (kser + 1).is_unique)

    def test_to_list(self):
        if LooseVersion(pd.__version__) >= LooseVersion("0.24.0"):
            self.assert_eq(self.kser.to_list(), self.pser.to_list())
        else:
            self.assert_eq(self.kser.tolist(), self.pser.tolist())

    def test_append(self):
        pser1 = pd.Series([1, 2, 3], name="0")
        pser2 = pd.Series([4, 5, 6], name="0")
        pser3 = pd.Series([4, 5, 6], index=[3, 4, 5], name="0")
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)
        kser3 = ks.from_pandas(pser3)

        self.assert_eq(kser1.append(kser2), pser1.append(pser2))
        self.assert_eq(kser1.append(kser3), pser1.append(pser3))
        self.assert_eq(
            kser1.append(kser2, ignore_index=True), pser1.append(pser2, ignore_index=True)
        )

        kser1.append(kser3, verify_integrity=True)
        msg = "Indices have overlapping values"
        with self.assertRaises(ValueError, msg=msg):
            kser1.append(kser2, verify_integrity=True)

    def test_map(self):
        pser = pd.Series(["cat", "dog", None, "rabbit"])
        kser = ks.from_pandas(pser)
        # Currently Koalas doesn't return NaN as pandas does.
        self.assert_eq(kser.map({}), pser.map({}).replace({pd.np.nan: None}))

        d = defaultdict(lambda: "abc")
        self.assertTrue("abc" in repr(kser.map(d)))
        self.assert_eq(kser.map(d), pser.map(d))

        def tomorrow(date) -> datetime:
            return date + timedelta(days=1)

        pser = pd.Series([datetime(2019, 10, 24)])
        kser = ks.from_pandas(pser)
        self.assert_eq(kser.map(tomorrow), pser.map(tomorrow))

    def test_add_prefix(self):
        pser = pd.Series([1, 2, 3, 4], name="0")
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.add_prefix("item_"), kser.add_prefix("item_"))

        pser = pd.Series(
            [1, 2, 3],
            name="0",
            index=pd.MultiIndex.from_tuples([("A", "X"), ("A", "Y"), ("B", "X")]),
        )
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.add_prefix("item_"), kser.add_prefix("item_"))

    def test_add_suffix(self):
        pser = pd.Series([1, 2, 3, 4], name="0")
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.add_suffix("_item"), kser.add_suffix("_item"))

        pser = pd.Series(
            [1, 2, 3],
            name="0",
            index=pd.MultiIndex.from_tuples([("A", "X"), ("A", "Y"), ("B", "X")]),
        )
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.add_suffix("_item"), kser.add_suffix("_item"))

    def test_hist(self):
        pdf = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50],}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10]
        )

        kdf = ks.from_pandas(pdf)

        def plot_to_base64(ax):
            bytes_data = BytesIO()
            ax.figure.savefig(bytes_data, format="png")
            bytes_data.seek(0)
            b64_data = base64.b64encode(bytes_data.read())
            plt.close(ax.figure)
            return b64_data

        _, ax1 = plt.subplots(1, 1)
        # Using plot.hist() because pandas changes ticks props when called hist()
        ax1 = pdf["a"].plot.hist()
        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf["a"].hist()
        self.assert_eq(plot_to_base64(ax1), plot_to_base64(ax2))

    def test_cummin(self):
        pser = pd.Series([1.0, None, 0.0, 4.0, 9.0])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.cummin(), kser.cummin())
        self.assert_eq(pser.cummin(skipna=False), kser.cummin(skipna=False))
        self.assert_eq(pser.cummin().sum(), kser.cummin().sum())

        # with reversed index
        pser.index = [4, 3, 2, 1, 0]
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.cummin(), kser.cummin())
        self.assert_eq(pser.cummin(skipna=False), kser.cummin(skipna=False))

    def test_cummax(self):
        pser = pd.Series([1.0, None, 0.0, 4.0, 9.0])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.cummax(), kser.cummax())
        self.assert_eq(pser.cummax(skipna=False), kser.cummax(skipna=False))
        self.assert_eq(pser.cummax().sum(), kser.cummax().sum())

        # with reversed index
        pser.index = [4, 3, 2, 1, 0]
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.cummax(), kser.cummax())
        self.assert_eq(pser.cummax(skipna=False), kser.cummax(skipna=False))

    def test_cumsum(self):
        pser = pd.Series([1.0, None, 0.0, 4.0, 9.0])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.cumsum(), kser.cumsum())
        self.assert_eq(pser.cumsum(skipna=False), kser.cumsum(skipna=False))
        self.assert_eq(pser.cumsum().sum(), kser.cumsum().sum())

        # with reversed index
        pser.index = [4, 3, 2, 1, 0]
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.cumsum(), kser.cumsum())
        self.assert_eq(pser.cumsum(skipna=False), kser.cumsum(skipna=False))

    def test_cumprod(self):
        pser = pd.Series([1.0, None, 1.0, 4.0, 9.0])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.cumprod(), kser.cumprod())
        self.assert_eq(pser.cumprod(skipna=False), kser.cumprod(skipna=False))
        self.assert_eq(pser.cumprod().sum(), kser.cumprod().sum())

        # with integer type
        pser = pd.Series([1, 10, 1, 4, 9])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.cumprod(), kser.cumprod())
        self.assert_eq(pser.cumprod(skipna=False), kser.cumprod(skipna=False))
        self.assert_eq(pser.cumprod().sum(), kser.cumprod().sum())

        # with reversed index
        pser.index = [4, 3, 2, 1, 0]
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.cumprod(), kser.cumprod())
        self.assert_eq(pser.cumprod(skipna=False), kser.cumprod(skipna=False))

        with self.assertRaisesRegex(Exception, "values should be bigger than 0"):
            ks.Series([0, 1]).cumprod().to_pandas()

    def test_median(self):
        with self.assertRaisesRegex(ValueError, "accuracy must be an integer; however"):
            ks.Series([24.0, 21.0, 25.0, 33.0, 26.0]).median(accuracy="a")

    def test_rank(self):
        pser = pd.Series([1, 2, 3, 1], name="x")
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.rank(), kser.rank().sort_index())
        self.assert_eq(pser.rank(ascending=False), kser.rank(ascending=False).sort_index())
        self.assert_eq(pser.rank(method="min"), kser.rank(method="min").sort_index())
        self.assert_eq(pser.rank(method="max"), kser.rank(method="max").sort_index())
        self.assert_eq(pser.rank(method="first"), kser.rank(method="first").sort_index())
        self.assert_eq(pser.rank(method="dense"), kser.rank(method="dense").sort_index())

        msg = "method must be one of 'average', 'min', 'max', 'first', 'dense'"
        with self.assertRaisesRegex(ValueError, msg):
            kser.rank(method="nothing")

    def test_round(self):
        pser = pd.Series([0.028208, 0.038683, 0.877076], name="x")
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.round(2), kser.round(2))
        msg = "decimals must be an integer"
        with self.assertRaisesRegex(ValueError, msg):
            kser.round(1.5)

    def test_quantile(self):
        with self.assertRaisesRegex(ValueError, "accuracy must be an integer; however"):
            ks.Series([24.0, 21.0, 25.0, 33.0, 26.0]).quantile(accuracy="a")
        with self.assertRaisesRegex(ValueError, "q must be a float of an array of floats;"):
            ks.Series([24.0, 21.0, 25.0, 33.0, 26.0]).quantile(q="a")
        with self.assertRaisesRegex(ValueError, "q must be a float of an array of floats;"):
            ks.Series([24.0, 21.0, 25.0, 33.0, 26.0]).quantile(q=["a"])

    def test_idxmax(self):
        pser = pd.Series(data=[1, 4, 5], index=["A", "B", "C"])
        kser = ks.Series(pser)

        self.assertEqual(kser.idxmax(), pser.idxmax())
        self.assertEqual(kser.idxmax(skipna=False), pser.idxmax(skipna=False))

        index = pd.MultiIndex.from_arrays(
            [["a", "a", "b", "b"], ["c", "d", "e", "f"]], names=("first", "second")
        )
        pser = pd.Series(data=[1, 2, 4, 5], index=index)
        kser = ks.Series(pser)

        self.assertEqual(kser.idxmax(), pser.idxmax())
        self.assertEqual(kser.idxmax(skipna=False), pser.idxmax(skipna=False))

        kser = ks.Series([])
        with self.assertRaisesRegex(ValueError, "an empty sequence"):
            kser.idxmax()

        pser = pd.Series([1, 100, None, 100, 1, 100], index=[10, 3, 5, 2, 1, 8])
        kser = ks.Series(pser)

        self.assertEqual(kser.idxmax(), pser.idxmax())
        self.assertEqual(repr(kser.idxmax(skipna=False)), repr(pser.idxmax(skipna=False)))

    def test_idxmin(self):
        pser = pd.Series(data=[1, 4, 5], index=["A", "B", "C"])
        kser = ks.Series(pser)

        self.assertEqual(kser.idxmin(), pser.idxmin())
        self.assertEqual(kser.idxmin(skipna=False), pser.idxmin(skipna=False))

        index = pd.MultiIndex.from_arrays(
            [["a", "a", "b", "b"], ["c", "d", "e", "f"]], names=("first", "second")
        )
        pser = pd.Series(data=[1, 2, 4, 5], index=index)
        kser = ks.Series(pser)

        self.assertEqual(kser.idxmin(), pser.idxmin())
        self.assertEqual(kser.idxmin(skipna=False), pser.idxmin(skipna=False))

        kser = ks.Series([])
        with self.assertRaisesRegex(ValueError, "an empty sequence"):
            kser.idxmin()

        pser = pd.Series([1, 100, None, 100, 1, 100], index=[10, 3, 5, 2, 1, 8])
        kser = ks.Series(pser)

        self.assertEqual(kser.idxmin(), pser.idxmin())
        self.assertEqual(repr(kser.idxmin(skipna=False)), repr(pser.idxmin(skipna=False)))

    def test_shift(self):
        pser = pd.Series([10, 20, 15, 30, 45], name="x")
        kser = ks.Series(pser)
        if LooseVersion(pd.__version__) < LooseVersion("0.24.2"):
            self.assert_eq(kser.shift(periods=2), pser.shift(periods=2))
        else:
            self.assert_eq(kser.shift(periods=2, fill_value=0), pser.shift(periods=2, fill_value=0))
        with self.assertRaisesRegex(ValueError, "periods should be an int; however"):
            kser.shift(periods=1.5)

    def test_astype(self):
        pser = pd.Series([10, 20, 15, 30, 45], name="x")
        kser = ks.Series(pser)

        self.assert_eq(kser.astype(np.int32), pser.astype(np.int32))
        self.assert_eq(kser.astype(bool), pser.astype(bool))

        pser = pd.Series([10, 20, 15, 30, 45, None, np.nan], name="x")
        kser = ks.Series(pser)

        self.assert_eq(kser.astype(bool), pser.astype(bool))

        pser = pd.Series(["hi", "hi ", " ", " \t", "", None], name="x")
        kser = ks.Series(pser)

        self.assert_eq(kser.astype(bool), pser.astype(bool))
        # TODO: restore after pandas 1.1.4 is released.
        # self.assert_eq(kser.astype(str).tolist(), pser.astype(str).tolist())
        self.assert_eq(kser.astype(str).tolist(), ["hi", "hi ", " ", " \t", "", "None"])
        self.assert_eq(kser.str.strip().astype(bool), pser.str.strip().astype(bool))

        pser = pd.Series([True, False, None], name="x")
        kser = ks.Series(pser)

        self.assert_eq(kser.astype(bool), pser.astype(bool))

        with self.assertRaisesRegex(TypeError, "not understood"):
            kser.astype("int63")

    def test_aggregate(self):
        pser = pd.Series([10, 20, 15, 30, 45], name="x")
        kser = ks.Series(pser)
        msg = "func must be a string or list of strings"
        with self.assertRaisesRegex(ValueError, msg):
            kser.aggregate({"x": ["min", "max"]})
        msg = (
            "If the given function is a list, it " "should only contains function names as strings."
        )
        with self.assertRaisesRegex(ValueError, msg):
            kser.aggregate(["min", max])

    def test_drop(self):
        pser = pd.Series([10, 20, 15, 30, 45], name="x")
        kser = ks.Series(pser)

        self.assert_eq(kser.drop(1), pser.drop(1))
        self.assert_eq(kser.drop([1, 4]), pser.drop([1, 4]))

        msg = "Need to specify at least one of 'labels' or 'index'"
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop()
        self.assertRaises(KeyError, lambda: kser.drop((0, 1)))

        # For MultiIndex
        midx = pd.MultiIndex(
            [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        pser = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.drop("lama"), pser.drop("lama"))
        self.assert_eq(kser.drop(labels="weight", level=1), pser.drop(labels="weight", level=1))
        self.assert_eq(kser.drop(("lama", "weight")), pser.drop(("lama", "weight")))
        self.assert_eq(
            kser.drop([("lama", "speed"), ("falcon", "weight")]),
            pser.drop([("lama", "speed"), ("falcon", "weight")]),
        )
        self.assert_eq(kser.drop({"lama": "speed"}), pser.drop({"lama": "speed"}))

        msg = "'level' should be less than the number of indexes"
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop(labels="weight", level=2)

        msg = (
            "If the given index is a list, it "
            "should only contains names as all tuples or all non tuples "
            "that contain index names"
        )
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop(["lama", ["cow", "falcon"]])

        msg = "Cannot specify both 'labels' and 'index'"
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop("lama", index="cow")

        msg = r"'Key length \(2\) exceeds index depth \(3\)'"
        with self.assertRaisesRegex(KeyError, msg):
            kser.drop(("lama", "speed", "x"))

    def test_pop(self):
        midx = pd.MultiIndex(
            [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        pdf = pd.DataFrame({"x": [45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3]}, index=midx)
        kdf = ks.from_pandas(pdf)

        pser = pdf.x
        kser = kdf.x

        self.assert_eq(kser.pop(("lama", "speed")), pser.pop(("lama", "speed")))
        self.assert_eq(kser, pser)
        self.assert_eq(kdf, pdf)

        msg = r"'Key length \(3\) exceeds index depth \(2\)'"
        with self.assertRaisesRegex(KeyError, msg):
            kser.pop(("lama", "speed", "x"))

    def test_replace(self):
        pser = pd.Series([10, 20, 15, 30, 45], name="x")
        kser = ks.Series(pser)

        self.assert_eq(kser.replace(), pser.replace())
        self.assert_eq(kser.replace({}), pser.replace({}))

        msg = "'to_replace' should be one of str, list, dict, int, float"
        with self.assertRaisesRegex(ValueError, msg):
            kser.replace(ks.range(5))
        msg = "Replacement lists must match in length. Expecting 3 got 2"
        with self.assertRaisesRegex(ValueError, msg):
            kser.replace([10, 20, 30], [1, 2])
        msg = "replace currently not support for regex"
        with self.assertRaisesRegex(NotImplementedError, msg):
            kser.replace(r"^1.$", regex=True)

    def test_xs(self):
        midx = pd.MultiIndex(
            [["a", "b", "c"], ["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        pser = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.xs(("a", "lama", "speed")), pser.xs(("a", "lama", "speed")))

    def test_duplicates(self):
        psers = {
            "test on texts": pd.Series(
                ["lama", "cow", "lama", "beetle", "lama", "hippo"], name="animal"
            ),
            "test on numbers": pd.Series([1, 1, 2, 4, 3]),
        }
        keeps = ["first", "last", False]

        for (msg, pser), keep in product(psers.items(), keeps):
            with self.subTest(msg, keep=keep):
                kser = ks.Series(pser)

                self.assert_eq(
                    pser.drop_duplicates(keep=keep).sort_values(),
                    kser.drop_duplicates(keep=keep).sort_values(),
                )

    def test_update(self):
        pser = pd.Series([10, 20, 15, 30, 45], name="x")
        kser = ks.Series(pser)

        msg = "'other' must be a Series"
        with self.assertRaisesRegex(ValueError, msg):
            kser.update(10)

    def test_where(self):
        pser1 = pd.Series([0, 1, 2, 3, 4])
        kser1 = ks.from_pandas(pser1)

        self.assert_eq(pser1.where(pser1 > 3), kser1.where(kser1 > 3).sort_index())

    def test_mask(self):
        pser1 = pd.Series([0, 1, 2, 3, 4])
        kser1 = ks.from_pandas(pser1)

        self.assert_eq(pser1.mask(pser1 > 3), kser1.mask(kser1 > 3).sort_index())

    def test_truncate(self):
        pser1 = pd.Series([10, 20, 30, 40, 50, 60, 70], index=[1, 2, 3, 4, 5, 6, 7])
        kser1 = ks.Series(pser1)
        pser2 = pd.Series([10, 20, 30, 40, 50, 60, 70], index=[7, 6, 5, 4, 3, 2, 1])
        kser2 = ks.Series(pser2)

        self.assert_eq(kser1.truncate(), pser1.truncate())
        self.assert_eq(kser1.truncate(before=2), pser1.truncate(before=2))
        self.assert_eq(kser1.truncate(after=5), pser1.truncate(after=5))
        self.assert_eq(kser1.truncate(copy=False), pser1.truncate(copy=False))
        self.assert_eq(kser1.truncate(2, 5, copy=False), pser1.truncate(2, 5, copy=False))
        # The bug for these tests has been fixed in pandas 1.1.0.
        if LooseVersion(pd.__version__) >= LooseVersion("1.1.0"):
            self.assert_eq(kser2.truncate(4, 6), pser2.truncate(4, 6))
            self.assert_eq(kser2.truncate(4, 6, copy=False), pser2.truncate(4, 6, copy=False))
        else:
            expected_kser = ks.Series([20, 30, 40], index=[6, 5, 4])
            self.assert_eq(kser2.truncate(4, 6), expected_kser)
            self.assert_eq(kser2.truncate(4, 6, copy=False), expected_kser)

        kser = ks.Series([10, 20, 30, 40, 50, 60, 70], index=[1, 2, 3, 4, 3, 2, 1])
        msg = "truncate requires a sorted index"
        with self.assertRaisesRegex(ValueError, msg):
            kser.truncate()

        kser = ks.Series([10, 20, 30, 40, 50, 60, 70], index=[1, 2, 3, 4, 5, 6, 7])
        msg = "Truncate: 2 must be after 5"
        with self.assertRaisesRegex(ValueError, msg):
            kser.truncate(5, 2)

    def test_getitem(self):
        pser = pd.Series([10, 20, 15, 30, 45], ["A", "A", "B", "C", "D"])
        kser = ks.Series(pser)

        self.assert_eq(kser["A"], pser["A"])
        self.assert_eq(kser["B"], pser["B"])
        self.assert_eq(kser[kser > 15], pser[pser > 15])

        # for MultiIndex
        midx = pd.MultiIndex(
            [["a", "b", "c"], ["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 0, 1, 2, 0, 1, 2]],
        )
        pser = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], name="0", index=midx)
        kser = ks.Series(pser)

        self.assert_eq(kser["a"], pser["a"])
        self.assert_eq(kser["a", "lama"], pser["a", "lama"])
        self.assert_eq(kser[kser > 1.5], pser[pser > 1.5])

        msg = r"'Key length \(4\) exceeds index depth \(3\)'"
        with self.assertRaisesRegex(KeyError, msg):
            kser[("a", "lama", "speed", "x")]

    def test_keys(self):
        midx = pd.MultiIndex(
            [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        pser = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.keys(), pser.keys())

    def test_index(self):
        # to check setting name of Index properly.
        idx = pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9])
        pser = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=idx)
        kser = ks.from_pandas(pser)

        kser.name = "koalas"
        pser.name = "koalas"
        self.assert_eq(kser.index.name, pser.index.name)

        # for check setting names of MultiIndex properly.
        kser.names = ["hello", "koalas"]
        pser.names = ["hello", "koalas"]
        self.assert_eq(kser.index.names, pser.index.names)

    def test_pct_change(self):
        pser = pd.Series([90, 91, 85], index=[2, 4, 1])
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.pct_change(), pser.pct_change(), check_exact=False)
        self.assert_eq(kser.pct_change(periods=2), pser.pct_change(periods=2), check_exact=False)
        self.assert_eq(kser.pct_change(periods=-1), pser.pct_change(periods=-1), check_exact=False)
        self.assert_eq(kser.pct_change(periods=-100000000), pser.pct_change(periods=-100000000))
        self.assert_eq(kser.pct_change(periods=100000000), pser.pct_change(periods=100000000))

        # for MultiIndex
        midx = pd.MultiIndex(
            [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        pser = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.pct_change(), pser.pct_change(), check_exact=False)
        self.assert_eq(kser.pct_change(periods=2), pser.pct_change(periods=2), check_exact=False)
        self.assert_eq(kser.pct_change(periods=-1), pser.pct_change(periods=-1), check_exact=False)
        self.assert_eq(kser.pct_change(periods=-100000000), pser.pct_change(periods=-100000000))
        self.assert_eq(kser.pct_change(periods=100000000), pser.pct_change(periods=100000000))

    def test_axes(self):
        pser = pd.Series([90, 91, 85], index=[2, 4, 1])
        kser = ks.from_pandas(pser)
        self.assert_eq(kser.axes, pser.axes)

        # for MultiIndex
        midx = pd.MultiIndex(
            [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        pser = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        kser = ks.from_pandas(pser)
        self.assert_eq(kser.axes, pser.axes)

    def test_combine_first(self):
        pser1 = pd.Series({"falcon": 330.0, "eagle": 160.0})
        pser2 = pd.Series({"falcon": 345.0, "eagle": 200.0, "duck": 30.0})
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)

        self.assert_eq(
            kser1.combine_first(kser2).sort_index(), pser1.combine_first(pser2).sort_index()
        )
        with self.assertRaisesRegex(
            ValueError, "`combine_first` only allows `Series` for parameter `other`"
        ):
            kser1.combine_first(50)

        kser1.name = ("X", "A")
        kser2.name = ("Y", "B")
        pser1.name = ("X", "A")
        pser2.name = ("Y", "B")
        self.assert_eq(
            kser1.combine_first(kser2).sort_index(), pser1.combine_first(pser2).sort_index()
        )

        # MultiIndex
        midx1 = pd.MultiIndex(
            [["lama", "cow", "falcon", "koala"], ["speed", "weight", "length", "power"]],
            [[0, 3, 1, 1, 1, 2, 2, 2], [0, 2, 0, 3, 2, 0, 1, 3]],
        )
        midx2 = pd.MultiIndex(
            [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        pser1 = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1], index=midx1)
        pser2 = pd.Series([-45, 200, -1.2, 30, -250, 1.5, 320, 1, -0.3], index=midx2)
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)

        self.assert_eq(
            kser1.combine_first(kser2).sort_index(), pser1.combine_first(pser2).sort_index()
        )

        # Series come from same DataFrame
        pdf = pd.DataFrame(
            {
                "A": {"falcon": 330.0, "eagle": 160.0},
                "B": {"falcon": 345.0, "eagle": 200.0, "duck": 30.0},
            }
        )
        pser1 = pdf.A
        pser2 = pdf.B
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)

        self.assert_eq(
            kser1.combine_first(kser2).sort_index(), pser1.combine_first(pser2).sort_index()
        )

        kser1.name = ("X", "A")
        kser2.name = ("Y", "B")
        pser1.name = ("X", "A")
        pser2.name = ("Y", "B")

        self.assert_eq(
            kser1.combine_first(kser2).sort_index(), pser1.combine_first(pser2).sort_index()
        )

    def test_udt(self):
        sparse_values = {0: 0.1, 1: 1.1}
        sparse_vector = SparseVector(len(sparse_values), sparse_values)
        pser = pd.Series([sparse_vector])

        if LooseVersion(pyspark.__version__) < LooseVersion("2.4"):
            with self.sql_conf({SPARK_CONF_ARROW_ENABLED: False}):
                kser = ks.from_pandas(pser)
                self.assert_eq(kser, pser)
        else:
            kser = ks.from_pandas(pser)
            self.assert_eq(kser, pser)

    def test_repeat(self):
        pser = pd.Series(["a", "b", "c"], name="0", index=np.random.rand(3))
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.repeat(3).sort_index(), pser.repeat(3).sort_index())
        self.assert_eq(kser.repeat(0).sort_index(), pser.repeat(0).sort_index())

        self.assertRaises(ValueError, lambda: kser.repeat(-1))
        self.assertRaises(ValueError, lambda: kser.repeat("abc"))

        pdf = pd.DataFrame({"a": ["a", "b", "c"], "rep": [10, 20, 30]}, index=np.random.rand(3))
        kdf = ks.from_pandas(pdf)

        if LooseVersion(pyspark.__version__) < LooseVersion("2.4"):
            self.assertRaises(ValueError, lambda: kdf.a.repeat(kdf.rep))
        else:
            self.assert_eq(kdf.a.repeat(kdf.rep).sort_index(), pdf.a.repeat(pdf.rep).sort_index())

    def test_take(self):
        pser = pd.Series([100, 200, 300, 400, 500], name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.take([0, 2, 4]).sort_values(), pser.take([0, 2, 4]).sort_values())
        self.assert_eq(
            kser.take(range(0, 5, 2)).sort_values(), pser.take(range(0, 5, 2)).sort_values()
        )
        self.assert_eq(kser.take([-4, -2, 0]).sort_values(), pser.take([-4, -2, 0]).sort_values())
        self.assert_eq(
            kser.take(range(-2, 1, 2)).sort_values(), pser.take(range(-2, 1, 2)).sort_values()
        )

        # Checking the type of indices.
        self.assertRaises(ValueError, lambda: kser.take(1))
        self.assertRaises(ValueError, lambda: kser.take("1"))
        self.assertRaises(ValueError, lambda: kser.take({1, 2}))
        self.assertRaises(ValueError, lambda: kser.take({1: None, 2: None}))

    def test_divmod(self):
        pser = pd.Series([100, None, 300, None, 500], name="Koalas")
        kser = ks.from_pandas(pser)

        if LooseVersion(pd.__version__) >= LooseVersion("1.0.0"):
            kdiv, kmod = kser.divmod(-100)
            pdiv, pmod = pser.divmod(-100)
            self.assert_eq(kdiv, pdiv)
            self.assert_eq(kmod, pmod)

            kdiv, kmod = kser.divmod(100)
            pdiv, pmod = pser.divmod(100)
            self.assert_eq(kdiv, pdiv)
            self.assert_eq(kmod, pmod)
        elif LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            kdiv, kmod = kser.divmod(-100)
            pdiv, pmod = pser.floordiv(-100), pser.mod(-100)
            self.assert_eq(kdiv, pdiv)
            self.assert_eq(kmod, pmod)

            kdiv, kmod = kser.divmod(100)
            pdiv, pmod = pser.floordiv(100), pser.mod(100)
            self.assert_eq(kdiv, pdiv)
            self.assert_eq(kmod, pmod)

    def test_rdivmod(self):
        pser = pd.Series([100, None, 300, None, 500])
        kser = ks.from_pandas(pser)

        if LooseVersion(pd.__version__) >= LooseVersion("1.0.0"):
            krdiv, krmod = kser.rdivmod(-100)
            prdiv, prmod = pser.rdivmod(-100)
            self.assert_eq(krdiv, prdiv)
            self.assert_eq(krmod, prmod)

            krdiv, krmod = kser.rdivmod(100)
            prdiv, prmod = pser.rdivmod(100)
            self.assert_eq(krdiv, prdiv)
            self.assert_eq(krmod, prmod)
        elif LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            krdiv, krmod = kser.rdivmod(-100)
            prdiv, prmod = pser.rfloordiv(-100), pser.rmod(-100)
            self.assert_eq(krdiv, prdiv)
            self.assert_eq(krmod, prmod)

            krdiv, krmod = kser.rdivmod(100)
            prdiv, prmod = pser.rfloordiv(100), pser.rmod(100)
            self.assert_eq(krdiv, prdiv)
            self.assert_eq(krmod, prmod)

    def test_mod(self):
        pser = pd.Series([100, None, -300, None, 500, -700], name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.mod(-150), pser.mod(-150))
        self.assert_eq(kser.mod(0), pser.mod(0))
        self.assert_eq(kser.mod(150), pser.mod(150))

        pdf = pd.DataFrame({"a": [100, None, -300, None, 500, -700], "b": [150] * 6})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.a.mod(kdf.b), pdf.a.mod(pdf.b))

    def test_rmod(self):
        pser = pd.Series([100, None, -300, None, 500, -700], name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.rmod(-150), pser.rmod(-150))
        self.assert_eq(kser.rmod(0), pser.rmod(0))
        self.assert_eq(kser.rmod(150), pser.rmod(150))

        pdf = pd.DataFrame({"a": [100, None, -300, None, 500, -700], "b": [150] * 6})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.a.rmod(kdf.b), pdf.a.rmod(pdf.b))

    def test_asof(self):
        pser = pd.Series([1, 2, np.nan, 4], index=[10, 20, 30, 40], name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.asof(20), pser.asof(20))
        self.assert_eq(kser.asof([5, 20]).sort_index(), pser.asof([5, 20]).sort_index())
        self.assert_eq(kser.asof(100), pser.asof(100))
        self.assert_eq(repr(kser.asof(-100)), repr(pser.asof(-100)))
        self.assert_eq(kser.asof([-100, 100]).sort_index(), pser.asof([-100, 100]).sort_index())

        # where cannot be an Index, Series or a DataFrame
        self.assertRaises(ValueError, lambda: kser.asof(ks.Index([-100, 100])))
        self.assertRaises(ValueError, lambda: kser.asof(ks.Series([-100, 100])))
        self.assertRaises(ValueError, lambda: kser.asof(ks.DataFrame({"A": [1, 2, 3]})))
        # asof is not supported for a MultiIndex
        pser.index = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c"), ("y", "d")])
        kser = ks.from_pandas(pser)
        self.assertRaises(ValueError, lambda: kser.asof(20))
        # asof requires a sorted index (More precisely, should be a monotonic increasing)
        kser = ks.Series([1, 2, np.nan, 4], index=[10, 30, 20, 40], name="Koalas")
        self.assertRaises(ValueError, lambda: kser.asof(20))
        kser = ks.Series([1, 2, np.nan, 4], index=[40, 30, 20, 10], name="Koalas")
        self.assertRaises(ValueError, lambda: kser.asof(20))

    def test_squeeze(self):
        # Single value
        pser = pd.Series([90])
        kser = ks.from_pandas(pser)
        self.assert_eq(kser.squeeze(), pser.squeeze())

        # Single value with MultiIndex
        midx = pd.MultiIndex.from_tuples([("a", "b", "c")])
        pser = pd.Series([90], index=midx)
        kser = ks.from_pandas(pser)
        self.assert_eq(kser.squeeze(), pser.squeeze())

        # Multiple values
        pser = pd.Series([90, 91, 85])
        kser = ks.from_pandas(pser)
        self.assert_eq(kser.squeeze(), pser.squeeze())

        # Multiple values with MultiIndex
        midx = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
        pser = pd.Series([90, 91, 85], index=midx)
        kser = ks.from_pandas(pser)
        self.assert_eq(kser.squeeze(), pser.squeeze())

    def test_div_zero_and_nan(self):
        pser = pd.Series([100, None, -300, None, 500, -700, np.inf, -np.inf], name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(pser.div(0), kser.div(0))
        self.assert_eq(pser.truediv(0), kser.truediv(0))
        self.assert_eq(pser / 0, kser / 0)
        self.assert_eq(pser.div(np.nan), kser.div(np.nan))
        self.assert_eq(pser.truediv(np.nan), kser.truediv(np.nan))
        self.assert_eq(pser / np.nan, kser / np.nan)

        # floordiv has different behavior in pandas > 1.0.0 when divide by 0
        if LooseVersion(pd.__version__) >= LooseVersion("1.0.0"):
            self.assert_eq(pser.floordiv(0), kser.floordiv(0))
            self.assert_eq(pser // 0, kser // 0)
        else:
            result = pd.Series(
                [np.inf, np.nan, -np.inf, np.nan, np.inf, -np.inf, np.inf, -np.inf], name="Koalas"
            )
            self.assert_eq(kser.floordiv(0), result)
            self.assert_eq(kser // 0, result)
        self.assert_eq(pser.floordiv(np.nan), kser.floordiv(np.nan))

    def test_mad(self):
        pser = pd.Series([1, 2, 3, 4], name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(pser.mad(), kser.mad())

        pser = pd.Series([None, -2, 5, 10, 50, np.nan, -20], name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(pser.mad(), kser.mad())

        pmidx = pd.MultiIndex.from_tuples(
            [("a", "1"), ("a", "2"), ("b", "1"), ("b", "2"), ("c", "1")]
        )
        pser = pd.Series([1, 2, 3, 4, 5], name="Koalas")
        pser.index = pmidx
        kser = ks.from_pandas(pser)

        self.assert_eq(pser.mad(), kser.mad())

        pmidx = pd.MultiIndex.from_tuples(
            [("a", "1"), ("a", "2"), ("b", "1"), ("b", "2"), ("c", "1")]
        )
        pser = pd.Series([None, -2, 5, 50, np.nan], name="Koalas")
        pser.index = pmidx
        kser = ks.from_pandas(pser)

        self.assert_eq(pser.mad(), kser.mad())

    def test_to_frame(self):
        pser = pd.Series(["a", "b", "c"])
        kser = ks.from_pandas(pser)

        self.assert_eq(pser.to_frame(name="a"), kser.to_frame(name="a"))

        # for MultiIndex
        midx = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
        pser = pd.Series(["a", "b", "c"], index=midx)
        kser = ks.from_pandas(pser)

        self.assert_eq(pser.to_frame(name="a"), kser.to_frame(name="a"))

    def test_shape(self):
        pser = pd.Series(["a", "b", "c"])
        kser = ks.from_pandas(pser)

        self.assert_eq(pser.shape, kser.shape)

        # for MultiIndex
        midx = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
        pser = pd.Series(["a", "b", "c"], index=midx)
        kser = ks.from_pandas(pser)

        self.assert_eq(pser.shape, kser.shape)

    def test_to_markdown(self):
        pser = pd.Series(["elk", "pig", "dog", "quetzal"], name="animal")
        kser = ks.from_pandas(pser)

        # `to_markdown()` is supported in pandas >= 1.0.0 since it's newly added in pandas 1.0.0.
        if LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            self.assertRaises(NotImplementedError, lambda: kser.to_markdown())
        else:
            self.assert_eq(pser.to_markdown(), kser.to_markdown())

    def test_unstack(self):
        pser = pd.Series(
            [10, -2, 4, 7],
            index=pd.MultiIndex.from_tuples(
                [("one", "a", "z"), ("one", "b", "x"), ("two", "a", "c"), ("two", "b", "v")],
                names=["A", "B", "C"],
            ),
        )
        kser = ks.from_pandas(pser)

        levels = [-3, -2, -1, 0, 1, 2]
        for level in levels:
            pandas_result = pser.unstack(level=level)
            koalas_result = kser.unstack(level=level).sort_index()
            self.assert_eq(pandas_result, koalas_result)
            self.assert_eq(pandas_result.index.names, koalas_result.index.names)
            self.assert_eq(pandas_result.columns.names, koalas_result.columns.names)

        # non-numeric datatypes
        pser = pd.Series(
            list("abcd"), index=pd.MultiIndex.from_product([["one", "two"], ["a", "b"]])
        )
        kser = ks.from_pandas(pser)

        levels = [-2, -1, 0, 1]
        for level in levels:
            pandas_result = pser.unstack(level=level)
            koalas_result = kser.unstack(level=level).sort_index()
            self.assert_eq(pandas_result, koalas_result)
            self.assert_eq(pandas_result.index.names, koalas_result.index.names)
            self.assert_eq(pandas_result.columns.names, koalas_result.columns.names)

        # Exceeding the range of level
        self.assertRaises(IndexError, lambda: kser.unstack(level=3))
        self.assertRaises(IndexError, lambda: kser.unstack(level=-4))
        # Only support for MultiIndex
        kser = ks.Series([10, -2, 4, 7])
        self.assertRaises(ValueError, lambda: kser.unstack())

    def test_item(self):
        kser = ks.Series([10, 20])
        self.assertRaises(ValueError, lambda: kser.item())

    def test_filter(self):
        pser = pd.Series([0, 1, 2], index=["one", "two", "three"])
        kser = ks.from_pandas(pser)

        self.assert_eq(pser.filter(items=["one", "three"]), kser.filter(items=["one", "three"]))
        self.assert_eq(pser.filter(regex="e$"), kser.filter(regex="e$"))
        self.assert_eq(pser.filter(like="hre"), kser.filter(like="hre"))

        with self.assertRaisesRegex(ValueError, "Series does not support columns axis."):
            kser.filter(like="hre", axis=1)

        # for MultiIndex
        midx = pd.MultiIndex.from_tuples([("one", "x"), ("two", "y"), ("three", "z")])
        pser = pd.Series([0, 1, 2], index=midx)
        kser = ks.from_pandas(pser)

        self.assert_eq(
            pser.filter(items=[("one", "x"), ("three", "z")]),
            kser.filter(items=[("one", "x"), ("three", "z")]),
        )

        with self.assertRaisesRegex(TypeError, "Unsupported type list"):
            kser.filter(items=[["one", "x"], ("three", "z")])

        with self.assertRaisesRegex(ValueError, "The item should not be empty."):
            kser.filter(items=[(), ("three", "z")])

    def test_abs(self):
        pser = pd.Series([-2, -1, 0, 1])
        kser = ks.from_pandas(pser)

        self.assert_eq(abs(kser), abs(pser))
        self.assert_eq(np.abs(kser), np.abs(pser))

    def test_bfill(self):
        pdf = pd.DataFrame({"x": [np.nan, 2, 3, 4, np.nan, 6], "y": [np.nan, 2, 3, 4, np.nan, 6]})
        kdf = ks.from_pandas(pdf)

        pser = pdf.x
        kser = kdf.x

        self.assert_eq(kser.bfill(), pser.bfill())
        self.assert_eq(kser.bfill()[0], pser.bfill()[0])

        kser.bfill(inplace=True)
        pser.bfill(inplace=True)
        self.assert_eq(kser, pser)
        self.assert_eq(kser[0], pser[0])
        self.assert_eq(kdf, pdf)

    def test_ffill(self):
        pdf = pd.DataFrame({"x": [np.nan, 2, 3, 4, np.nan, 6], "y": [np.nan, 2, 3, 4, np.nan, 6]})
        kdf = ks.from_pandas(pdf)

        pser = pdf.x
        kser = kdf.x

        self.assert_eq(kser.ffill(), pser.ffill())
        self.assert_eq(kser.ffill()[4], pser.ffill()[4])

        kser.ffill(inplace=True)
        pser.ffill(inplace=True)
        self.assert_eq(kser, pser)
        self.assert_eq(kser[4], pser[4])
        self.assert_eq(kdf, pdf)

    def test_iteritems(self):
        pser = pd.Series(["A", "B", "C"])
        kser = ks.from_pandas(pser)

        for (p_name, p_items), (k_name, k_items) in zip(pser.iteritems(), kser.iteritems()):
            self.assert_eq(p_name, k_name)
            self.assert_eq(p_items, k_items)

    def test_droplevel(self):
        # droplevel is new in pandas 0.24.0
        if LooseVersion(pd.__version__) >= LooseVersion("0.24.0"):
            pser = pd.Series(
                [1, 2, 3],
                index=pd.MultiIndex.from_tuples(
                    [("x", "a", "q"), ("x", "b", "w"), ("y", "c", "e")],
                    names=["level_1", "level_2", "level_3"],
                ),
            )
            kser = ks.from_pandas(pser)

            self.assert_eq(pser.droplevel(0), kser.droplevel(0))
            self.assert_eq(pser.droplevel("level_1"), kser.droplevel("level_1"))
            self.assert_eq(pser.droplevel(-1), kser.droplevel(-1))
            self.assert_eq(pser.droplevel([0]), kser.droplevel([0]))
            self.assert_eq(pser.droplevel(["level_1"]), kser.droplevel(["level_1"]))
            self.assert_eq(pser.droplevel((0,)), kser.droplevel((0,)))
            self.assert_eq(pser.droplevel(("level_1",)), kser.droplevel(("level_1",)))
            self.assert_eq(pser.droplevel([0, 2]), kser.droplevel([0, 2]))
            self.assert_eq(
                pser.droplevel(["level_1", "level_3"]), kser.droplevel(["level_1", "level_3"])
            )
            self.assert_eq(pser.droplevel((1, 2)), kser.droplevel((1, 2)))
            self.assert_eq(
                pser.droplevel(("level_2", "level_3")), kser.droplevel(("level_2", "level_3"))
            )

            with self.assertRaisesRegex(KeyError, "Level {0, 1, 2} not found"):
                kser.droplevel({0, 1, 2})
            with self.assertRaisesRegex(KeyError, "Level level_100 not found"):
                kser.droplevel(["level_1", "level_100"])
            with self.assertRaisesRegex(
                IndexError, "Too many levels: Index has only 3 levels, not 11"
            ):
                kser.droplevel(10)
            with self.assertRaisesRegex(
                IndexError,
                "Too many levels: Index has only 3 levels, -10 is not a valid level number",
            ):
                kser.droplevel(-10)
            with self.assertRaisesRegex(
                ValueError,
                "Cannot remove 3 levels from an index with 3 levels: "
                "at least one level must be left.",
            ):
                kser.droplevel([0, 1, 2])
            with self.assertRaisesRegex(
                ValueError,
                "Cannot remove 5 levels from an index with 3 levels: "
                "at least one level must be left.",
            ):
                kser.droplevel([1, 1, 1, 1, 1])

            # Tupled names
            pser.index.names = [("a", "1"), ("b", "2"), ("c", "3")]
            kser = ks.from_pandas(pser)

            self.assert_eq(
                pser.droplevel([("a", "1"), ("c", "3")]), kser.droplevel([("a", "1"), ("c", "3")])
            )

    @unittest.skipIf(
        LooseVersion(pyspark.__version__) < LooseVersion("3.0"),
        "tail won't work properly with PySpark<3.0",
    )
    def test_tail(self):
        pser = pd.Series(range(1000), name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(pser.tail(), kser.tail())
        self.assert_eq(pser.tail(10), kser.tail(10))
        self.assert_eq(pser.tail(-990), kser.tail(-990))
        self.assert_eq(pser.tail(0), kser.tail(0))
        self.assert_eq(pser.tail(1001), kser.tail(1001))
        self.assert_eq(pser.tail(-1001), kser.tail(-1001))
        with self.assertRaisesRegex(TypeError, "bad operand type for unary -: 'str'"):
            kser.tail("10")

    def test_product(self):
        pser = pd.Series([10, 20, 30, 40, 50])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.prod(), kser.prod())

        # Containing NA values
        pser = pd.Series([10, np.nan, 30, np.nan, 50])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.prod(), kser.prod(), almost=True)

        # All-NA values
        pser = pd.Series([np.nan, np.nan, np.nan])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.prod(), kser.prod())

        # Empty Series
        pser = pd.Series([])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.prod(), kser.prod())

        # Boolean Series
        pser = pd.Series([True, True, True])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.prod(), kser.prod())

        pser = pd.Series([False, False, False])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.prod(), kser.prod())

        pser = pd.Series([True, False, True])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.prod(), kser.prod())

        # With `min_count` parameter
        pser = pd.Series([10, 20, 30, 40, 50])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.prod(min_count=5), kser.prod(min_count=5))
        # Using `repr` since the result of below will be `np.nan`.
        self.assert_eq(repr(pser.prod(min_count=6)), repr(kser.prod(min_count=6)))

        pser = pd.Series([10, np.nan, 30, np.nan, 50])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.prod(min_count=3), kser.prod(min_count=3), almost=True)
        # ditto.
        self.assert_eq(repr(pser.prod(min_count=4)), repr(kser.prod(min_count=4)))

        pser = pd.Series([np.nan, np.nan, np.nan])
        kser = ks.from_pandas(pser)
        # ditto.
        self.assert_eq(repr(pser.prod(min_count=1)), repr(kser.prod(min_count=1)))

        pser = pd.Series([])
        kser = ks.from_pandas(pser)
        # ditto.
        self.assert_eq(repr(pser.prod(min_count=1)), repr(kser.prod(min_count=1)))

        with self.assertRaisesRegex(TypeError, "cannot perform prod with type object"):
            ks.Series(["a", "b", "c"]).prod()
        with self.assertRaisesRegex(TypeError, "cannot perform prod with type datetime64"):
            ks.Series([pd.Timestamp("2016-01-01") for _ in range(3)]).prod()

    def test_hasnans(self):
        # BooleanType
        pser = pd.Series([True, False, True, True])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.hasnans, kser.hasnans)

        pser = pd.Series([True, False, np.nan, True])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.hasnans, kser.hasnans)

        # TimestampType
        pser = pd.Series([pd.Timestamp("2020-07-30") for _ in range(3)])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.hasnans, kser.hasnans)

        pser = pd.Series([pd.Timestamp("2020-07-30"), np.nan, pd.Timestamp("2020-07-30")])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.hasnans, kser.hasnans)

    def test_last_valid_index(self):
        # `pyspark.sql.dataframe.DataFrame.tail` is new in pyspark >= 3.0.
        if LooseVersion(pyspark.__version__) >= LooseVersion("3.0"):
            pser = pd.Series([250, 1.5, 320, 1, 0.3, None, None, None, None])
            kser = ks.from_pandas(pser)
            self.assert_eq(pser.last_valid_index(), kser.last_valid_index())

            # MultiIndex columns
            midx = pd.MultiIndex(
                [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
                [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
            )
            pser.index = midx
            kser = ks.from_pandas(pser)
            self.assert_eq(pser.last_valid_index(), kser.last_valid_index())

            # Empty Series
            pser = pd.Series([])
            kser = ks.from_pandas(pser)
            self.assert_eq(pser.last_valid_index(), kser.last_valid_index())

    def test_first_valid_index(self):
        # Empty Series
        pser = pd.Series([])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.first_valid_index(), kser.first_valid_index())

    def test_pad(self):
        pser = pd.Series([np.nan, 2, 3, 4, np.nan, 6], name="x")
        kser = ks.from_pandas(pser)

        if LooseVersion(pd.__version__) >= LooseVersion("1.1"):
            self.assert_eq(pser.pad(), kser.pad())

            # Test `inplace=True`
            pser.pad(inplace=True)
            kser.pad(inplace=True)
            self.assert_eq(pser, kser)
        else:
            expected = ks.Series([np.nan, 2, 3, 4, 4, 6], name="x")
            self.assert_eq(expected, kser.pad())

            # Test `inplace=True`
            kser.pad(inplace=True)
            self.assert_eq(expected, kser)

    def test_explode(self):
        if LooseVersion(pd.__version__) >= LooseVersion("0.25"):
            pser = pd.Series([[1, 2, 3], [], None, [3, 4]])
            kser = ks.from_pandas(pser)
            self.assert_eq(pser.explode(), kser.explode(), almost=True)

            # MultiIndex
            pser.index = pd.MultiIndex.from_tuples([("a", "w"), ("b", "x"), ("c", "y"), ("d", "z")])
            kser = ks.from_pandas(pser)
            self.assert_eq(pser.explode(), kser.explode(), almost=True)

            # non-array type Series
            pser = pd.Series([1, 2, 3, 4])
            kser = ks.from_pandas(pser)
            self.assert_eq(pser.explode(), kser.explode())
        else:
            pser = pd.Series([[1, 2, 3], [], None, [3, 4]])
            kser = ks.from_pandas(pser)
            expected = pd.Series([1.0, 2.0, 3.0, None, None, 3.0, 4.0], index=[0, 0, 0, 1, 2, 3, 3])
            self.assert_eq(kser.explode(), expected)

            # MultiIndex
            pser.index = pd.MultiIndex.from_tuples([("a", "w"), ("b", "x"), ("c", "y"), ("d", "z")])
            kser = ks.from_pandas(pser)
            expected = pd.Series(
                [1.0, 2.0, 3.0, None, None, 3.0, 4.0],
                index=pd.MultiIndex.from_tuples(
                    [
                        ("a", "w"),
                        ("a", "w"),
                        ("a", "w"),
                        ("b", "x"),
                        ("c", "y"),
                        ("d", "z"),
                        ("d", "z"),
                    ]
                ),
            )
            self.assert_eq(kser.explode(), expected)

            # non-array type Series
            pser = pd.Series([1, 2, 3, 4])
            kser = ks.from_pandas(pser)
            expected = pser
            self.assert_eq(kser.explode(), expected)

    def test_argsort(self):
        # Without null values
        pser = pd.Series([0, -100, 50, 100, 20], index=["A", "B", "C", "D", "E"])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.argsort().sort_index(), kser.argsort().sort_index())
        self.assert_eq((-pser).argsort().sort_index(), (-kser).argsort().sort_index())

        # MultiIndex
        pser.index = pd.MultiIndex.from_tuples(
            [("a", "v"), ("b", "w"), ("c", "x"), ("d", "y"), ("e", "z")]
        )
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.argsort().sort_index(), kser.argsort().sort_index())
        self.assert_eq((-pser).argsort().sort_index(), (-kser).argsort().sort_index())

        # With name
        pser.name = "Koalas"
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.argsort().sort_index(), kser.argsort().sort_index())
        self.assert_eq((-pser).argsort().sort_index(), (-kser).argsort().sort_index())

        # Series from Index
        pidx = pd.Index([4.0, -6.0, 2.0, -100.0, 11.0, 20.0, 1.0, -99.0])
        kidx = ks.from_pandas(pidx)
        self.assert_eq(
            pidx.to_series().argsort().sort_index(), kidx.to_series().argsort().sort_index()
        )
        self.assert_eq(
            (-pidx.to_series()).argsort().sort_index(), (-kidx.to_series()).argsort().sort_index()
        )

        # Series from Index with name
        pidx.name = "Koalas"
        kidx = ks.from_pandas(pidx)
        self.assert_eq(
            pidx.to_series().argsort().sort_index(), kidx.to_series().argsort().sort_index()
        )
        self.assert_eq(
            (-pidx.to_series()).argsort().sort_index(), (-kidx.to_series()).argsort().sort_index()
        )

        # Series from DataFrame
        pdf = pd.DataFrame({"A": [4.0, -6.0, 2.0, np.nan, -100.0, 11.0, 20.0, np.nan, 1.0, -99.0]})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.A.argsort().sort_index(), kdf.A.argsort().sort_index())
        self.assert_eq((-pdf.A).argsort().sort_index(), (-kdf.A).argsort().sort_index())

        # With null values
        pser = pd.Series([0, -100, np.nan, 100, np.nan], index=["A", "B", "C", "D", "E"])
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.argsort().sort_index(), kser.argsort().sort_index())
        self.assert_eq((-pser).argsort().sort_index(), (-kser).argsort().sort_index())

        # MultiIndex with null values
        pser.index = pd.MultiIndex.from_tuples(
            [("a", "v"), ("b", "w"), ("c", "x"), ("d", "y"), ("e", "z")]
        )
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.argsort().sort_index(), kser.argsort().sort_index())
        self.assert_eq((-pser).argsort().sort_index(), (-kser).argsort().sort_index())

        # With name with null values
        pser.name = "Koalas"
        kser = ks.from_pandas(pser)
        self.assert_eq(pser.argsort().sort_index(), kser.argsort().sort_index())
        self.assert_eq((-pser).argsort().sort_index(), (-kser).argsort().sort_index())

        # Series from Index with null values
        pidx = pd.Index([4.0, -6.0, 2.0, np.nan, -100.0, 11.0, 20.0, np.nan, 1.0, -99.0])
        kidx = ks.from_pandas(pidx)
        self.assert_eq(
            pidx.to_series().argsort().sort_index(), kidx.to_series().argsort().sort_index()
        )
        self.assert_eq(
            (-pidx.to_series()).argsort().sort_index(), (-kidx.to_series()).argsort().sort_index()
        )

        # Series from Index with name with null values
        pidx.name = "Koalas"
        kidx = ks.from_pandas(pidx)
        self.assert_eq(
            pidx.to_series().argsort().sort_index(), kidx.to_series().argsort().sort_index()
        )
        self.assert_eq(
            (-pidx.to_series()).argsort().sort_index(), (-kidx.to_series()).argsort().sort_index()
        )

        # Series from DataFrame with null values
        pdf = pd.DataFrame({"A": [4.0, -6.0, 2.0, np.nan, -100.0, 11.0, 20.0, np.nan, 1.0, -99.0]})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf.A.argsort().sort_index(), kdf.A.argsort().sort_index())
        self.assert_eq((-pdf.A).argsort().sort_index(), (-kdf.A).argsort().sort_index())

    def test_argmin_argmax(self):
        pser = pd.Series(
            {
                "Corn Flakes": 100.0,
                "Almond Delight": 110.0,
                "Cinnamon Toast Crunch": 120.0,
                "Cocoa Puff": 110.0,
                "Expensive Flakes": 120.0,
                "Cheap Flakes": 100.0,
            },
            name="Koalas",
        )
        kser = ks.from_pandas(pser)

        if LooseVersion(pd.__version__) >= LooseVersion("1.0"):
            self.assert_eq(pser.argmin(), kser.argmin())
            self.assert_eq(pser.argmax(), kser.argmax())

            # MultiIndex
            pser.index = pd.MultiIndex.from_tuples(
                [("a", "t"), ("b", "u"), ("c", "v"), ("d", "w"), ("e", "x"), ("f", "u")]
            )
            kser = ks.from_pandas(pser)
            self.assert_eq(pser.argmin(), kser.argmin())
            self.assert_eq(pser.argmax(), kser.argmax())

            # Null Series
            self.assert_eq(pd.Series([np.nan]).argmin(), ks.Series([np.nan]).argmin())
            self.assert_eq(pd.Series([np.nan]).argmax(), ks.Series([np.nan]).argmax())
        else:
            self.assert_eq(pser.values.argmin(), kser.argmin())
            self.assert_eq(pser.values.argmax(), kser.argmax())

            # MultiIndex
            pser.index = pd.MultiIndex.from_tuples(
                [("a", "t"), ("b", "u"), ("c", "v"), ("d", "w"), ("e", "x"), ("f", "u")]
            )
            kser = ks.from_pandas(pser)
            self.assert_eq(pser.values.argmin(), kser.argmin())
            self.assert_eq(pser.values.argmax(), kser.argmax())

            # Null Series
            self.assert_eq(-1, ks.Series([np.nan]).argmin())
            self.assert_eq(-1, ks.Series([np.nan]).argmax())

        with self.assertRaisesRegex(ValueError, "attempt to get argmin of an empty sequence"):
            ks.Series([]).argmin()
        with self.assertRaisesRegex(ValueError, "attempt to get argmax of an empty sequence"):
            ks.Series([]).argmax()

    def test_backfill(self):
        pser = pd.Series([np.nan, 2, 3, 4, np.nan, 6], name="x")
        kser = ks.from_pandas(pser)

        if LooseVersion(pd.__version__) >= LooseVersion("1.1"):
            self.assert_eq(pser.backfill(), kser.backfill())

            # Test `inplace=True`
            pser.backfill(inplace=True)
            kser.backfill(inplace=True)
            self.assert_eq(pser, kser)
        else:
            expected = ks.Series([2.0, 2.0, 3.0, 4.0, 6.0, 6.0], name="x")
            self.assert_eq(expected, kser.backfill())

            # Test `inplace=True`
            kser.backfill(inplace=True)
            self.assert_eq(expected, kser)

    def test_compare(self):
        if LooseVersion(pd.__version__) >= LooseVersion("1.1"):
            pser1 = pd.Series(["b", "c", np.nan, "g", np.nan])
            pser2 = pd.Series(["a", "c", np.nan, np.nan, "h"])
            kser1 = ks.from_pandas(pser1)
            kser2 = ks.from_pandas(pser2)
            self.assert_eq(
                pser1.compare(pser2).sort_index(), kser1.compare(kser2).sort_index(),
            )

            # `keep_shape=True`
            self.assert_eq(
                pser1.compare(pser2, keep_shape=True).sort_index(),
                kser1.compare(kser2, keep_shape=True).sort_index(),
            )
            # `keep_equal=True`
            self.assert_eq(
                pser1.compare(pser2, keep_equal=True).sort_index(),
                kser1.compare(kser2, keep_equal=True).sort_index(),
            )
            # `keep_shape=True` and `keep_equal=True`
            self.assert_eq(
                pser1.compare(pser2, keep_shape=True, keep_equal=True).sort_index(),
                kser1.compare(kser2, keep_shape=True, keep_equal=True).sort_index(),
            )

            # MultiIndex
            pser1.index = pd.MultiIndex.from_tuples(
                [("a", "x"), ("b", "y"), ("c", "z"), ("x", "k"), ("q", "l")]
            )
            pser2.index = pd.MultiIndex.from_tuples(
                [("a", "x"), ("b", "y"), ("c", "z"), ("x", "k"), ("q", "l")]
            )
            kser1 = ks.from_pandas(pser1)
            kser2 = ks.from_pandas(pser2)
            self.assert_eq(
                pser1.compare(pser2).sort_index(), kser1.compare(kser2).sort_index(),
            )

            # `keep_shape=True` with MultiIndex
            self.assert_eq(
                pser1.compare(pser2, keep_shape=True).sort_index(),
                kser1.compare(kser2, keep_shape=True).sort_index(),
            )
            # `keep_equal=True` with MultiIndex
            self.assert_eq(
                pser1.compare(pser2, keep_equal=True).sort_index(),
                kser1.compare(kser2, keep_equal=True).sort_index(),
            )
            # `keep_shape=True` and `keep_equal=True` with MultiIndex
            self.assert_eq(
                pser1.compare(pser2, keep_shape=True, keep_equal=True).sort_index(),
                kser1.compare(kser2, keep_shape=True, keep_equal=True).sort_index(),
            )
        else:
            kser1 = ks.Series(["b", "c", np.nan, "g", np.nan])
            kser2 = ks.Series(["a", "c", np.nan, np.nan, "h"])
            expected = ks.DataFrame(
                [["b", "a"], ["g", None], [None, "h"]], index=[0, 3, 4], columns=["self", "other"]
            )
            self.assert_eq(expected, kser1.compare(kser2).sort_index())

            # `keep_shape=True`
            expected = ks.DataFrame(
                [["b", "a"], [None, None], [None, None], ["g", None], [None, "h"]],
                index=[0, 1, 2, 3, 4],
                columns=["self", "other"],
            )
            self.assert_eq(
                expected, kser1.compare(kser2, keep_shape=True).sort_index(),
            )
            # `keep_equal=True`
            expected = ks.DataFrame(
                [["b", "a"], ["g", None], [None, "h"]], index=[0, 3, 4], columns=["self", "other"]
            )
            self.assert_eq(
                expected, kser1.compare(kser2, keep_equal=True).sort_index(),
            )
            # `keep_shape=True` and `keep_equal=True`
            expected = ks.DataFrame(
                [["b", "a"], ["c", "c"], [None, None], ["g", None], [None, "h"]],
                index=[0, 1, 2, 3, 4],
                columns=["self", "other"],
            )
            self.assert_eq(
                expected, kser1.compare(kser2, keep_shape=True, keep_equal=True).sort_index(),
            )

            # MultiIndex
            kser1 = ks.Series(
                ["b", "c", np.nan, "g", np.nan],
                index=pd.MultiIndex.from_tuples(
                    [("a", "x"), ("b", "y"), ("c", "z"), ("x", "k"), ("q", "l")]
                ),
            )
            kser2 = ks.Series(
                ["a", "c", np.nan, np.nan, "h"],
                index=pd.MultiIndex.from_tuples(
                    [("a", "x"), ("b", "y"), ("c", "z"), ("x", "k"), ("q", "l")]
                ),
            )
            expected = ks.DataFrame(
                [["b", "a"], [None, "h"], ["g", None]],
                index=pd.MultiIndex.from_tuples([("a", "x"), ("q", "l"), ("x", "k")]),
                columns=["self", "other"],
            )
            self.assert_eq(expected, kser1.compare(kser2).sort_index())

            # `keep_shape=True`
            expected = ks.DataFrame(
                [["b", "a"], [None, None], [None, None], [None, "h"], ["g", None]],
                index=pd.MultiIndex.from_tuples(
                    [("a", "x"), ("b", "y"), ("c", "z"), ("q", "l"), ("x", "k")]
                ),
                columns=["self", "other"],
            )
            self.assert_eq(
                expected, kser1.compare(kser2, keep_shape=True).sort_index(),
            )
            # `keep_equal=True`
            expected = ks.DataFrame(
                [["b", "a"], [None, "h"], ["g", None]],
                index=pd.MultiIndex.from_tuples([("a", "x"), ("q", "l"), ("x", "k")]),
                columns=["self", "other"],
            )
            self.assert_eq(
                expected, kser1.compare(kser2, keep_equal=True).sort_index(),
            )
            # `keep_shape=True` and `keep_equal=True`
            expected = ks.DataFrame(
                [["b", "a"], ["c", "c"], [None, None], [None, "h"], ["g", None]],
                index=pd.MultiIndex.from_tuples(
                    [("a", "x"), ("b", "y"), ("c", "z"), ("q", "l"), ("x", "k")]
                ),
                columns=["self", "other"],
            )
            self.assert_eq(
                expected, kser1.compare(kser2, keep_shape=True, keep_equal=True).sort_index(),
            )

        # Different Index
        with self.assertRaisesRegex(
            ValueError, "Can only compare identically-labeled Series objects"
        ):
            kser1 = ks.Series([1, 2, 3, 4, 5], index=pd.Index([1, 2, 3, 4, 5]),)
            kser2 = ks.Series([2, 2, 3, 4, 1], index=pd.Index([5, 4, 3, 2, 1]),)
            kser1.compare(kser2)
        # Different MultiIndex
        with self.assertRaisesRegex(
            ValueError, "Can only compare identically-labeled Series objects"
        ):
            kser1 = ks.Series(
                [1, 2, 3, 4, 5],
                index=pd.MultiIndex.from_tuples(
                    [("a", "x"), ("b", "y"), ("c", "z"), ("x", "k"), ("q", "l")]
                ),
            )
            kser2 = ks.Series(
                [2, 2, 3, 4, 1],
                index=pd.MultiIndex.from_tuples(
                    [("a", "x"), ("b", "y"), ("c", "a"), ("x", "k"), ("q", "l")]
                ),
            )
            kser1.compare(kser2)
