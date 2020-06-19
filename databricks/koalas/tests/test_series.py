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
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils
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

        with self.sql_conf({"spark.sql.execution.arrow.enabled": False}):
            self.assert_eq(ks.from_pandas(a), a)
            self.assertRaises(ValueError, lambda: ks.from_pandas(b))

    def test_all_null_series(self):
        a = pd.Series([None, None, None], dtype="float64")
        b = pd.Series([None, None, None], dtype="str")

        self.assert_eq(ks.from_pandas(a).dtype, a.dtype)
        self.assertTrue(ks.from_pandas(a).toPandas().isnull().all())
        self.assertRaises(ValueError, lambda: ks.from_pandas(b))

        with self.sql_conf({"spark.sql.execution.arrow.enabled": False}):
            self.assert_eq(ks.from_pandas(a).dtype, a.dtype)
            self.assertTrue(ks.from_pandas(a).toPandas().isnull().all())
            self.assertRaises(ValueError, lambda: ks.from_pandas(b))

    def test_head_tail(self):
        kser = self.kser
        pser = self.pser

        self.assert_eq(kser.head(3), pser.head(3))
        self.assert_eq(kser.head(0), pser.head(0))
        self.assert_eq(kser.head(-3), pser.head(-3))
        self.assert_eq(kser.head(-10), pser.head(-10))

        # TODO: self.assert_eq(kser.tail(3), pser.tail(3))

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

        self.assert_eq(pdf["left"] & pdf["right"], kdf["left"] & kdf["right"])

    def test_to_numpy(self):
        pser = pd.Series([1, 2, 3, 4, 5, 6, 7], name="x")

        kser = ks.from_pandas(pser)
        np.testing.assert_equal(kser.to_numpy(), pser.values)

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

    def test_fillna(self):
        pdf = pd.DataFrame({"x": [np.nan, 2, 3, 4, np.nan, 6]})
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

        pser = pd.Series([1, 2, 3, 4, 5, 6], name="x")
        kser = ks.from_pandas(pser)

        pser.loc[3] = np.nan
        kser.loc[3] = np.nan

        self.assert_eq(kser.fillna(0), pser.fillna(0))
        self.assert_eq(kser.fillna(method="ffill"), pser.fillna(method="ffill"))
        self.assert_eq(kser.fillna(method="bfill"), pser.fillna(method="bfill"))

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
        pser = pd.Series([1, 2, 1, 3, 3, np.nan, 1, 4], name="x")
        kser = ks.from_pandas(pser)

        exp = pser.value_counts()
        res = kser.value_counts()
        self.assertEqual(res.name, exp.name)
        self.assert_eq(res, exp, almost=True)

        self.assert_eq(
            kser.value_counts(normalize=True), pser.value_counts(normalize=True), almost=True
        )
        self.assert_eq(
            kser.value_counts(ascending=True), pser.value_counts(ascending=True), almost=True
        )
        self.assert_eq(
            kser.value_counts(normalize=True, dropna=False),
            pser.value_counts(normalize=True, dropna=False),
            almost=True,
        )
        self.assert_eq(
            kser.value_counts(ascending=True, dropna=False),
            pser.value_counts(ascending=True, dropna=False),
            almost=True,
        )

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

        with self.assertRaisesRegex(
            NotImplementedError, "value_counts currently does not support bins"
        ):
            kser.value_counts(bins=3)

        pser.name = "index"
        kser.name = "index"
        self.assert_eq(kser.value_counts(), pser.value_counts(), almost=True)

        # Series from DataFrame
        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [None, 1, None]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.a.value_counts(normalize=True), pdf.a.value_counts(normalize=True), almost=True
        )
        self.assert_eq(
            kdf.a.value_counts(ascending=True), pdf.a.value_counts(ascending=True), almost=True
        )
        self.assert_eq(
            kdf.a.value_counts(normalize=True, dropna=False),
            pdf.a.value_counts(normalize=True, dropna=False),
            almost=True,
        )
        self.assert_eq(
            kdf.a.value_counts(ascending=True, dropna=False),
            pdf.a.value_counts(ascending=True, dropna=False),
            almost=True,
        )

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

        # Series with NaN index
        pser = pd.Series([1, 2, 3], index=[2, None, 5])
        kser = ks.from_pandas(pser)

        self.assert_eq(
            kser.value_counts(normalize=True), pser.value_counts(normalize=True), almost=True
        )
        self.assert_eq(
            kser.value_counts(ascending=True), pser.value_counts(ascending=True), almost=True
        )
        self.assert_eq(
            kser.value_counts(normalize=True, dropna=False),
            pser.value_counts(normalize=True, dropna=False),
            almost=True,
        )
        self.assert_eq(
            kser.value_counts(ascending=True, dropna=False),
            pser.value_counts(ascending=True, dropna=False),
            almost=True,
        )

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

        # Series with MultiIndex
        pser.index = pd.MultiIndex.from_tuples([("x", "a"), ("x", "b"), ("y", "c")])
        kser = ks.from_pandas(pser)

        self.assert_eq(
            kser.value_counts(normalize=True), pser.value_counts(normalize=True), almost=True
        )
        self.assert_eq(
            kser.value_counts(ascending=True), pser.value_counts(ascending=True), almost=True
        )
        self.assert_eq(
            kser.value_counts(normalize=True, dropna=False),
            pser.value_counts(normalize=True, dropna=False),
            almost=True,
        )
        self.assert_eq(
            kser.value_counts(ascending=True, dropna=False),
            pser.value_counts(ascending=True, dropna=False),
            almost=True,
        )

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
        pser.index = pd.MultiIndex.from_tuples([("x", "a"), ("x", None), ("y", "c")])
        kser = ks.from_pandas(pser)

        self.assert_eq(
            kser.value_counts(normalize=True), pser.value_counts(normalize=True), almost=True
        )
        self.assert_eq(
            kser.value_counts(ascending=True), pser.value_counts(ascending=True), almost=True
        )
        self.assert_eq(
            kser.value_counts(normalize=True, dropna=False),
            pser.value_counts(normalize=True, dropna=False),
            almost=True,
        )
        self.assert_eq(
            kser.value_counts(ascending=True, dropna=False),
            pser.value_counts(ascending=True, dropna=False),
            almost=True,
        )

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
            pser.index = pd.MultiIndex.from_tuples([("x", "a"), None, ("y", "c")])
            kser = ks.from_pandas(pser)

            self.assert_eq(
                kser.value_counts(normalize=True), pser.value_counts(normalize=True), almost=True
            )
            self.assert_eq(
                kser.value_counts(ascending=True), pser.value_counts(ascending=True), almost=True
            )
            self.assert_eq(
                kser.value_counts(normalize=True, dropna=False),
                pser.value_counts(normalize=True, dropna=False),
                almost=True,
            )
            self.assert_eq(
                kser.value_counts(ascending=True, dropna=False),
                pser.value_counts(ascending=True, dropna=False),
                almost=True,
            )

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
            with self.sql_conf({"spark.sql.execution.arrow.enabled": False}):
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

        self.assert_eq(kser.sort_values(), pser.sort_values(), almost=True)
        self.assert_eq(
            kser.sort_values(ascending=False), pser.sort_values(ascending=False), almost=True
        )
        self.assert_eq(
            kser.sort_values(na_position="first"),
            pser.sort_values(na_position="first"),
            almost=True,
        )

        self.assertRaises(ValueError, lambda: kser.sort_values(na_position="invalid"))

        # inplace
        # pandas raises an exception when the Series is derived from DataFrame
        kser.sort_values(inplace=True)
        self.assert_eq(kser, pser.sort_values(), almost=True)
        self.assert_eq(kdf, pdf)

        pser = pdf.x.copy()
        kser = kdf.x.copy()

        kser.sort_values(inplace=True)
        pser.sort_values(inplace=True)
        self.assert_eq(kser, pser, almost=True)
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
        self.assert_eq(kser.sort_index(), pser.sort_index(), almost=True)
        # Assert sorting descending
        self.assert_eq(
            kser.sort_index(ascending=False), pser.sort_index(ascending=False), almost=True
        )
        # Assert sorting NA indices first
        self.assert_eq(
            kser.sort_index(na_position="first"), pser.sort_index(na_position="first"), almost=True
        )

        # Assert sorting inplace
        # pandas sorts pdf.x by the index and update the column only
        # when the Series is derived from DataFrame.
        kser.sort_index(inplace=True)
        self.assert_eq(kser, pser.sort_index(), almost=True)
        self.assert_eq(kdf, pdf)

        pser = pdf.x.copy()
        kser = kdf.x.copy()

        kser.sort_index(inplace=True)
        pser.sort_index(inplace=True)
        self.assert_eq(kser, pser, almost=True)
        self.assert_eq(kdf, pdf)

        # Assert multi-indices
        pser = pd.Series(range(4), index=[["b", "b", "a", "a"], [1, 0, 1, 0]], name="0")
        kser = ks.from_pandas(pser)
        self.assert_eq(kser.sort_index(), pser.sort_index(), almost=True)
        self.assert_eq(kser.sort_index(level=[1, 0]), pser.sort_index(level=[1, 0]), almost=True)

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
            self.assertEqual(self.kser.to_list(), self.pser.to_list())

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
        self.assertEqual(
            repr(kser.map({})), repr(pser.map({}).replace({pd.np.nan: None}).rename(0))
        )

        d = defaultdict(lambda: "abc")
        self.assertTrue("abc" in repr(kser.map(d)))
        self.assertEqual(repr(kser.map(d)), repr(pser.map(d).rename(0)))

        def tomorrow(date) -> datetime:
            return date + timedelta(days=1)

        pser = pd.Series([datetime(2019, 10, 24)])
        kser = ks.from_pandas(pser)
        self.assertEqual(repr(kser.map(tomorrow)), repr(pser.map(tomorrow).rename(0)))

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
        pser = pd.Series([1.0, None, 0.0, 4.0, 9.0]).rename("a")
        kser = ks.from_pandas(pser)
        self.assertEqual(repr(pser.cummin()), repr(kser.cummin()))
        self.assertEqual(repr(pser.cummin(skipna=False)), repr(kser.cummin(skipna=False)))

        # with reversed index
        pser.index = [4, 3, 2, 1, 0]
        kser = ks.from_pandas(pser)
        self.assertEqual(repr(pser.cummin()), repr(kser.cummin()))
        self.assertEqual(repr(pser.cummin(skipna=False)), repr(kser.cummin(skipna=False)))

    def test_cummax(self):
        pser = pd.Series([1.0, None, 0.0, 4.0, 9.0]).rename("a")
        kser = ks.from_pandas(pser)
        self.assertEqual(repr(pser.cummax()), repr(kser.cummax()))
        self.assertEqual(repr(pser.cummax(skipna=False)), repr(kser.cummax(skipna=False)))

        # with reversed index
        pser.index = [4, 3, 2, 1, 0]
        kser = ks.from_pandas(pser)
        self.assertEqual(repr(pser.cummax()), repr(kser.cummax()))
        self.assertEqual(repr(pser.cummax(skipna=False)), repr(kser.cummax(skipna=False)))

    def test_cumsum(self):
        pser = pd.Series([1.0, None, 0.0, 4.0, 9.0]).rename("a")
        kser = ks.from_pandas(pser)
        self.assertEqual(repr(pser.cumsum()), repr(kser.cumsum()))
        self.assertEqual(repr(pser.cumsum(skipna=False)), repr(kser.cumsum(skipna=False)))

        # with reversed index
        pser.index = [4, 3, 2, 1, 0]
        kser = ks.from_pandas(pser)
        self.assertEqual(repr(pser.cumsum()), repr(kser.cumsum()))
        self.assertEqual(repr(pser.cumsum(skipna=False)), repr(kser.cumsum(skipna=False)))

    def test_cumprod(self):
        pser = pd.Series([1.0, None, 1.0, 4.0, 9.0]).rename("a")
        kser = ks.from_pandas(pser)
        self.assertEqual(repr(pser.cumprod()), repr(kser.cumprod()))
        self.assertEqual(repr(pser.cumprod(skipna=False)), repr(kser.cumprod(skipna=False)))

        # with reversed index
        pser.index = [4, 3, 2, 1, 0]
        kser = ks.from_pandas(pser)
        self.assertEqual(repr(pser.cumprod()), repr(kser.cumprod()))
        self.assertEqual(repr(pser.cumprod(skipna=False)), repr(kser.cumprod(skipna=False)))

        with self.assertRaisesRegex(Exception, "values should be bigger than 0"):
            repr(ks.Series([0, 1]).cumprod())

    def test_median(self):
        with self.assertRaisesRegex(ValueError, "accuracy must be an integer; however"):
            ks.Series([24.0, 21.0, 25.0, 33.0, 26.0]).median(accuracy="a")

    def test_rank(self):
        pser = pd.Series([1, 2, 3, 1], name="x")
        kser = ks.from_pandas(pser)
        self.assertEqual(repr(pser.rank()), repr(kser.rank().sort_index()))
        self.assertEqual(repr(pser.rank()), repr(kser.rank().sort_index()))
        self.assertEqual(
            repr(pser.rank(ascending=False)), repr(kser.rank(ascending=False).sort_index())
        )
        self.assertEqual(repr(pser.rank(method="min")), repr(kser.rank(method="min").sort_index()))
        self.assertEqual(repr(pser.rank(method="max")), repr(kser.rank(method="max").sort_index()))
        self.assertEqual(
            repr(pser.rank(method="first")), repr(kser.rank(method="first").sort_index())
        )
        self.assertEqual(
            repr(pser.rank(method="dense")), repr(kser.rank(method="dense").sort_index())
        )

        msg = "method must be one of 'average', 'min', 'max', 'first', 'dense'"
        with self.assertRaisesRegex(ValueError, msg):
            kser.rank(method="nothing")

    def test_round(self):
        pser = pd.Series([0.028208, 0.038683, 0.877076], name="x")
        kser = ks.from_pandas(pser)
        self.assertEqual(repr(pser.round(2)), repr(kser.round(2)))
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
            self.assertEqual(repr(kser.shift(periods=2)), repr(pser.shift(periods=2)))
        else:
            self.assertEqual(
                repr(kser.shift(periods=2, fill_value=0)), repr(pser.shift(periods=2, fill_value=0))
            )
        with self.assertRaisesRegex(ValueError, "periods should be an int; however"):
            kser.shift(periods=1.5)

    def test_astype(self):
        pser = pd.Series([10, 20, 15, 30, 45], name="x")
        kser = ks.Series(pser)

        self.assert_eq(kser.astype(int), pser.astype(int))
        self.assert_eq(kser.astype(bool), pser.astype(bool))

        pser = pd.Series([10, 20, 15, 30, 45, None, np.nan], name="x")
        kser = ks.Series(pser)

        self.assert_eq(kser.astype(bool), pser.astype(bool))

        pser = pd.Series(["hi", "hi ", " ", " \t", "", None], name="x")
        kser = ks.Series(pser)

        self.assert_eq(kser.astype(bool), pser.astype(bool))
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
        msg = "Need to specify at least one of 'labels' or 'index'"
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop()

        # For MultiIndex
        midx = pd.MultiIndex(
            [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        kser = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        msg = "'level' should be less than the number of indexes"
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop(labels="weight", level=2)
        msg = (
            "If the given index is a list, it "
            "should only contains names as strings, "
            "or a list of tuples that contain "
            "index names as strings"
        )
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop(["lama", ["cow", "falcon"]])
        msg = "'index' type should be one of str, list, tuple"
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop({"lama": "speed"})
        msg = "Cannot specify both 'labels' and 'index'"
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop("lama", index="cow")
        msg = r"'Key length \(2\) exceeds index depth \(3\)'"
        with self.assertRaisesRegex(KeyError, msg):
            kser.drop(("lama", "speed", "x"))
        self.assert_eq(kser.drop(("lama", "speed", "x"), level=1), kser)

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

        msg = "'key' should be string or tuple that contains strings"
        with self.assertRaisesRegex(ValueError, msg):
            kser.pop(0)
        msg = (
            "'key' should have index names as only strings "
            "or a tuple that contain index names as only strings"
        )
        with self.assertRaisesRegex(ValueError, msg):
            kser.pop(("lama", 0))
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
        kser = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        pser = kser.to_pandas()

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
        pser1 = pd.Series([0, 1, 2, 3, 4], name=0)
        kser1 = ks.from_pandas(pser1)

        self.assert_eq(pser1.where(pser1 > 3), kser1.where(kser1 > 3).sort_index(), almost=True)

    def test_mask(self):
        pser1 = pd.Series([0, 1, 2, 3, 4], name=0)
        kser1 = ks.from_pandas(pser1)

        self.assert_eq(pser1.mask(pser1 > 3), kser1.mask(kser1 > 3).sort_index(), almost=True)

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
        self.assert_eq(kser2.truncate(4, 6), pser2.truncate(4, 6))
        self.assert_eq(kser2.truncate(4, 6, copy=False), pser2.truncate(4, 6, copy=False))

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
        kser = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        pser = kser.to_pandas()

        self.assert_eq(kser.keys(), pser.keys())

    def test_index(self):
        # to check setting name of Index properly.
        idx = pd.Index([1, 2, 3, 4, 5, 6, 7, 8, 9])
        kser = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=idx)
        pser = kser.to_pandas()

        kser.name = "koalas"
        pser.name = "koalas"
        self.assert_eq(kser.index.name, pser.index.name)

        # for check setting names of MultiIndex properly.
        kser.names = ["hello", "koalas"]
        pser.names = ["hello", "koalas"]
        self.assert_eq(kser.index.names, pser.index.names)

    def test_pct_change(self):
        kser = ks.Series([90, 91, 85], index=[2, 4, 1])
        pser = kser.to_pandas()

        self.assert_eq(kser.pct_change(periods=-1), pser.pct_change(periods=-1), almost=True)
        self.assert_eq(
            kser.pct_change(periods=-100000000), pser.pct_change(periods=-100000000), almost=True
        )
        self.assert_eq(
            kser.pct_change(periods=100000000), pser.pct_change(periods=100000000), almost=True
        )

        # for MultiIndex
        midx = pd.MultiIndex(
            [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        kser = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        pser = kser.to_pandas()

        self.assert_eq(kser.pct_change(), pser.pct_change(), almost=True)
        self.assert_eq(kser.pct_change(periods=2), pser.pct_change(periods=2), almost=True)
        self.assert_eq(kser.pct_change(periods=-1), pser.pct_change(periods=-1), almost=True)
        self.assert_eq(
            kser.pct_change(periods=-100000000), pser.pct_change(periods=-100000000), almost=True
        )
        self.assert_eq(
            kser.pct_change(periods=100000000), pser.pct_change(periods=100000000), almost=True
        )

    def test_axes(self):
        kser = ks.Series([90, 91, 85], index=[2, 4, 1])
        pser = kser.to_pandas()
        self.assert_list_eq(kser.axes, pser.axes)

        # for MultiIndex
        midx = pd.MultiIndex(
            [["lama", "cow", "falcon"], ["speed", "weight", "length"]],
            [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
        )
        kser = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)
        pser = kser.to_pandas()
        self.assert_list_eq(kser.axes, pser.axes)

    def test_combine_first(self):
        kser1 = ks.Series({"falcon": 330.0, "eagle": 160.0})
        kser2 = ks.Series({"falcon": 345.0, "eagle": 200.0, "duck": 30.0})
        pser1 = kser1.to_pandas()
        pser2 = kser2.to_pandas()

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
        kser1 = ks.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1], index=midx1)
        kser2 = ks.Series([-45, 200, -1.2, 30, -250, 1.5, 320, 1, -0.3], index=midx2)
        pser1 = kser1.to_pandas()
        pser2 = kser2.to_pandas()

        self.assert_eq(
            kser1.combine_first(kser2).sort_index(), pser1.combine_first(pser2).sort_index()
        )

        # Series come from same DataFrame
        kdf = ks.DataFrame(
            {
                "A": {"falcon": 330.0, "eagle": 160.0},
                "B": {"falcon": 345.0, "eagle": 200.0, "duck": 30.0},
            }
        )
        kser1 = kdf.A
        kser2 = kdf.B
        pser1 = kser1.to_pandas()
        pser2 = kser2.to_pandas()

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
            with self.sql_conf({"spark.sql.execution.arrow.enabled": False}):
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
            self.assert_eq(repr(kser.divmod(-100)), repr(pser.divmod(-100)))
            self.assert_eq(repr(kser.divmod(100)), repr(pser.divmod(100)))
        elif LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            expected_result = repr((pser.floordiv(-100), pser.mod(-100)))
            self.assert_eq(repr(kser.divmod(-100)), expected_result)
            expected_result = repr((pser.floordiv(100), pser.mod(100)))
            self.assert_eq(repr(kser.divmod(100)), expected_result)

    def test_rdivmod(self):
        pser = pd.Series([100, None, 300, None, 500], name="Koalas")
        kser = ks.from_pandas(pser)

        if LooseVersion(pd.__version__) >= LooseVersion("1.0.0"):
            self.assert_eq(repr(kser.rdivmod(-100)), repr(pser.rdivmod(-100)))
            self.assert_eq(repr(kser.rdivmod(100)), repr(pser.rdivmod(100)))
        elif LooseVersion(pd.__version__) < LooseVersion("1.0.0"):
            expected_result = repr((pser.rfloordiv(-100), pser.rmod(-100)))
            self.assert_eq(repr(kser.rdivmod(-100)), expected_result)
            expected_result = repr((pser.rfloordiv(100), pser.rmod(100)))
            self.assert_eq(repr(kser.rdivmod(100)), expected_result)

    def test_mod(self):
        pser = pd.Series([100, None, -300, None, 500, -700], name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(repr(kser.mod(-150)), repr(pser.mod(-150)))
        self.assert_eq(repr(kser.mod(0)), repr(pser.mod(0)))
        self.assert_eq(repr(kser.mod(150)), repr(pser.mod(150)))

        pdf = pd.DataFrame({"a": [100, None, -300, None, 500, -700], "b": [150] * 6})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(repr(kdf.a.mod(kdf.b)), repr(pdf.a.mod(pdf.b).rename("a")))

    def test_rmod(self):
        pser = pd.Series([100, None, -300, None, 500, -700], name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(repr(kser.rmod(-150)), repr(pser.rmod(-150)))
        self.assert_eq(repr(kser.rmod(0)), repr(pser.rmod(0)))
        self.assert_eq(repr(kser.rmod(150)), repr(pser.rmod(150)))

        pdf = pd.DataFrame({"a": [100, None, -300, None, 500, -700], "b": [150] * 6})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(repr(kdf.a.rmod(kdf.b)), repr(pdf.a.rmod(pdf.b).rename("a")))

    def test_asof(self):
        pser = pd.Series([1, 2, np.nan, 4], index=[10, 20, 30, 40], name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(repr(kser.asof(20)), repr(pser.asof(20)))
        self.assert_eq(repr(kser.asof([5, 20]).sort_index()), repr(pser.asof([5, 20]).sort_index()))
        self.assert_eq(repr(kser.asof(100)), repr(pser.asof(100)))
        self.assert_eq(repr(kser.asof(-100)), repr(pser.asof(-100)))
        self.assert_eq(repr(kser.asof(-100)), repr(pser.asof(-100)))
        self.assert_eq(
            repr(kser.asof([-100, 100]).sort_index()), repr(pser.asof([-100, 100]).sort_index())
        )

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
        kser = ks.Series([90])
        pser = kser.to_pandas()
        self.assert_eq(kser.squeeze(), pser.squeeze())

        # Single value with MultiIndex
        midx = pd.MultiIndex.from_tuples([("a", "b", "c")])
        kser = ks.Series([90], index=midx)
        pser = kser.to_pandas()
        self.assert_eq(kser.squeeze(), pser.squeeze())

        # Multiple values
        kser = ks.Series([90, 91, 85])
        pser = kser.to_pandas()
        self.assert_eq(kser.squeeze(), pser.squeeze())

        # Multiple values with MultiIndex
        midx = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
        kser = ks.Series([90, 91, 85], index=midx)
        pser = kser.to_pandas()
        self.assert_eq(kser.squeeze(), pser.squeeze())

    def test_div_zero_and_nan(self):
        pser = pd.Series([100, None, -300, None, 500, -700, np.inf, -np.inf], name="Koalas")
        kser = ks.from_pandas(pser)

        self.assert_eq(repr(pser.div(0)), repr(kser.div(0)))
        self.assert_eq(repr(pser.truediv(0)), repr(kser.truediv(0)))
        self.assert_eq(repr(pser / 0), repr(kser / 0))
        self.assert_eq(repr(pser.div(np.nan)), repr(kser.div(np.nan)))
        self.assert_eq(repr(pser.truediv(np.nan)), repr(kser.truediv(np.nan)))
        self.assert_eq(repr(pser / np.nan), repr(kser / np.nan))

        # floordiv has different behavior in pandas > 1.0.0 when divide by 0
        if LooseVersion(pd.__version__) >= LooseVersion("1.0.0"):
            self.assert_eq(repr(pser.floordiv(0)), repr(kser.floordiv(0)))
            self.assert_eq(repr(pser // 0), repr(kser // 0))
        else:
            result = pd.Series(
                [np.inf, np.nan, -np.inf, np.nan, np.inf, -np.inf, np.inf, -np.inf], name="Koalas"
            )
            self.assert_eq(repr(kser.floordiv(0)), repr(result))
            self.assert_eq(repr(kser // 0), repr(result))
        self.assert_eq(repr(pser.floordiv(np.nan)), repr(kser.floordiv(np.nan)))

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
        kser = ks.Series(["a", "b", "c"])
        pser = kser.to_pandas()

        self.assert_eq(pser.to_frame(name="a"), kser.to_frame(name="a"))

        # for MultiIndex
        midx = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
        kser = ks.Series(["a", "b", "c"], index=midx)
        pser = kser.to_pandas()

        self.assert_eq(pser.to_frame(name="a"), kser.to_frame(name="a"))

    def test_shape(self):
        kser = ks.Series(["a", "b", "c"])
        pser = kser.to_pandas()

        self.assert_eq(pser.shape, kser.shape)

        # for MultiIndex
        midx = pd.MultiIndex.from_tuples([("a", "x"), ("b", "y"), ("c", "z")])
        kser = ks.Series(["a", "b", "c"], index=midx)
        pser = kser.to_pandas()

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

        with self.assertRaisesRegex(TypeError, "Unsupported type <class 'list'>"):
            kser.filter(items=[["one", "x"], ("three", "z")])

        with self.assertRaisesRegex(ValueError, "The item should not be empty."):
            kser.filter(items=[(), ("three", "z")])

    def test_abs(self):
        pser = pd.Series([-2, -1, 0, 1])
        kser = ks.from_pandas(pser)

        self.assert_eq(abs(kser), abs(pser))
        self.assert_eq(np.abs(kser), np.abs(pser))

    def test_bfill(self):
        pser = pd.Series([np.nan, 2, 3, 4, np.nan, 6], name="x")
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.bfill(), pser.bfill())

        kser.bfill(inplace=True)
        pser.bfill(inplace=True)
        self.assert_eq(kser, pser)

    def test_ffill(self):
        pser = pd.Series([np.nan, 2, 3, 4, np.nan, 6], name="x")
        kser = ks.from_pandas(pser)

        self.assert_eq(repr(kser.ffill()), repr(pser.ffill()))

        kser.ffill(inplace=True)
        pser.ffill(inplace=True)
        self.assert_eq(repr(kser), repr(pser))
