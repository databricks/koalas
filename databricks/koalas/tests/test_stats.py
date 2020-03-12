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

import numpy as np
import pandas as pd
import unittest
from distutils.version import LooseVersion

import pyspark

from databricks import koalas as ks
from databricks.koalas.config import option_context
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class StatsTest(ReusedSQLTestCase, SQLTestUtils):
    def _test_stat_functions(self, pdf, kdf):
        functions = ["max", "min", "mean", "sum"]
        for funcname in functions:
            self.assert_eq(getattr(kdf.A, funcname)(), getattr(pdf.A, funcname)(), almost=True)
            self.assert_eq(getattr(kdf, funcname)(), getattr(pdf, funcname)(), almost=True)

        functions = ["std", "var"]
        for funcname in functions:
            self.assert_eq(getattr(kdf.A, funcname)(), getattr(pdf.A, funcname)(), almost=True)
            self.assert_eq(getattr(kdf, funcname)(), getattr(pdf, funcname)(), almost=True)

        # NOTE: To test skew, kurt, and median, just make sure they run.
        #       The numbers are different in spark and pandas.
        functions = ["skew", "kurt", "median"]
        for funcname in functions:
            getattr(kdf.A, funcname)()
            getattr(kdf, funcname)()

    @unittest.skipIf(
        LooseVersion(pyspark.__version__) < LooseVersion("2.4"),
        "transpose() doesn't work property with PySpark<2.4",
    )
    def test_stat_functions(self):
        pdf = pd.DataFrame({"A": [1, 2, 3, 4], "B": [1.0, 2.1, 3, 4]})
        kdf = ks.from_pandas(pdf)
        self._test_stat_functions(pdf, kdf)

    def test_stat_functions_multiindex_column(self):
        arrays = [np.array(["A", "A", "B", "B"]), np.array(["one", "two", "one", "two"])]
        pdf = pd.DataFrame(np.random.randn(3, 4), index=["A", "B", "C"], columns=arrays)
        kdf = ks.from_pandas(pdf)
        self._test_stat_functions(pdf, kdf)

    def test_abs(self):
        pdf = pd.DataFrame(
            {
                "A": [1, -2, 3, -4, 5],
                "B": [1.0, -2, 3, -4, 5],
                "C": [-6.0, -7, -8, -9, 10],
                "D": ["a", "b", "c", "d", "e"],
            }
        )
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.A.abs(), pdf.A.abs())
        self.assert_eq(kdf.B.abs(), pdf.B.abs())
        self.assert_eq(kdf[["B", "C"]].abs(), pdf[["B", "C"]].abs())

    def test_axis_on_dataframe(self):
        # The number of each count is intentionally big
        # because when data is small, it executes a shortcut.
        # Less than 'compute.shortcut_limit' will execute a shortcut
        # by using collected pandas dataframe directly.
        # now we set the 'compute.shortcut_limit' as 1000 explicitly
        with option_context("compute.shortcut_limit", 1000):
            pdf = pd.DataFrame(
                {
                    "A": [1, -2, 3, -4, 5] * 300,
                    "B": [1.0, -2, 3, -4, 5] * 300,
                    "C": [-6.0, -7, -8, -9, 10] * 300,
                    "D": [True, False, True, False, False] * 300,
                }
            )
            kdf = ks.from_pandas(pdf)
            self.assert_eq(kdf.count(axis=1), pdf.count(axis=1))
            self.assert_eq(kdf.var(axis=1), pdf.var(axis=1))
            self.assert_eq(kdf.std(axis=1), pdf.std(axis=1))
            self.assert_eq(kdf.max(axis=1), pdf.max(axis=1))
            self.assert_eq(kdf.min(axis=1), pdf.min(axis=1))
            self.assert_eq(kdf.sum(axis=1), pdf.sum(axis=1))
            self.assert_eq(kdf.kurtosis(axis=1), pdf.kurtosis(axis=1))
            self.assert_eq(kdf.skew(axis=1), pdf.skew(axis=1))
            self.assert_eq(kdf.mean(axis=1), pdf.mean(axis=1))

    def test_corr(self):
        # Disable arrow execution since corr() is using UDT internally which is not supported.
        with self.sql_conf({"spark.sql.execution.arrow.enabled": False}):
            # DataFrame
            # we do not handle NaNs for now
            if LooseVersion(pd.__version__) >= LooseVersion("1.0.0"):
                pdf = pd._testing.makeMissingDataframe(0.3, 42).fillna(0)
            else:
                pdf = pd.util.testing.makeMissingDataframe(0.3, 42).fillna(0)
            kdf = ks.from_pandas(pdf)

            self.assert_eq(kdf.corr(), pdf.corr(), almost=True)

            # Series
            pser_a = pdf.A
            pser_b = pdf.B
            kser_a = kdf.A
            kser_b = kdf.B

            self.assertAlmostEqual(kser_a.corr(kser_b), pser_a.corr(pser_b))
            self.assertRaises(TypeError, lambda: kser_a.corr(kdf))

            # multi-index columns
            columns = pd.MultiIndex.from_tuples([("X", "A"), ("X", "B"), ("Y", "C"), ("Z", "D")])
            pdf.columns = columns
            kdf.columns = columns

            self.assert_eq(kdf.corr(), pdf.corr(), almost=True)

            # Series
            pser_xa = pdf[("X", "A")]
            pser_xb = pdf[("X", "B")]
            kser_xa = kdf[("X", "A")]
            kser_xb = kdf[("X", "B")]

            self.assertAlmostEqual(kser_xa.corr(kser_xb), pser_xa.corr(pser_xb))

    def test_cov_corr_meta(self):
        # Disable arrow execution since corr() is using UDT internally which is not supported.
        with self.sql_conf({"spark.sql.execution.arrow.enabled": False}):
            pdf = pd.DataFrame(
                {
                    "a": np.array([1, 2, 3], dtype="i1"),
                    "b": np.array([1, 2, 3], dtype="i2"),
                    "c": np.array([1, 2, 3], dtype="i4"),
                    "d": np.array([1, 2, 3]),
                    "e": np.array([1.0, 2.0, 3.0], dtype="f4"),
                    "f": np.array([1.0, 2.0, 3.0]),
                    "g": np.array([True, False, True]),
                    "h": np.array(list("abc")),
                },
                index=pd.Index([1, 2, 3], name="myindex"),
            )
            kdf = ks.from_pandas(pdf)
            self.assert_eq(kdf.corr(), pdf.corr())

    def test_stats_on_boolean_dataframe(self):
        pdf = pd.DataFrame({"A": [True, False, True], "B": [False, False, True]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.min(), pdf.min())
        self.assert_eq(kdf.max(), pdf.max())

        self.assert_eq(kdf.sum(), pdf.sum())
        self.assert_eq(kdf.mean(), pdf.mean())

        self.assert_eq(kdf.var(), pdf.var(), almost=True)
        self.assert_eq(kdf.std(), pdf.std(), almost=True)

    def test_stats_on_boolean_series(self):
        pser = pd.Series([True, False, True])
        kser = ks.from_pandas(pser)

        self.assert_eq(kser.min(), pser.min())
        self.assert_eq(kser.max(), pser.max())

        self.assert_eq(kser.sum(), pser.sum())
        self.assert_eq(kser.mean(), pser.mean())

        self.assert_eq(kser.var(), pser.var(), almost=True)
        self.assert_eq(kser.std(), pser.std(), almost=True)

    @unittest.skip(
        "PySpark does not allow object type. "
        "Should we support these case by casting to string? or just disallow?"
    )
    def test_some_stats_functions_should_discard_non_numeric_columns_by_default(self):
        pdf = pd.DataFrame({"i": [0, 1, 2], "b": [False, False, True], "s": ["x", "y", "z"]})
        kdf = ks.from_pandas(pdf)

        # min and max do not discard non-numeric columns by default
        self.assertEqual(len(kdf.min()), len(pdf.min()))
        self.assertEqual(len(kdf.max()), len(pdf.max()))
        self.assertEqual(len(kdf.sum()), len(pdf.sum()))

        # all the others do
        self.assertEqual(len(kdf.mean()), len(pdf.mean()))

        self.assertEqual(len(kdf.var()), len(pdf.var()))
        self.assertEqual(len(kdf.std()), len(pdf.std()))

        self.assertEqual(len(kdf.kurtosis()), len(pdf.kurtosis()))
        self.assertEqual(len(kdf.skew()), len(pdf.skew()))

    def test_stats_on_non_numeric_columns_should_be_discarded_if_numeric_only_is_true(self):
        pdf = pd.DataFrame({"i": [0, 1, 2], "b": [False, False, True], "s": ["x", "y", "z"]})
        kdf = ks.from_pandas(pdf)

        self.assertTrue(isinstance(kdf.sum(numeric_only=True), ks.Series))

        self.assertEqual(len(kdf.sum(numeric_only=True)), len(pdf.sum(numeric_only=True)))
        self.assertEqual(len(kdf.mean(numeric_only=True)), len(pdf.mean(numeric_only=True)))

        self.assertEqual(len(kdf.var(numeric_only=True)), len(pdf.var(numeric_only=True)))
        self.assertEqual(len(kdf.std(numeric_only=True)), len(pdf.std(numeric_only=True)))

        self.assertEqual(len(kdf.kurtosis(numeric_only=True)), len(pdf.kurtosis(numeric_only=True)))
        self.assertEqual(len(kdf.skew(numeric_only=True)), len(pdf.skew(numeric_only=True)))

    @unittest.skipIf(
        LooseVersion(pyspark.__version__) < LooseVersion("2.4"),
        "transpose() doesn't work property with PySpark<2.4",
    )
    def test_stats_on_non_numeric_columns_should_not_be_discarded_if_numeric_only_is_false(self):
        pdf = pd.DataFrame({"i": [0, 1, 2], "b": [False, False, True], "s": ["x", "y", "z"]})
        kdf = ks.from_pandas(pdf)

        # the lengths are the same, but the results are different.
        self.assertEqual(len(kdf.sum(numeric_only=False)), len(pdf.sum(numeric_only=False)))

        # pandas fails belows.
        # self.assertEqual(len(kdf.mean(numeric_only=False)), len(pdf.mean(numeric_only=False)))

        # self.assertEqual(len(kdf.var(numeric_only=False)), len(pdf.var(numeric_only=False)))
        # self.assertEqual(len(kdf.std(numeric_only=False)), len(pdf.std(numeric_only=False)))

        # self.assertEqual(len(kdf.kurtosis(numeric_only=False)),
        #                  len(pdf.kurtosis(numeric_only=False)))
        # self.assertEqual(len(kdf.skew(numeric_only=False)), len(pdf.skew(numeric_only=False)))
