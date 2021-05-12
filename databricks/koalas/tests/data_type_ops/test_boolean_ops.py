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

import datetime
from distutils.version import LooseVersion

import pandas as pd

from databricks import koalas as ks
from databricks.koalas.config import option_context
from databricks.koalas.tests.data_type_ops.testing_utils import TestCasesUtils
from databricks.koalas.testing.utils import ReusedSQLTestCase


class BooleanOpsTest(ReusedSQLTestCase, TestCasesUtils):
    @property
    def pser(self):
        return pd.Series([True, True, False])

    @property
    def kser(self):
        return ks.from_pandas(self.pser)

    @property
    def float_pser(self):
        return pd.Series([1, 2, 3], dtype=float)

    @property
    def float_kser(self):
        return ks.from_pandas(self.float_pser)

    def test_add(self):
        self.assert_eq(self.pser + 1, self.kser + 1)
        self.assert_eq(self.pser + 0.1, self.kser + 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assert_eq(self.pser + pser, (self.kser + kser).sort_index())

            for k, kser in self.non_numeric_ksers.items():
                self.assertRaises(TypeError, lambda: self.kser + kser)

    def test_sub(self):
        self.assert_eq(self.pser - 1, self.kser - 1)
        self.assert_eq(self.pser - 0.1, self.kser - 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assert_eq(self.pser - pser, (self.kser - kser).sort_index())

            for k, kser in self.non_numeric_ksers.items():
                self.assertRaises(TypeError, lambda: self.kser - kser)

    def test_mul(self):
        self.assert_eq(self.pser * 1, self.kser * 1)
        self.assert_eq(self.pser * 0.1, self.kser * 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assert_eq(self.pser * pser, (self.kser * kser).sort_index())

            for k, kser in self.non_numeric_ksers.items():
                self.assertRaises(TypeError, lambda: self.kser * kser)

    def test_truediv(self):
        self.assert_eq(self.pser / 1, self.kser / 1)
        self.assert_eq(self.pser / 0.1, self.kser / 0.1)

        with option_context("compute.ops_on_diff_frames", True):

            self.assert_eq(self.pser / self.float_pser, (self.kser / self.float_kser).sort_index())

            for k, kser in self.non_numeric_ksers.items():
                self.assertRaises(TypeError, lambda: self.kser / kser)

    def test_floordiv(self):
        # float is always returned in Koalas
        self.assert_eq((self.pser // 1).astype("float"), self.kser // 1)
        # in pandas, 1 // 0.1 = 9.0; in Koalas, 1 // 0.1 = 10.0
        # self.assert_eq(self.pser // 0.1, self.kser // 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            self.assert_eq(
                self.pser // self.float_pser, (self.kser // self.float_kser).sort_index()
            )

            for k, kser in self.non_numeric_ksers.items():
                self.assertRaises(TypeError, lambda: self.kser // kser)

    def test_mod(self):
        self.assert_eq(self.pser % 1, self.kser % 1)
        self.assert_eq(self.pser % 0.1, self.kser % 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assert_eq(self.pser % pser, (self.kser % kser).sort_index())

            for k, kser in self.non_numeric_ksers.items():
                self.assertRaises(TypeError, lambda: self.kser % kser)

    def test_pow(self):
        # float is always returned in Koalas
        self.assert_eq((self.pser ** 1).astype("float"), self.kser ** 1)
        self.assert_eq(self.pser ** 0.1, self.kser ** 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            self.assert_eq(
                self.pser ** self.float_pser, (self.kser ** self.float_kser).sort_index()
            )

            for k, kser in self.non_numeric_ksers.items():
                self.assertRaises(TypeError, lambda: self.kser ** kser)

    def test_radd(self):
        self.assert_eq(1 + self.pser, 1 + self.kser)
        self.assert_eq(0.1 + self.pser, 0.1 + self.kser)
        self.assertRaises(TypeError, lambda: "x" + self.kser)
        self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) + self.kser)
        self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) + self.kser)

    def test_rsub(self):
        self.assert_eq(1 - self.pser, 1 - self.kser)
        self.assert_eq(0.1 - self.pser, 0.1 - self.kser)
        self.assertRaises(TypeError, lambda: "x" - self.kser)
        self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) - self.kser)
        self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) - self.kser)

    def test_rmul(self):
        self.assert_eq(1 * self.pser, 1 * self.kser)
        self.assert_eq(0.1 * self.pser, 0.1 * self.kser)
        self.assertRaises(TypeError, lambda: "x" * self.kser)
        self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) * self.kser)
        self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) * self.kser)

    def test_rtruediv(self):
        self.assert_eq(1 / self.pser, 1 / self.kser)
        self.assert_eq(0.1 / self.pser, 0.1 / self.kser)
        self.assertRaises(TypeError, lambda: "x" / self.kser)
        self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) / self.kser)
        self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) / self.kser)

    def test_rfloordiv(self):
        if LooseVersion(pd.__version__) >= LooseVersion("0.25.3"):
            self.assert_eq(1 // self.pser, 1 // self.kser)
            self.assert_eq(0.1 // self.pser, 0.1 // self.kser)
        self.assertRaises(TypeError, lambda: "x" + self.kser)
        self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) // self.kser)
        self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) // self.kser)

    def test_rpow(self):
        # float is returned always in Koalas
        self.assert_eq((1 ** self.pser).astype(float), 1 ** self.kser)
        self.assert_eq(0.1 ** self.pser, 0.1 ** self.kser)
        self.assertRaises(TypeError, lambda: "x" ** self.kser)
        self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) ** self.kser)
        self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) ** self.kser)

    def test_rmod(self):
        # 1 % False is 0.0 in pandas
        self.assert_eq(ks.Series([0, 0, None], dtype=float), 1 % self.kser)
        # 0.1 / True is 0.1 in pandas
        self.assert_eq(
            ks.Series([0.10000000000000009, 0.10000000000000009, None], dtype=float),
            0.1 % self.kser,
        )
        self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) % self.kser)
        self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) % self.kser)
