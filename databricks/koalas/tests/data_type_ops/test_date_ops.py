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

import pandas as pd

from pyspark.sql.types import DateType

from databricks import koalas as ks
from databricks.koalas.config import option_context
from databricks.koalas.tests.data_type_ops.testing_utils import TestCasesUtils
from databricks.koalas.testing.utils import ReusedSQLTestCase


class DateOpsTest(ReusedSQLTestCase, TestCasesUtils):
    @property
    def pser(self):
        return pd.Series(
            [datetime.date(1994, 1, 31), datetime.date(1994, 2, 1), datetime.date(1994, 2, 2)]
        )

    @property
    def kser(self):
        return ks.from_pandas(self.pser)

    @property
    def some_date(self):
        return datetime.date(1994, 1, 1)

    def test_add(self):
        self.assertRaises(TypeError, lambda: self.kser + "x")
        self.assertRaises(TypeError, lambda: self.kser + 1)
        self.assertRaises(TypeError, lambda: self.kser + self.some_date)

        with option_context("compute.ops_on_diff_frames", True):
            for kser in self.ksers:
                self.assertRaises(TypeError, lambda: self.kser + kser)

    def test_sub(self):
        self.assertRaises(TypeError, lambda: self.kser - "x")
        self.assertRaises(TypeError, lambda: self.kser - 1)
        self.assert_eq(
            (self.pser - self.some_date).dt.days, self.kser - self.some_date,
        )
        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.pser_kser_pairs:
                if isinstance(kser.spark.data_type, DateType):
                    self.assert_eq((self.pser - pser).dt.days, (self.kser - kser).sort_index())
                else:
                    self.assertRaises(TypeError, lambda: self.kser - kser)

    def test_mul(self):
        self.assertRaises(TypeError, lambda: self.kser * "x")
        self.assertRaises(TypeError, lambda: self.kser * 1)
        self.assertRaises(TypeError, lambda: self.kser * self.some_date)

        with option_context("compute.ops_on_diff_frames", True):
            for kser in self.ksers:
                self.assertRaises(TypeError, lambda: self.kser * kser)

    def test_truediv(self):
        self.assertRaises(TypeError, lambda: self.kser / "x")
        self.assertRaises(TypeError, lambda: self.kser / 1)
        self.assertRaises(TypeError, lambda: self.kser / self.some_date)

        with option_context("compute.ops_on_diff_frames", True):
            for kser in self.ksers:
                self.assertRaises(TypeError, lambda: self.kser / kser)

    def test_floordiv(self):
        self.assertRaises(TypeError, lambda: self.kser // "x")
        self.assertRaises(TypeError, lambda: self.kser // 1)
        self.assertRaises(TypeError, lambda: self.kser // self.some_date)

        with option_context("compute.ops_on_diff_frames", True):
            for kser in self.ksers:
                self.assertRaises(TypeError, lambda: self.kser // kser)

    def test_mod(self):
        self.assertRaises(TypeError, lambda: self.kser % "x")
        self.assertRaises(TypeError, lambda: self.kser % 1)
        self.assertRaises(TypeError, lambda: self.kser % self.some_date)

        with option_context("compute.ops_on_diff_frames", True):
            for kser in self.ksers:
                self.assertRaises(TypeError, lambda: self.kser % kser)

    def test_pow(self):
        self.assertRaises(TypeError, lambda: self.kser ** "x")
        self.assertRaises(TypeError, lambda: self.kser ** 1)
        self.assertRaises(TypeError, lambda: self.kser ** self.some_date)

        with option_context("compute.ops_on_diff_frames", True):
            for kser in self.ksers:
                self.assertRaises(TypeError, lambda: self.kser ** kser)

    def test_radd(self):
        self.assertRaises(TypeError, lambda: "x" + self.kser)
        self.assertRaises(TypeError, lambda: 1 + self.kser)
        self.assertRaises(TypeError, lambda: self.some_date + self.kser)

    def test_rsub(self):
        self.assertRaises(TypeError, lambda: "x" - self.kser)
        self.assertRaises(TypeError, lambda: 1 - self.kser)
        self.assert_eq(
            (self.some_date - self.pser).dt.days, self.some_date - self.kser,
        )

    def test_rmul(self):
        self.assertRaises(TypeError, lambda: "x" * self.kser)
        self.assertRaises(TypeError, lambda: 1 * self.kser)
        self.assertRaises(TypeError, lambda: self.some_date * self.kser)

    def test_rtruediv(self):
        self.assertRaises(TypeError, lambda: "x" / self.kser)
        self.assertRaises(TypeError, lambda: 1 / self.kser)
        self.assertRaises(TypeError, lambda: self.some_date / self.kser)

    def test_rfloordiv(self):
        self.assertRaises(TypeError, lambda: "x" // self.kser)
        self.assertRaises(TypeError, lambda: 1 // self.kser)
        self.assertRaises(TypeError, lambda: self.some_date // self.kser)

    def test_rmod(self):
        self.assertRaises(TypeError, lambda: 1 % self.kser)
        self.assertRaises(TypeError, lambda: self.some_date % self.kser)

    def test_rpow(self):
        self.assertRaises(TypeError, lambda: "x" ** self.kser)
        self.assertRaises(TypeError, lambda: 1 ** self.kser)
        self.assertRaises(TypeError, lambda: self.some_date ** self.kser)
