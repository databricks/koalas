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

from databricks import koalas as ks
from databricks.koalas.config import option_context
from databricks.koalas.data_type_ops.testing_utils import TestCasesUtils
from databricks.koalas.testing.utils import ReusedSQLTestCase


class StringOpsTest(ReusedSQLTestCase, TestCasesUtils):
    @property
    def pser(self):
        return pd.Series(["x", "y", "z"])

    @property
    def kser(self):
        return ks.from_pandas(self.pser)

    def test_add(self):
        self.assert_eq(self.pser + "x", self.kser + "x")
        self.assertRaises(TypeError, lambda: self.kser + 1)

        with option_context("compute.ops_on_diff_frames", True):
            self.assert_eq(
                self.pser + self.non_numeric_psers["string"],
                self.kser + self.non_numeric_ksers["string"],
            )
            self.assertRaises(TypeError, lambda: self.kser + self.non_numeric_ksers["datetime"])
            self.assertRaises(TypeError, lambda: self.kser + self.non_numeric_ksers["date"])
            self.assertRaises(TypeError, lambda: self.kser + self.non_numeric_ksers["categorical"])
            self.assertRaises(TypeError, lambda: self.kser + self.non_numeric_ksers["bool"])
            for kser in self.numeric_ksers:
                self.assertRaises(TypeError, lambda: self.kser + kser)

    def test_sub(self):
        self.assertRaises(TypeError, lambda: self.kser - "x")
        self.assertRaises(TypeError, lambda: self.kser - 1)

        with option_context("compute.ops_on_diff_frames", True):
            for kser in self.ksers:
                self.assertRaises(TypeError, lambda: self.kser - kser)

    def test_mul(self):
        self.assertRaises(TypeError, lambda: self.kser * "x")
        self.assert_eq(self.pser * 1, self.kser * 1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.pser_kser_pairs:
                if kser.dtype in [np.int64, np.int32]:
                    self.assert_eq(self.pser * pser, self.kser * kser)
                else:
                    self.assertRaises(TypeError, lambda: self.kser * kser)

    def test_truediv(self):
        self.assertRaises(TypeError, lambda: self.kser / "x")
        self.assertRaises(TypeError, lambda: self.kser / 1)

        with option_context("compute.ops_on_diff_frames", True):
            for kser in self.ksers:
                self.assertRaises(TypeError, lambda: self.kser / kser)

    def test_floordiv(self):
        self.assertRaises(TypeError, lambda: self.kser // "x")
        self.assertRaises(TypeError, lambda: self.kser // 1)

        with option_context("compute.ops_on_diff_frames", True):
            for kser in self.ksers:
                self.assertRaises(TypeError, lambda: self.kser // kser)

    def test_mod(self):
        self.assertRaises(TypeError, lambda: self.kser % "x")
        self.assertRaises(TypeError, lambda: self.kser % 1)

        with option_context("compute.ops_on_diff_frames", True):
            for kser in self.ksers:
                self.assertRaises(TypeError, lambda: self.kser % kser)

    def test_pow(self):
        self.assertRaises(TypeError, lambda: self.kser ** "x")
        self.assertRaises(TypeError, lambda: self.kser ** 1)

        with option_context("compute.ops_on_diff_frames", True):
            for kser in self.ksers:
                self.assertRaises(TypeError, lambda: self.kser ** kser)

    def test_radd(self):
        self.assert_eq("x" + self.pser, "x" + self.kser)
        self.assertRaises(TypeError, lambda: 1 + self.kser)

    def test_rsub(self):
        self.assertRaises(TypeError, lambda: "x" - self.kser)
        self.assertRaises(TypeError, lambda: 1 - self.kser)

    def test_rmul(self):
        self.assertRaises(TypeError, lambda: "x" * self.kser)
        self.assert_eq(1 * self.pser, 1 * self.kser)

    def test_rtruediv(self):
        self.assertRaises(TypeError, lambda: "x" / self.kser)
        self.assertRaises(TypeError, lambda: 1 / self.kser)

    def test_rfloordiv(self):
        self.assertRaises(TypeError, lambda: "x" // self.kser)
        self.assertRaises(TypeError, lambda: 1 // self.kser)

    def test_rmod(self):
        self.assertRaises(TypeError, lambda: 1 % self.kser)

    def test_rpow(self):
        self.assertRaises(TypeError, lambda: "x" ** self.kser)
        self.assertRaises(TypeError, lambda: 1 ** self.kser)
