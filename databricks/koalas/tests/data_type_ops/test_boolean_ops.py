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
import numpy as np
import pandas as pd

from databricks import koalas as ks
from databricks.koalas.config import option_context
from databricks.koalas.testing.utils import ReusedSQLTestCase


class BooleanOpsTest(ReusedSQLTestCase):
    @property
    def pser(self):
        return pd.Series([True, True, False])

    @property
    def kser(self):
        return ks.from_pandas(self.pser)

    @property
    def numeric_psers(self):
        dtypes = [np.float32, float, int, np.int32]
        sers = [pd.Series([1, 2, 3], dtype=dtype) for dtype in dtypes]
        # TODO: enable DecimalType series
        # sers.append(pd.Series([decimal.Decimal(1), decimal.Decimal(2), decimal.Decimal(3)]))
        return sers

    @property
    def numeric_ksers(self):
        return [ks.from_pandas(pser) for pser in self.numeric_psers]

    @property
    def non_numeric_psers(self):
        psers = {
            "string": pd.Series(["x", "y", "z"]),
            "datetime": pd.to_datetime(pd.Series([1, 2, 3])),
            "bool": pd.Series([True, True, False]),
            "date": pd.Series(
                [datetime.date(1994, 1, 1), datetime.date(1994, 1, 2), datetime.date(1994, 1, 3)]
            ),
            "categorical": pd.Series(["a", "b", "a"], dtype="category"),
        }
        return psers

    @property
    def non_numeric_ksers(self):
        ksers = {}

        for k, v in self.non_numeric_psers.items():

            ksers[k] = ks.from_pandas(v)
        return ksers

    @property
    def numeric_pser_kser_pairs(self):
        return zip(self.numeric_psers, self.numeric_ksers)

    def test_add(self):
        self.assert_eq(self.pser + 1, self.kser + 1)
        self.assert_eq(self.pser + 0.1, self.kser + 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assert_eq(self.pser + pser, self.kser + kser)

            for k, kser in self.non_numeric_ksers.items():
                if k != "bool":  # TODO: handle bool case
                    self.assertRaises(TypeError, lambda: self.kser + kser)

    def test_sub(self):
        self.assert_eq(self.pser - 1, self.kser - 1)
        self.assert_eq(self.pser - 0.1, self.kser - 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assert_eq(self.pser - pser, self.kser - kser)

            for k, kser in self.non_numeric_ksers.items():
                if k != "bool":  # TODO: handle bool case
                    self.assertRaises(TypeError, lambda: self.kser - kser)

    def test_mul(self):
        self.assert_eq(self.pser * 1, self.kser * 1)
        self.assert_eq(self.pser * 0.1, self.kser * 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assert_eq(self.pser * pser, self.kser * kser)

            for k, kser in self.non_numeric_ksers.items():
                if k != "bool":  # TODO: handle bool case
                    self.assertRaises(TypeError, lambda: self.kser * kser)

    def test_truediv(self):
        self.assert_eq(self.pser / 1, self.kser / 1)
        self.assert_eq(self.pser / 0.1, self.kser / 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                if kser.dtype == float:  # DoubleType is returned always
                    self.assert_eq(self.pser / pser, self.kser / kser)

            for k, kser in self.non_numeric_ksers.items():
                if k != "bool":  # TODO: handle bool case
                    self.assertRaises(TypeError, lambda: self.kser / kser)

    def test_floordiv(self):
        # DoubleType is returned always
        # self.assert_eq(self.pser // 1, self.kser // 1)

        # in pandas, 1 // 0.1 = 9.0; in Koalas, 1 // 0.1 = 10.0
        # self.assert_eq(self.pser // 0.1, self.kser // 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                if kser.dtype == float:  # DoubleType is returned always
                    self.assert_eq(self.pser // pser, self.kser // kser)

            for k, kser in self.non_numeric_ksers.items():
                if k != "bool":  # TODO: handle bool case
                    self.assertRaises(TypeError, lambda: self.kser // kser)

    def test_mod(self):
        self.assert_eq(self.pser % 1, self.kser % 1)
        self.assert_eq(self.pser % 0.1, self.kser % 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assert_eq(self.pser % pser, self.kser % kser)

            for k, kser in self.non_numeric_ksers.items():
                if k != "bool":  # TODO: handle bool case
                    self.assertRaises(TypeError, lambda: self.kser % kser)

    def test_pow(self):
        # DoubleType is returned always
        # self.assert_eq(self.pser ** 1, self.kser ** 1)
        self.assert_eq(self.pser ** 0.1, self.kser ** 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                if kser.dtype == float:  # DoubleType is returned always
                    self.assert_eq(self.pser ** pser, self.kser ** kser)

            for k, kser in self.non_numeric_ksers.items():
                if k != "bool":  # TODO: handle bool case
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
        self.assert_eq(1 // self.pser, 1 // self.kser)
        self.assert_eq(0.1 // self.pser, 0.1 // self.kser)
        self.assertRaises(TypeError, lambda: "x" + self.kser)
        self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) // self.kser)
        self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) // self.kser)

    def test_rpow(self):
        self.assert_eq(1 ** self.pser, 1 ** self.kser)
        self.assert_eq(0.1 ** self.pser, 0.1 ** self.kser)
        self.assertRaises(TypeError, lambda: "x" ** self.kser)
        self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) ** self.kser)
        self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) ** self.kser)

    def test_rmod(self):
        # self.assert_eq(1 % self.pser, 1 % self.kser)
        # in pandas 0.1 / True is 0.1; in Koalas 0.1 / True is 0.10000000000000009
        # self.assert_eq(0.1 % self.pser, 0.1 % self.kser)
        self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) % self.kser)
        self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) % self.kser)
