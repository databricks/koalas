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
import decimal

from databricks import koalas as ks
from databricks.koalas.config import option_context
from databricks.koalas.testing.utils import ReusedSQLTestCase


class NumOpsTest(ReusedSQLTestCase):
    @property
    def numeric_psers(self):
        dtypes = [np.float32, float, int, np.int32]
        sers = [pd.Series([1, 2, 3], dtype=dtype) for dtype in dtypes]
        sers.append(pd.Series([decimal.Decimal(1), decimal.Decimal(2), decimal.Decimal(3)]))
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
        for pser, kser in self.numeric_pser_kser_pairs:
            self.assert_eq(pser + pser, kser + kser)
            self.assert_eq(pser + 1, kser + 1)
            # self.assert_eq(pser + 0.1, kser + 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assertRaises(TypeError, lambda: kser + self.non_numeric_ksers["string"])
                self.assertRaises(TypeError, lambda: kser + self.non_numeric_ksers["datetime"])
                self.assertRaises(TypeError, lambda: kser + self.non_numeric_ksers["date"])
                self.assertRaises(TypeError, lambda: kser + self.non_numeric_ksers["categorical"])
                # self.assertEqual(
                # kser + self.non_numeric_ksers['bool'], pser + self.non_numeric_psers['bool'])

    def test_sub(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            self.assert_eq(pser - pser, kser - kser)
            self.assert_eq(pser - 1, kser - 1)
            # self.assert_eq(pser - 0.1, kser - 0.1)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assertRaises(TypeError, lambda: kser - self.non_numeric_ksers["string"])
                self.assertRaises(TypeError, lambda: kser - self.non_numeric_ksers["datetime"])
                self.assertRaises(TypeError, lambda: kser - self.non_numeric_ksers["date"])
                self.assertRaises(TypeError, lambda: kser - self.non_numeric_ksers["categorical"])
                # self.assertEqual(
                # kser - self.non_numeric_ksers['bool'], pser - self.non_numeric_psers['bool'])

    def test_mul(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            self.assert_eq(pser * pser, kser * kser)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                if kser.dtype in [int, np.int32]:
                    self.assert_eq(
                        kser * self.non_numeric_ksers["string"],
                        pser * self.non_numeric_psers["string"],
                    )
                else:
                    self.assertRaises(TypeError, lambda: kser * self.non_numeric_ksers["string"])
                self.assertRaises(TypeError, lambda: kser - self.non_numeric_ksers["datetime"])
                self.assertRaises(TypeError, lambda: kser - self.non_numeric_ksers["date"])
                self.assertRaises(TypeError, lambda: kser - self.non_numeric_ksers["categorical"])
                # self.assertEqual(
                # kser * self.non_numeric_ksers['bool'], pser * self.non_numeric_psers['bool'])

    def test_truediv(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            # FloatType is coverted to DoubleType
            if kser.dtype in [float, int, np.int32]:
                self.assert_eq(pser / pser, kser / kser)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assertRaises(TypeError, lambda: kser / self.non_numeric_ksers["string"])
                self.assertRaises(TypeError, lambda: kser / self.non_numeric_ksers["datetime"])
                self.assertRaises(TypeError, lambda: kser / self.non_numeric_ksers["date"])
                self.assertRaises(TypeError, lambda: kser / self.non_numeric_ksers["categorical"])
                # self.assertEqual(
                # kser / self.non_numeric_ksers['bool'], pser / self.non_numeric_psers['bool'])

    def test_floordiv(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            # DoubleType is returned always
            if kser.dtype == float:
                self.assert_eq(pser // pser, kser // kser)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assertRaises(TypeError, lambda: kser // self.non_numeric_ksers["string"])
                self.assertRaises(TypeError, lambda: kser // self.non_numeric_ksers["datetime"])
                self.assertRaises(TypeError, lambda: kser // self.non_numeric_ksers["date"])
                self.assertRaises(TypeError, lambda: kser // self.non_numeric_ksers["categorical"])
                # self.assertEqual(
                # kser // self.non_numeric_ksers['bool'], pser // self.non_numeric_psers['bool'])

    def test_mod(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            self.assert_eq(pser % pser, kser % kser)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assertRaises(TypeError, lambda: kser % self.non_numeric_ksers["string"])
                self.assertRaises(TypeError, lambda: kser % self.non_numeric_ksers["datetime"])
                self.assertRaises(TypeError, lambda: kser % self.non_numeric_ksers["date"])
                self.assertRaises(TypeError, lambda: kser % self.non_numeric_ksers["categorical"])
                # self.assertEqual(
                # kser % self.non_numeric_ksers['bool'], pser % self.non_numeric_psers['bool'])

    def test_pow(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            # DoubleType is returned always
            if kser.dtype == float:
                self.assert_eq(pser ** pser, kser ** kser)

        with option_context("compute.ops_on_diff_frames", True):
            for pser, kser in self.numeric_pser_kser_pairs:
                self.assertRaises(TypeError, lambda: kser ** self.non_numeric_ksers["string"])
                self.assertRaises(TypeError, lambda: kser ** self.non_numeric_ksers["datetime"])
                self.assertRaises(TypeError, lambda: kser ** self.non_numeric_ksers["date"])
                self.assertRaises(TypeError, lambda: kser ** self.non_numeric_ksers["categorical"])
                # self.assertEqual(
                # kser ** self.non_numeric_ksers['bool'], pser ** self.non_numeric_psers['bool'])

    def test_radd(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            self.assert_eq(1 + pser, 1 + kser)
            # self.assert_eq(0.1 + pser, 0.1 + kser)
            self.assertRaises(TypeError, lambda: "x" + kser)
            self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) + kser)
            self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) + kser)

    def test_rsub(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            self.assert_eq(1 - pser, 1 - kser)
            # self.assert_eq(0.1 - pser, 0.1 - kser)
            self.assertRaises(TypeError, lambda: "x" - kser)
            self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) - kser)
            self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) - kser)

    def test_rmul(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            self.assert_eq(1 * pser, 1 * kser)
            # self.assert_eq(0.1 * pser, 0.1 * kser)
            self.assertRaises(TypeError, lambda: "x" * kser)
            self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) * kser)
            self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) * kser)

    def test_rtruediv(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            # self.assert_eq(5 / pser, 5 / kser)
            # self.assert_eq(0.1 / pser, 0.1 / kser)
            self.assertRaises(TypeError, lambda: "x" + kser)
            self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) / kser)
            self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) / kser)

    def test_rfloordiv(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            # self.assert_eq(5 // pser, 5 // kser)
            # self.assert_eq(0.1 // pser, 0.1 // kser)
            self.assertRaises(TypeError, lambda: "x" // kser)
            self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) // kser)
            self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) // kser)

    def test_rpow(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            # self.assert_eq(1 ** pser, 1 ** kser)
            # self.assert_eq(0.1 ** pser, 0.1 ** kser)
            self.assertRaises(TypeError, lambda: "x" ** kser)
            self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) ** kser)
            self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) ** kser)

    def test_rmod(self):
        for pser, kser in self.numeric_pser_kser_pairs:
            self.assert_eq(1 % pser, 1 % kser)
            # self.assert_eq(0.1 % pser, 0.1 % kser)
            self.assertRaises(TypeError, lambda: datetime.date(1994, 1, 1) % kser)
            self.assertRaises(TypeError, lambda: datetime.datetime(1994, 1, 1) % kser)
