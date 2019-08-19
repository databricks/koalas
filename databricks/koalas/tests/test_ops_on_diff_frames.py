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
import os

import pandas as pd

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class OpsOnDiffFramesEnabledTest(ReusedSQLTestCase, SQLTestUtils):

    @classmethod
    def setUpClass(cls):
        super(OpsOnDiffFramesEnabledTest, cls).setUpClass()
        cls.should_ops_on_diff_frames = os.environ.get('OPS_ON_DIFF_FRAMES', 'false')
        os.environ['OPS_ON_DIFF_FRAMES'] = 'true'

    @classmethod
    def tearDownClass(cls):
        super(OpsOnDiffFramesEnabledTest, cls).tearDownClass()
        os.environ['OPS_ON_DIFF_FRAMES'] = cls.should_ops_on_diff_frames

    @property
    def pdf1(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 10, 11])

    @property
    def pdf2(self):
        return pd.DataFrame({
            'a': [9, 8, 7, 6, 5, 4, 3, 2, 1],
            'b': [0, 0, 0, 4, 5, 6, 1, 2, 3],
        }, index=list(range(9)))

    @property
    def pdf3(self):
        return pd.DataFrame({
            'b': [1, 1, 1, 1, 1, 1, 1, 1, 1],
            'c': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        }, index=list(range(9)))

    @property
    def pdf4(self):
        return pd.DataFrame({
            'e': [2, 2, 2, 2, 2, 2, 2, 2, 2],
            'f': [2, 2, 2, 2, 2, 2, 2, 2, 2],
        }, index=list(range(9)))

    @property
    def pdf5(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
            'c': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 10, 11]).set_index(['a', 'b'])

    @property
    def pdf6(self):
        return pd.DataFrame({
            'a': [9, 8, 7, 6, 5, 4, 3, 2, 1],
            'b': [0, 0, 0, 4, 5, 6, 1, 2, 3],
            'c': [9, 8, 7, 6, 5, 4, 3, 2, 1],
            'e': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=list(range(9))).set_index(['a', 'b'])

    @property
    def kdf1(self):
        return ks.from_pandas(self.pdf1)

    @property
    def kdf2(self):
        return ks.from_pandas(self.pdf2)

    @property
    def kdf3(self):
        return ks.from_pandas(self.pdf3)

    @property
    def kdf4(self):
        return ks.from_pandas(self.pdf4)

    @property
    def kdf5(self):
        return ks.from_pandas(self.pdf5)

    @property
    def kdf6(self):
        return ks.from_pandas(self.pdf6)

    def test_ranges(self):
        self.assertEqual(
            ks.range(10) + ks.range(10),
            ks.DataFrame({'id': list(range(10))}) + ks.DataFrame({'id': list(range(10))}))

    def test_no_matched_index(self):
        with self.assertRaisesRegex(ValueError, "Index names must be exactly matched"):
            ks.DataFrame({'a': [1, 2, 3]}).set_index('a') + \
                ks.DataFrame({'b': [1, 2, 3]}).set_index('b')

    def test_arithmetic(self):
        # Series
        self.assertEqual(
            repr((self.kdf1.a - self.kdf2.b).sort_index()),
            repr((self.pdf1.a - self.pdf2.b).rename("a").sort_index()))

        self.assertEqual(
            repr((self.kdf1.a * self.kdf2.a).sort_index()),
            repr((self.pdf1.a * self.pdf2.a).rename("a").sort_index()))

        self.assertEqual(
            repr((self.kdf1["a"] / self.kdf2["a"]).sort_index()),
            repr((self.pdf1["a"] / self.pdf2["a"]).rename("a").sort_index()))

        # DataFrame
        self.assertEqual(
            repr((self.kdf1 + self.kdf2).sort_index()),
            repr((self.pdf1 + self.pdf2).sort_index()))

    def test_arithmetic_chain(self):
        # Series
        self.assertEqual(
            repr((self.kdf1.a - self.kdf2.b - self.kdf3.c).sort_index()),
            repr((self.pdf1.a - self.pdf2.b - self.pdf3.c).rename("a").sort_index()))

        self.assertEqual(
            repr((self.kdf1.a * self.kdf2.a * self.kdf3.c).sort_index()),
            repr((self.pdf1.a * self.pdf2.a * self.pdf3.c).rename("a").sort_index()))

        self.assertEqual(
            repr((self.kdf1["a"] / self.kdf2["a"] / self.kdf3["c"]).sort_index()),
            repr((self.pdf1["a"] / self.pdf2["a"] / self.pdf3["c"]).rename("a").sort_index()))

        # DataFrame
        self.assertEqual(
            repr((self.kdf1 + self.kdf2 - self.kdf3).sort_index()),
            repr((self.pdf1 + self.pdf2 - self.pdf3).sort_index()))

    def test_different_columns(self):
        self.assertEqual(
            repr((self.kdf1 + self.kdf4).sort_index()),
            repr((self.pdf1 + self.pdf4).sort_index()))

    def test_assignment_series(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1.copy()
        kdf['a'] = self.kdf2.a
        pdf['a'] = self.pdf2.a

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1.copy()
        kdf['a'] = self.kdf2.b
        pdf['a'] = self.pdf2.b

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1.copy()
        kdf['c'] = self.kdf2.a

        pdf['c'] = self.pdf2.a

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_assignment_frame(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1.copy()
        kdf[['a', 'b']] = self.kdf1
        pdf[['a', 'b']] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        # 'c' does not exist in `kdf`.
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1.copy()
        kdf[['b', 'c']] = self.kdf1
        pdf[['b', 'c']] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        # 'c' and 'd' do not exist in `kdf`.
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1.copy()
        kdf[['c', 'd']] = self.kdf1
        pdf[['c', 'd']] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_assignment_series_chain(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1.copy()
        kdf['a'] = self.kdf1.a
        pdf['a'] = self.pdf1.a

        kdf['a'] = self.kdf2.b
        pdf['a'] = self.pdf2.b

        kdf['d'] = self.kdf3.c
        pdf['d'] = self.pdf3.c

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_assignment_frame_chain(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1.copy()
        kdf[['a', 'b']] = self.kdf1
        pdf[['a', 'b']] = self.pdf1

        kdf[['e', 'f']] = self.kdf3
        pdf[['e', 'f']] = self.pdf3

        kdf[['b', 'c']] = self.kdf2
        pdf[['b', 'c']] = self.pdf2

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_multi_index_arithmetic(self):
        # Series
        self.assertEqual(
            repr((self.kdf5.c - self.kdf6.e).sort_index()),
            repr((self.pdf5.c - self.pdf6.e).rename("c").sort_index()))

        self.assertEqual(
            repr((self.kdf5["c"] / self.kdf6["e"]).sort_index()),
            repr((self.pdf5["c"] / self.pdf6["e"]).rename("c").sort_index()))

        # DataFrame
        self.assertEqual(
            repr((self.kdf5 + self.kdf6).sort_index()),
            repr((self.pdf5 + self.pdf6).sort_index()))

    def test_multi_index_assignment_series(self):
        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5.copy()
        kdf['x'] = self.kdf6.e
        pdf['x'] = self.pdf6.e

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5.copy()
        kdf['e'] = self.kdf6.e
        pdf['e'] = self.pdf6.e

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5.copy()
        kdf['c'] = self.kdf6.e

        pdf['c'] = self.pdf6.e

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_multi_index_assignment_frame(self):
        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5.copy()
        kdf[['c']] = self.kdf5
        pdf[['c']] = self.pdf5

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5.copy()
        kdf[['x']] = self.kdf5
        pdf[['x']] = self.pdf5

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf6)
        pdf = self.pdf6.copy()
        kdf[['x', 'y']] = self.kdf6
        pdf[['x', 'y']] = self.pdf6

        self.assert_eq(kdf.sort_index(), pdf.sort_index())


class OpsOnDiffFramesDisabledTest(ReusedSQLTestCase, SQLTestUtils):

    @classmethod
    def setUpClass(cls):
        super(OpsOnDiffFramesDisabledTest, cls).setUpClass()
        cls.should_ops_on_diff_frames = os.environ.get('OPS_ON_DIFF_FRAMES', 'false')
        os.environ['OPS_ON_DIFF_FRAMES'] = 'false'

    @classmethod
    def tearDownClass(cls):
        super(OpsOnDiffFramesDisabledTest, cls).tearDownClass()
        os.environ['OPS_ON_DIFF_FRAMES'] = cls.should_ops_on_diff_frames

    @property
    def pdf1(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def pdf2(self):
        return pd.DataFrame({
            'a': [9, 8, 7, 6, 5, 4, 3, 2, 1],
            'b': [0, 0, 0, 4, 5, 6, 1, 2, 3],
        }, index=list(range(9)))

    @property
    def kdf1(self):
        return ks.from_pandas(self.pdf1)

    @property
    def kdf2(self):
        return ks.from_pandas(self.pdf2)

    def test_arithmetic(self):
        with self.assertRaisesRegex(ValueError, "Cannot combine column argument"):
            self.kdf1.a - self.kdf2.b

        with self.assertRaisesRegex(ValueError, "Cannot combine column argument"):
            self.kdf1.a - self.kdf2.a

        with self.assertRaisesRegex(ValueError, "Cannot combine column argument"):
            self.kdf1["a"] - self.kdf2["a"]

        with self.assertRaisesRegex(ValueError, "Cannot combine column argument"):
            self.kdf1 - self.kdf2

    def test_assignment(self):
        with self.assertRaisesRegex(ValueError, "Cannot combine column argument"):
            kdf = ks.from_pandas(self.pdf1)
            kdf['c'] = self.kdf1.a
