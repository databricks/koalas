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
import pandas as pd

from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class OpsOnDiffFramesEnabledTest(ReusedSQLTestCase, SQLTestUtils):

    @classmethod
    def setUpClass(cls):
        super(OpsOnDiffFramesEnabledTest, cls).setUpClass()
        set_option('compute.ops_on_diff_frames', True)

    @classmethod
    def tearDownClass(cls):
        super(OpsOnDiffFramesEnabledTest, cls).tearDownClass()
        reset_option('compute.ops_on_diff_frames')

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
        self.assert_eq(
            (ks.range(10) + ks.range(10)).sort_index(),
            (ks.DataFrame({'id': list(range(10))})
             + ks.DataFrame({'id': list(range(10))})).sort_index())

    def test_no_matched_index(self):
        with self.assertRaisesRegex(ValueError, "Index names must be exactly matched"):
            ks.DataFrame({'a': [1, 2, 3]}).set_index('a') + \
                ks.DataFrame({'b': [1, 2, 3]}).set_index('b')

    def test_arithmetic(self):
        kdf1 = self.kdf1
        kdf2 = self.kdf2
        pdf1 = self.pdf1
        pdf2 = self.pdf2

        # Series
        self.assert_eq(
            (kdf1.a - kdf2.b).sort_index(),
            (pdf1.a - pdf2.b).rename("a").sort_index(), almost=True)

        self.assert_eq(
            (kdf1.a * kdf2.a).sort_index(),
            (pdf1.a * pdf2.a).rename("a").sort_index(), almost=True)

        self.assert_eq(
            (kdf1["a"] / kdf2["a"]).sort_index(),
            (pdf1["a"] / pdf2["a"]).rename("a").sort_index(), almost=True)

        # DataFrame
        self.assert_eq(
            (kdf1 + kdf2).sort_index(),
            (pdf1 + pdf2).sort_index(), almost=True)

        # Multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b')])
        kdf1.columns = columns
        kdf2.columns = columns
        pdf1.columns = columns
        pdf2.columns = columns

        self.assert_eq(
            (kdf1 + kdf2).sort_index(),
            (pdf1 + pdf2).sort_index(), almost=True)

    def test_arithmetic_chain(self):
        kdf1 = self.kdf1
        kdf2 = self.kdf2
        kdf3 = self.kdf3
        pdf1 = self.pdf1
        pdf2 = self.pdf2
        pdf3 = self.pdf3

        # Series
        self.assert_eq(
            (kdf1.a - kdf2.b - kdf3.c).sort_index(),
            (pdf1.a - pdf2.b - pdf3.c).rename("a").sort_index(), almost=True)

        self.assert_eq(
            (kdf1.a * kdf2.a * kdf3.c).sort_index(),
            (pdf1.a * pdf2.a * pdf3.c).rename("a").sort_index(), almost=True)

        self.assert_eq(
            (kdf1["a"] / kdf2["a"] / kdf3["c"]).sort_index(),
            (pdf1["a"] / pdf2["a"] / pdf3["c"]).rename("a").sort_index(),
            almost=True)

        # DataFrame
        self.assert_eq(
            (kdf1 + kdf2 - kdf3).sort_index(),
            (pdf1 + pdf2 - pdf3).sort_index(), almost=True)

        # Multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b')])
        kdf1.columns = columns
        kdf2.columns = columns
        pdf1.columns = columns
        pdf2.columns = columns
        columns = pd.MultiIndex.from_tuples([('x', 'b'), ('y', 'c')])
        kdf3.columns = columns
        pdf3.columns = columns

        self.assert_eq(
            (kdf1 + kdf2 - kdf3).sort_index(),
            (pdf1 + pdf2 - pdf3).sort_index(), almost=True)

    def test_different_columns(self):
        kdf1 = self.kdf1
        kdf4 = self.kdf4
        pdf1 = self.pdf1
        pdf4 = self.pdf4

        self.assert_eq(
            (kdf1 + kdf4).sort_index(),
            (pdf1 + pdf4).sort_index(), almost=True)

        # Multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b')])
        kdf1.columns = columns
        pdf1.columns = columns
        columns = pd.MultiIndex.from_tuples([('z', 'e'), ('z', 'f')])
        kdf4.columns = columns
        pdf4.columns = columns

        self.assert_eq(
            (kdf1 + kdf4).sort_index(),
            (pdf1 + pdf4).sort_index(), almost=True)

    def test_assignment_series(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf['a'] = self.kdf2.a
        pdf['a'] = self.pdf2.a

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf['a'] = self.kdf2.b
        pdf['a'] = self.pdf2.b

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf['c'] = self.kdf2.a
        pdf['c'] = self.pdf2.a

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        # Multi-index columns
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b')])
        kdf.columns = columns
        pdf.columns = columns
        kdf[('y', 'c')] = self.kdf2.a
        pdf[('y', 'c')] = self.pdf2.a

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_assignment_frame(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf[['a', 'b']] = self.kdf1
        pdf[['a', 'b']] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        # 'c' does not exist in `kdf`.
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf[['b', 'c']] = self.kdf1
        pdf[['b', 'c']] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        # 'c' and 'd' do not exist in `kdf`.
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf[['c', 'd']] = self.kdf1
        pdf[['c', 'd']] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        # Multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b')])
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf.columns = columns
        pdf.columns = columns
        kdf[[('y', 'c'), ('z', 'd')]] = self.kdf1
        pdf[[('y', 'c'), ('z', 'd')]] = self.pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf1 = ks.from_pandas(self.pdf1)
        pdf1 = self.pdf1
        kdf1.columns = columns
        pdf1.columns = columns
        kdf[['c', 'd']] = kdf1
        pdf[['c', 'd']] = pdf1

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_assignment_series_chain(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf['a'] = self.kdf1.a
        pdf['a'] = self.pdf1.a

        kdf['a'] = self.kdf2.b
        pdf['a'] = self.pdf2.b

        kdf['d'] = self.kdf3.c
        pdf['d'] = self.pdf3.c

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_assignment_frame_chain(self):
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf[['a', 'b']] = self.kdf1
        pdf[['a', 'b']] = self.pdf1

        kdf[['e', 'f']] = self.kdf3
        pdf[['e', 'f']] = self.pdf3

        kdf[['b', 'c']] = self.kdf2
        pdf[['b', 'c']] = self.pdf2

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_multi_index_arithmetic(self):
        kdf5 = self.kdf5
        kdf6 = self.kdf6
        pdf5 = self.pdf5
        pdf6 = self.pdf6

        # Series
        self.assert_eq(
            (kdf5.c - kdf6.e).sort_index(),
            (pdf5.c - pdf6.e).rename("c").sort_index(), almost=True)

        self.assert_eq(
            (kdf5["c"] / kdf6["e"]).sort_index(),
            (pdf5["c"] / pdf6["e"]).rename("c").sort_index(), almost=True)

        # DataFrame
        self.assert_eq(
            (kdf5 + kdf6).sort_index(),
            (pdf5 + pdf6).sort_index(), almost=True)

    def test_multi_index_assignment_series(self):
        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5
        kdf['x'] = self.kdf6.e
        pdf['x'] = self.pdf6.e

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5
        kdf['e'] = self.kdf6.e
        pdf['e'] = self.pdf6.e

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5
        kdf['c'] = self.kdf6.e
        pdf['c'] = self.pdf6.e

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_multi_index_assignment_frame(self):
        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5
        kdf[['c']] = self.kdf5
        pdf[['c']] = self.pdf5

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf5)
        pdf = self.pdf5
        kdf[['x']] = self.kdf5
        pdf[['x']] = self.pdf5

        self.assert_eq(kdf.sort_index(), pdf.sort_index())

        kdf = ks.from_pandas(self.pdf6)
        pdf = self.pdf6
        kdf[['x', 'y']] = self.kdf6
        pdf[['x', 'y']] = self.pdf6

        self.assert_eq(kdf.sort_index(), pdf.sort_index())


class OpsOnDiffFramesDisabledTest(ReusedSQLTestCase, SQLTestUtils):

    @classmethod
    def setUpClass(cls):
        super(OpsOnDiffFramesDisabledTest, cls).setUpClass()
        set_option('compute.ops_on_diff_frames', False)

    @classmethod
    def tearDownClass(cls):
        super(OpsOnDiffFramesDisabledTest, cls).tearDownClass()
        reset_option('compute.ops_on_diff_frames')

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
