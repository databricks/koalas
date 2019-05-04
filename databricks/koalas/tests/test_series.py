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

from databricks import koalas
from databricks.koalas.series import Series
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class DataFrameTest(ReusedSQLTestCase, SQLTestUtils):
    @property
    def pdf(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def kdf(self):
        return koalas.from_pandas(self.pdf)

    def test_Series(self):
        kdf = self.kdf
        pdf = self.pdf

        self.assertTrue(isinstance(kdf.a, Series))
        self.assertTrue(isinstance(kdf.a + 1, Series))
        self.assertTrue(isinstance(1 + kdf.a, Series))
        # TODO: self.assert_eq(d + 1, pdf + 1)

    def test_rename_series(self):
        ps = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')
        ks = koalas.from_pandas(ps)

        ps.name = 'renamed'
        ks.name = 'renamed'
        self.assertEqual(ks.name, 'renamed')
        self.assert_eq(ks, ps)

        ind = ps.index
        dind = ks.index
        ind.name = 'renamed'
        dind.name = 'renamed'
        self.assertEqual(ind.name, 'renamed')
        self.assert_eq(list(dind.toPandas()), list(ind))

    def test_rename_series_method(self):
        # Series name
        ps = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')
        ks = koalas.from_pandas(ps)

        self.assert_eq(ks.rename('y'), ps.rename('y'))
        self.assertEqual(ks.name, 'x')  # no mutation
        # self.assert_eq(ks.rename(), ps.rename())

        ks.rename('z', inplace=True)
        ps.rename('z', inplace=True)
        self.assertEqual(ks.name, 'z')
        self.assert_eq(ks, ps)

        # Series index
        ps = pd.Series(['a', 'b', 'c', 'd', 'e', 'f', 'g'], name='x')
        # ks = koalas.from_pandas(s)

        # TODO: index
        # res = ks.rename(lambda x: x ** 2)
        # self.assert_eq(res, ps.rename(lambda x: x ** 2))

        # res = ks.rename(ps)
        # self.assert_eq(res, ps.rename(ps))

        # res = ks.rename(ks)
        # self.assert_eq(res, ps.rename(ps))

        # res = ks.rename(lambda x: x**2, inplace=True)
        # self.assertis(res, ks)
        # s.rename(lambda x: x**2, inplace=True)
        # self.assert_eq(ks, ps)

    def test_to_numpy(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

        ddf = koalas.from_pandas(s)
        np.testing.assert_equal(ddf.to_numpy(), s.values)

    def test_isin_series(self):
        s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'], name='animal')

        ds = koalas.from_pandas(s)

        self.assert_eq(ds.isin(['cow', 'lama']), s.isin(['cow', 'lama']))
        self.assert_eq(ds.isin(set(['cow'])), s.isin(set(['cow'])))

        msg = "Values should be list or set"
        with self.assertRaisesRegex(TypeError, msg):
            ds.isin(s)

        # test list sanitizer
        value_list = [s, s]
        msg = "List contains unsupported type <class 'pandas.core.series.Series'>"
        with self.assertRaisesRegex(TypeError, msg):
            ds.isin(value_list)
