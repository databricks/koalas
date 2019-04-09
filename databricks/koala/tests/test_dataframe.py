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

import unittest

import numpy as np
import pandas as pd
import pyspark
from pyspark.sql import Column

from databricks.koala.testing.utils import ReusedSQLTestCase, TestUtils


class DataFrameTest(ReusedSQLTestCase, TestUtils):

    @property
    def full(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def df(self):
        return self.spark.from_pandas(self.full)

    def test_Dataframe(self):
        d = self.df
        full = self.full

        expected = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10],
                             index=[0, 1, 3, 5, 6, 8, 9, 9, 9],
                             name='(a + 1)')  # TODO: name='a'

        self.assert_eq(d['a'] + 1, expected)

        self.assert_eq(d.columns, pd.Index(['a', 'b']))

        self.assert_eq(d[d['b'] > 2], full[full['b'] > 2])
        self.assert_eq(d[['a', 'b']], full[['a', 'b']])
        self.assert_eq(d.a, full.a)
        # TODO: assert d.b.mean().compute() == full.b.mean()
        # TODO: assert np.allclose(d.b.var().compute(), full.b.var())
        # TODO: assert np.allclose(d.b.std().compute(), full.b.std())

        assert repr(d)

        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        })
        ddf = self.spark.createDataFrame(df)
        self.assert_eq(df[['a', 'b']], ddf[['a', 'b']])

    def test_head_tail(self):
        d = self.df
        full = self.full

        self.assert_eq(d.head(2), full.head(2))
        self.assert_eq(d.head(3), full.head(3))
        self.assert_eq(d['a'].head(2), full['a'].head(2))
        self.assert_eq(d['a'].head(3), full['a'].head(3))

        # TODO: self.assert_eq(d.tail(2), full.tail(2))
        # TODO: self.assert_eq(d.tail(3), full.tail(3))
        # TODO: self.assert_eq(d['a'].tail(2), full['a'].tail(2))
        # TODO: self.assert_eq(d['a'].tail(3), full['a'].tail(3))

    @unittest.skip('TODO: support index')
    def test_index_head(self):
        d = self.df
        full = self.full

        self.assert_eq(d.index[:2], full.index[:2])
        self.assert_eq(d.index[:3], full.index[:3])

    def test_Series(self):
        d = self.df
        full = self.full

        self.assertTrue(isinstance(d.a, Column))
        self.assertTrue(isinstance(d.a + 1, Column))
        # TODO: self.assert_eq(d + 1, full + 1)

    @unittest.skip('TODO: support index')
    def test_Index(self):
        for case in [pd.DataFrame(np.random.randn(10, 5), index=list('abcdefghij')),
                     pd.DataFrame(np.random.randn(10, 5),
                                  index=pd.date_range('2011-01-01', freq='D',
                                                      periods=10))]:
            ddf = self.spark.from_pandas(case)
            self.assert_eq(ddf.index, case.index)

    def test_attributes(self):
        d = self.df

        self.assertIn('a', dir(d))
        self.assertNotIn('foo', dir(d))
        self.assertRaises(AttributeError, lambda: d.foo)

        df = self.spark.from_pandas(pd.DataFrame({'a b c': [1, 2, 3]}))
        self.assertNotIn('a b c', dir(df))
        df = self.spark.from_pandas(pd.DataFrame({'a': [1, 2], 5: [1, 2]}))
        self.assertIn('a', dir(df))
        self.assertNotIn(5, dir(df))

    def test_column_names(self):
        d = self.df

        self.assert_eq(d.columns, pd.Index(['a', 'b']))
        # TODO: self.assert_eq(d[['b', 'a']].columns, pd.Index(['b', 'a']))
        self.assertEqual(d['a'].name, 'a')
        self.assertEqual((d['a'] + 1).name, '(a + 1)')  # TODO: 'a'
        self.assertEqual((d['a'] + d['b']).name, '(a + b)')  # TODO: None

    @unittest.skip('TODO: support index')
    def test_index_names(self):
        d = self.df

        self.assertIsNone(d.index.name)

        idx = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name='x')
        df = pd.DataFrame(np.random.randn(10, 5), idx)
        ddf = self.spark.from_pandas(df)
        self.assertEqual(ddf.index.name, 'x')

    def test_rename_columns(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7],
                           'b': [7, 6, 5, 4, 3, 2, 1]})
        ddf = self.spark.from_pandas(df)

        ddf.columns = ['x', 'y']
        df.columns = ['x', 'y']
        self.assert_eq(ddf.columns, pd.Index(['x', 'y']))
        self.assert_eq(ddf, df)

        msg = "Length mismatch: Expected axis has 2 elements, new values have 4 elements"
        with self.assertRaisesRegex(ValueError, msg):
            ddf.columns = [1, 2, 3, 4]

        # Multi-index columns
        df = pd.DataFrame({('A', '0'): [1, 2, 2, 3], ('B', 1): [1, 2, 3, 4]})
        ddf = self.spark.from_pandas(df)

        df.columns = ['x', 'y']
        ddf.columns = ['x', 'y']
        self.assert_eq(ddf.columns, pd.Index(['x', 'y']))
        self.assert_eq(ddf, df)

    def test_rename_series(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')
        ds = self.spark.from_pandas(pd.DataFrame(s)).x

        s.name = 'renamed'
        ds.name = 'renamed'
        self.assertEqual(ds.name, 'renamed')
        self.assert_eq(ds, s)

        # TODO: index
        # ind = s.index
        # dind = ds.index
        # ind.name = 'renamed'
        # dind.name = 'renamed'
        # self.assertEqual(ind.name, 'renamed')
        # self.assert_eq(dind, ind)

    def test_rename_series_method(self):
        # Series name
        s = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')
        ds = self.spark.from_pandas(pd.DataFrame(s)).x

        self.assert_eq(ds.rename('y'), s.rename('y'))
        self.assertEqual(ds.name, 'x')  # no mutation
        # self.assert_eq(ds.rename(), s.rename())

        ds.rename('z', inplace=True)
        s.rename('z', inplace=True)
        self.assertEqual(ds.name, 'z')
        self.assert_eq(ds, s)

        # Series index
        s = pd.Series(['a', 'b', 'c', 'd', 'e', 'f', 'g'], name='x')
        ds = self.spark.from_pandas(pd.DataFrame(s)).x

        # TODOD: index
        # res = ds.rename(lambda x: x ** 2)
        # self.assert_eq(res, s.rename(lambda x: x ** 2))

        # res = ds.rename(s)
        # self.assert_eq(res, s.rename(s))

        # res = ds.rename(ds)
        # self.assert_eq(res, s.rename(s))

        # res = ds.rename(lambda x: x**2, inplace=True)
        # self.assertis(res, ds)
        # s.rename(lambda x: x**2, inplace=True)
        # self.assert_eq(ds, s)

    def test_dropna(self):
        df = pd.DataFrame({'x': [np.nan, 2, 3, 4, np.nan, 6],
                           'y': [1, 2, np.nan, 4, np.nan, np.nan],
                           'z': [1, 2, 3, 4, np.nan, np.nan]},
                          index=[10, 20, 30, 40, 50, 60])
        ddf = self.spark.from_pandas(df)

        self.assert_eq(ddf.x.dropna(), df.x.dropna())
        self.assert_eq(ddf.y.dropna(), df.y.dropna())
        self.assert_eq(ddf.z.dropna(), df.z.dropna())

        self.assert_eq(ddf.dropna(), df.dropna())
        self.assert_eq(ddf.dropna(how='all'), df.dropna(how='all'))
        self.assert_eq(ddf.dropna(subset=['x']), df.dropna(subset=['x']))
        self.assert_eq(ddf.dropna(subset=['y', 'z']), df.dropna(subset=['y', 'z']))
        self.assert_eq(ddf.dropna(subset=['y', 'z'], how='all'),
                       df.dropna(subset=['y', 'z'], how='all'))

        self.assert_eq(ddf.dropna(thresh=2), df.dropna(thresh=2))
        self.assert_eq(ddf.dropna(thresh=1, subset=['y', 'z']),
                       df.dropna(thresh=1, subset=['y', 'z']))

        ddf2 = ddf.copy()
        x = ddf2.x
        x.dropna(inplace=True)
        self.assert_eq(x, df.x.dropna())
        ddf2.dropna(inplace=True)
        self.assert_eq(ddf2, df.dropna())

        msg = "dropna currently only works for axis=0 or axis='index'"
        with self.assertRaisesRegex(NotImplementedError, msg):
            ddf.dropna(axis=1)
        with self.assertRaisesRegex(NotImplementedError, msg):
            ddf.dropna(axis='column')
        with self.assertRaisesRegex(NotImplementedError, msg):
            ddf.dropna(axis='foo')

    def test_value_counts(self):
        df = pd.DataFrame({'x': [1, 2, 1, 3, 3, np.nan, 1, 4]})
        ddf = self.spark.from_pandas(df)

        self.assertPandasAlmostEqual(ddf.x.value_counts().toPandas(), df.x.value_counts())
        self.assertPandasAlmostEqual(ddf.x.value_counts(normalize=True).toPandas(),
                                     df.x.value_counts(normalize=True))
        self.assertPandasAlmostEqual(ddf.x.value_counts(ascending=True).toPandas(),
                                     df.x.value_counts(ascending=True))
        self.assertPandasAlmostEqual(ddf.x.value_counts(normalize=True, dropna=False).toPandas(),
                                     df.x.value_counts(normalize=True, dropna=False))
        self.assertPandasAlmostEqual(ddf.x.value_counts(ascending=True, dropna=False).toPandas(),
                                     df.x.value_counts(ascending=True, dropna=False))

        with self.assertRaisesRegex(NotImplementedError,
                                    "value_counts currently does not support bins"):
            ddf.x.value_counts(bins=3)

    def test_to_datetime(self):
        df = pd.DataFrame({'year': [2015, 2016],
                           'month': [2, 3],
                           'day': [4, 5]})
        ddf = self.spark.from_pandas(df)

        self.assert_eq(pd.to_datetime(df), pyspark.to_datetime(ddf))

        s = pd.Series(['3/11/2000', '3/12/2000', '3/13/2000'] * 100)
        ds = self.spark.from_pandas(pd.DataFrame({'s': s}))['s']

        self.assert_eq(pd.to_datetime(s, infer_datetime_format=True),
                       pyspark.to_datetime(ds, infer_datetime_format=True))

    def test_abs(self):
        df = pd.DataFrame({'A': [1, -2, 3, -4, 5],
                           'B': [1., -2, 3, -4, 5],
                           'C': [-6., -7, -8, -9, 10],
                           'D': ['a', 'b', 'c', 'd', 'e']})
        ddf = self.spark.from_pandas(df)
        self.assert_eq(ddf.A.abs(), df.A.abs())
        self.assert_eq(ddf.B.abs(), df.B.abs())
        self.assert_eq(ddf.select('B', 'C').abs(), df[['B', 'C']].abs())
        # self.assert_eq(ddf.select('A', 'B').abs(), df[['A', 'B']].abs())


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
