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

import inspect

import numpy as np
import pandas as pd

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.missing.frame import _MissingPandasLikeDataFrame
from databricks.koalas.missing.series import _MissingPandasLikeSeries
from databricks.koalas.series import Series


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

    def test_dataframe(self):
        kdf = self.kdf
        pdf = self.pdf

        expected = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10],
                             index=[0, 1, 3, 5, 6, 8, 9, 9, 9],
                             name='(a + 1)')  # TODO: name='a'

        self.assert_eq(kdf['a'] + 1, expected)

        self.assert_eq(kdf.columns, pd.Index(['a', 'b']))

        self.assert_eq(kdf[kdf['b'] > 2], pdf[pdf['b'] > 2])
        self.assert_eq(kdf[['a', 'b']], pdf[['a', 'b']])
        self.assert_eq(kdf.a, pdf.a)
        # TODO: assert d.b.mean().compute() == pdf.b.mean()
        # TODO: assert np.allclose(d.b.var().compute(), pdf.b.var())
        # TODO: assert np.allclose(d.b.std().compute(), pdf.b.std())

        assert repr(kdf)

        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        })
        ddf = koalas.from_pandas(df)
        self.assert_eq(df[['a', 'b']], ddf[['a', 'b']])

        self.assertEqual(ddf.a.notnull().alias("x").name, "x")

    def test_empty_dataframe(self):
        a = pd.Series([], dtype='i1')
        b = pd.Series([], dtype='str')
        pdf = pd.DataFrame({'a': a, 'b': b})

        self.assert_eq(koalas.from_pandas(a), a)
        self.assertRaises(ValueError, lambda: koalas.from_pandas(b))
        self.assertRaises(ValueError, lambda: koalas.from_pandas(pdf))

        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            self.assert_eq(koalas.from_pandas(a), a)
            self.assertRaises(ValueError, lambda: koalas.from_pandas(b))
            self.assertRaises(ValueError, lambda: koalas.from_pandas(pdf))

    def test_all_null_dataframe(self):
        a = pd.Series([None, None, None], dtype='float64')
        b = pd.Series([None, None, None], dtype='str')
        pdf = pd.DataFrame({'a': a, 'b': b})

        self.assert_eq(koalas.from_pandas(a).dtype, a.dtype)
        self.assertTrue(koalas.from_pandas(a).toPandas().isnull().all())
        self.assertRaises(ValueError, lambda: koalas.from_pandas(b))
        self.assertRaises(ValueError, lambda: koalas.from_pandas(pdf))

        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            self.assert_eq(koalas.from_pandas(a).dtype, a.dtype)
            self.assertTrue(koalas.from_pandas(a).toPandas().isnull().all())
            self.assertRaises(ValueError, lambda: koalas.from_pandas(b))
            self.assertRaises(ValueError, lambda: koalas.from_pandas(pdf))

    def test_nullable_object(self):
        pdf = pd.DataFrame({'a': list('abc') + [np.nan],
                            'b': list(range(1, 4)) + [np.nan],
                            'c': list(np.arange(3, 6).astype('i1')) + [np.nan],
                            'd': list(np.arange(4.0, 7.0, dtype='float64')) + [np.nan],
                            'e': [True, False, True, np.nan],
                            'f': list(pd.date_range('20130101', periods=3)) + [np.nan]})

        kdf = koalas.from_pandas(pdf)
        self.assert_eq(kdf, pdf)

        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            kdf = koalas.from_pandas(pdf)
            self.assert_eq(kdf, pdf)

    def test_assign(self):
        kdf = self.kdf.copy()
        pdf = self.pdf.copy()

        kdf['w'] = 1.0
        pdf['w'] = 1.0

        self.assert_eq(kdf, pdf)

        kdf['a'] = 'abc'
        pdf['a'] = 'abc'

        self.assert_eq(kdf, pdf)

    def test_head_tail(self):
        kdf = self.kdf
        pdf = self.pdf

        self.assert_eq(kdf.head(2), pdf.head(2))
        self.assert_eq(kdf.head(3), pdf.head(3))
        self.assert_eq(kdf['a'].head(2), pdf['a'].head(2))
        self.assert_eq(kdf['a'].head(3), pdf['a'].head(3))

        # TODO: self.assert_eq(d.tail(2), pdf.tail(2))
        # TODO: self.assert_eq(d.tail(3), pdf.tail(3))
        # TODO: self.assert_eq(d['a'].tail(2), pdf['a'].tail(2))
        # TODO: self.assert_eq(d['a'].tail(3), pdf['a'].tail(3))

    def test_index_head(self):
        kdf = self.kdf
        pdf = self.pdf

        self.assert_eq(list(kdf.index.head(2).toPandas()), list(pdf.index[:2]))
        self.assert_eq(list(kdf.index.head(3).toPandas()), list(pdf.index[:3]))

    def test_Series(self):
        kdf = self.kdf
        pdf = self.pdf

        self.assertTrue(isinstance(kdf.a, Series))
        self.assertTrue(isinstance(kdf.a + 1, Series))
        self.assertTrue(isinstance(1 + kdf.a, Series))
        # TODO: self.assert_eq(d + 1, pdf + 1)

    def test_Index(self):
        for case in [pd.DataFrame(np.random.randn(10, 5), index=list('abcdefghij')),
                     pd.DataFrame(np.random.randn(10, 5),
                                  index=pd.date_range('2011-01-01', freq='D',
                                                      periods=10))]:
            ddf = koalas.from_pandas(case)
            self.assert_eq(list(ddf.index.toPandas()), list(case.index))

    def test_attributes(self):
        kdf = self.kdf

        self.assertIn('a', dir(kdf))
        self.assertNotIn('foo', dir(kdf))
        self.assertRaises(AttributeError, lambda: kdf.foo)

        kdf = koalas.DataFrame({'a b c': [1, 2, 3]})
        self.assertNotIn('a b c', dir(kdf))
        kdf = koalas.DataFrame({'a': [1, 2], 5: [1, 2]})
        self.assertIn('a', dir(kdf))
        self.assertNotIn(5, dir(kdf))

    def test_column_names(self):
        kdf = self.kdf

        self.assert_eq(kdf.columns, pd.Index(['a', 'b']))
        self.assert_eq(kdf[['b', 'a']].columns, pd.Index(['b', 'a']))
        self.assertEqual(kdf['a'].name, 'a')
        self.assertEqual((kdf['a'] + 1).name, '(a + 1)')  # TODO: 'a'
        self.assertEqual((kdf['a'] + kdf['b']).name, '(a + b)')  # TODO: None

    def test_index_names(self):
        # kdf = self.kdf
        # TODO?: self.assertIsNone(kdf.index.name)

        idx = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name='x')
        df = pd.DataFrame(np.random.randn(10, 5), idx)
        ddf = koalas.from_pandas(df)
        self.assertEqual(ddf.index.name, 'x')

    def test_rename_columns(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7],
                            'b': [7, 6, 5, 4, 3, 2, 1]})
        kdf = koalas.from_pandas(pdf)

        kdf.columns = ['x', 'y']
        pdf.columns = ['x', 'y']
        self.assert_eq(kdf.columns, pd.Index(['x', 'y']))
        self.assert_eq(kdf, pdf)

        msg = "Length mismatch: Expected axis has 2 elements, new values have 4 elements"
        with self.assertRaisesRegex(ValueError, msg):
            kdf.columns = [1, 2, 3, 4]

        # Multi-index columns
        pdf = pd.DataFrame({('A', '0'): [1, 2, 2, 3], ('B', 1): [1, 2, 3, 4]})
        kdf = koalas.from_pandas(pdf)

        pdf.columns = ['x', 'y']
        kdf.columns = ['x', 'y']
        self.assert_eq(kdf.columns, pd.Index(['x', 'y']))
        self.assert_eq(kdf, pdf)

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

    def test_dropna(self):
        pdf = pd.DataFrame({'x': [np.nan, 2, 3, 4, np.nan, 6],
                            'y': [1, 2, np.nan, 4, np.nan, np.nan],
                            'z': [1, 2, 3, 4, np.nan, np.nan]},
                           index=[10, 20, 30, 40, 50, 60])
        kdf = koalas.from_pandas(pdf)

        self.assert_eq(kdf.x.dropna(), pdf.x.dropna())
        self.assert_eq(kdf.y.dropna(), pdf.y.dropna())
        self.assert_eq(kdf.z.dropna(), pdf.z.dropna())

        self.assert_eq(kdf.dropna(), pdf.dropna())
        self.assert_eq(kdf.dropna(how='all'), pdf.dropna(how='all'))
        self.assert_eq(kdf.dropna(subset=['x']), pdf.dropna(subset=['x']))
        self.assert_eq(kdf.dropna(subset=['y', 'z']), pdf.dropna(subset=['y', 'z']))
        self.assert_eq(kdf.dropna(subset=['y', 'z'], how='all'),
                       pdf.dropna(subset=['y', 'z'], how='all'))

        self.assert_eq(kdf.dropna(thresh=2), pdf.dropna(thresh=2))
        self.assert_eq(kdf.dropna(thresh=1, subset=['y', 'z']),
                       pdf.dropna(thresh=1, subset=['y', 'z']))

        ddf2 = kdf.copy()
        x = ddf2.x
        x.dropna(inplace=True)
        self.assert_eq(x, pdf.x.dropna())
        ddf2.dropna(inplace=True)
        self.assert_eq(ddf2, pdf.dropna())

        msg = "dropna currently only works for axis=0 or axis='index'"
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.dropna(axis=1)
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.dropna(axis='column')
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.dropna(axis='foo')

    def test_dtype(self):
        pdf = pd.DataFrame({'a': list('abc'),
                            'b': list(range(1, 4)),
                            'c': np.arange(3, 6).astype('i1'),
                            'd': np.arange(4.0, 7.0, dtype='float64'),
                            'e': [True, False, True],
                            'f': pd.date_range('20130101', periods=3)})
        kdf = koalas.from_pandas(pdf)
        self.assert_eq(kdf, pdf)
        self.assertTrue((kdf.dtypes == pdf.dtypes).all())

    def test_value_counts(self):
        pdf = pd.DataFrame({'x': [1, 2, 1, 3, 3, np.nan, 1, 4]})
        kdf = koalas.from_pandas(pdf)

        exp = pdf.x.value_counts()
        res = kdf.x.value_counts()
        self.assertEqual(res.name, exp.name)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        self.assertPandasAlmostEqual(kdf.x.value_counts(normalize=True).toPandas(),
                                     pdf.x.value_counts(normalize=True))
        self.assertPandasAlmostEqual(kdf.x.value_counts(ascending=True).toPandas(),
                                     pdf.x.value_counts(ascending=True))
        self.assertPandasAlmostEqual(kdf.x.value_counts(normalize=True, dropna=False).toPandas(),
                                     pdf.x.value_counts(normalize=True, dropna=False))
        self.assertPandasAlmostEqual(kdf.x.value_counts(ascending=True, dropna=False).toPandas(),
                                     pdf.x.value_counts(ascending=True, dropna=False))

        with self.assertRaisesRegex(NotImplementedError,
                                    "value_counts currently does not support bins"):
            kdf.x.value_counts(bins=3)

        s = pdf.x
        s.name = 'index'
        ds = kdf.x
        ds.name = 'index'
        self.assertPandasAlmostEqual(ds.value_counts().toPandas(), s.value_counts())

    def test_isnull(self):
        pdf = pd.DataFrame({'x': [1, 2, 3, 4, None, 6], 'y': list('abdabd')},
                           index=[10, 20, 30, 40, 50, 60])
        kdf = koalas.from_pandas(pdf)

        self.assert_eq(kdf.x.notnull(), pdf.x.notnull())
        self.assert_eq(kdf.x.isnull(), pdf.x.isnull())
        self.assert_eq(kdf.notnull(), pdf.notnull())
        self.assert_eq(kdf.isnull(), pdf.isnull())

    def test_to_datetime(self):
        pdf = pd.DataFrame({'year': [2015, 2016],
                            'month': [2, 3],
                            'day': [4, 5]})
        kdf = koalas.from_pandas(pdf)

        self.assert_eq(pd.to_datetime(pdf), koalas.to_datetime(kdf))

        s = pd.Series(['3/11/2000', '3/12/2000', '3/13/2000'] * 100)
        ds = koalas.from_pandas(s)

        self.assert_eq(pd.to_datetime(s, infer_datetime_format=True),
                       koalas.to_datetime(ds, infer_datetime_format=True))

    def test_sort_values(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, None, 7],
                           'b': [7, 6, 5, 4, 3, 2, 1]})
        kdf = koalas.from_pandas(pdf)
        self.assert_eq(kdf.sort_values('b'), pdf.sort_values('b'))
        self.assert_eq(kdf.sort_values(['b', 'a']), pdf.sort_values(['b', 'a']))
        self.assert_eq(
            repr(kdf.sort_values(['b', 'a'], ascending=[False, True])),
            repr(pdf.sort_values(['b', 'a'], ascending=[False, True])))

        self.assertRaises(ValueError, lambda: kdf.sort_values(['b', 'a'], ascending=[False]))

        self.assert_eq(
            kdf.sort_values(['b', 'a'], na_position='first'),
            pdf.sort_values(['b', 'a'], na_position='first'))

        self.assertRaises(ValueError, lambda: kdf.sort_values(['b', 'a'], na_position='invalid'))

        self.assert_eq(kdf.sort_values('b', inplace=True), pdf.sort_values('b', inplace=True))
        self.assert_eq(kdf, pdf)

    def test_missing(self):
        kdf = self.kdf

        missing_functions = inspect.getmembers(_MissingPandasLikeDataFrame, inspect.isfunction)
        for name, _ in missing_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "DataFrame.*{}.*not implemented".format(name)):
                getattr(kdf, name)()

        missing_functions = inspect.getmembers(_MissingPandasLikeSeries, inspect.isfunction)
        for name, _ in missing_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "Series.*{}.*not implemented".format(name)):
                getattr(kdf.a, name)()

    def test_to_numpy(self):
        pdf = pd.DataFrame({'a': [4, 2, 3, 4, 8, 6],
                            'b': [1, 2, 9, 4, 2, 4],
                            'c': ["one", "three", "six", "seven", "one", "5"]},
                           index=[10, 20, 30, 40, 50, 60])

        kdf = koalas.from_pandas(pdf)

        np.testing.assert_equal(kdf.to_numpy(), pdf.values)

        s = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

        ddf = koalas.from_pandas(s)
        np.testing.assert_equal(ddf.to_numpy(), s.values)

    def test_to_pandas(self):
        kdf = self.kdf
        pdf = self.pdf
        self.assert_eq(kdf.toPandas(), pdf)
        self.assert_eq(kdf.to_pandas(), pdf)
