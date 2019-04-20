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

from distutils.version import LooseVersion
import unittest

import numpy as np
import pandas as pd

from databricks import koalas
from databricks.koalas.exceptions import SparkPandasIndexingError
from databricks.koalas.testing.utils import ComparisonTestBase, ReusedSQLTestCase, compare_both


class BasicIndexingTest(ComparisonTestBase):

    @property
    def pdf(self):
        return pd.DataFrame({'month': [1, 4, 7, 10],
                             'year': [2012, 2014, 2013, 2014],
                             'sale': [55, 40, 84, 31]})

    @compare_both(almost=False)
    def test_indexing(self, df):
        df1 = df.set_index('month')
        yield df1

        yield df.set_index('month', drop=False)
        yield df.set_index('month', append=True)
        yield df.set_index(['year', 'month'])
        yield df.set_index(['year', 'month'], drop=False)
        yield df.set_index(['year', 'month'], append=True)

        yield df1.set_index('year', drop=False, append=True)

        df2 = df1.copy()
        df2.set_index('year', append=True, inplace=True)
        yield df2

        self.assertRaisesRegex(KeyError, 'unknown', lambda: df.set_index('unknown'))
        self.assertRaisesRegex(KeyError, 'unknown', lambda: df.set_index(['month', 'unknown']))

        for d in [df, df1, df2]:
            yield d.reset_index()
            yield d.reset_index(drop=True)

        yield df1.reset_index(level=0)
        yield df2.reset_index(level=1)
        yield df2.reset_index(level=[1, 0])
        yield df1.reset_index(level='month')
        yield df2.reset_index(level='year')
        yield df2.reset_index(level=['month', 'year'])
        yield df2.reset_index(level='month', drop=True)
        yield df2.reset_index(level=['month', 'year'], drop=True)

        if LooseVersion("0.20.0") <= LooseVersion(pd.__version__):
            self.assertRaisesRegex(IndexError, 'Too many levels: Index has only 1 level, not 3',
                                   lambda: df1.reset_index(level=2))
            self.assertRaisesRegex(IndexError, 'Too many levels: Index has only 1 level, not 4',
                                   lambda: df1.reset_index(level=[3, 2]))
            self.assertRaisesRegex(KeyError, 'Level unknown must be same as name \\(month\\)',
                                   lambda: df1.reset_index(level='unknown'))
        self.assertRaisesRegex(KeyError, 'Level unknown not found',
                               lambda: df2.reset_index(level='unknown'))

        df3 = df2.copy()
        df3.reset_index(inplace=True)
        yield df3

        yield df1.sale.reset_index()
        yield df1.sale.reset_index(level=0)
        yield df2.sale.reset_index(level=[1, 0])
        yield df1.sale.reset_index(drop=True)
        yield df1.sale.reset_index(name='s')
        yield df1.sale.reset_index(name='s', drop=True)

        s = df1.sale
        self.assertRaisesRegex(TypeError,
                               'Cannot reset_index inplace on a Series to create a DataFrame',
                               lambda: s.reset_index(inplace=True))
        s.reset_index(drop=True, inplace=True)
        yield s
        yield df1

    def test_from_pandas_with_explicit_index(self):
        pdf = self.pdf

        df1 = koalas.from_pandas(pdf.set_index('month'))
        self.assertPandasEqual(df1.toPandas(), pdf.set_index('month'))

        df2 = koalas.from_pandas(pdf.set_index(['year', 'month']))
        self.assertPandasEqual(df2.toPandas(), pdf.set_index(['year', 'month']))

    def test_limitations(self):
        df = self.df.set_index('month')

        self.assertRaisesRegex(ValueError, 'Level should be all int or all string.',
                               lambda: df.reset_index([1, 'month']))
        self.assertRaisesRegex(NotImplementedError, 'Can\'t reset index because there is no index.',
                               lambda: df.reset_index().reset_index())


class IndexingTest(ReusedSQLTestCase):

    @property
    def full(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0]
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def df(self):
        return koalas.from_pandas(self.full)

    def test_loc(self):
        d = self.df
        full = self.full

        self.assert_eq(d.loc[5:5], full.loc[5:5])
        self.assert_eq(d.loc[3:8], full.loc[3:8])
        self.assert_eq(d.loc[:8], full.loc[:8])
        self.assert_eq(d.loc[3:], full.loc[3:])
        self.assert_eq(d.loc[[5]], full.loc[[5]])
        self.assert_eq(d.loc[:], full.loc[:])

        # TODO?: self.assert_eq(d.loc[[3, 4, 1, 8]], full.loc[[3, 4, 1, 8]])
        # TODO?: self.assert_eq(d.loc[[3, 4, 1, 9]], full.loc[[3, 4, 1, 9]])
        # TODO?: self.assert_eq(d.loc[np.array([3, 4, 1, 9])], full.loc[np.array([3, 4, 1, 9])])

        self.assert_eq(d.a.loc[5:5], full.a.loc[5:5])
        self.assert_eq(d.a.loc[3:8], full.a.loc[3:8])
        self.assert_eq(d.a.loc[:8], full.a.loc[:8])
        self.assert_eq(d.a.loc[3:], full.a.loc[3:])
        self.assert_eq(d.a.loc[[5]], full.a.loc[[5]])

        # TODO?: self.assert_eq(d.a.loc[[3, 4, 1, 8]], full.a.loc[[3, 4, 1, 8]])
        # TODO?: self.assert_eq(d.a.loc[[3, 4, 1, 9]], full.a.loc[[3, 4, 1, 9]])
        # TODO?: self.assert_eq(d.a.loc[np.array([3, 4, 1, 9])], full.a.loc[np.array([3, 4, 1, 9])])

        self.assert_eq(d.a.loc[[]], full.a.loc[[]])
        self.assert_eq(d.a.loc[np.array([])], full.a.loc[np.array([])])

        self.assert_eq(d.loc[1000:], full.loc[1000:])
        self.assert_eq(d.loc[-2000:-1000], full.loc[-2000:-1000])

    def test_loc_non_informative_index(self):
        df = pd.DataFrame({'x': [1, 2, 3, 4]}, index=[10, 20, 30, 40])
        ddf = koalas.from_pandas(df)

        self.assert_eq(ddf.loc[20:30], df.loc[20:30])

        df = pd.DataFrame({'x': [1, 2, 3, 4]}, index=[10, 20, 20, 40])
        ddf = koalas.from_pandas(df)
        self.assert_eq(ddf.loc[20:20], df.loc[20:20])

    def test_loc_with_series(self):
        d = self.df
        full = self.full

        self.assert_eq(d.loc[d.a % 2 == 0], full.loc[full.a % 2 == 0])

    def test_loc_noindex(self):
        d = self.df
        d = d.reset_index()
        full = self.full
        full = full.reset_index()

        self.assert_eq(d[['a']], full[['a']])

        self.assert_eq(d.loc[:], full.loc[:])
        self.assertRaises(NotImplementedError, lambda: d.loc[5:5])

    def test_loc_multiindex(self):
        d = self.df
        d = d.set_index('b', append=True)
        full = self.full
        full = full.set_index('b', append=True)

        self.assert_eq(d[['a']], full[['a']])

        self.assert_eq(d.loc[:], full.loc[:])
        self.assertRaises(NotImplementedError, lambda: d.loc[5:5])

    def test_loc2d_multiindex(self):
        d = self.df
        d = d.set_index('b', append=True)
        full = self.full
        full = full.set_index('b', append=True)

        self.assert_eq(d.loc[:, :], full.loc[:, :])
        self.assert_eq(d.loc[:, 'a'], full.loc[:, 'a'])
        self.assertRaises(NotImplementedError, lambda: d.loc[5:5, 'a'])

    def test_loc2d(self):
        d = self.df
        full = self.full

        # index indexer is always regarded as slice for duplicated values
        self.assert_eq(d.loc[5:5, 'a'], full.loc[5:5, 'a'])
        self.assert_eq(d.loc[[5], 'a'], full.loc[[5], 'a'])
        self.assert_eq(d.loc[5:5, ['a']], full.loc[5:5, ['a']])
        self.assert_eq(d.loc[[5], ['a']], full.loc[[5], ['a']])
        self.assert_eq(d.loc[:, :], full.loc[:, :])

        self.assert_eq(d.loc[3:8, 'a'], full.loc[3:8, 'a'])
        self.assert_eq(d.loc[:8, 'a'], full.loc[:8, 'a'])
        self.assert_eq(d.loc[3:, 'a'], full.loc[3:, 'a'])
        self.assert_eq(d.loc[[8], 'a'], full.loc[[8], 'a'])

        self.assert_eq(d.loc[3:8, ['a']], full.loc[3:8, ['a']])
        self.assert_eq(d.loc[:8, ['a']], full.loc[:8, ['a']])
        self.assert_eq(d.loc[3:, ['a']], full.loc[3:, ['a']])
        # TODO?: self.assert_eq(d.loc[[3, 4, 3], ['a']], full.loc[[3, 4, 3], ['a']])

        self.assertRaises(SparkPandasIndexingError, lambda: d.loc[3, 3, 3])
        self.assertRaises(SparkPandasIndexingError, lambda: d.a.loc[3, 3])
        self.assertRaises(SparkPandasIndexingError, lambda: d.a.loc[3:, 3])
        self.assertRaises(SparkPandasIndexingError, lambda: d.a.loc[d.a % 2 == 0, 3])

    def test_loc2d_with_known_divisions(self):
        df = pd.DataFrame(np.random.randn(20, 5),
                          index=list('abcdefghijklmnopqrst'),
                          columns=list('ABCDE'))
        ddf = koalas.from_pandas(df)

        self.assert_eq(ddf.loc[['a'], 'A'], df.loc[['a'], 'A'])
        self.assert_eq(ddf.loc[['a'], ['A']], df.loc[['a'], ['A']])
        self.assert_eq(ddf.loc['a':'o', 'A'], df.loc['a':'o', 'A'])
        self.assert_eq(ddf.loc['a':'o', ['A']], df.loc['a':'o', ['A']])
        self.assert_eq(ddf.loc[['n'], ['A']], df.loc[['n'], ['A']])
        self.assert_eq(ddf.loc[['a', 'c', 'n'], ['A']], df.loc[['a', 'c', 'n'], ['A']])
        # TODO?: self.assert_eq(ddf.loc[['t', 'b'], ['A']], df.loc[['t', 'b'], ['A']])
        # TODO?: self.assert_eq(ddf.loc[['r', 'r', 'c', 'g', 'h'], ['A']],
        # TODO?:                df.loc[['r', 'r', 'c', 'g', 'h'], ['A']])

    def test_loc2d_duplicated_columns(self):
        df = pd.DataFrame(np.random.randn(20, 5),
                          index=list('abcdefghijklmnopqrst'),
                          columns=list('AABCD'))
        ddf = koalas.from_pandas(df)

        # TODO?: self.assert_eq(ddf.loc[['a'], 'A'], df.loc[['a'], 'A'])
        # TODO?: self.assert_eq(ddf.loc[['a'], ['A']], df.loc[['a'], ['A']])
        self.assert_eq(ddf.loc[['j'], 'B'], df.loc[['j'], 'B'])
        self.assert_eq(ddf.loc[['j'], ['B']], df.loc[['j'], ['B']])

        # TODO?: self.assert_eq(ddf.loc['a':'o', 'A'], df.loc['a':'o', 'A'])
        # TODO?: self.assert_eq(ddf.loc['a':'o', ['A']], df.loc['a':'o', ['A']])
        self.assert_eq(ddf.loc['j':'q', 'B'], df.loc['j':'q', 'B'])
        self.assert_eq(ddf.loc['j':'q', ['B']], df.loc['j':'q', ['B']])

        # TODO?: self.assert_eq(ddf.loc['a':'o', 'B':'D'], df.loc['a':'o', 'B':'D'])
        # TODO?: self.assert_eq(ddf.loc['a':'o', 'B':'D'], df.loc['a':'o', 'B':'D'])
        # TODO?: self.assert_eq(ddf.loc['j':'q', 'B':'A'], df.loc['j':'q', 'B':'A'])
        # TODO?: self.assert_eq(ddf.loc['j':'q', 'B':'A'], df.loc['j':'q', 'B':'A'])

        self.assert_eq(ddf.loc[ddf.B > 0, 'B'], df.loc[df.B > 0, 'B'])
        # TODO?: self.assert_eq(ddf.loc[ddf.B > 0, ['A', 'C']], df.loc[df.B > 0, ['A', 'C']])

    def test_getitem(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                           'B': [9, 8, 7, 6, 5, 4, 3, 2, 1],
                           'C': [True, False, True] * 3},
                          columns=list('ABC'))
        ddf = koalas.from_pandas(df)
        self.assert_eq(ddf['A'], df['A'])

        self.assert_eq(ddf[['A', 'B']], df[['A', 'B']])

        self.assert_eq(ddf[ddf.C], df[df.C])

        self.assertRaises(KeyError, lambda: ddf['X'])
        self.assertRaises(KeyError, lambda: ddf[['A', 'X']])
        self.assertRaises(AttributeError, lambda: ddf.X)

        # not str/unicode
        # TODO?: df = pd.DataFrame(np.random.randn(10, 5))
        # TODO?: ddf = koalas.from_pandas(df)
        # TODO?: self.assert_eq(ddf[0], df[0])
        # TODO?: self.assert_eq(ddf[[1, 2]], df[[1, 2]])

        # TODO?: self.assertRaises(KeyError, lambda: df[8])
        # TODO?: self.assertRaises(KeyError, lambda: df[[1, 8]])

    def test_getitem_slice(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                           'B': [9, 8, 7, 6, 5, 4, 3, 2, 1],
                           'C': [True, False, True] * 3},
                          index=list('abcdefghi'))
        ddf = koalas.from_pandas(df)
        self.assert_eq(ddf['a':'e'], df['a':'e'])
        self.assert_eq(ddf['a':'b'], df['a':'b'])
        self.assert_eq(ddf['f':], df['f':])

    def test_loc_on_numpy_datetimes(self):
        df = pd.DataFrame({'x': [1, 2, 3]},
                          index=list(map(np.datetime64, ['2014', '2015', '2016'])))
        a = koalas.from_pandas(df)

        self.assert_eq(a.loc['2014':'2015'], df.loc['2014':'2015'])

    def test_loc_on_pandas_datetimes(self):
        df = pd.DataFrame({'x': [1, 2, 3]},
                          index=list(map(pd.Timestamp, ['2014', '2015', '2016'])))
        a = koalas.from_pandas(df)

        self.assert_eq(a.loc['2014':'2015'], df.loc['2014':'2015'])

    @unittest.skip('TODO?: the behavior of slice for datetime')
    def test_loc_datetime_no_freq(self):
        datetime_index = pd.date_range('2016-01-01', '2016-01-31', freq='12h')
        datetime_index.freq = None  # FORGET FREQUENCY
        df = pd.DataFrame({'num': range(len(datetime_index))}, index=datetime_index)
        ddf = koalas.from_pandas(df)

        slice_ = slice('2016-01-03', '2016-01-05')
        result = ddf.loc[slice_, :]
        expected = df.loc[slice_, :]
        self.assert_eq(result, expected)

    @unittest.skip('TODO?: the behavior of slice for datetime')
    def test_loc_timestamp_str(self):
        df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                          index=pd.date_range('2011-01-01', freq='H', periods=100))
        ddf = koalas.from_pandas(df)

        # partial string slice
        # TODO?: self.assert_eq(df.loc['2011-01-02'],
        # TODO?:                ddf.loc['2011-01-02'])
        self.assert_eq(df.loc['2011-01-02':'2011-01-05'],
                       ddf.loc['2011-01-02':'2011-01-05'])

        # series
        # TODO?: self.assert_eq(df.A.loc['2011-01-02'],
        # TODO?:                ddf.A.loc['2011-01-02'])
        self.assert_eq(df.A.loc['2011-01-02':'2011-01-05'],
                       ddf.A.loc['2011-01-02':'2011-01-05'])

        df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                          index=pd.date_range('2011-01-01', freq='M', periods=100))
        ddf = koalas.from_pandas(df)
        # TODO?: self.assert_eq(df.loc['2011-01'], ddf.loc['2011-01'])
        # TODO?: self.assert_eq(df.loc['2011'], ddf.loc['2011'])

        self.assert_eq(df.loc['2011-01':'2012-05'], ddf.loc['2011-01':'2012-05'])
        self.assert_eq(df.loc['2011':'2015'], ddf.loc['2011':'2015'])

        # series
        # TODO?: self.assert_eq(df.B.loc['2011-01'], ddf.B.loc['2011-01'])
        # TODO?: self.assert_eq(df.B.loc['2011'], ddf.B.loc['2011'])

        self.assert_eq(df.B.loc['2011-01':'2012-05'], ddf.B.loc['2011-01':'2012-05'])
        self.assert_eq(df.B.loc['2011':'2015'], ddf.B.loc['2011':'2015'])

    @unittest.skip('TODO?: the behavior of slice for datetime')
    def test_getitem_timestamp_str(self):
        df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                          index=pd.date_range('2011-01-01', freq='H', periods=100))
        ddf = koalas.from_pandas(df)

        # partial string slice
        # TODO?: self.assert_eq(df['2011-01-02'],
        # TODO?:                ddf['2011-01-02'])
        self.assert_eq(df['2011-01-02':'2011-01-05'],
                       ddf['2011-01-02':'2011-01-05'])

        df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                          index=pd.date_range('2011-01-01', freq='M', periods=100))
        ddf = koalas.from_pandas(df)

        # TODO?: self.assert_eq(df['2011-01'], ddf['2011-01'])
        # TODO?: self.assert_eq(df['2011'], ddf['2011'])

        self.assert_eq(df['2011-01':'2012-05'], ddf['2011-01':'2012-05'])
        self.assert_eq(df['2011':'2015'], ddf['2011':'2015'])

    @unittest.skip('TODO?: period index can\'t convert to DataFrame correctly')
    def test_getitem_period_str(self):
        df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                          index=pd.period_range('2011-01-01', freq='H', periods=100))
        ddf = koalas.from_pandas(df)

        # partial string slice
        # TODO?: self.assert_eq(df['2011-01-02'],
        # TODO?:                ddf['2011-01-02'])
        self.assert_eq(df['2011-01-02':'2011-01-05'],
                       ddf['2011-01-02':'2011-01-05'])

        df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                          index=pd.period_range('2011-01-01', freq='M', periods=100))
        ddf = koalas.from_pandas(df)

        # TODO?: self.assert_eq(df['2011-01'], ddf['2011-01'])
        # TODO?: self.assert_eq(df['2011'], ddf['2011'])

        self.assert_eq(df['2011-01':'2012-05'], ddf['2011-01':'2012-05'])
        self.assert_eq(df['2011':'2015'], ddf['2011':'2015'])


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
