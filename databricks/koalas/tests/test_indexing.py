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

from databricks import koalas as ks
from databricks.koalas.exceptions import SparkPandasIndexingError, SparkPandasNotImplementedError
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

        df1 = ks.from_pandas(pdf.set_index('month'))
        self.assertPandasEqual(df1.toPandas(), pdf.set_index('month'))

        df2 = ks.from_pandas(pdf.set_index(['year', 'month']))
        self.assertPandasEqual(df2.toPandas(), pdf.set_index(['year', 'month']))

    def test_limitations(self):
        df = self.kdf.set_index('month')

        self.assertRaisesRegex(ValueError, 'Level should be all int or all string.',
                               lambda: df.reset_index([1, 'month']))


class IndexingTest(ReusedSQLTestCase):

    @property
    def pdf(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0]
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def kdf(self):
        return ks.from_pandas(self.pdf)

    def test_at(self):
        pdf = self.pdf
        kdf = self.kdf
        # Create the equivalent of pdf.loc[3] as a Koalas Series
        # This is necessary because .loc[n] does not currently work with Koalas DataFrames (#383)
        test_series = ks.Series([3, 6], index=['a', 'b'], name='3')

        # Assert invalided signatures raise TypeError
        with self.assertRaises(TypeError, msg="Use DataFrame.at like .at[row_index, column_name]"):
            kdf.at[3]
        with self.assertRaises(TypeError, msg="Use DataFrame.at like .at[row_index, column_name]"):
            kdf.at['ab']  # 'ab' is of length 2 but str type instead of tuple
        with self.assertRaises(TypeError, msg="Use Series.at like .at[column_name]"):
            test_series.at[3, 'b']

        # Assert .at for DataFrames
        self.assertEqual(kdf.at[3, 'b'], 6)
        self.assertEqual(kdf.at[3, 'b'], pdf.at[3, 'b'])
        np.testing.assert_array_equal(kdf.at[9, 'b'], np.array([0, 0, 0]))
        np.testing.assert_array_equal(kdf.at[9, 'b'], pdf.at[9, 'b'])

        # Assert .at for Series
        self.assertEqual(test_series.at['b'], 6)
        self.assertEqual(test_series.at['b'], pdf.loc[3].at['b'])

        # Assert multi-character indices
        self.assertEqual(ks.Series([0, 1], index=['ab', 'cd']).at['ab'],
                         pd.Series([0, 1], index=['ab', 'cd']).at['ab'])

        # Assert invalid column or index names result in a KeyError like with pandas
        with self.assertRaises(KeyError, msg='x'):
            kdf.at[3, 'x']
        with self.assertRaises(KeyError, msg=99):
            kdf.at[99, 'b']

        # Assert setting values fails
        with self.assertRaises(TypeError):
            kdf.at[3, 'b'] = 10

    def test_at_multiindex_columns(self):
        arrays = [np.array(['bar', 'bar', 'baz', 'baz']),
                  np.array(['one', 'two', 'one', 'two'])]

        pdf = pd.DataFrame(np.random.randn(3, 4), index=['A', 'B', 'C'], columns=arrays)
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.at['B', ('bar', 'one')], pdf.at['B', ('bar', 'one')])

    def test_loc(self):
        kdf = self.kdf
        pdf = self.pdf

        self.assert_eq(kdf.loc[5:5], pdf.loc[5:5])
        self.assert_eq(kdf.loc[3:8], pdf.loc[3:8])
        self.assert_eq(kdf.loc[:8], pdf.loc[:8])
        self.assert_eq(kdf.loc[3:], pdf.loc[3:])
        self.assert_eq(kdf.loc[[5]], pdf.loc[[5]])
        self.assert_eq(kdf.loc[:], pdf.loc[:])

        # TODO?: self.assert_eq(kdf.loc[[3, 4, 1, 8]], pdf.loc[[3, 4, 1, 8]])
        # TODO?: self.assert_eq(kdf.loc[[3, 4, 1, 9]], pdf.loc[[3, 4, 1, 9]])
        # TODO?: self.assert_eq(kdf.loc[np.array([3, 4, 1, 9])], pdf.loc[np.array([3, 4, 1, 9])])

        self.assert_eq(kdf.a.loc[5:5], pdf.a.loc[5:5])
        self.assert_eq(kdf.a.loc[3:8], pdf.a.loc[3:8])
        self.assert_eq(kdf.a.loc[:8], pdf.a.loc[:8])
        self.assert_eq(kdf.a.loc[3:], pdf.a.loc[3:])
        self.assert_eq(kdf.a.loc[[5]], pdf.a.loc[[5]])

        # TODO?: self.assert_eq(kdf.a.loc[[3, 4, 1, 8]], pdf.a.loc[[3, 4, 1, 8]])
        # TODO?: self.assert_eq(kdf.a.loc[[3, 4, 1, 9]], pdf.a.loc[[3, 4, 1, 9]])
        # TODO?: self.assert_eq(kdf.a.loc[np.array([3, 4, 1, 9])],
        #                       pdf.a.loc[np.array([3, 4, 1, 9])])

        self.assert_eq(kdf.a.loc[[]], pdf.a.loc[[]])
        self.assert_eq(kdf.a.loc[np.array([])], pdf.a.loc[np.array([])])

        self.assert_eq(kdf.loc[1000:], pdf.loc[1000:])
        self.assert_eq(kdf.loc[-2000:-1000], pdf.loc[-2000:-1000])

    def test_loc_non_informative_index(self):
        pdf = pd.DataFrame({'x': [1, 2, 3, 4]}, index=[10, 20, 30, 40])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.loc[20:30], pdf.loc[20:30])

        pdf = pd.DataFrame({'x': [1, 2, 3, 4]}, index=[10, 20, 20, 40])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.loc[20:20], pdf.loc[20:20])

    def test_loc_with_series(self):
        kdf = self.kdf
        pdf = self.pdf

        self.assert_eq(kdf.loc[kdf.a % 2 == 0], pdf.loc[pdf.a % 2 == 0])

    def test_loc_noindex(self):
        kdf = self.kdf
        kdf = kdf.reset_index()
        pdf = self.pdf
        pdf = pdf.reset_index()

        self.assert_eq(kdf[['a']], pdf[['a']])

        self.assert_eq(kdf.loc[:], pdf.loc[:])
        self.assert_eq(kdf.loc[5:5], pdf.loc[5:5])

    def test_loc_multiindex(self):
        kdf = self.kdf
        kdf = kdf.set_index('b', append=True)
        pdf = self.pdf
        pdf = pdf.set_index('b', append=True)

        self.assert_eq(kdf[['a']], pdf[['a']])

        self.assert_eq(kdf.loc[:], pdf.loc[:])
        self.assertRaises(NotImplementedError, lambda: kdf.loc[5:5])

    def test_loc2d_multiindex(self):
        kdf = self.kdf
        kdf = kdf.set_index('b', append=True)
        pdf = self.pdf
        pdf = pdf.set_index('b', append=True)

        self.assert_eq(kdf.loc[:, :], pdf.loc[:, :])
        self.assert_eq(kdf.loc[:, 'a'], pdf.loc[:, 'a'])
        self.assertRaises(NotImplementedError, lambda: kdf.loc[5:5, 'a'])

    def test_loc2d(self):
        kdf = self.kdf
        pdf = self.pdf

        # index indexer is always regarded as slice for duplicated values
        self.assert_eq(kdf.loc[5:5, 'a'], pdf.loc[5:5, 'a'])
        self.assert_eq(kdf.loc[[5], 'a'], pdf.loc[[5], 'a'])
        self.assert_eq(kdf.loc[5:5, ['a']], pdf.loc[5:5, ['a']])
        self.assert_eq(kdf.loc[[5], ['a']], pdf.loc[[5], ['a']])
        self.assert_eq(kdf.loc[:, :], pdf.loc[:, :])

        self.assert_eq(kdf.loc[3:8, 'a'], pdf.loc[3:8, 'a'])
        self.assert_eq(kdf.loc[:8, 'a'], pdf.loc[:8, 'a'])
        self.assert_eq(kdf.loc[3:, 'a'], pdf.loc[3:, 'a'])
        self.assert_eq(kdf.loc[[8], 'a'], pdf.loc[[8], 'a'])

        self.assert_eq(kdf.loc[3:8, ['a']], pdf.loc[3:8, ['a']])
        self.assert_eq(kdf.loc[:8, ['a']], pdf.loc[:8, ['a']])
        self.assert_eq(kdf.loc[3:, ['a']], pdf.loc[3:, ['a']])
        # TODO?: self.assert_eq(kdf.loc[[3, 4, 3], ['a']], pdf.loc[[3, 4, 3], ['a']])

        self.assertRaises(SparkPandasIndexingError, lambda: kdf.loc[3, 3, 3])
        self.assertRaises(SparkPandasIndexingError, lambda: kdf.a.loc[3, 3])
        self.assertRaises(SparkPandasIndexingError, lambda: kdf.a.loc[3:, 3])
        self.assertRaises(SparkPandasIndexingError, lambda: kdf.a.loc[kdf.a % 2 == 0, 3])

    def test_loc2d_multiindex_columns(self):
        arrays = [np.array(['bar', 'bar', 'baz', 'baz']),
                  np.array(['one', 'two', 'one', 'two'])]

        pdf = pd.DataFrame(np.random.randn(3, 4), index=['A', 'B', 'C'], columns=arrays)
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.loc['B':'B', 'bar'], pdf.loc['B':'B', 'bar'])
        self.assert_eq(kdf.loc['B':'B', ['bar']], pdf.loc['B':'B', ['bar']])

    def test_loc2d_with_known_divisions(self):
        pdf = pd.DataFrame(np.random.randn(20, 5),
                           index=list('abcdefghijklmnopqrst'),
                           columns=list('ABCDE'))
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.loc[['a'], 'A'], pdf.loc[['a'], 'A'])
        self.assert_eq(kdf.loc[['a'], ['A']], pdf.loc[['a'], ['A']])
        self.assert_eq(kdf.loc['a':'o', 'A'], pdf.loc['a':'o', 'A'])
        self.assert_eq(kdf.loc['a':'o', ['A']], pdf.loc['a':'o', ['A']])
        self.assert_eq(kdf.loc[['n'], ['A']], pdf.loc[['n'], ['A']])
        self.assert_eq(kdf.loc[['a', 'c', 'n'], ['A']], pdf.loc[['a', 'c', 'n'], ['A']])
        # TODO?: self.assert_eq(kdf.loc[['t', 'b'], ['A']], pdf.loc[['t', 'b'], ['A']])
        # TODO?: self.assert_eq(kdf.loc[['r', 'r', 'c', 'g', 'h'], ['A']],
        # TODO?:                pdf.loc[['r', 'r', 'c', 'g', 'h'], ['A']])

    def test_loc2d_duplicated_columns(self):
        pdf = pd.DataFrame(np.random.randn(20, 5),
                           index=list('abcdefghijklmnopqrst'),
                           columns=list('AABCD'))
        pdf = ks.from_pandas(pdf)

        # TODO?: self.assert_eq(pdf.loc[['a'], 'A'], pdf.loc[['a'], 'A'])
        # TODO?: self.assert_eq(pdf.loc[['a'], ['A']], pdf.loc[['a'], ['A']])
        self.assert_eq(pdf.loc[['j'], 'B'], pdf.loc[['j'], 'B'])
        self.assert_eq(pdf.loc[['j'], ['B']], pdf.loc[['j'], ['B']])

        # TODO?: self.assert_eq(pdf.loc['a':'o', 'A'], pdf.loc['a':'o', 'A'])
        # TODO?: self.assert_eq(pdf.loc['a':'o', ['A']], pdf.loc['a':'o', ['A']])
        self.assert_eq(pdf.loc['j':'q', 'B'], pdf.loc['j':'q', 'B'])
        self.assert_eq(pdf.loc['j':'q', ['B']], pdf.loc['j':'q', ['B']])

        # TODO?: self.assert_eq(pdf.loc['a':'o', 'B':'D'], pdf.loc['a':'o', 'B':'D'])
        # TODO?: self.assert_eq(pdf.loc['a':'o', 'B':'D'], pdf.loc['a':'o', 'B':'D'])
        # TODO?: self.assert_eq(pdf.loc['j':'q', 'B':'A'], pdf.loc['j':'q', 'B':'A'])
        # TODO?: self.assert_eq(pdf.loc['j':'q', 'B':'A'], pdf.loc['j':'q', 'B':'A'])

        self.assert_eq(pdf.loc[pdf.B > 0, 'B'], pdf.loc[pdf.B > 0, 'B'])
        # TODO?: self.assert_eq(pdf.loc[pdf.B > 0, ['A', 'C']], pdf.loc[pdf.B > 0, ['A', 'C']])

    def test_getitem(self):
        pdf = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                            'B': [9, 8, 7, 6, 5, 4, 3, 2, 1],
                            'C': [True, False, True] * 3},
                           columns=list('ABC'))
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf['A'], pdf['A'])

        self.assert_eq(kdf[['A', 'B']], pdf[['A', 'B']])

        self.assert_eq(kdf[kdf.C], pdf[pdf.C])

        self.assertRaises(KeyError, lambda: kdf['X'])
        self.assertRaises(KeyError, lambda: kdf[['A', 'X']])
        self.assertRaises(AttributeError, lambda: kdf.X)

        # not str/unicode
        # TODO?: pdf = pd.DataFrame(np.random.randn(10, 5))
        # TODO?: kdf = ks.from_pandas(pdf)
        # TODO?: self.assert_eq(kdf[0], pdf[0])
        # TODO?: self.assert_eq(kdf[[1, 2]], pdf[[1, 2]])

        # TODO?: self.assertRaises(KeyError, lambda: pdf[8])
        # TODO?: self.assertRaises(KeyError, lambda: pdf[[1, 8]])

    def test_getitem_slice(self):
        pdf = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                            'B': [9, 8, 7, 6, 5, 4, 3, 2, 1],
                            'C': [True, False, True] * 3},
                           index=list('abcdefghi'))
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf['a':'e'], pdf['a':'e'])
        self.assert_eq(kdf['a':'b'], pdf['a':'b'])
        self.assert_eq(kdf['f':], pdf['f':])

    def test_loc_on_numpy_datetimes(self):
        pdf = pd.DataFrame({'x': [1, 2, 3]},
                           index=list(map(np.datetime64, ['2014', '2015', '2016'])))
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.loc['2014':'2015'], pdf.loc['2014':'2015'])

    def test_loc_on_pandas_datetimes(self):
        pdf = pd.DataFrame({'x': [1, 2, 3]},
                           index=list(map(pd.Timestamp, ['2014', '2015', '2016'])))
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.loc['2014':'2015'], pdf.loc['2014':'2015'])

    @unittest.skip('TODO?: the behavior of slice for datetime')
    def test_loc_datetime_no_freq(self):
        datetime_index = pd.date_range('2016-01-01', '2016-01-31', freq='12h')
        datetime_index.freq = None  # FORGET FREQUENCY
        pdf = pd.DataFrame({'num': range(len(datetime_index))}, index=datetime_index)
        kdf = ks.from_pandas(pdf)

        slice_ = slice('2016-01-03', '2016-01-05')
        result = kdf.loc[slice_, :]
        expected = pdf.loc[slice_, :]
        self.assert_eq(result, expected)

    @unittest.skip('TODO?: the behavior of slice for datetime')
    def test_loc_timestamp_str(self):
        pdf = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                           index=pd.date_range('2011-01-01', freq='H', periods=100))
        kdf = ks.from_pandas(pdf)

        # partial string slice
        # TODO?: self.assert_eq(pdf.loc['2011-01-02'],
        # TODO?:                kdf.loc['2011-01-02'])
        self.assert_eq(pdf.loc['2011-01-02':'2011-01-05'],
                       kdf.loc['2011-01-02':'2011-01-05'])

        # series
        # TODO?: self.assert_eq(pdf.A.loc['2011-01-02'],
        # TODO?:                kdf.A.loc['2011-01-02'])
        self.assert_eq(pdf.A.loc['2011-01-02':'2011-01-05'],
                       kdf.A.loc['2011-01-02':'2011-01-05'])

        pdf = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                           index=pd.date_range('2011-01-01', freq='M', periods=100))
        kdf = ks.from_pandas(pdf)
        # TODO?: self.assert_eq(pdf.loc['2011-01'], kdf.loc['2011-01'])
        # TODO?: self.assert_eq(pdf.loc['2011'], kdf.loc['2011'])

        self.assert_eq(pdf.loc['2011-01':'2012-05'], kdf.loc['2011-01':'2012-05'])
        self.assert_eq(pdf.loc['2011':'2015'], kdf.loc['2011':'2015'])

        # series
        # TODO?: self.assert_eq(pdf.B.loc['2011-01'], kdf.B.loc['2011-01'])
        # TODO?: self.assert_eq(pdf.B.loc['2011'], kdf.B.loc['2011'])

        self.assert_eq(pdf.B.loc['2011-01':'2012-05'], kdf.B.loc['2011-01':'2012-05'])
        self.assert_eq(pdf.B.loc['2011':'2015'], kdf.B.loc['2011':'2015'])

    @unittest.skip('TODO?: the behavior of slice for datetime')
    def test_getitem_timestamp_str(self):
        pdf = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                           index=pd.date_range('2011-01-01', freq='H', periods=100))
        kdf = ks.from_pandas(pdf)

        # partial string slice
        # TODO?: self.assert_eq(pdf['2011-01-02'],
        # TODO?:                kdf['2011-01-02'])
        self.assert_eq(pdf['2011-01-02':'2011-01-05'],
                       kdf['2011-01-02':'2011-01-05'])

        pdf = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                           index=pd.date_range('2011-01-01', freq='M', periods=100))
        kdf = ks.from_pandas(pdf)

        # TODO?: self.assert_eq(pdf['2011-01'], kdf['2011-01'])
        # TODO?: self.assert_eq(pdf['2011'], kdf['2011'])

        self.assert_eq(pdf['2011-01':'2012-05'], kdf['2011-01':'2012-05'])
        self.assert_eq(pdf['2011':'2015'], kdf['2011':'2015'])

    @unittest.skip('TODO?: period index can\'t convert to DataFrame correctly')
    def test_getitem_period_str(self):
        pdf = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                           index=pd.period_range('2011-01-01', freq='H', periods=100))
        kdf = ks.from_pandas(pdf)

        # partial string slice
        # TODO?: self.assert_eq(pdf['2011-01-02'],
        # TODO?:                kdf['2011-01-02'])
        self.assert_eq(pdf['2011-01-02':'2011-01-05'],
                       kdf['2011-01-02':'2011-01-05'])

        pdf = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100)},
                           index=pd.period_range('2011-01-01', freq='M', periods=100))
        kdf = ks.from_pandas(pdf)

        # TODO?: self.assert_eq(pdf['2011-01'], kdf['2011-01'])
        # TODO?: self.assert_eq(pdf['2011'], kdf['2011'])

        self.assert_eq(pdf['2011-01':'2012-05'], kdf['2011-01':'2012-05'])
        self.assert_eq(pdf['2011':'2015'], kdf['2011':'2015'])

    def test_iloc(self):
        pdf = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        kdf = ks.from_pandas(pdf)

        for indexer in [0,
                        [0],
                        [0, 1],
                        [1, 0],
                        [False, True, True],
                        slice(0, 1)]:
            self.assert_eq(kdf.iloc[:, indexer], pdf.iloc[:, indexer])
            self.assert_eq(kdf.iloc[:1, indexer], pdf.iloc[:1, indexer])
            self.assert_eq(kdf.iloc[:-1, indexer], pdf.iloc[:-1, indexer])
            self.assert_eq(kdf.iloc[kdf.index == 2, indexer], pdf.iloc[pdf.index == 2, indexer])

    def test_iloc_multiindex_columns(self):
        arrays = [np.array(['bar', 'bar', 'baz', 'baz']),
                  np.array(['one', 'two', 'one', 'two'])]

        pdf = pd.DataFrame(np.random.randn(3, 4), index=['A', 'B', 'C'], columns=arrays)
        kdf = ks.from_pandas(pdf)

        for indexer in [0,
                        [0],
                        [0, 1],
                        [1, 0],
                        [False, True, True, True],
                        slice(0, 1)]:
            self.assert_eq(kdf.iloc[:, indexer], pdf.iloc[:, indexer])
            self.assert_eq(kdf.iloc[:1, indexer], pdf.iloc[:1, indexer])
            self.assert_eq(kdf.iloc[:-1, indexer], pdf.iloc[:-1, indexer])
            self.assert_eq(kdf.iloc[kdf.index == 'B', indexer], pdf.iloc[pdf.index == 'B', indexer])

    def test_iloc_series(self):
        pseries = pd.Series([1, 2, 3])
        kseries = ks.from_pandas(pseries)

        self.assert_eq(kseries.iloc[:], pseries.iloc[:])
        self.assert_eq(kseries.iloc[:1], pseries.iloc[:1])
        self.assert_eq(kseries.iloc[:-1], pseries.iloc[:-1])

    def test_iloc_raises(self):
        pdf = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        kdf = ks.from_pandas(pdf)

        with self.assertRaisesRegex(SparkPandasNotImplementedError,
                                    'Cannot use start or step with Spark.'):
            kdf.iloc[0:]

        with self.assertRaisesRegex(SparkPandasNotImplementedError,
                                    'Cannot use start or step with Spark.'):
            kdf.iloc[:2:2]

        with self.assertRaisesRegex(SparkPandasNotImplementedError,
                                    '.iloc requires numeric slice or conditional boolean Index'):
            kdf.iloc[[0, 1], :]

        with self.assertRaisesRegex(SparkPandasNotImplementedError,
                                    '.iloc requires numeric slice or conditional boolean Index'):
            kdf.A.iloc[[0, 1]]

        with self.assertRaisesRegex(SparkPandasIndexingError,
                                    'Only accepts pairs of candidates'):
            kdf.iloc[[0, 1], [0, 1], [1, 2]]

        with self.assertRaisesRegex(SparkPandasIndexingError,
                                    'Too many indexers'):
            kdf.A.iloc[[0, 1], [0, 1]]

        with self.assertRaisesRegex(TypeError,
                                    'cannot do slice indexing with these indexers'):
            kdf.iloc[:'b', :]

        with self.assertRaisesRegex(TypeError,
                                    'cannot do slice indexing with these indexers'):
            kdf.iloc[:, :'b']

        with self.assertRaisesRegex(TypeError,
                                    'cannot perform reduce with flexible type'):
            kdf.iloc[:, ['A']]

        with self.assertRaisesRegex(ValueError,
                                    'Location based indexing can only have'):
            kdf.iloc[:, 'A']

        with self.assertRaisesRegex(IndexError,
                                    'index 5 is out of bounds'):
            kdf.iloc[:, [5, 6]]
