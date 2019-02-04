import unittest

import pandas as pd
import pandorable_sparky
import pyspark
from pyspark.sql import Column, DataFrame

from pandorable_sparky.testing.utils import ComparisonTestBase, compare_both


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

    def test_limitations(self):
        df = self.df.set_index('month')

        self.assertRaisesRegex(ValueError, 'Level should be all int or all string.',
                               lambda: df.reset_index([1, 'month']))
        self.assertRaisesRegex(NotImplementedError, 'Can\'t reset index because there is no index.',
                               lambda: df.reset_index().reset_index())


if __name__ == "__main__":
    from pandorable_sparky.tests.test_indexing import *

    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
