import unittest

import pandas as pd
import pandorable_sparky
import pyspark

from pandorable_sparky.testing.utils import ComparisonTestBase, compare_both


class TestETL(ComparisonTestBase):

    @property
    def pdf(self):
        return pd.read_csv('data/sample_stocks.csv')

    @compare_both
    def test_etl(self, df):
        df1 = df.loc[:, 'Symbol Date Open High Low Close'.split()]
        yield df1

        df2 = df1.sort_values(by=["Symbol", "Date"])
        yield df2

        df3 = df2.groupby(by="Symbol").agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        })
        yield df3

        df4 = df2.copy()

        df4.loc[:, 'signal_1'] = df4.Close - df4.Open
        df4.loc[:, 'signal_2'] = df4.High - df4.Low

        # df4.loc[:, 'signal_3'] = (df4.signal_2 - df4.signal_2.mean()) / df4.signal_2.std()
        yield df4

        df5 = df4.loc[df4.signal_1 > 0, ['Symbol', 'Date']]
        yield df5

        df6 = df4.loc[df4.signal_2 > 0, ['Symbol', 'Date']]
        yield df6

        # df7 = df4.loc[df4.signal_3 > 0, ['Symbol', 'Date']]
        # yield df7


if __name__ == "__main__":
    from pandorable_sparky.tests.test_etl import *

    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
