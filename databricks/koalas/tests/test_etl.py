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
import unittest

import pandas as pd

from databricks.koalas.testing.utils import ComparisonTestBase, compare_both


class EtlTest(ComparisonTestBase):

    @property
    def pdf(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        return pd.read_csv('%s/../../../data/sample_stocks.csv' % test_dir)

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
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
