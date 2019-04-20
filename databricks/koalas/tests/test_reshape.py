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

import datetime
from decimal import Decimal
from distutils.version import LooseVersion
import unittest

import numpy as np
import pandas as pd

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase


class ReshapeTest(ReusedSQLTestCase):

    def test_get_dummies(self):
        for data in [pd.Series([1, 1, 1, 2, 2, 1, 3, 4]),
                     # pd.Series([1, 1, 1, 2, 2, 1, 3, 4], dtype='category'),
                     # pd.Series(pd.Categorical([1, 1, 1, 2, 2, 1, 3, 4], categories=[4, 3, 2, 1])),
                     pd.DataFrame({'a': [1, 2, 3, 4, 4, 3, 2, 1],
                                   # 'b': pd.Categorical(list('abcdabcd')),
                                   'b': list('abcdabcd')})]:
            exp = pd.get_dummies(data)

            ddata = koalas.from_pandas(data)
            res = koalas.get_dummies(ddata)
            self.assertPandasAlmostEqual(res.toPandas(), exp)

    def test_get_dummies_object(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4, 4, 3, 2, 1],
                           # 'a': pd.Categorical([1, 2, 3, 4, 4, 3, 2, 1]),
                           'b': list('abcdabcd'),
                           # 'c': pd.Categorical(list('abcdabcd')),
                           'c': list('abcdabcd')})
        ddf = koalas.from_pandas(df)

        # Explicitly exclude object columns
        exp = pd.get_dummies(df, columns=['a', 'c'])
        res = koalas.get_dummies(ddf, columns=['a', 'c'])
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        exp = pd.get_dummies(df)
        res = koalas.get_dummies(ddf)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        exp = pd.get_dummies(df.b)
        res = koalas.get_dummies(ddf.b)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        exp = pd.get_dummies(df, columns=['b'])
        res = koalas.get_dummies(ddf, columns=['b'])
        self.assertPandasAlmostEqual(res.toPandas(), exp)

    def test_get_dummies_date_datetime(self):
        df = pd.DataFrame({'d': [datetime.date(2019, 1, 1),
                                 datetime.date(2019, 1, 2),
                                 datetime.date(2019, 1, 1)],
                           'dt': [datetime.datetime(2019, 1, 1, 0, 0, 0),
                                  datetime.datetime(2019, 1, 1, 0, 0, 1),
                                  datetime.datetime(2019, 1, 1, 0, 0, 0)]})
        ddf = koalas.from_pandas(df)

        exp = pd.get_dummies(df)
        res = koalas.get_dummies(ddf)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        exp = pd.get_dummies(df.d)
        res = koalas.get_dummies(ddf.d)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        exp = pd.get_dummies(df.dt)
        res = koalas.get_dummies(ddf.dt)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

    def test_get_dummies_boolean(self):
        df = pd.DataFrame({'b': [True, False, True]})
        ddf = koalas.from_pandas(df)

        exp = pd.get_dummies(df)
        res = koalas.get_dummies(ddf)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        exp = pd.get_dummies(df.b)
        res = koalas.get_dummies(ddf.b)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

    def test_get_dummies_decimal(self):
        df = pd.DataFrame({'d': [Decimal(1.0), Decimal(2.0), Decimal(1)]})
        ddf = koalas.from_pandas(df)

        exp = pd.get_dummies(df)
        res = koalas.get_dummies(ddf)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        exp = pd.get_dummies(df.d)
        res = koalas.get_dummies(ddf.d)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

    def test_get_dummies_kwargs(self):
        # s = pd.Series([1, 1, 1, 2, 2, 1, 3, 4], dtype='category')
        s = pd.Series([1, 1, 1, 2, 2, 1, 3, 4])
        exp = pd.get_dummies(s, prefix='X', prefix_sep='-')

        ds = koalas.from_pandas(s)
        res = koalas.get_dummies(ds, prefix='X', prefix_sep='-')
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        exp = pd.get_dummies(s, drop_first=True)

        ds = koalas.from_pandas(s)
        res = koalas.get_dummies(ds, drop_first=True)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        # nan
        # s = pd.Series([1, 1, 1, 2, np.nan, 3, np.nan, 5], dtype='category')
        s = pd.Series([1, 1, 1, 2, np.nan, 3, np.nan, 5])
        exp = pd.get_dummies(s)

        ds = koalas.from_pandas(s)
        res = koalas.get_dummies(ds)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        # dummy_na
        exp = pd.get_dummies(s, dummy_na=True)

        ds = koalas.from_pandas(s)
        res = koalas.get_dummies(ds, dummy_na=True)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

    def test_get_dummies_prefix(self):
        df = pd.DataFrame({
            "A": ['a', 'b', 'a'],
            "B": ['b', 'a', 'c'],
            "D": [0, 0, 1],
        })
        ddf = koalas.from_pandas(df)

        exp = pd.get_dummies(df, prefix=['foo', 'bar'])
        res = koalas.get_dummies(ddf, prefix=['foo', 'bar'])
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        exp = pd.get_dummies(df, prefix=['foo'], columns=['B'])
        res = koalas.get_dummies(ddf, prefix=['foo'], columns=['B'])
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        with self.assertRaisesRegex(ValueError, "string types"):
            koalas.get_dummies(ddf, prefix='foo')
        with self.assertRaisesRegex(ValueError, "Length of 'prefix' \\(1\\) .* \\(2\\)"):
            koalas.get_dummies(ddf, prefix=['foo'])
        with self.assertRaisesRegex(ValueError, "Length of 'prefix' \\(2\\) .* \\(1\\)"):
            koalas.get_dummies(ddf, prefix=['foo', 'bar'], columns=['B'])

        s = pd.Series([1, 1, 1, 2, 2, 1, 3, 4], name='A')
        ds = koalas.from_pandas(s)

        exp = pd.get_dummies(s, prefix='foo')
        res = koalas.get_dummies(ds, prefix='foo')
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        # columns are ignored.
        exp = pd.get_dummies(s, prefix=['foo'], columns=['B'])
        res = koalas.get_dummies(ds, prefix=['foo'], columns=['B'])
        self.assertPandasAlmostEqual(res.toPandas(), exp)

    def test_get_dummies_dtype(self):
        df = pd.DataFrame({
            # "A": pd.Categorical(['a', 'b', 'a'], categories=['a', 'b', 'c']),
            "A": ['a', 'b', 'a'],
            "B": [0, 0, 1],
        })
        ddf = koalas.from_pandas(df)

        if LooseVersion("0.23.0") <= LooseVersion(pd.__version__):
            exp = pd.get_dummies(df, dtype='float64')
        else:
            exp = pd.get_dummies(df)
            exp = exp.astype({'A_a': 'float64', 'A_b': 'float64'})
        res = koalas.get_dummies(ddf, dtype='float64')
        self.assertPandasAlmostEqual(exp, res.toPandas())


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
