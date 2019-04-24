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
import unittest

import numpy as np
import pandas as pd

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.missing.frame import _MissingPandasLikeDataFrame
from databricks.koalas.missing.series import _MissingPandasLikeSeries
from databricks.koalas.series import Series


class DataFrameTest(ReusedSQLTestCase, TestUtils):

    @property
    def full(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def df(self):
        return koalas.from_pandas(self.full)

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
        ddf = koalas.from_pandas(df)
        self.assert_eq(df[['a', 'b']], ddf[['a', 'b']])

        self.assertEqual(ddf.a.notnull().alias("x").name, "x")

    def test_corr(self):
        # DataFrame
        df = pd.util.testing.makeMissingDataframe(0.3, 42).fillna(0)  # We do not handle NaNs for now
        print("df", df)
        ddf = koalas.from_pandas(df)

        res = ddf.corr()
        sol = df.corr()
        self.assertPandasAlmostEqual(res, sol)

        # Series
        a = df.A
        b = df.B
        da = ddf.A
        db = ddf.B

        res = da.corr(db)
        sol = a.corr(b)
        self.assertAlmostEqual(res, sol)
        self.assertRaises(TypeError, lambda: da.corr(ddf))

    def test_cov_corr_meta(self):
        df = pd.DataFrame({'a': np.array([1, 2, 3]),
                           'b': np.array([1.0, 2.0, 3.0], dtype='f4'),
                           'c': np.array([1.0, 2.0, 3.0])},
                          index=pd.Index([1, 2, 3], name='myindex'))
        ddf = koalas.from_pandas(df)
        self.assert_eq(ddf.corr(), df.corr())
        # self.assert_eq(ddf.cov(), df.cov())
        # assert ddf.a.cov(ddf.b)._meta.dtype == 'f8'
        # assert ddf.a.corr(ddf.b)._meta.dtype == 'f8'


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports')
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
