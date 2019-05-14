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

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class StatsTest(ReusedSQLTestCase, SQLTestUtils):

    def test_stat_functions(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4],
                           'B': [1.0, 2.1, 3, 4]})
        ddf = koalas.from_pandas(df)

        functions = ['max', 'min', 'mean', 'sum']
        for funcname in functions:
            self.assertEqual(getattr(ddf.A, funcname)(), getattr(df.A, funcname)())
            self.assert_eq(getattr(ddf, funcname)(), getattr(df, funcname)())

        functions = ['std', 'var']
        for funcname in functions:
            self.assertAlmostEqual(getattr(ddf.A, funcname)(), getattr(df.A, funcname)())
            self.assertPandasAlmostEqual(getattr(ddf, funcname)(), getattr(df, funcname)())

        # NOTE: To test skew and kurt, just make sure they run.
        #       The numbers are different in spark and pandas.
        functions = ['skew', 'kurt']
        for funcname in functions:
            getattr(ddf.A, funcname)()
            getattr(ddf, funcname)()

    def test_abs(self):
        df = pd.DataFrame({'A': [1, -2, 3, -4, 5],
                           'B': [1., -2, 3, -4, 5],
                           'C': [-6., -7, -8, -9, 10],
                           'D': ['a', 'b', 'c', 'd', 'e']})
        ddf = koalas.from_pandas(df)
        self.assert_eq(ddf.A.abs(), df.A.abs())
        self.assert_eq(ddf.B.abs(), df.B.abs())
        self.assert_eq(ddf[['B', 'C']].abs(), df[['B', 'C']].abs())
        # self.assert_eq(ddf.select('A', 'B').abs(), df[['A', 'B']].abs())

    def test_corr(self):
        # Disable arrow execution since corr() is using UDT internally which is not supported.
        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            # DataFrame
            # we do not handle NaNs for now
            df = pd.util.testing.makeMissingDataframe(0.3, 42).fillna(0)
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
        # Disable arrow execution since corr() is using UDT internally which is not supported.
        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            df = pd.DataFrame({'a': np.array([1, 2, 3], dtype='i1'),
                               'b': np.array([1, 2, 3], dtype='i2'),
                               'c': np.array([1, 2, 3], dtype='i4'),
                               'd': np.array([1, 2, 3]),
                               'e': np.array([1.0, 2.0, 3.0], dtype='f4'),
                               'f': np.array([1.0, 2.0, 3.0]),
                               'g': np.array([True, False, True]),
                               'h': np.array(list('abc'))},
                              index=pd.Index([1, 2, 3], name='myindex'))
            ddf = koalas.from_pandas(df)
            self.assert_eq(ddf.corr(), df.corr())

    def test_stats_on_boolean_dataframe(self):
        df = pd.DataFrame({'A': [True, False, True],
                           'B': [False, False, True]})
        ddf = koalas.from_pandas(df)

        pd.testing.assert_series_equal(ddf.min(), df.min())
        pd.testing.assert_series_equal(ddf.max(), df.max())

        pd.testing.assert_series_equal(ddf.sum(), df.sum())
        pd.testing.assert_series_equal(ddf.mean(), df.mean())

        pd.testing.assert_series_equal(ddf.var(), df.var())
        pd.testing.assert_series_equal(ddf.std(), df.std())

    def test_stats_on_boolean_series(self):
        s = pd.Series([True, False, True])
        ds = koalas.from_pandas(s)

        self.assertEqual(ds.min(), s.min())
        self.assertEqual(ds.max(), s.max())

        self.assertEqual(ds.sum(), s.sum())
        self.assertEqual(ds.mean(), s.mean())

        self.assertAlmostEqual(ds.var(), s.var())
        self.assertAlmostEqual(ds.std(), s.std())
