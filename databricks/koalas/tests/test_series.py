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
from databricks.koalas import Series
from databricks.koalas.generic import max_display_count
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.missing.series import _MissingPandasLikeSeries


class SeriesTest(ReusedSQLTestCase, SQLTestUtils):

    @property
    def ps(self):
        return pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    @property
    def ks(self):
        return koalas.from_pandas(self.ps)

    def test_series(self):
        ks = self.ks

        self.assertTrue(isinstance(ks['x'], Series))

        # TODO: self.assert_eq(d + 1, pdf + 1)

    def test_repr_cache_invalidation(self):
        # If there is any cache, inplace operations should invalidate it.
        s = koalas.range(10)['id']
        s.__repr__()
        s.rename('a', inplace=True)
        self.assertEqual(s.__repr__(), s.rename("a").__repr__())

    def test_empty_series(self):
        a = pd.Series([], dtype='i1')
        b = pd.Series([], dtype='str')

        self.assert_eq(koalas.from_pandas(a), a)
        self.assertRaises(ValueError, lambda: koalas.from_pandas(b))

        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            self.assert_eq(koalas.from_pandas(a), a)
            self.assertRaises(ValueError, lambda: koalas.from_pandas(b))

    def test_all_null_series(self):
        a = pd.Series([None, None, None], dtype='float64')
        b = pd.Series([None, None, None], dtype='str')

        self.assert_eq(koalas.from_pandas(a).dtype, a.dtype)
        self.assertTrue(koalas.from_pandas(a).toPandas().isnull().all())
        self.assertRaises(ValueError, lambda: koalas.from_pandas(b))

        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            self.assert_eq(koalas.from_pandas(a).dtype, a.dtype)
            self.assertTrue(koalas.from_pandas(a).toPandas().isnull().all())
            self.assertRaises(ValueError, lambda: koalas.from_pandas(b))

    def test_head_tail(self):
        ks = self.ks
        ps = self.ps

        self.assert_eq(ks.head(3), ps.head(3))

        # TODO: self.assert_eq(ks.tail(3), ps.tail(3))

    def test_rename(self):
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

    def test_rename_method(self):
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
        # ps = pd.Series(['a', 'b', 'c', 'd', 'e', 'f', 'g'], name='x')
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

    def test_to_numpy(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

        ddf = koalas.from_pandas(s)
        np.testing.assert_equal(ddf.to_numpy(), s.values)

    def test_isin(self):
        s = pd.Series(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'], name='animal')

        ds = koalas.from_pandas(s)

        self.assert_eq(ds.isin(['cow', 'lama']), s.isin(['cow', 'lama']))
        self.assert_eq(ds.isin({'cow'}), s.isin({'cow'}))

        msg = "only list-like objects are allowed to be passed to isin()"
        with self.assertRaisesRegex(TypeError, msg):
            ds.isin(1)

    def test_fillna(self):
        ps = pd.Series([np.nan, 2, 3, 4, np.nan, 6], name='x')
        ks = koalas.from_pandas(ps)

        self.assert_eq(ks.fillna(0), ps.fillna(0))

        ks.fillna(0, inplace=True)
        ps.fillna(0, inplace=True)
        self.assert_eq(ks, ps)

    def test_dropna(self):
        ps = pd.Series([np.nan, 2, 3, 4, np.nan, 6], name='x')

        ks = koalas.from_pandas(ps)

        self.assert_eq(ks.dropna(), ps.dropna())

        ks.dropna(inplace=True)
        self.assert_eq(ks, ps.dropna())

    def test_value_counts(self):
        ps = pd.Series([1, 2, 1, 3, 3, np.nan, 1, 4], name="x")
        ks = koalas.from_pandas(ps)

        exp = ps.value_counts()
        res = ks.value_counts()
        self.assertEqual(res.name, exp.name)
        self.assertPandasAlmostEqual(res.toPandas(), exp)

        self.assertPandasAlmostEqual(ks.value_counts(normalize=True).toPandas(),
                                     ps.value_counts(normalize=True))
        self.assertPandasAlmostEqual(ks.value_counts(ascending=True).toPandas(),
                                     ps.value_counts(ascending=True))
        self.assertPandasAlmostEqual(ks.value_counts(normalize=True, dropna=False).toPandas(),
                                     ps.value_counts(normalize=True, dropna=False))
        self.assertPandasAlmostEqual(ks.value_counts(ascending=True, dropna=False).toPandas(),
                                     ps.value_counts(ascending=True, dropna=False))

        with self.assertRaisesRegex(NotImplementedError,
                                    "value_counts currently does not support bins"):
            ks.value_counts(bins=3)

        ps.name = 'index'
        ks.name = 'index'
        self.assertPandasAlmostEqual(ks.value_counts().toPandas(), ps.value_counts())

    def test_isnull(self):
        ps = pd.Series([1, 2, 3, 4, np.nan, 6], name='x')
        ks = koalas.from_pandas(ps)

        self.assert_eq(ks.notnull(), ps.notnull())
        self.assert_eq(ks.isnull(), ps.isnull())

        ps = self.ps
        ks = self.ks

        self.assert_eq(ks.notnull(), ps.notnull())
        self.assert_eq(ks.isnull(), ps.isnull())

    def test_to_datetime(self):
        ps = pd.Series(['3/11/2000', '3/12/2000', '3/13/2000'] * 100)
        ks = koalas.from_pandas(ps)

        self.assert_eq(pd.to_datetime(ps, infer_datetime_format=True),
                       koalas.to_datetime(ks, infer_datetime_format=True))

    def test_missing(self):
        ks = self.ks

        missing_functions = inspect.getmembers(_MissingPandasLikeSeries, inspect.isfunction)
        for name, _ in missing_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Series.*{}.*not implemented".format(name)):
                getattr(ks, name)()

        missing_properties = inspect.getmembers(_MissingPandasLikeSeries,
                                                lambda o: isinstance(o, property))
        for name, _ in missing_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Series.*{}.*not implemented".format(name)):
                getattr(ks, name)

    def test_clip(self):
        ps = pd.Series([0, 2, 4])
        ks = koalas.from_pandas(ps)

        # Assert list-like values are not accepted for 'lower' and 'upper'
        msg = "List-like value are not supported for 'lower' and 'upper' at the moment"
        with self.assertRaises(ValueError, msg=msg):
            ks.clip(lower=[1])
        with self.assertRaises(ValueError, msg=msg):
            ks.clip(upper=[1])

        # Assert no lower or upper
        self.assert_eq(ks.clip(), ps.clip())
        # Assert lower only
        self.assert_eq(ks.clip(1), ps.clip(1))
        # Assert upper only
        self.assert_eq(ks.clip(upper=3), ps.clip(upper=3))
        # Assert lower and upper
        self.assert_eq(ks.clip(1, 3), ps.clip(1, 3))

        # Assert behavior on string values
        str_ks = koalas.Series(['a', 'b', 'c'])
        self.assert_eq(str_ks.clip(1, 3), str_ks)
