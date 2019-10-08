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
from collections import defaultdict

import base64
from io import BytesIO
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from databricks import koalas
from distutils.version import LooseVersion
from databricks.koalas import Series
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

        self.assertTrue(isinstance(ks, Series))

        self.assert_eq(ks + 1, self.ps + 1)

    def test_series_tuple_name(self):
        ps = self.ps
        ps.name = ('x', 'a')

        ks = koalas.from_pandas(ps)

        self.assert_eq(ks, ps)
        self.assert_eq(ks.name, ps.name)

        ps.name = ('y', 'z')
        ks.name = ('y', 'z')

        self.assert_eq(ks, ps)
        self.assert_eq(ks.name, ps.name)

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

        ps.name = None
        ks.name = None
        self.assertEqual(ks.name, None)
        self.assert_eq(ks, ps)

        pidx = ps.index
        kidx = ks.index
        pidx.name = 'renamed'
        kidx.name = 'renamed'
        self.assertEqual(kidx.name, 'renamed')
        self.assert_eq(kidx, pidx)

    def test_rename_method(self):
        # Series name
        ps = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')
        ks = koalas.from_pandas(ps)

        self.assert_eq(ks.rename('y'), ps.rename('y'))
        self.assertEqual(ks.name, 'x')  # no mutation
        self.assert_eq(ks.rename(), ps.rename())

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

    def test_values_property(self):
        ks = self.ks
        msg = ("Koalas does not support the 'values' property. If you want to collect your data " +
               "as an NumPy array, use 'to_numpy()' instead.")
        with self.assertRaises(NotImplementedError, msg=msg):
            ks.values

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

    def test_nunique(self):
        ps = pd.Series([1, 2, 1, np.nan])
        ks = koalas.from_pandas(ps)

        # Assert NaNs are dropped by default
        nunique_result = ks.nunique()
        self.assertEqual(nunique_result, 2)
        self.assert_eq(nunique_result, ps.nunique())

        # Assert including NaN values
        nunique_result = ks.nunique(dropna=False)
        self.assertEqual(nunique_result, 3)
        self.assert_eq(nunique_result, ps.nunique(dropna=False))

        # Assert approximate counts
        self.assertEqual(koalas.Series(range(100)).nunique(approx=True), 103)
        self.assertEqual(koalas.Series(range(100)).nunique(approx=True, rsd=0.01), 100)

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

    def test_nsmallest(self):
        sample_lst = [1, 2, 3, 4, np.nan, 6]
        ps = pd.Series(sample_lst, name='x')
        ks = koalas.Series(sample_lst, name='x')
        self.assert_eq(ks.nsmallest(n=3), ps.nsmallest(n=3))
        self.assert_eq(ks.nsmallest(), ps.nsmallest())
        self.assert_eq((ks + 1).nsmallest(), (ps + 1).nsmallest())

    def test_nlargest(self):
        sample_lst = [1, 2, 3, 4, np.nan, 6]
        ps = pd.Series(sample_lst, name='x')
        ks = koalas.Series(sample_lst, name='x')
        self.assert_eq(ks.nlargest(n=3), ps.nlargest(n=3))
        self.assert_eq(ks.nlargest(), ps.nlargest())
        self.assert_eq((ks + 1).nlargest(), (ps + 1).nlargest())

    def test_isnull(self):
        ps = pd.Series([1, 2, 3, 4, np.nan, 6], name='x')
        ks = koalas.from_pandas(ps)

        self.assert_eq(ks.notnull(), ps.notnull())
        self.assert_eq(ks.isnull(), ps.isnull())

        ps = self.ps
        ks = self.ks

        self.assert_eq(ks.notnull(), ps.notnull())
        self.assert_eq(ks.isnull(), ps.isnull())

    def test_all(self):
        for ps in [pd.Series([True, True], name='x'),
                   pd.Series([True, False], name='x'),
                   pd.Series([0, 1], name='x'),
                   pd.Series([1, 2, 3], name='x'),
                   pd.Series([True, True, None], name='x'),
                   pd.Series([True, False, None], name='x'),
                   pd.Series([], name='x'),
                   pd.Series([np.nan], name='x')]:
            ks = koalas.from_pandas(ps)
            self.assert_eq(ks.all(), ps.all())

        ps = pd.Series([1, 2, 3, 4], name='x')
        ks = koalas.from_pandas(ps)

        self.assert_eq((ks % 2 == 0).all(), (ps % 2 == 0).all())

    def test_any(self):
        for ps in [pd.Series([False, False], name='x'),
                   pd.Series([True, False], name='x'),
                   pd.Series([0, 1], name='x'),
                   pd.Series([1, 2, 3], name='x'),
                   pd.Series([True, True, None], name='x'),
                   pd.Series([True, False, None], name='x'),
                   pd.Series([], name='x'),
                   pd.Series([np.nan], name='x')]:
            ks = koalas.from_pandas(ps)
            self.assert_eq(ks.any(), ps.any())

        ps = pd.Series([1, 2, 3, 4], name='x')
        ks = koalas.from_pandas(ps)

        self.assert_eq((ks % 2 == 0).any(), (ps % 2 == 0).any())

    def test_sort_values(self):
        ps = pd.Series([1, 2, 3, 4, 5, None, 7], name='0')
        ks = koalas.from_pandas(ps)
        self.assert_eq(repr(ks.sort_values()), repr(ps.sort_values()))
        self.assert_eq(repr(ks.sort_values(ascending=False)),
                       repr(ps.sort_values(ascending=False)))
        self.assert_eq(repr(ks.sort_values(na_position='first')),
                       repr(ps.sort_values(na_position='first')))
        self.assertRaises(ValueError, lambda: ks.sort_values(na_position='invalid'))
        self.assert_eq(ks.sort_values(inplace=True), ps.sort_values(inplace=True))
        self.assert_eq(repr(ks), repr(ps))

    def test_sort_index(self):
        ps = pd.Series([2, 1, np.nan], index=['b', 'a', np.nan], name='0')
        ks = koalas.from_pandas(ps)

        # Assert invalid parameters
        self.assertRaises(ValueError, lambda: ks.sort_index(axis=1))
        self.assertRaises(ValueError, lambda: ks.sort_index(kind='mergesort'))
        self.assertRaises(ValueError, lambda: ks.sort_index(na_position='invalid'))

        # Assert default behavior without parameters
        self.assert_eq(ks.sort_index(), ps.sort_index(), almost=True)
        # Assert sorting descending
        self.assert_eq(ks.sort_index(ascending=False), ps.sort_index(ascending=False), almost=True)
        # Assert sorting NA indices first
        self.assert_eq(ks.sort_index(na_position='first'), ps.sort_index(na_position='first'),
                       almost=True)
        # Assert sorting inplace
        self.assertEqual(ks.sort_index(inplace=True), ps.sort_index(inplace=True))
        self.assert_eq(ks, ps, almost=True)

        # Assert multi-indices
        ps = pd.Series(range(4), index=[['b', 'b', 'a', 'a'], [1, 0, 1, 0]], name='0')
        ks = koalas.from_pandas(ps)
        self.assert_eq(ks.sort_index(), ps.sort_index(), almost=True)
        self.assert_eq(ks.sort_index(level=[1, 0]), ps.sort_index(level=[1, 0]), almost=True)

        self.assert_eq(ks.reset_index().sort_index(), ps.reset_index().sort_index())

    def test_to_datetime(self):
        ps = pd.Series(['3/11/2000', '3/12/2000', '3/13/2000'] * 100)
        ks = koalas.from_pandas(ps)

        self.assert_eq(pd.to_datetime(ps, infer_datetime_format=True),
                       koalas.to_datetime(ks, infer_datetime_format=True))

    def test_missing(self):
        ks = self.ks

        missing_functions = inspect.getmembers(_MissingPandasLikeSeries, inspect.isfunction)
        unsupported_functions = [name for (name, type_) in missing_functions
                                 if type_.__name__ == 'unsupported_function']
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*Series.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(ks, name)()

        deprecated_functions = [name for (name, type_) in missing_functions
                                if type_.__name__ == 'deprecated_function']
        for name in deprecated_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Series.*{}.*is deprecated".format(name)):
                getattr(ks, name)()

        missing_properties = inspect.getmembers(_MissingPandasLikeSeries,
                                                lambda o: isinstance(o, property))
        unsupported_properties = [name for (name, type_) in missing_properties
                                  if type_.fget.__name__ == 'unsupported_property']
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*Series.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(ks, name)
        deprecated_properties = [name for (name, type_) in missing_properties
                                 if type_.fget.__name__ == 'deprecated_property']
        for name in deprecated_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Series.*{}.*is deprecated".format(name)):
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

    def test_is_unique(self):
        # We can't use pandas' is_unique for comparison. pandas 0.23 ignores None
        pser = pd.Series([1, 2, 2, None, None])
        kser = koalas.from_pandas(pser)
        self.assertEqual(False, kser.is_unique)

        pser = pd.Series([1, None, None])
        kser = koalas.from_pandas(pser)
        self.assertEqual(False, kser.is_unique)

        pser = pd.Series([1])
        kser = koalas.from_pandas(pser)
        self.assertEqual(pser.is_unique, kser.is_unique)

        pser = pd.Series([1, 1, 1])
        kser = koalas.from_pandas(pser)
        self.assertEqual(pser.is_unique, kser.is_unique)

    def test_to_list(self):
        if LooseVersion(pd.__version__) >= LooseVersion("0.24.0"):
            self.assertEqual(self.ks.to_list(), self.ps.to_list())

    def test_append(self):
        ps1 = pd.Series([1, 2, 3], name='0')
        ps2 = pd.Series([4, 5, 6], name='0')
        ps3 = pd.Series([4, 5, 6], index=[3, 4, 5], name='0')
        ks1 = koalas.from_pandas(ps1)
        ks2 = koalas.from_pandas(ps2)
        ks3 = koalas.from_pandas(ps3)

        self.assert_eq(ks1.append(ks2), ps1.append(ps2))
        self.assert_eq(ks1.append(ks3), ps1.append(ps3))
        self.assert_eq(ks1.append(ks2, ignore_index=True), ps1.append(ps2, ignore_index=True))

        ks1.append(ks3, verify_integrity=True)
        msg = "Indices have overlapping values"
        with self.assertRaises(ValueError, msg=msg):
            ks1.append(ks2, verify_integrity=True)

    def test_map(self):
        pser = pd.Series(['cat', 'dog', None, 'rabbit'])
        kser = koalas.from_pandas(pser)
        # Currently Koalas doesn't return NaN as Pandas does.
        self.assertEqual(
            repr(kser.map({})),
            repr(pser.map({}).replace({pd.np.nan: None}).rename(0)))

        d = defaultdict(lambda: "abc")
        self.assertTrue("abc" in repr(kser.map(d)))
        self.assertEqual(
            repr(kser.map(d)),
            repr(pser.map(d).rename(0)))

    def test_add_prefix(self):
        ps = pd.Series([1, 2, 3, 4], name='0')
        ks = koalas.from_pandas(ps)
        self.assert_eq(ps.add_prefix('item_'), ks.add_prefix('item_'))

        ps = pd.Series([1, 2, 3], name='0',
                       index=pd.MultiIndex.from_tuples([('A', 'X'), ('A', 'Y'), ('B', 'X')]))
        ks = koalas.from_pandas(ps)
        self.assert_eq(ps.add_prefix('item_'), ks.add_prefix('item_'))

    def test_add_suffix(self):
        ps = pd.Series([1, 2, 3, 4], name='0')
        ks = koalas.from_pandas(ps)
        self.assert_eq(ps.add_suffix('_item'), ks.add_suffix('_item'))

        ps = pd.Series([1, 2, 3], name='0',
                       index=pd.MultiIndex.from_tuples([('A', 'X'), ('A', 'Y'), ('B', 'X')]))
        ks = koalas.from_pandas(ps)
        self.assert_eq(ps.add_suffix('_item'), ks.add_suffix('_item'))

    def test_pandas_wraps(self):
        # This test checks the return column name of `isna()`. Previously it returned the column
        # name as its internal expression which contains, for instance, '`f(x)`' in the middle of
        # column name which currently cannot be recognized in PySpark.
        @koalas.pandas_wraps
        def f(x) -> koalas.Series[int]:
            return 2 * x

        df = koalas.DataFrame({"x": [1, None]})
        self.assert_eq(
            f(df["x"]).isna(),
            pd.Series([False, True]).rename("f(x)"))

    def test_hist(self):
        pdf = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10])

        kdf = koalas.from_pandas(pdf)

        def plot_to_base64(ax):
            bytes_data = BytesIO()
            ax.figure.savefig(bytes_data, format='png')
            bytes_data.seek(0)
            b64_data = base64.b64encode(bytes_data.read())
            plt.close(ax.figure)
            return b64_data

        _, ax1 = plt.subplots(1, 1)
        # Using plot.hist() because pandas changes ticks props when called hist()
        ax1 = pdf['a'].plot.hist()
        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf['a'].hist()
        self.assert_eq(plot_to_base64(ax1), plot_to_base64(ax2))

    def test_cummin(self):
        pser = pd.Series([1.0, None, 0.0, 4.0, 9.0]).rename("a")
        kser = koalas.from_pandas(pser)
        self.assertEqual(repr(pser.cummin()), repr(kser.cummin()))
        self.assertEqual(repr(pser.cummin(skipna=False)), repr(kser.cummin(skipna=False)))

    def test_cummax(self):
        pser = pd.Series([1.0, None, 0.0, 4.0, 9.0]).rename("a")
        kser = koalas.from_pandas(pser)
        self.assertEqual(repr(pser.cummax()), repr(kser.cummax()))
        self.assertEqual(repr(pser.cummax(skipna=False)), repr(kser.cummax(skipna=False)))

    def test_cumsum(self):
        pser = pd.Series([1.0, None, 0.0, 4.0, 9.0]).rename("a")
        kser = koalas.from_pandas(pser)
        self.assertEqual(repr(pser.cumsum()), repr(kser.cumsum()))
        self.assertEqual(repr(pser.cumsum(skipna=False)), repr(kser.cumsum(skipna=False)))

    def test_cumprod(self):
        pser = pd.Series([1.0, None, 1.0, 4.0, 9.0]).rename("a")
        kser = koalas.from_pandas(pser)
        self.assertEqual(repr(pser.cumprod()), repr(kser.cumprod()))
        self.assertEqual(repr(pser.cumprod(skipna=False)), repr(kser.cumprod(skipna=False)))

        # TODO: due to unknown reason, this test passes in Travis CI. Unable to reproduce in local.
        # with self.assertRaisesRegex(Exception, "values should be bigger than 0"):
        #     repr(koalas.Series([0, 1]).cumprod())

    def test_median(self):
        with self.assertRaisesRegex(ValueError, "accuracy must be an integer; however"):
            koalas.Series([24., 21., 25., 33., 26.]).median(accuracy="a")

    def test_rank(self):
        pser = pd.Series([1, 2, 3, 1], name='x')
        kser = koalas.from_pandas(pser)
        self.assertEqual(repr(pser.rank()),
                         repr(kser.rank().sort_index()))
        self.assertEqual(repr(pser.rank()),
                         repr(kser.rank().sort_index()))
        self.assertEqual(repr(pser.rank(ascending=False)),
                         repr(kser.rank(ascending=False).sort_index()))
        self.assertEqual(repr(pser.rank(method='min')),
                         repr(kser.rank(method='min').sort_index()))
        self.assertEqual(repr(pser.rank(method='max')),
                         repr(kser.rank(method='max').sort_index()))
        self.assertEqual(repr(pser.rank(method='first')),
                         repr(kser.rank(method='first').sort_index()))
        self.assertEqual(repr(pser.rank(method='dense')),
                         repr(kser.rank(method='dense').sort_index()))

        msg = "method must be one of 'average', 'min', 'max', 'first', 'dense'"
        with self.assertRaisesRegex(ValueError, msg):
            kser.rank(method='nothing')

    def test_round(self):
        pser = pd.Series([0.028208, 0.038683, 0.877076], name='x')
        kser = koalas.from_pandas(pser)
        self.assertEqual(repr(pser.round(2)), repr(kser.round(2)))
        msg = "decimals must be an integer"
        with self.assertRaisesRegex(ValueError, msg):
            kser.round(1.5)

    def test_quantile(self):
        with self.assertRaisesRegex(ValueError, "accuracy must be an integer; however"):
            koalas.Series([24., 21., 25., 33., 26.]).quantile(accuracy="a")
        with self.assertRaisesRegex(ValueError, "q must be a float of an array of floats;"):
            koalas.Series([24., 21., 25., 33., 26.]).quantile(q="a")
        with self.assertRaisesRegex(ValueError, "q must be a float of an array of floats;"):
            koalas.Series([24., 21., 25., 33., 26.]).quantile(q=["a"])

    def test_idxmax(self):
        pser = pd.Series(data=[1, 4, 5], index=['A', 'B', 'C'])
        kser = koalas.Series(pser)

        self.assertEqual(kser.idxmax(), pser.idxmax())
        self.assertEqual(kser.idxmax(skipna=False), pser.idxmax(skipna=False))

        index = pd.MultiIndex.from_arrays([
            ['a', 'a', 'b', 'b'], ['c', 'd', 'e', 'f']], names=('first', 'second'))
        pser = pd.Series(data=[1, 2, 4, 5], index=index)
        kser = koalas.Series(pser)

        self.assertEqual(kser.idxmax(), pser.idxmax())
        self.assertEqual(kser.idxmax(skipna=False), pser.idxmax(skipna=False))

        kser = koalas.Series([])
        with self.assertRaisesRegex(ValueError, "an empty sequence"):
            kser.idxmax()

    def test_idxmin(self):
        pser = pd.Series(data=[1, 4, 5], index=['A', 'B', 'C'])
        kser = koalas.Series(pser)

        self.assertEqual(kser.idxmin(), pser.idxmin())
        self.assertEqual(kser.idxmin(skipna=False), pser.idxmin(skipna=False))

        index = pd.MultiIndex.from_arrays([
            ['a', 'a', 'b', 'b'], ['c', 'd', 'e', 'f']], names=('first', 'second'))
        pser = pd.Series(data=[1, 2, 4, 5], index=index)
        kser = koalas.Series(pser)

        self.assertEqual(kser.idxmin(), pser.idxmin())
        self.assertEqual(kser.idxmin(skipna=False), pser.idxmin(skipna=False))

        kser = koalas.Series([])
        with self.assertRaisesRegex(ValueError, "an empty sequence"):
            kser.idxmin()

    def test_shift(self):
        pser = pd.Series([10, 20, 15, 30, 45], name='x')
        kser = koalas.Series(pser)
        if LooseVersion(pd.__version__) < LooseVersion('0.24.2'):
            self.assertEqual(repr(kser.shift(periods=2)),
                             repr(pser.shift(periods=2)))
        else:
            self.assertEqual(repr(kser.shift(periods=2, fill_value=0)),
                             repr(pser.shift(periods=2, fill_value=0)))
        with self.assertRaisesRegex(ValueError, 'periods should be an int; however'):
            kser.shift(periods=1.5)

    def test_astype(self):
        pser = pd.Series([10, 20, 15, 30, 45], name='x')
        kser = koalas.Series(pser)
        with self.assertRaisesRegex(ValueError, 'Type int63 not understood'):
            kser.astype('int63')

    def test_aggregate(self):
        pser = pd.Series([10, 20, 15, 30, 45], name='x')
        kser = koalas.Series(pser)
        msg = 'func must be a string or list of strings'
        with self.assertRaisesRegex(ValueError, msg):
            kser.aggregate({'x': ['min', 'max']})
        msg = ('If the given function is a list, it '
               'should only contains function names as strings.')
        with self.assertRaisesRegex(ValueError, msg):
            kser.aggregate(['min', max])

    def test_drop(self):
        pser = pd.Series([10, 20, 15, 30, 45], name='x')
        kser = koalas.Series(pser)
        msg = "Need to specify at least one of 'labels' or 'index'"
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop()

        # For MultiIndex
        midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
                              ['speed', 'weight', 'length']],
                             [[0, 0, 0, 1, 1, 1, 2, 2, 2],
                              [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        kser = koalas.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
                             index=midx)
        msg = "'level' should be less than the number of indexes"
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop(labels='weight', level=2)
        msg = ("If the given index is a list, it "
               "should only contains names as strings, "
               "or a list of tuples that contain "
               "index names as strings")
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop(['lama', ['cow', 'falcon']])
        msg = "'index' type should be one of str, list, tuple"
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop({'lama': 'speed'})
        msg = "Cannot specify both 'labels' and 'index'"
        with self.assertRaisesRegex(ValueError, msg):
            kser.drop('lama', index='cow')
        msg = r"'Key length \(2\) exceeds index depth \(3\)'"
        with self.assertRaisesRegex(KeyError, msg):
            kser.drop(('lama', 'speed', 'x'))
        self.assert_eq(kser.drop(('lama', 'speed', 'x'), level=1), kser)

    def test_pop(self):
        midx = pd.MultiIndex([['lama', 'cow', 'falcon'],
                              ['speed', 'weight', 'length']],
                             [[0, 0, 0, 1, 1, 1, 2, 2, 2],
                              [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        kser = koalas.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3],
                             index=midx)
        pser = kser.to_pandas()

        self.assert_eq(kser.pop(('lama', 'speed')), pser.pop(('lama', 'speed')))

        msg = "'key' should be string or tuple that contains strings"
        with self.assertRaisesRegex(ValueError, msg):
            kser.pop(0)
        msg = ("'key' should have index names as only strings "
               "or a tuple that contain index names as only strings")
        with self.assertRaisesRegex(ValueError, msg):
            kser.pop(('lama', 0))
