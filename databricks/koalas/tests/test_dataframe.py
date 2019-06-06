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
from pyspark.sql.utils import AnalysisException

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.missing.frame import _MissingPandasLikeDataFrame


class DataFrameTest(ReusedSQLTestCase, SQLTestUtils):

    @property
    def pdf(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def kdf(self):
        return ks.from_pandas(self.pdf)

    def test_dataframe(self):
        kdf = self.kdf
        pdf = self.pdf

        expected = pd.Series([2, 3, 4, 5, 6, 7, 8, 9, 10],
                             index=[0, 1, 3, 5, 6, 8, 9, 9, 9],
                             name='(a + 1)')  # TODO: name='a'

        self.assert_eq(kdf['a'] + 1, expected)

        self.assert_eq(kdf.columns, pd.Index(['a', 'b']))

        self.assert_eq(kdf[kdf['b'] > 2], pdf[pdf['b'] > 2])
        self.assert_eq(kdf[['a', 'b']], pdf[['a', 'b']])
        self.assert_eq(kdf.a, pdf.a)
        # TODO: assert d.b.mean().compute() == pdf.b.mean()
        # TODO: assert np.allclose(d.b.var().compute(), pdf.b.var())
        # TODO: assert np.allclose(d.b.std().compute(), pdf.b.std())

        assert repr(kdf)

        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        })
        ddf = ks.from_pandas(df)
        self.assert_eq(df[['a', 'b']], ddf[['a', 'b']])

        self.assertEqual(ddf.a.notnull().alias("x").name, "x")

    def test_repr_cache_invalidation(self):
        # If there is any cache, inplace operations should invalidate it.
        df = ks.range(10)
        df.__repr__()
        df['a'] = df['id']
        self.assertEqual(df.__repr__(), df.to_pandas().__repr__())

    def test_repr_html_cache_invalidation(self):
        # If there is any cache, inplace operations should invalidate it.
        df = ks.range(10)
        df._repr_html_()
        df['a'] = df['id']
        self.assertEqual(df._repr_html_(), df.to_pandas()._repr_html_())

    def test_empty_dataframe(self):
        pdf = pd.DataFrame({'a': pd.Series([], dtype='i1'),
                            'b': pd.Series([], dtype='str')})

        self.assertRaises(ValueError, lambda: ks.from_pandas(pdf))

        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            self.assertRaises(ValueError, lambda: ks.from_pandas(pdf))

    def test_all_null_dataframe(self):

        pdf = pd.DataFrame({'a': pd.Series([None, None, None], dtype='float64'),
                            'b': pd.Series([None, None, None], dtype='str')})

        self.assertRaises(ValueError, lambda: ks.from_pandas(pdf))

        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            self.assertRaises(ValueError, lambda: ks.from_pandas(pdf))

    def test_nullable_object(self):
        pdf = pd.DataFrame({'a': list('abc') + [np.nan],
                            'b': list(range(1, 4)) + [np.nan],
                            'c': list(np.arange(3, 6).astype('i1')) + [np.nan],
                            'd': list(np.arange(4.0, 7.0, dtype='float64')) + [np.nan],
                            'e': [True, False, True, np.nan],
                            'f': list(pd.date_range('20130101', periods=3)) + [np.nan]})

        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf, pdf)

        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            kdf = ks.from_pandas(pdf)
            self.assert_eq(kdf, pdf)

    def test_assign(self):
        kdf = self.kdf.copy()
        pdf = self.pdf.copy()

        kdf['w'] = 1.0
        pdf['w'] = 1.0

        self.assert_eq(kdf, pdf)

        kdf['a'] = 'abc'
        pdf['a'] = 'abc'

        self.assert_eq(kdf, pdf)

    def test_head_tail(self):
        kdf = self.kdf
        pdf = self.pdf

        self.assert_eq(kdf.head(2), pdf.head(2))
        self.assert_eq(kdf.head(3), pdf.head(3))

    def test_attributes(self):
        kdf = self.kdf

        self.assertIn('a', dir(kdf))
        self.assertNotIn('foo', dir(kdf))
        self.assertRaises(AttributeError, lambda: kdf.foo)

        kdf = ks.DataFrame({'a b c': [1, 2, 3]})
        self.assertNotIn('a b c', dir(kdf))
        kdf = ks.DataFrame({'a': [1, 2], 5: [1, 2]})
        self.assertIn('a', dir(kdf))
        self.assertNotIn(5, dir(kdf))

    def test_column_names(self):
        kdf = self.kdf

        self.assert_eq(kdf.columns, pd.Index(['a', 'b']))
        self.assert_eq(kdf[['b', 'a']].columns, pd.Index(['b', 'a']))
        self.assertEqual(kdf['a'].name, 'a')
        self.assertEqual((kdf['a'] + 1).name, '(a + 1)')  # TODO: 'a'
        self.assertEqual((kdf['a'] + kdf['b']).name, '(a + b)')  # TODO: None

    def test_rename_columns(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7],
                            'b': [7, 6, 5, 4, 3, 2, 1]})
        kdf = ks.from_pandas(pdf)

        kdf.columns = ['x', 'y']
        pdf.columns = ['x', 'y']
        self.assert_eq(kdf.columns, pd.Index(['x', 'y']))
        self.assert_eq(kdf, pdf)

        msg = "Length mismatch: Expected axis has 2 elements, new values have 4 elements"
        with self.assertRaisesRegex(ValueError, msg):
            kdf.columns = [1, 2, 3, 4]

        # Multi-index columns
        pdf = pd.DataFrame({('A', '0'): [1, 2, 2, 3], ('B', 1): [1, 2, 3, 4]})
        kdf = ks.from_pandas(pdf)

        pdf.columns = ['x', 'y']
        kdf.columns = ['x', 'y']
        self.assert_eq(kdf.columns, pd.Index(['x', 'y']))
        self.assert_eq(kdf, pdf)

    def test_drop(self):
        kdf = ks.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6]})

        # Assert 'labels' or 'columns' parameter is set
        expected_error_message = "Need to specify at least one of 'labels' or 'columns'"
        with self.assertRaisesRegex(ValueError, expected_error_message):
            kdf.drop()
        # Assert axis cannot be 0
        with self.assertRaisesRegex(NotImplementedError, "Drop currently only works for axis=1"):
            kdf.drop('x', axis=0)
        # Assert using a str for 'labels' works
        self.assert_eq(kdf.drop('x', axis=1), pd.DataFrame({'y': [3, 4], 'z': [5, 6]}))
        # Assert axis is 1 by default
        self.assert_eq(kdf.drop('x'), pd.DataFrame({'y': [3, 4], 'z': [5, 6]}))
        # Assert using a list for 'labels' works
        self.assert_eq(kdf.drop(['y', 'z'], axis=1), pd.DataFrame({'x': [1, 2]}))
        # Assert using 'columns' instead of 'labels' produces the same results
        self.assert_eq(kdf.drop(columns='x'), pd.DataFrame({'y': [3, 4], 'z': [5, 6]}))
        self.assert_eq(kdf.drop(columns=['y', 'z']), pd.DataFrame({'x': [1, 2]}))
        # Assert 'labels' being used when both 'labels' and 'columns' are specified
        expected_output = pd.DataFrame({'y': [3, 4], 'z': [5, 6]})
        self.assert_eq(kdf.drop(labels=['x'], columns=['y']), expected_output)

    def test_dropna(self):
        pdf = pd.DataFrame({'x': [np.nan, 2, 3, 4, np.nan, 6],
                            'y': [1, 2, np.nan, 4, np.nan, np.nan],
                            'z': [1, 2, 3, 4, np.nan, np.nan]},
                           index=[10, 20, 30, 40, 50, 60])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.dropna(), pdf.dropna())
        self.assert_eq(kdf.dropna(how='all'), pdf.dropna(how='all'))
        self.assert_eq(kdf.dropna(subset=['x']), pdf.dropna(subset=['x']))
        self.assert_eq(kdf.dropna(subset=['y', 'z']), pdf.dropna(subset=['y', 'z']))
        self.assert_eq(kdf.dropna(subset=['y', 'z'], how='all'),
                       pdf.dropna(subset=['y', 'z'], how='all'))

        self.assert_eq(kdf.dropna(thresh=2), pdf.dropna(thresh=2))
        self.assert_eq(kdf.dropna(thresh=1, subset=['y', 'z']),
                       pdf.dropna(thresh=1, subset=['y', 'z']))

        ddf2 = kdf.copy()
        ddf2.dropna(inplace=True)
        self.assert_eq(ddf2, pdf.dropna())

        msg = "dropna currently only works for axis=0 or axis='index'"
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.dropna(axis=1)
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.dropna(axis='column')
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.dropna(axis='foo')

    def test_dtype(self):
        pdf = pd.DataFrame({'a': list('abc'),
                            'b': list(range(1, 4)),
                            'c': np.arange(3, 6).astype('i1'),
                            'd': np.arange(4.0, 7.0, dtype='float64'),
                            'e': [True, False, True],
                            'f': pd.date_range('20130101', periods=3)})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf, pdf)
        self.assertTrue((kdf.dtypes == pdf.dtypes).all())

    def test_fillna(self):
        pdf = pd.DataFrame({'x': [np.nan, 2, 3, 4, np.nan, 6],
                            'y': [1, 2, np.nan, 4, np.nan, np.nan],
                            'z': [1, 2, 3, 4, np.nan, np.nan]},
                           index=[10, 20, 30, 40, 50, 60])

        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf, pdf)
        self.assert_eq(kdf.fillna(-1), pdf.fillna(-1))
        self.assert_eq(kdf.fillna({'x': -1, 'y': -2, 'z': -5}),
                       pdf.fillna({'x': -1, 'y': -2, 'z': -5}))

        pdf.fillna({'x': -1, 'y': -2, 'z': -5}, inplace=True)
        kdf.fillna({'x': -1, 'y': -2, 'z': -5}, inplace=True)
        self.assert_eq(kdf, pdf)

        s_nan = pd.Series([-1, -2, -5], index=['x', 'y', 'z'], dtype=int)
        self.assert_eq(kdf.fillna(s_nan),
                       pdf.fillna(s_nan))

        with self.assertRaisesRegex(NotImplementedError, "fillna currently only"):
            kdf.fillna(-1, axis=1)
        with self.assertRaisesRegex(NotImplementedError, "fillna currently only"):
            kdf.fillna(-1, axis='column')
        with self.assertRaisesRegex(ValueError, "must specify value"):
            kdf.fillna()
        with self.assertRaisesRegex(TypeError, "Unsupported.*DataFrame"):
            kdf.fillna(pd.DataFrame({'x': [-1], 'y': [-1], 'z': [-1]}))
        with self.assertRaisesRegex(TypeError, "Unsupported.*numpy.int64"):
            kdf.fillna({'x': np.int64(-6), 'y': np.int64(-4), 'z': -5})

    def test_isnull(self):
        pdf = pd.DataFrame({'x': [1, 2, 3, 4, None, 6], 'y': list('abdabd')},
                           index=[10, 20, 30, 40, 50, 60])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.notnull(), pdf.notnull())
        self.assert_eq(kdf.isnull(), pdf.isnull())

    def test_to_datetime(self):
        pdf = pd.DataFrame({'year': [2015, 2016],
                            'month': [2, 3],
                            'day': [4, 5]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pd.to_datetime(pdf), ks.to_datetime(kdf))

    def test_nunique(self):
        pdf = pd.DataFrame({'A': [1, 2, 3], 'B': [np.nan, 3, np.nan]})
        kdf = ks.from_pandas(pdf)

        # Assert NaNs are dropped by default
        nunique_result = kdf.nunique()
        self.assert_eq(nunique_result, pd.Series([3, 1], index=['A', 'B'], name='0'))
        self.assert_eq(nunique_result, pdf.nunique())

        # Assert including NaN values
        nunique_result = kdf.nunique(dropna=False)
        self.assert_eq(nunique_result, pd.Series([3, 2], index=['A', 'B'], name='0'))
        self.assert_eq(nunique_result, pdf.nunique(dropna=False))

        # Assert approximate counts
        self.assert_eq(ks.DataFrame({'A': range(100)}).nunique(approx=True),
                       pd.Series([103], index=['A'], name='0'))
        self.assert_eq(ks.DataFrame({'A': range(100)}).nunique(approx=True, rsd=0.01),
                       pd.Series([100], index=['A'], name='0'))

    def test_sort_values(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, None, 7],
                            'b': [7, 6, 5, 4, 3, 2, 1]})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(repr(kdf.sort_values('b')), repr(pdf.sort_values('b')))
        self.assert_eq(repr(kdf.sort_values(['b', 'a'])), repr(pdf.sort_values(['b', 'a'])))
        self.assert_eq(
            repr(kdf.sort_values(['b', 'a'], ascending=[False, True])),
            repr(pdf.sort_values(['b', 'a'], ascending=[False, True])))

        self.assertRaises(ValueError, lambda: kdf.sort_values(['b', 'a'], ascending=[False]))

        self.assert_eq(
            repr(kdf.sort_values(['b', 'a'], na_position='first')),
            repr(pdf.sort_values(['b', 'a'], na_position='first')))

        self.assertRaises(ValueError, lambda: kdf.sort_values(['b', 'a'], na_position='invalid'))

        self.assert_eq(kdf.sort_values('b', inplace=True), pdf.sort_values('b', inplace=True))
        self.assert_eq(repr(kdf), repr(pdf))

    def test_sort_index(self):
        pdf = pd.DataFrame({'A': [2, 1, np.nan], 'B': [np.nan, 0, np.nan]},
                           index=['b', 'a', np.nan])
        kdf = ks.from_pandas(pdf)

        # Assert invalid parameters
        self.assertRaises(ValueError, lambda: kdf.sort_index(axis=1))
        self.assertRaises(ValueError, lambda: kdf.sort_index(level=42))
        self.assertRaises(ValueError, lambda: kdf.sort_index(kind='mergesort'))
        self.assertRaises(ValueError, lambda: kdf.sort_index(na_position='invalid'))

        # Assert default behavior without parameters
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        # Assert sorting descending
        self.assert_eq(kdf.sort_index(ascending=False), pdf.sort_index(ascending=False))
        # Assert sorting NA indices first
        self.assert_eq(kdf.sort_index(na_position='first'), pdf.sort_index(na_position='first'))
        # Assert sorting inplace
        self.assertEqual(kdf.sort_index(inplace=True), pdf.sort_index(inplace=True))
        self.assert_eq(kdf, pdf)

        # Assert multi-indices
        pdf = pd.DataFrame({'A': range(4), 'B': range(4)[::-1]},
                           index=[['b', 'b', 'a', 'a'], [1, 0, 1, 0]])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_nlargest(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, None, 7],
                            'b': [7, 6, 5, 4, 3, 2, 1]})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.nlargest(n=5, columns='a'), pdf.nlargest(5, columns='a'))
        self.assert_eq(kdf.nlargest(n=5, columns=['a', 'b']), pdf.nlargest(5, columns=['a', 'b']))

    def test_nsmallest(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, None, 7],
                            'b': [7, 6, 5, 4, 3, 2, 1]})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.nsmallest(n=5, columns='a'), pdf.nsmallest(5, columns='a'))
        self.assert_eq(kdf.nsmallest(n=5, columns=['a', 'b']), pdf.nsmallest(5, columns=['a', 'b']))

    def test_missing(self):
        kdf = self.kdf

        missing_functions = inspect.getmembers(_MissingPandasLikeDataFrame, inspect.isfunction)
        unsupported_functions = [name for (name, type_) in missing_functions
                                 if type_.__name__ == 'unsupported_function']
        for name in unsupported_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*DataFrame.*{}.*not implemented".format(name)):
                getattr(kdf, name)()

        deprecated_functions = [name for (name, type_) in missing_functions
                                if type_.__name__ == 'deprecated_function']
        for name in deprecated_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*DataFrame.*{}.*is deprecated".format(name)):
                getattr(kdf, name)()

        missing_properties = inspect.getmembers(_MissingPandasLikeDataFrame,
                                                lambda o: isinstance(o, property))
        unsupported_properties = [name for (name, type_) in missing_properties
                                  if type_.fget.__name__ == 'unsupported_property']
        for name in unsupported_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*DataFrame.*{}.*not implemented".format(name)):
                getattr(kdf, name)
        deprecated_properties = [name for (name, type_) in missing_properties
                                 if type_.fget.__name__ == 'deprecated_property']
        for name in deprecated_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*DataFrame.*{}.*is deprecated".format(name)):
                getattr(kdf, name)

    def test_values_property(self):
        kdf = self.kdf
        msg = ("Koalas does not support the 'values' property. If you want to collect your data " +
               "as an NumPy array, use 'to_numpy()' instead.")
        with self.assertRaises(NotImplementedError, msg=msg):
            kdf.values

    def test_to_numpy(self):
        pdf = pd.DataFrame({'a': [4, 2, 3, 4, 8, 6],
                            'b': [1, 2, 9, 4, 2, 4],
                            'c': ["one", "three", "six", "seven", "one", "5"]},
                           index=[10, 20, 30, 40, 50, 60])

        kdf = ks.from_pandas(pdf)

        np.testing.assert_equal(kdf.to_numpy(), pdf.values)

    def test_to_pandas(self):
        kdf = self.kdf
        pdf = self.pdf
        self.assert_eq(kdf.toPandas(), pdf)
        self.assert_eq(kdf.to_pandas(), pdf)

    def test_isin(self):
        df = pd.DataFrame({'a': [4, 2, 3, 4, 8, 6],
                           'b': [1, 2, 9, 4, 2, 4],
                           'c': ["one", "three", "six", "seven", "one", "5"]},
                          index=[10, 20, 30, 40, 50, 60])

        kdf = ks.from_pandas(df)
        self.assert_eq(kdf.isin([4, 'six']), df.isin([4, 'six']))
        self.assert_eq(kdf.isin({"a": [2, 8], "c": ['three', "one"]}),
                       df.isin({"a": [2, 8], "c": ['three', "one"]}))

        msg = "'DataFrame' object has no attribute {'e'}"
        with self.assertRaisesRegex(AttributeError, msg):
            kdf.isin({"e": [5, 7], "a": [1, 6]})

        msg = "DataFrame and Series are not supported"
        with self.assertRaisesRegex(NotImplementedError, msg):
            kdf.isin(df)

        msg = "Values should be iterable, Series, DataFrame or dict."
        with self.assertRaisesRegex(TypeError, msg):
            kdf.isin(1)

    def test_merge(self):
        left_pdf = pd.DataFrame({'lkey': ['foo', 'bar', 'baz', 'foo', 'bar', 'l'],
                                 'value': [1, 2, 3, 5, 6, 7],
                                 'x': list('abcdef')},
                                columns=['lkey', 'value', 'x'])
        right_pdf = pd.DataFrame({'rkey': ['baz', 'foo', 'bar', 'baz', 'foo', 'r'],
                                  'value': [4, 5, 6, 7, 8, 9],
                                  'y': list('efghij')},
                                 columns=['rkey', 'value', 'y'])

        left_kdf = ks.from_pandas(left_pdf)
        right_kdf = ks.from_pandas(right_pdf)

        def check(op):
            k_res = op(left_kdf, right_kdf)
            k_res = k_res.sort_values(by=list(k_res.columns))
            p_res = op(left_pdf, right_pdf)
            p_res = p_res.sort_values(by=list(p_res.columns))
            p_res = p_res.reset_index(drop=True)
            self.assert_eq(k_res, p_res)

        check(lambda left, right: left.merge(right))
        check(lambda left, right: left.merge(right, on='value'))
        check(lambda left, right: left.merge(right, left_on='lkey', right_on='rkey'))
        check(lambda left, right: left.set_index('lkey').merge(right.set_index('rkey')))
        check(lambda left, right: left.set_index('lkey').merge(right,
                                                               left_index=True, right_on='rkey'))
        check(lambda left, right: left.merge(right.set_index('rkey'),
                                             left_on='lkey', right_index=True))
        check(lambda left, right: left.set_index('lkey').merge(right.set_index('rkey'),
                                                               left_index=True, right_index=True))

        # MultiIndex
        check(lambda left, right: left.merge(right,
                                             left_on=['lkey', 'value'], right_on=['rkey', 'value']))
        check(lambda left, right: left.set_index(['lkey', 'value'])
              .merge(right, left_index=True, right_on=['rkey', 'value']))
        check(lambda left, right: left.merge(
            right.set_index(['rkey', 'value']), left_on=['lkey', 'value'], right_index=True))
        # TODO: when both left_index=True and right_index=True with multi-index
        # check(lambda left, right: left.set_index(['lkey', 'value']).merge(
        #     right.set_index(['rkey', 'value']), left_index=True, right_index=True))

        # join types
        for how in ['inner', 'left', 'right', 'outer']:
            check(lambda left, right: left.merge(right, left_on='lkey', right_on='rkey', how=how))

        # suffix
        check(lambda left, right: left.merge(right, left_on='lkey', right_on='rkey',
                                             suffixes=['_left', '_right']))

    def test_merge_raises(self):
        left = ks.DataFrame({'value': [1, 2, 3, 5, 6],
                             'x': list('abcde')},
                            columns=['value', 'x'],
                            index=['foo', 'bar', 'baz', 'foo', 'bar'])
        right = ks.DataFrame({'value': [4, 5, 6, 7, 8],
                              'y': list('fghij')},
                             columns=['value', 'y'],
                             index=['baz', 'foo', 'bar', 'baz', 'foo'])

        with self.assertRaisesRegex(ValueError,
                                    'No common columns to perform merge on'):
            left[['x']].merge(right[['y']])

        with self.assertRaisesRegex(ValueError,
                                    'not a combination of both'):
            left.merge(right, on='value', left_on='x')

        with self.assertRaisesRegex(ValueError,
                                    'Must pass right_on or right_index=True'):
            left.merge(right, left_on='x')

        with self.assertRaisesRegex(ValueError,
                                    'Must pass right_on or right_index=True'):
            left.merge(right, left_index=True)

        with self.assertRaisesRegex(ValueError,
                                    'Must pass left_on or left_index=True'):
            left.merge(right, right_on='y')

        with self.assertRaisesRegex(ValueError,
                                    'Must pass left_on or left_index=True'):
            left.merge(right, right_index=True)

        with self.assertRaisesRegex(ValueError,
                                    'len\\(left_keys\\) must equal len\\(right_keys\\)'):
            left.merge(right, left_on='value', right_on=['value', 'y'])

        with self.assertRaisesRegex(ValueError,
                                    'len\\(left_keys\\) must equal len\\(right_keys\\)'):
            left.merge(right, left_on=['value', 'x'], right_on='value')

        with self.assertRaisesRegex(ValueError,
                                    "['inner', 'left', 'right', 'full', 'outer']"):
            left.merge(right, left_index=True, right_index=True, how='foo')

        with self.assertRaisesRegex(AnalysisException,
                                    'Cannot resolve column name "id"'):
            left.merge(right, on='id')

    def test_append(self):
        pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
        kdf = ks.from_pandas(pdf)
        other_pdf = pd.DataFrame([[3, 4], [5, 6]], columns=list('BC'), index=[2, 3])
        other_kdf = ks.from_pandas(other_pdf)

        self.assert_eq(kdf.append(kdf), pdf.append(pdf))
        self.assert_eq(kdf.append(kdf, ignore_index=True), pdf.append(pdf, ignore_index=True))

        # Assert DataFrames with non-matching columns
        self.assert_eq(kdf.append(other_kdf), pdf.append(other_pdf))

        # Assert appending a Series fails
        msg = "DataFrames.append() does not support appending Series to DataFrames"
        with self.assertRaises(ValueError, msg=msg):
            kdf.append(kdf['A'])

        # Assert using the sort parameter raises an exception
        msg = "The 'sort' parameter is currently not supported"
        with self.assertRaises(ValueError, msg=msg):
            kdf.append(kdf, sort=True)

        # Assert using 'verify_integrity' only raises an exception for overlapping indices
        kdf.append(other_kdf, verify_integrity=True)
        msg = "Indices have overlapping values"
        with self.assertRaises(ValueError, msg=msg):
            kdf.append(kdf, verify_integrity=True)

        self.assert_eq(kdf.append(kdf, ignore_index=True, verify_integrity=True),
                       pdf.append(pdf, ignore_index=True, verify_integrity=True))

        # Assert appending multi-index DataFrames
        multi_index_pdf = pd.DataFrame([[1, 2], [3, 4]], columns=list('AB'),
                                       index=[[2, 3], [4, 5]])
        multi_index_kdf = ks.from_pandas(multi_index_pdf)
        self.assert_eq(multi_index_kdf.append(multi_index_kdf),
                       multi_index_pdf.append(multi_index_pdf))

        # Assert trying to append DataFrames with different index levels
        msg = "Both DataFrames have to have the same number of index levels"
        with self.assertRaises(ValueError, msg=msg):
            kdf.append(multi_index_kdf)

        self.assert_eq(kdf.append(multi_index_kdf, ignore_index=True),
                       pdf.append(multi_index_pdf, ignore_index=True))

    def test_clip(self):
        pdf = pd.DataFrame({'A': [0, 2, 4]})
        kdf = ks.from_pandas(pdf)

        # Assert list-like values are not accepted for 'lower' and 'upper'
        msg = "List-like value are not supported for 'lower' and 'upper' at the moment"
        with self.assertRaises(ValueError, msg=msg):
            kdf.clip(lower=[1])
        with self.assertRaises(ValueError, msg=msg):
            kdf.clip(upper=[1])

        # Assert no lower or upper
        self.assert_eq(kdf.clip(), pdf.clip())
        # Assert lower only
        self.assert_eq(kdf.clip(1), pdf.clip(1))
        # Assert upper only
        self.assert_eq(kdf.clip(upper=3), pdf.clip(upper=3))
        # Assert lower and upper
        self.assert_eq(kdf.clip(1, 3), pdf.clip(1, 3))

        # Assert behavior on string values
        str_kdf = ks.DataFrame({'A': ['a', 'b', 'c']})
        self.assert_eq(str_kdf.clip(1, 3), str_kdf)

    def test_binary_operators(self):
        self.assertRaisesRegex(
            ValueError,
            'with another DataFrame or a sequence is currently not supported',
            lambda: ks.range(10).add(ks.range(10)))

        self.assertRaisesRegex(
            ValueError,
            'with another DataFrame or a sequence is currently not supported',
            lambda: ks.range(10).add(ks.range(10).id))

    def test_sample(self):
        pdf = pd.DataFrame({'A': [0, 2, 4]})
        kdf = ks.from_pandas(pdf)

        # Make sure the tests run, but we can't check the result because they are non-deterministic.
        kdf.sample(frac=0.1)
        kdf.sample(frac=0.2, replace=True)
        kdf.sample(frac=0.2, random_state=5)
        kdf['A'].sample(frac=0.2)
        kdf['A'].sample(frac=0.2, replace=True)
        kdf['A'].sample(frac=0.2, random_state=5)

        with self.assertRaises(ValueError):
            kdf.sample()
        with self.assertRaises(NotImplementedError):
            kdf.sample(n=1)

    def test_add_prefix(self):
        pdf = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
        kdf = ks.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
        self.assert_eq(pdf.add_prefix('col_'), kdf.add_prefix('col_'))

    def test_add_suffix(self):
        pdf = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
        kdf = ks.DataFrame({'A': [1, 2, 3, 4], 'B': [3, 4, 5, 6]})
        self.assert_eq(pdf.add_suffix('_col'), kdf.add_suffix('_col'))
