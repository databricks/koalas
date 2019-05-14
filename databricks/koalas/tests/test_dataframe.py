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
from databricks.koalas.generic import max_display_count
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
        return koalas.from_pandas(self.pdf)

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
        ddf = koalas.from_pandas(df)
        self.assert_eq(df[['a', 'b']], ddf[['a', 'b']])

        self.assertEqual(ddf.a.notnull().alias("x").name, "x")

    def test_repr_cache_invalidation(self):
        # If there is any cache, inplace operations should invalidate it.
        df = koalas.range(10)
        df.__repr__()
        df['a'] = df['id']
        self.assertEqual(df.__repr__(), df.to_pandas().__repr__())

    def test_repr_html_cache_invalidation(self):
        # If there is any cache, inplace operations should invalidate it.
        df = koalas.range(10)
        df._repr_html_()
        df['a'] = df['id']
        self.assertEqual(df._repr_html_(), df.to_pandas()._repr_html_())

    def test_empty_dataframe(self):
        pdf = pd.DataFrame({'a': pd.Series([], dtype='i1'),
                            'b': pd.Series([], dtype='str')})

        self.assertRaises(ValueError, lambda: koalas.from_pandas(pdf))

        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            self.assertRaises(ValueError, lambda: koalas.from_pandas(pdf))

    def test_all_null_dataframe(self):

        pdf = pd.DataFrame({'a': pd.Series([None, None, None], dtype='float64'),
                            'b': pd.Series([None, None, None], dtype='str')})

        self.assertRaises(ValueError, lambda: koalas.from_pandas(pdf))

        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            self.assertRaises(ValueError, lambda: koalas.from_pandas(pdf))

    def test_nullable_object(self):
        pdf = pd.DataFrame({'a': list('abc') + [np.nan],
                            'b': list(range(1, 4)) + [np.nan],
                            'c': list(np.arange(3, 6).astype('i1')) + [np.nan],
                            'd': list(np.arange(4.0, 7.0, dtype='float64')) + [np.nan],
                            'e': [True, False, True, np.nan],
                            'f': list(pd.date_range('20130101', periods=3)) + [np.nan]})

        kdf = koalas.from_pandas(pdf)
        self.assert_eq(kdf, pdf)

        with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
            kdf = koalas.from_pandas(pdf)
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

    def test_index_head(self):
        kdf = self.kdf
        pdf = self.pdf

        self.assert_eq(list(kdf.index.head(2).toPandas()), list(pdf.index[:2]))
        self.assert_eq(list(kdf.index.head(3).toPandas()), list(pdf.index[:3]))

    def test_index(self):
        for case in [pd.DataFrame(np.random.randn(10, 5), index=list('abcdefghij')),
                     pd.DataFrame(np.random.randn(10, 5),
                                  index=pd.date_range('2011-01-01', freq='D',
                                                      periods=10))]:
            ddf = koalas.from_pandas(case)
            self.assert_eq(list(ddf.index.toPandas()), list(case.index))

    def test_attributes(self):
        kdf = self.kdf

        self.assertIn('a', dir(kdf))
        self.assertNotIn('foo', dir(kdf))
        self.assertRaises(AttributeError, lambda: kdf.foo)

        kdf = koalas.DataFrame({'a b c': [1, 2, 3]})
        self.assertNotIn('a b c', dir(kdf))
        kdf = koalas.DataFrame({'a': [1, 2], 5: [1, 2]})
        self.assertIn('a', dir(kdf))
        self.assertNotIn(5, dir(kdf))

    def test_column_names(self):
        kdf = self.kdf

        self.assert_eq(kdf.columns, pd.Index(['a', 'b']))
        self.assert_eq(kdf[['b', 'a']].columns, pd.Index(['b', 'a']))
        self.assertEqual(kdf['a'].name, 'a')
        self.assertEqual((kdf['a'] + 1).name, '(a + 1)')  # TODO: 'a'
        self.assertEqual((kdf['a'] + kdf['b']).name, '(a + b)')  # TODO: None

    def test_index_names(self):
        # kdf = self.kdf
        # TODO?: self.assertIsNone(kdf.index.name)

        idx = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name='x')
        df = pd.DataFrame(np.random.randn(10, 5), idx)
        ddf = koalas.from_pandas(df)
        self.assertEqual(ddf.index.name, 'x')

    def test_rename_columns(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7],
                            'b': [7, 6, 5, 4, 3, 2, 1]})
        kdf = koalas.from_pandas(pdf)

        kdf.columns = ['x', 'y']
        pdf.columns = ['x', 'y']
        self.assert_eq(kdf.columns, pd.Index(['x', 'y']))
        self.assert_eq(kdf, pdf)

        msg = "Length mismatch: Expected axis has 2 elements, new values have 4 elements"
        with self.assertRaisesRegex(ValueError, msg):
            kdf.columns = [1, 2, 3, 4]

        # Multi-index columns
        pdf = pd.DataFrame({('A', '0'): [1, 2, 2, 3], ('B', 1): [1, 2, 3, 4]})
        kdf = koalas.from_pandas(pdf)

        pdf.columns = ['x', 'y']
        kdf.columns = ['x', 'y']
        self.assert_eq(kdf.columns, pd.Index(['x', 'y']))
        self.assert_eq(kdf, pdf)

    def test_drop(self):
        kdf = koalas.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6]})

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
        kdf = koalas.from_pandas(pdf)

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
        kdf = koalas.from_pandas(pdf)
        self.assert_eq(kdf, pdf)
        self.assertTrue((kdf.dtypes == pdf.dtypes).all())

    def test_fillna(self):
        pdf = pd.DataFrame({'x': [np.nan, 2, 3, 4, np.nan, 6],
                            'y': [1, 2, np.nan, 4, np.nan, np.nan],
                            'z': [1, 2, 3, 4, np.nan, np.nan]},
                           index=[10, 20, 30, 40, 50, 60])

        kdf = koalas.from_pandas(pdf)

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
        kdf = koalas.from_pandas(pdf)

        self.assert_eq(kdf.notnull(), pdf.notnull())
        self.assert_eq(kdf.isnull(), pdf.isnull())

    def test_to_datetime(self):
        pdf = pd.DataFrame({'year': [2015, 2016],
                            'month': [2, 3],
                            'day': [4, 5]})
        kdf = koalas.from_pandas(pdf)

        self.assert_eq(pd.to_datetime(pdf), koalas.to_datetime(kdf))

    def test_sort_values(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, None, 7],
                            'b': [7, 6, 5, 4, 3, 2, 1]})
        kdf = koalas.from_pandas(pdf)
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

    def test_missing(self):
        kdf = self.kdf

        missing_functions = inspect.getmembers(_MissingPandasLikeDataFrame, inspect.isfunction)
        for name, _ in missing_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*DataFrame.*{}.*not implemented".format(name)):
                getattr(kdf, name)()

        missing_properties = inspect.getmembers(_MissingPandasLikeDataFrame,
                                                lambda o: isinstance(o, property))
        for name, _ in missing_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*DataFrame.*{}.*not implemented".format(name)):
                getattr(kdf, name)

    def test_to_numpy(self):
        pdf = pd.DataFrame({'a': [4, 2, 3, 4, 8, 6],
                            'b': [1, 2, 9, 4, 2, 4],
                            'c': ["one", "three", "six", "seven", "one", "5"]},
                           index=[10, 20, 30, 40, 50, 60])

        kdf = koalas.from_pandas(pdf)

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

        kdf = koalas.from_pandas(df)
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
        left_kdf = koalas.DataFrame({'A': [1, 2]})
        right_kdf = koalas.DataFrame({'B': ['x', 'y']}, index=[1, 2])

        # Assert only 'on' or 'left_index' and 'right_index' parameters are set
        msg = "At least 'on' or 'left_index' and 'right_index' have to be set"
        with self.assertRaises(ValueError, msg=msg):
            left_kdf.merge(right_kdf)
        msg = "Only 'on' or 'left_index' and 'right_index' can be set"
        with self.assertRaises(ValueError, msg=msg):
            left_kdf.merge(right_kdf, on='id', left_index=True)

        # Assert a valid option for the 'how' parameter is used
        msg = ("The 'how' parameter has to be amongst the following values: ['inner', 'left', " +
               "'right', 'full', 'outer']")
        with self.assertRaises(ValueError, msg=msg):
            left_kdf.merge(right_kdf, how='foo', left_index=True, right_index=True)

        # Assert inner join
        res = left_kdf.merge(right_kdf, left_index=True, right_index=True)
        self.assert_eq(res, pd.DataFrame({'A': [2], 'B': ['x']}))

        # Assert inner join on non-default column
        left_kdf_with_id = koalas.DataFrame({'A': [1, 2], 'id': [0, 1]})
        right_kdf_with_id = koalas.DataFrame({'B': ['x', 'y'], 'id': [0, 1]}, index=[1, 2])
        res = left_kdf_with_id.merge(right_kdf_with_id, on='id')
        # Explicitly set columns to also assure their correct order with Python 3.5
        self.assert_eq(res, pd.DataFrame({'A': [1, 2], 'id': [0, 1], 'B': ['x', 'y']},
                                         columns=['A', 'id', 'B']))

        # Assert left join
        res = left_kdf.merge(right_kdf, left_index=True, right_index=True, how='left')
        # FIXME Replace None with np.nan once #263 is solved
        self.assert_eq(res, pd.DataFrame({'A': [1, 2], 'B': [None, 'x']}))

        # Assert right join
        res = left_kdf.merge(right_kdf, left_index=True, right_index=True, how='right')
        self.assert_eq(res, pd.DataFrame({'A': [2, np.nan], 'B': ['x', 'y']}))

        # Assert full outer join
        res = left_kdf.merge(right_kdf, left_index=True, right_index=True, how='outer')
        # FIXME Replace None with np.nan once #263 is solved
        self.assert_eq(res, pd.DataFrame({'A': [1, 2, np.nan], 'B': [None, 'x', 'y']}))

        # Assert full outer join also works with 'full' keyword
        res = left_kdf.merge(right_kdf, left_index=True, right_index=True, how='full')
        # FIXME Replace None with np.nan once #263 is solved
        self.assert_eq(res, pd.DataFrame({'A': [1, 2, np.nan], 'B': [None, 'x', 'y']}))

        # Assert suffixes create the expected column names
        res = left_kdf.merge(koalas.DataFrame({'A': [3, 4]}), left_index=True, right_index=True,
                             suffixes=('_left', '_right'))
        self.assert_eq(res, pd.DataFrame({'A_left': [1, 2], 'A_right': [3, 4]}))

    def test_clip(self):
        pdf = pd.DataFrame({'A': [0, 2, 4]})
        kdf = koalas.from_pandas(pdf)

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
        str_kdf = koalas.DataFrame({'A': ['a', 'b', 'c']})
        self.assert_eq(str_kdf.clip(1, 3), str_kdf)
