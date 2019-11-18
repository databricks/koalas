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
import inspect
from distutils.version import LooseVersion
import pandas as pd

from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.missing.groupby import _MissingPandasLikeDataFrameGroupBy, \
    _MissingPandasLikeSeriesGroupBy
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils
from databricks.koalas.groupby import _is_multi_agg_with_relabel


class GroupByTest(ReusedSQLTestCase, TestUtils):

    def test_groupby(self):
        pdf = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7],
                            'b': [4, 2, 7, 3, 3, 1, 1, 1, 2],
                            'c': [4, 2, 7, 3, None, 1, 1, 1, 2],
                            'd': list('abcdefght')},
                           index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
        kdf = ks.from_pandas(pdf)

        for as_index in [True, False]:
            self.assert_eq(kdf.groupby('a', as_index=as_index).sum(),
                           pdf.groupby('a', as_index=as_index).sum())
            self.assert_eq(kdf.groupby('a', as_index=as_index).b.sum(),
                           pdf.groupby('a', as_index=as_index).b.sum())
            self.assert_eq(kdf.groupby('a', as_index=as_index)['b'].sum(),
                           pdf.groupby('a', as_index=as_index)['b'].sum())
            self.assert_eq(kdf.groupby('a', as_index=as_index)[['b', 'c']].sum(),
                           pdf.groupby('a', as_index=as_index)[['b', 'c']].sum())
            self.assert_eq(kdf.groupby('a', as_index=as_index)[[]].sum(),
                           pdf.groupby('a', as_index=as_index)[[]].sum())
            self.assert_eq(kdf.groupby('a', as_index=as_index)['c'].sum(),
                           pdf.groupby('a', as_index=as_index)['c'].sum())

        self.assert_eq(kdf.groupby('a').a.sum(), pdf.groupby('a').a.sum())
        self.assert_eq(kdf.groupby('a')['a'].sum(), pdf.groupby('a')['a'].sum())
        self.assert_eq(kdf.groupby('a')[['a']].sum(), pdf.groupby('a')[['a']].sum())
        self.assert_eq(kdf.groupby('a')[['a', 'c']].sum(), pdf.groupby('a')[['a', 'c']].sum())

        self.assert_eq(kdf.a.groupby(kdf.b).sum(), pdf.a.groupby(pdf.b).sum())

        self.assertRaises(ValueError, lambda: kdf.groupby('a', as_index=False).a)
        self.assertRaises(ValueError, lambda: kdf.groupby('a', as_index=False)['a'])
        self.assertRaises(ValueError, lambda: kdf.groupby('a', as_index=False)[['a']])
        self.assertRaises(ValueError, lambda: kdf.groupby('a', as_index=False)[['a', 'c']])
        self.assertRaises(ValueError, lambda: kdf.groupby(0, as_index=False)[['a', 'c']])
        self.assertRaises(KeyError, lambda: kdf.groupby([0], as_index=False)[['a', 'c']])

        self.assertRaises(TypeError, lambda: kdf.a.groupby(kdf.b, as_index=False))

    def test_groupby_multiindex_columns(self):
        pdf = pd.DataFrame({('x', 'a'): [1, 2, 6, 4, 4, 6, 4, 3, 7],
                            ('x', 'b'): [4, 2, 7, 3, 3, 1, 1, 1, 2],
                            ('y', 'c'): [4, 2, 7, 3, None, 1, 1, 1, 2],
                            ('z', 'd'): list('abcdefght')},
                           index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.groupby(('x', 'a')).sum(),
                       pdf.groupby(('x', 'a')).sum())
        self.assert_eq(kdf.groupby(('x', 'a'), as_index=False).sum(),
                       pdf.groupby(('x', 'a'), as_index=False).sum())
        self.assert_eq(kdf.groupby(('x', 'a'))[[('y', 'c')]].sum(),
                       pdf.groupby(('x', 'a'))[[('y', 'c')]].sum())
        self.assert_eq(kdf[('x', 'a')].groupby(kdf[('x', 'b')]).sum(),
                       pdf[('x', 'a')].groupby(pdf[('x', 'b')]).sum())

    def test_split_apply_combine_on_series(self):
        pdf = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7],
                            'b': [4, 2, 7, 3, 3, 1, 1, 1, 2],
                            'c': [4, 2, 7, 3, None, 1, 1, 1, 2],
                            'd': list('abcdefght')},
                           index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
        kdf = ks.from_pandas(pdf)

        funcs = [(False, ['sum', 'min', 'max', 'count', 'mean', 'first', 'last']),
                 (True, ['var', 'std'])]
        funcs = [(almost, f) for almost, fs in funcs for f in fs]
        for ddkey, pdkey in [('b', 'b'), (kdf.b, pdf.b), (kdf.b + 1, pdf.b + 1)]:
            for almost, func in funcs:
                self.assert_eq(getattr(kdf.groupby(ddkey).a, func)(),
                               getattr(pdf.groupby(pdkey).a, func)(), almost=almost)
                self.assert_eq(getattr(kdf.groupby(ddkey), func)(),
                               getattr(pdf.groupby(pdkey), func)(), almost=almost)

        for ddkey, pdkey in [(kdf.b, pdf.b), (kdf.b + 1, pdf.b + 1)]:
            for almost, func in funcs:
                self.assert_eq(getattr(kdf.a.groupby(ddkey), func)(),
                               getattr(pdf.a.groupby(pdkey), func)(), almost=almost)
                self.assert_eq(getattr((kdf.a + 1).groupby(ddkey), func)(),
                               getattr((pdf.a + 1).groupby(pdkey), func)(), almost=almost)

        for i in [0, 4, 7]:
            for almost, func in funcs:
                self.assert_eq(getattr(kdf.groupby(kdf.b > i).a, func)(),
                               getattr(pdf.groupby(pdf.b > i).a, func)(), almost=almost)
                self.assert_eq(getattr(kdf.groupby(kdf.b > i), func)(),
                               getattr(pdf.groupby(pdf.b > i), func)(), almost=almost)

    def test_aggregate(self):
        pdf = pd.DataFrame({'A': [1, 1, 2, 2],
                            'B': [1, 2, 3, 4],
                            'C': [0.362, 0.227, 1.267, -0.562]})
        kdf = ks.from_pandas(pdf)

        for as_index in [True, False]:
            self.assert_eq(kdf.groupby('A', as_index=as_index).agg({'B': 'min', 'C': 'sum'}),
                           pdf.groupby('A', as_index=as_index).agg({'B': 'min', 'C': 'sum'}))

            self.assert_eq(kdf.groupby('A', as_index=as_index).agg({'B': ['min', 'max'],
                                                                    'C': 'sum'}),
                           pdf.groupby('A', as_index=as_index).agg({'B': ['min', 'max'],
                                                                    'C': 'sum'}))

        expected_error_message = (r"aggs must be a dict mapping from column name \(string or "
                                  r"tuple\) to aggregate functions \(string or list of strings\).")
        with self.assertRaisesRegex(ValueError, expected_error_message):
            kdf.groupby('A', as_index=as_index).agg(0)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'), ('Y', 'C')])
        pdf.columns = columns
        kdf.columns = columns

        for as_index in [True, False]:
            self.assert_eq(kdf.groupby(('X', 'A'), as_index=as_index)
                           .agg({('X', 'B'): 'min', ('Y', 'C'): 'sum'}),
                           pdf.groupby(('X', 'A'), as_index=as_index)
                           .agg({('X', 'B'): 'min', ('Y', 'C'): 'sum'}))

        self.assert_eq(kdf.groupby(('X', 'A')).agg({('X', 'B'): ['min', 'max'],
                                                    ('Y', 'C'): 'sum'}),
                       pdf.groupby(('X', 'A')).agg({('X', 'B'): ['min', 'max'],
                                                    ('Y', 'C'): 'sum'}))

    def test_aggregate_func_str_list(self):
        # this is test for cases where only string or list is assigned
        pdf = pd.DataFrame({'kind': ['cat', 'dog', 'cat', 'dog'],
                            'height': [9.1, 6.0, 9.5, 34.0],
                            'weight': [7.9, 7.5, 9.9, 198.0]}
                           )
        kdf = ks.from_pandas(pdf)

        agg_funcs = ['max', 'min', ['min', 'max']]
        for aggfunc in agg_funcs:

            # Since in koalas groupby, the order of rows might be different
            # so sort on index to ensure they have same output
            sorted_agg_kdf = kdf.groupby('kind').agg(aggfunc).sort_index()
            sorted_agg_pdf = pdf.groupby('kind').agg(aggfunc).sort_index()
            self.assert_eq(sorted_agg_kdf, sorted_agg_pdf)

        # test on multi index column case
        pdf = pd.DataFrame({'A': [1, 1, 2, 2],
                            'B': [1, 2, 3, 4],
                            'C': [0.362, 0.227, 1.267, -0.562]})
        kdf = ks.from_pandas(pdf)

        columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'), ('Y', 'C')])
        pdf.columns = columns
        kdf.columns = columns

        for aggfunc in agg_funcs:
            sorted_agg_kdf = kdf.groupby(('X', 'A')).agg(aggfunc).sort_index()
            sorted_agg_pdf = pdf.groupby(('X', 'A')).agg(aggfunc).sort_index()
            self.assert_eq(sorted_agg_kdf, sorted_agg_pdf)

    @unittest.skipIf(pd.__version__ < "0.25.0", "not supported before pandas 0.25.0")
    def test_aggregate_relabel(self):
        # this is to test named aggregation in groupby
        pdf = pd.DataFrame({"group": ['a', 'a', 'b', 'b'],
                            "A": [0, 1, 2, 3],
                            "B": [5, 6, 7, 8]})
        kdf = ks.from_pandas(pdf)

        # different agg column, same function
        agg_pdf = pdf.groupby("group").agg(a_max=("A", "max"), b_max=("B", "max")).sort_index()
        agg_kdf = kdf.groupby("group").agg(a_max=("A", "max"), b_max=("B", "max")).sort_index()
        self.assert_eq(agg_pdf, agg_kdf)

        # same agg column, different functions
        agg_pdf = pdf.groupby("group").agg(b_max=("B", "max"), b_min=("B", "min")).sort_index()
        agg_kdf = kdf.groupby("group").agg(b_max=("B", "max"), b_min=("B", "min")).sort_index()
        self.assert_eq(agg_pdf, agg_kdf)

        # test on NamedAgg
        agg_pdf = (
            pdf.groupby("group")
               .agg(b_max=pd.NamedAgg(column="B", aggfunc="max"))
               .sort_index()
        )
        agg_kdf = (
            kdf.groupby("group")
               .agg(b_max=ks.NamedAgg(column="B", aggfunc="max"))
               .sort_index()
        )
        self.assert_eq(agg_kdf, agg_pdf)

        # test on NamedAgg multi columns aggregation
        agg_pdf = (
            pdf.groupby("group")
               .agg(b_max=pd.NamedAgg(column="B", aggfunc="max"),
                    b_min=pd.NamedAgg(column="B", aggfunc="min"))
               .sort_index()
        )
        agg_kdf = (
            kdf.groupby("group")
               .agg(b_max=ks.NamedAgg(column="B", aggfunc="max"),
                    b_min=ks.NamedAgg(column="B", aggfunc="min"))
               .sort_index()
        )
        self.assert_eq(agg_kdf, agg_pdf)

    def test_all_any(self):
        pdf = pd.DataFrame({'A': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                            'B': [True, True, True, False, False, False, None, True, None, False]})
        kdf = ks.from_pandas(pdf)

        for as_index in [True, False]:
            self.assert_eq(kdf.groupby('A', as_index=as_index).all(),
                           pdf.groupby('A', as_index=as_index).all())
            self.assert_eq(kdf.groupby('A', as_index=as_index).any(),
                           pdf.groupby('A', as_index=as_index).any())

            self.assert_eq(kdf.groupby('A', as_index=as_index).all().B,
                           pdf.groupby('A', as_index=as_index).all().B)
            self.assert_eq(kdf.groupby('A', as_index=as_index).any().B,
                           pdf.groupby('A', as_index=as_index).any().B)

        self.assert_eq(kdf.B.groupby(kdf.A).all(), pdf.B.groupby(pdf.A).all())
        self.assert_eq(kdf.B.groupby(kdf.A).any(), pdf.B.groupby(pdf.A).any())

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('X', 'A'), ('Y', 'B')])
        pdf.columns = columns
        kdf.columns = columns

        for as_index in [True, False]:
            self.assert_eq(kdf.groupby(('X', 'A'), as_index=as_index).all(),
                           pdf.groupby(('X', 'A'), as_index=as_index).all())
            self.assert_eq(kdf.groupby(('X', 'A'), as_index=as_index).any(),
                           pdf.groupby(('X', 'A'), as_index=as_index).any())

    def test_raises(self):
        kdf = ks.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7],
                            'b': [4, 2, 7, 3, 3, 1, 1, 1, 2]},
                           index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
        # test raises with incorrect key
        self.assertRaises(ValueError, lambda: kdf.groupby([]))
        self.assertRaises(KeyError, lambda: kdf.groupby('x'))
        self.assertRaises(KeyError, lambda: kdf.groupby(['a', 'x']))
        self.assertRaises(KeyError, lambda: kdf.groupby('a')['x'])
        self.assertRaises(KeyError, lambda: kdf.groupby('a')['b', 'x'])
        self.assertRaises(KeyError, lambda: kdf.groupby('a')[['b', 'x']])

    def test_nunique(self):
        pdf = pd.DataFrame({'a': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            'b': [2, 2, 2, 3, 3, 4, 4, 5, 5, 5]})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.groupby("a").agg({"b": "nunique"}),
                       pdf.groupby("a").agg({"b": "nunique"}))
        self.assert_eq(kdf.groupby("a").nunique(),
                       pdf.groupby("a").nunique())
        self.assert_eq(kdf.groupby("a").nunique(dropna=False),
                       pdf.groupby("a").nunique(dropna=False))
        self.assert_eq(kdf.groupby("a")['b'].nunique(),
                       pdf.groupby("a")['b'].nunique())
        self.assert_eq(kdf.groupby("a")['b'].nunique(dropna=False),
                       pdf.groupby("a")['b'].nunique(dropna=False))

        for as_index in [True, False]:
            self.assert_eq(kdf.groupby("a", as_index=as_index).agg({"b": "nunique"}),
                           pdf.groupby("a", as_index=as_index).agg({"b": "nunique"}))

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('y', 'b')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("x", "a")).nunique(),
                       pdf.groupby(("x", "a")).nunique())
        self.assert_eq(kdf.groupby(("x", "a")).nunique(dropna=False),
                       pdf.groupby(("x", "a")).nunique(dropna=False))

    def test_value_counts(self):
        pdf = pd.DataFrame({'A': [1, 2, 2, 3, 3, 3],
                            'B': [1, 1, 2, 3, 3, 3]}, columns=['A', 'B'])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(repr(kdf.groupby("A")['B'].value_counts().sort_index()),
                       repr(pdf.groupby("A")['B'].value_counts().sort_index()))
        self.assert_eq(repr(kdf.groupby("A")['B']
                            .value_counts(sort=True, ascending=False).sort_index()),
                       repr(pdf.groupby("A")['B']
                            .value_counts(sort=True, ascending=False).sort_index()))
        self.assert_eq(repr(kdf.groupby("A")['B']
                            .value_counts(sort=True, ascending=True).sort_index()),
                       repr(pdf.groupby("A")['B']
                            .value_counts(sort=True, ascending=True).sort_index()))

    def test_size(self):
        pdf = pd.DataFrame({'A': [1, 2, 2, 3, 3, 3],
                            'B': [1, 1, 2, 3, 3, 3]})
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.groupby("A").size().sort_index(),
                       pdf.groupby("A").size().sort_index())
        self.assert_eq(kdf.groupby("A")['B'].size().sort_index(),
                       pdf.groupby("A")['B'].size().sort_index())
        self.assert_eq(kdf.groupby(['A', 'B']).size().sort_index(),
                       pdf.groupby(['A', 'B']).size().sort_index())

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('X', 'A'), ('Y', 'B')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("X", "A")).size().sort_index(),
                       pdf.groupby(("X", "A")).size().sort_index())
        self.assert_eq(kdf.groupby([('X', 'A'), ('Y', 'B')]).size().sort_index(),
                       pdf.groupby([('X', 'A'), ('Y', 'B')]).size().sort_index())

    def test_diff(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
                            'b': [1, 1, 2, 3, 5, 8],
                            'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.groupby("b").diff().sort_index(),
                       pdf.groupby("b").diff().sort_index())
        self.assert_eq(kdf.groupby(['a', 'b']).diff().sort_index(),
                       pdf.groupby(['a', 'b']).diff().sort_index())
        self.assert_eq(kdf.groupby(['b'])['a'].diff().sort_index(),
                       pdf.groupby(['b'])['a'].diff().sort_index(), almost=True)
        self.assert_eq(kdf.groupby(['b'])[['a', 'b']].diff().sort_index(),
                       pdf.groupby(['b'])[['a', 'b']].diff().sort_index(), almost=True)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("x", "b")).diff().sort_index(),
                       pdf.groupby(("x", "b")).diff().sort_index())
        self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')]).diff().sort_index(),
                       pdf.groupby([('x', 'a'), ('x', 'b')]).diff().sort_index())

    def test_rank(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
                            'b': [1, 1, 2, 3, 5, 8],
                            'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.groupby("b").rank().sort_index(),
                       pdf.groupby("b").rank().sort_index())
        self.assert_eq(kdf.groupby(['a', 'b']).rank().sort_index(),
                       pdf.groupby(['a', 'b']).rank().sort_index())
        self.assert_eq(kdf.groupby(['b'])['a'].rank().sort_index(),
                       pdf.groupby(['b'])['a'].rank().sort_index(), almost=True)
        self.assert_eq(kdf.groupby(['b'])[['a', 'c']].rank().sort_index(),
                       pdf.groupby(['b'])[['a', 'c']].rank().sort_index(), almost=True)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("x", "b")).rank().sort_index(),
                       pdf.groupby(("x", "b")).rank().sort_index())
        self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')]).rank().sort_index(),
                       pdf.groupby([('x', 'a'), ('x', 'b')]).rank().sort_index())

    def test_cummin(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
                            'b': [1, 1, 2, 3, 5, 8],
                            'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.groupby("b").cummin().sort_index(),
                       pdf.groupby("b").cummin().sort_index())
        self.assert_eq(kdf.groupby(['a', 'b']).cummin().sort_index(),
                       pdf.groupby(['a', 'b']).cummin().sort_index())
        self.assert_eq(kdf.groupby(['b'])['a'].cummin().sort_index(),
                       pdf.groupby(['b'])['a'].cummin().sort_index(), almost=True)
        self.assert_eq(kdf.groupby(['b'])[['a', 'c']].cummin().sort_index(),
                       pdf.groupby(['b'])[['a', 'c']].cummin().sort_index(), almost=True)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("x", "b")).cummin().sort_index(),
                       pdf.groupby(("x", "b")).cummin().sort_index())
        self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')]).cummin().sort_index(),
                       pdf.groupby([('x', 'a'), ('x', 'b')]).cummin().sort_index())

    def test_cummax(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
                            'b': [1, 1, 2, 3, 5, 8],
                            'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.groupby("b").cummax().sort_index(),
                       pdf.groupby("b").cummax().sort_index())
        self.assert_eq(kdf.groupby(['a', 'b']).cummax().sort_index(),
                       pdf.groupby(['a', 'b']).cummax().sort_index())
        self.assert_eq(kdf.groupby(['b'])['a'].cummax().sort_index(),
                       pdf.groupby(['b'])['a'].cummax().sort_index(), almost=True)
        self.assert_eq(kdf.groupby(['b'])[['a', 'c']].cummax().sort_index(),
                       pdf.groupby(['b'])[['a', 'c']].cummax().sort_index(), almost=True)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("x", "b")).cummax().sort_index(),
                       pdf.groupby(("x", "b")).cummax().sort_index())
        self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')]).cummax().sort_index(),
                       pdf.groupby([('x', 'a'), ('x', 'b')]).cummax().sort_index())

    def test_cumsum(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
                            'b': [1, 1, 2, 3, 5, 8],
                            'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.groupby("b").cumsum().sort_index(),
                       pdf.groupby("b").cumsum().sort_index())
        self.assert_eq(kdf.groupby(['a', 'b']).cumsum().sort_index(),
                       pdf.groupby(['a', 'b']).cumsum().sort_index())
        self.assert_eq(kdf.groupby(['b'])['a'].cumsum().sort_index(),
                       pdf.groupby(['b'])['a'].cumsum().sort_index(), almost=True)
        self.assert_eq(kdf.groupby(['b'])[['a', 'c']].cumsum().sort_index(),
                       pdf.groupby(['b'])[['a', 'c']].cumsum().sort_index(), almost=True)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("x", "b")).cumsum().sort_index(),
                       pdf.groupby(("x", "b")).cumsum().sort_index())
        self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')]).cumsum().sort_index(),
                       pdf.groupby([('x', 'a'), ('x', 'b')]).cumsum().sort_index())

    def test_cumprod(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
                            'b': [1, 1, 2, 3, 5, 8],
                            'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.groupby("b").cumprod().sort_index(),
                       pdf.groupby("b").cumprod().sort_index(), almost=True)
        self.assert_eq(kdf.groupby(['a', 'b']).cumprod().sort_index(),
                       pdf.groupby(['a', 'b']).cumprod().sort_index(), almost=True)
        self.assert_eq(kdf.groupby(['b'])['a'].cumprod().sort_index(),
                       pdf.groupby(['b'])['a'].cumprod().sort_index(), almost=True)
        self.assert_eq(kdf.groupby(['b'])[['a', 'c']].cumprod().sort_index(),
                       pdf.groupby(['b'])[['a', 'c']].cumprod().sort_index(), almost=True)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("x", "b")).cumprod().sort_index(),
                       pdf.groupby(("x", "b")).cumprod().sort_index(), almost=True)
        self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')]).cumprod().sort_index(),
                       pdf.groupby([('x', 'a'), ('x', 'b')]).cumprod().sort_index(), almost=True)

    def test_nsmallest(self):
        pdf = pd.DataFrame({'a': [1, 1, 1, 2, 2, 2, 3, 3, 3],
                            'b': [1, 2, 2, 2, 3, 3, 3, 4, 4],
                            'c': [1, 2, 2, 2, 3, 3, 3, 4, 4],
                            'd': [1, 2, 2, 2, 3, 3, 3, 4, 4]}, columns=['a', 'b', 'c', 'd'])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(repr(kdf.groupby(['a'])['b'].nsmallest(1).sort_values()),
                       repr(pdf.groupby(['a'])['b'].nsmallest(1).sort_values()))
        self.assert_eq(repr(kdf.groupby(['a'])['b'].nsmallest(2).sort_index()),
                       repr(pdf.groupby(['a'])['b'].nsmallest(2).sort_index()))
        with self.assertRaisesRegex(ValueError, "idxmax do not support multi-index now"):
            kdf.set_index(['a', 'b']).groupby(['c'])['d'].nsmallest(1)

    def test_nlargest(self):
        pdf = pd.DataFrame({'a': [1, 1, 1, 2, 2, 2, 3, 3, 3],
                            'b': [1, 2, 2, 2, 3, 3, 3, 4, 4],
                            'c': [1, 2, 2, 2, 3, 3, 3, 4, 4],
                            'd': [1, 2, 2, 2, 3, 3, 3, 4, 4]}, columns=['a', 'b', 'c', 'd'])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(repr(kdf.groupby(['a'])['b'].nlargest(1).sort_values()),
                       repr(pdf.groupby(['a'])['b'].nlargest(1).sort_values()))
        self.assert_eq(repr(kdf.groupby(['a'])['b'].nlargest(2).sort_index()),
                       repr(pdf.groupby(['a'])['b'].nlargest(2).sort_index()))
        with self.assertRaisesRegex(ValueError, "idxmax do not support multi-index now"):
            kdf.set_index(['a', 'b']).groupby(['c'])['d'].nlargest(1)

    def test_fillna(self):
        pdf = pd.DataFrame({'A': [1, 1, 2, 2],
                            'B': [2, 4, None, 3],
                            'C': [None, None, None, 1],
                            'D': [0, 1, 5, 4]}, columns=['A', 'B', 'C', 'D'], index=[0, 1, 2, 3])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.groupby("A").fillna(0),
                       pdf.groupby("A").fillna(0))
        self.assert_eq(kdf.groupby("A").fillna(method='bfill'),
                       pdf.groupby("A").fillna(method='bfill'))
        self.assert_eq(kdf.groupby("A").fillna(method='ffill'),
                       pdf.groupby("A").fillna(method='ffill'))

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'), ('Y', 'C'), ('Z', 'D')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("X", "A")).fillna(0),
                       pdf.groupby(("X", "A")).fillna(0))
        self.assert_eq(kdf.groupby(("X", "A")).fillna(method='bfill'),
                       pdf.groupby(("X", "A")).fillna(method='bfill'))
        self.assert_eq(kdf.groupby(("X", "A")).fillna(method='ffill'),
                       pdf.groupby(("X", "A")).fillna(method='ffill'))

    def test_ffill(self):
        pdf = pd.DataFrame({'A': [1, 1, 2, 2],
                            'B': [2, 4, None, 3],
                            'C': [None, None, None, 1],
                            'D': [0, 1, 5, 4]}, columns=['A', 'B', 'C', 'D'], index=[0, 1, 2, 3])
        kdf = ks.from_pandas(pdf)

        if LooseVersion(pd.__version__) <= LooseVersion("0.24.2"):
            self.assert_eq(kdf.groupby("A").ffill(),
                           pdf.groupby("A").ffill().drop('A', 1))
        else:
            self.assert_eq(kdf.groupby("A").ffill(),
                           pdf.groupby("A").ffill())
        self.assert_eq(repr(kdf.groupby("A")['B'].ffill()),
                       repr(pdf.groupby("A")['B'].ffill()))

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'), ('Y', 'C'), ('Z', 'D')])
        pdf.columns = columns
        kdf.columns = columns

        if LooseVersion(pd.__version__) <= LooseVersion("0.24.2"):
            self.assert_eq(kdf.groupby(("X", "A")).ffill(),
                           pdf.groupby(("X", "A")).ffill().drop(('X', 'A'), 1))
        else:
            self.assert_eq(kdf.groupby(("X", "A")).ffill(),
                           pdf.groupby(("X", "A")).ffill())

    def test_bfill(self):
        pdf = pd.DataFrame({'A': [1, 1, 2, 2],
                            'B': [2, 4, None, 3],
                            'C': [None, None, None, 1],
                            'D': [0, 1, 5, 4]}, columns=['A', 'B', 'C', 'D'], index=[0, 1, 2, 3])
        kdf = ks.from_pandas(pdf)
        if LooseVersion(pd.__version__) <= LooseVersion("0.24.2"):
            self.assert_eq(kdf.groupby("A").bfill(),
                           pdf.groupby("A").bfill().drop('A', 1))
        else:
            self.assert_eq(kdf.groupby("A").bfill(),
                           pdf.groupby("A").bfill())
        self.assert_eq(repr(kdf.groupby("A")['B'].bfill()),
                       repr(pdf.groupby("A")['B'].bfill()))

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B'), ('Y', 'C'), ('Z', 'D')])
        pdf.columns = columns
        kdf.columns = columns

        if LooseVersion(pd.__version__) <= LooseVersion("0.24.2"):
            self.assert_eq(kdf.groupby(("X", "A")).bfill(),
                           pdf.groupby(("X", "A")).bfill().drop(('X', 'A'), 1))
        else:
            self.assert_eq(kdf.groupby(("X", "A")).bfill(),
                           pdf.groupby(("X", "A")).bfill())

    @unittest.skipIf(pd.__version__ < '0.24.0', "not supported before pandas 0.24.0")
    def test_shift(self):
        pdf = pd.DataFrame({'a': [1, 1, 2, 2, 3, 3],
                            'b': [1, 1, 2, 2, 3, 4],
                            'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.groupby('a').shift().sort_index(),
                       pdf.groupby('a').shift().sort_index())
        # TODO: seems like a pandas' bug when fill_value is not None?
        # self.assert_eq(kdf.groupby(['a', 'b']).shift(periods=-1, fill_value=0).sort_index(),
        #                pdf.groupby(['a', 'b']).shift(periods=-1, fill_value=0).sort_index())
        self.assert_eq(kdf.groupby(['b'])['a'].shift().sort_index(),
                       pdf.groupby(['b'])['a'].shift().sort_index(), almost=True)
        self.assert_eq(kdf.groupby(['a', 'b'])['c'].shift().sort_index(),
                       pdf.groupby(['a', 'b'])['c'].shift().sort_index(), almost=True)
        self.assert_eq(kdf.groupby(['b'])[['a', 'c']].shift(periods=-1, fill_value=0).sort_index(),
                       pdf.groupby(['b'])[['a', 'c']].shift(periods=-1, fill_value=0).sort_index(),
                       almost=True)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(('x', 'a')).shift().sort_index(),
                       pdf.groupby(('x', 'a')).shift().sort_index())
        # TODO: seems like a pandas' bug when fill_value is not None?
        # self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')]).shift(periods=-1,
        #                                                            fill_value=0).sort_index(),
        #                pdf.groupby([('x', 'a'), ('x', 'b')]).shift(periods=-1,
        #                                                            fill_value=0).sort_index())

    def test_apply(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
                            'b': [1, 1, 2, 3, 5, 8],
                            'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.groupby("b").apply(lambda x: x + 1).sort_index(),
                       pdf.groupby("b").apply(lambda x: x + 1).sort_index())
        self.assert_eq(kdf.groupby(['a', 'b']).apply(lambda x: x * x).sort_index(),
                       pdf.groupby(['a', 'b']).apply(lambda x: x * x).sort_index())
        self.assert_eq(kdf.groupby(['b'])['a'].apply(lambda x: x).sort_index(),
                       pdf.groupby(['b'])['a'].apply(lambda x: x).sort_index())

        with self.assertRaisesRegex(TypeError, "<class 'int'> object is not callable"):
            kdf.groupby("b").apply(1)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("x", "b")).apply(lambda x: x + 1).sort_index(),
                       pdf.groupby(("x", "b")).apply(lambda x: x + 1).sort_index())
        self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')]).apply(lambda x: x * x).sort_index(),
                       pdf.groupby([('x', 'a'), ('x', 'b')]).apply(lambda x: x * x).sort_index())

    def test_apply_with_new_dataframe(self):
        pdf = pd.DataFrame({
            "timestamp": [0.0, 0.5, 1.0, 0.0, 0.5],
            "car_id": ['A', 'A', 'A', 'B', 'B']
        })
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby('car_id').apply(lambda _: pd.DataFrame({"column": [0.0]})).sort_index(),
            pdf.groupby('car_id').apply(lambda _: pd.DataFrame({"column": [0.0]})).sort_index())

        self.assert_eq(
            kdf.groupby('car_id')
            .apply(lambda df: pd.DataFrame({'mean': [df['timestamp'].mean()]})).sort_index(),
            pdf.groupby('car_id')
            .apply(lambda df: pd.DataFrame({"mean": [df['timestamp'].mean()]})).sort_index())

        # dataframe with 1000+ records
        pdf = pd.DataFrame({
            "timestamp": [0.0, 0.5, 1.0, 0.0, 0.5] * 300,
            "car_id": ['A', 'A', 'A', 'B', 'B'] * 300
        })
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            kdf.groupby('car_id').apply(lambda _: pd.DataFrame({"column": [0.0]})).sort_index(),
            pdf.groupby('car_id').apply(lambda _: pd.DataFrame({"column": [0.0]})).sort_index())

        self.assert_eq(
            kdf.groupby('car_id')
            .apply(lambda df: pd.DataFrame({"mean": [df['timestamp'].mean()]})).sort_index(),
            pdf.groupby('car_id')
            .apply(lambda df: pd.DataFrame({"mean": [df['timestamp'].mean()]})).sort_index())

    def test_transform(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
                            'b': [1, 1, 2, 3, 5, 8],
                            'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)
        self.assert_eq(kdf.groupby("b").transform(lambda x: x + 1).sort_index(),
                       pdf.groupby("b").transform(lambda x: x + 1).sort_index())
        self.assert_eq(kdf.groupby(['a', 'b']).transform(lambda x: x * x).sort_index(),
                       pdf.groupby(['a', 'b']).transform(lambda x: x * x).sort_index())
        self.assert_eq(kdf.groupby(['b'])['a'].transform(lambda x: x).sort_index(),
                       pdf.groupby(['b'])['a'].transform(lambda x: x).sort_index())

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("x", "b")).transform(lambda x: x + 1).sort_index(),
                       pdf.groupby(("x", "b")).transform(lambda x: x + 1).sort_index())
        self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')])
                       .transform(lambda x: x * x).sort_index(),
                       pdf.groupby([('x', 'a'), ('x', 'b')])
                       .transform(lambda x: x * x).sort_index())

        set_option('compute.shortcut_limit', 1000)
        try:
            pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6] * 300,
                                'b': [1, 1, 2, 3, 5, 8] * 300,
                                'c': [1, 4, 9, 16, 25, 36] * 300}, columns=['a', 'b', 'c'])
            kdf = ks.from_pandas(pdf)
            self.assert_eq(kdf.groupby("b").transform(lambda x: x + 1).sort_index(),
                           pdf.groupby("b").transform(lambda x: x + 1).sort_index())
            self.assert_eq(kdf.groupby(['a', 'b']).transform(lambda x: x * x).sort_index(),
                           pdf.groupby(['a', 'b']).transform(lambda x: x * x).sort_index())
            self.assert_eq(kdf.groupby(['b'])['a'].transform(lambda x: x).sort_index(),
                           pdf.groupby(['b'])['a'].transform(lambda x: x).sort_index())
            with self.assertRaisesRegex(TypeError, "<class 'int'> object is not callable"):
                kdf.groupby("b").transform(1)

            # multi-index columns
            columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
            pdf.columns = columns
            kdf.columns = columns

            self.assert_eq(kdf.groupby(("x", "b")).transform(lambda x: x + 1).sort_index(),
                           pdf.groupby(("x", "b")).transform(lambda x: x + 1).sort_index())
            self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')])
                           .transform(lambda x: x * x).sort_index(),
                           pdf.groupby([('x', 'a'), ('x', 'b')])
                           .transform(lambda x: x * x).sort_index())
        finally:
            reset_option('compute.shortcut_limit')

    def test_filter(self):
        pdf = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6],
                            'b': [1, 1, 2, 3, 5, 8],
                            'c': [1, 4, 9, 16, 25, 36]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.groupby("b").filter(lambda x: x.b.mean() < 4).sort_index(),
                       pdf.groupby("b").filter(lambda x: x.b.mean() < 4).sort_index())
        self.assert_eq(kdf.groupby(['a', 'b']).filter(lambda x: any(x.a == 2)).sort_index(),
                       pdf.groupby(['a', 'b']).filter(lambda x: any(x.a == 2)).sort_index())

        with self.assertRaisesRegex(TypeError, "<class 'int'> object is not callable"):
            kdf.groupby("b").filter(1)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(kdf.groupby(("x", "b"))
                       .filter(lambda x: x[('x', 'b')].mean() < 4).sort_index(),
                       pdf.groupby(("x", "b"))
                       .filter(lambda x: x[('x', 'b')].mean() < 4).sort_index())
        self.assert_eq(kdf.groupby([('x', 'a'), ('x', 'b')])
                       .filter(lambda x: any(x[('x', 'a')] == 2)).sort_index(),
                       pdf.groupby([('x', 'a'), ('x', 'b')])
                       .filter(lambda x: any(x[('x', 'a')] == 2)).sort_index())

    def test_idxmax(self):
        pdf = pd.DataFrame({'a': [1, 1, 2, 2, 3],
                            'b': [1, 2, 3, 4, 5],
                            'c': [5, 4, 3, 2, 1]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.groupby(['a']).idxmax(),
                       kdf.groupby(['a']).idxmax().sort_index())
        self.assert_eq(pdf.groupby(['a']).idxmax(skipna=False),
                       kdf.groupby(['a']).idxmax(skipna=False).sort_index())

        with self.assertRaisesRegex(ValueError, 'idxmax only support one-level index now'):
            kdf.set_index(['a', 'b']).groupby(['c']).idxmax()

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(pdf.groupby(('x', 'a')).idxmax(),
                       kdf.groupby(('x', 'a')).idxmax().sort_index())
        self.assert_eq(pdf.groupby(('x', 'a')).idxmax(skipna=False),
                       kdf.groupby(('x', 'a')).idxmax(skipna=False).sort_index())

    def test_idxmin(self):
        pdf = pd.DataFrame({'a': [1, 1, 2, 2, 3],
                            'b': [1, 2, 3, 4, 5],
                            'c': [5, 4, 3, 2, 1]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.groupby(['a']).idxmin(),
                       kdf.groupby(['a']).idxmin().sort_index())
        self.assert_eq(pdf.groupby(['a']).idxmin(skipna=False),
                       kdf.groupby(['a']).idxmin(skipna=False).sort_index())

        with self.assertRaisesRegex(ValueError, 'idxmin only support one-level index now'):
            kdf.set_index(['a', 'b']).groupby(['c']).idxmin()

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c')])
        pdf.columns = columns
        kdf.columns = columns

        self.assert_eq(pdf.groupby(('x', 'a')).idxmin(),
                       kdf.groupby(('x', 'a')).idxmin().sort_index())
        self.assert_eq(pdf.groupby(('x', 'a')).idxmin(skipna=False),
                       kdf.groupby(('x', 'a')).idxmin(skipna=False).sort_index())

    def test_head(self):
        pdf = pd.DataFrame({'a': [1, 1, 2, 2, 3],
                            'b': [1, 2, 3, 4, 5],
                            'c': [5, 4, 3, 2, 1]}, columns=['a', 'b', 'c'])
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pdf.groupby(['a', 'b']).idxmin(),
                       kdf.groupby(['a', 'b']).idxmin().sort_index())

    def test_missing(self):
        kdf = ks.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9]})

        # DataFrameGroupBy functions
        missing_functions = inspect.getmembers(_MissingPandasLikeDataFrameGroupBy,
                                               inspect.isfunction)
        unsupported_functions = [name for (name, type_) in missing_functions
                                 if type_.__name__ == 'unsupported_function']
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*GroupBy.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.groupby('a'), name)()

        deprecated_functions = [name for (name, type_) in missing_functions
                                if type_.__name__ == 'deprecated_function']
        for name in deprecated_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*GroupBy.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.groupby('a'), name)()

        # SeriesGroupBy functions
        missing_functions = inspect.getmembers(_MissingPandasLikeSeriesGroupBy,
                                               inspect.isfunction)
        unsupported_functions = [name for (name, type_) in missing_functions
                                 if type_.__name__ == 'unsupported_function']
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*GroupBy.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.a.groupby('a'), name)()

        deprecated_functions = [name for (name, type_) in missing_functions
                                if type_.__name__ == 'deprecated_function']
        for name in deprecated_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*GroupBy.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.a.groupby('a'), name)()

        # DataFrameGroupBy properties
        missing_properties = inspect.getmembers(_MissingPandasLikeDataFrameGroupBy,
                                                lambda o: isinstance(o, property))
        unsupported_properties = [name for (name, type_) in missing_properties
                                  if type_.fget.__name__ == 'unsupported_property']
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*GroupBy.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.groupby('a'), name)
        deprecated_properties = [name for (name, type_) in missing_properties
                                 if type_.fget.__name__ == 'deprecated_property']
        for name in deprecated_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*GroupBy.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.groupby('a'), name)

        # SeriesGroupBy properties
        missing_properties = inspect.getmembers(_MissingPandasLikeSeriesGroupBy,
                                                lambda o: isinstance(o, property))
        unsupported_properties = [name for (name, type_) in missing_properties
                                  if type_.fget.__name__ == 'unsupported_property']
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*GroupBy.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.a.groupby('a'), name)
        deprecated_properties = [name for (name, type_) in missing_properties
                                 if type_.fget.__name__ == 'deprecated_property']
        for name in deprecated_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*GroupBy.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.a.groupby('a'), name)

    @staticmethod
    def test_is_multi_agg_with_relabel():

        assert _is_multi_agg_with_relabel(a='max') is False
        assert _is_multi_agg_with_relabel(a_min=('a', 'max'), a_max=('a', 'min')) is True
