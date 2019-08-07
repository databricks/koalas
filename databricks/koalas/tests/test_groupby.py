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

import pandas as pd

from databricks import koalas
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.missing.groupby import _MissingPandasLikeDataFrameGroupBy, \
    _MissingPandasLikeSeriesGroupBy
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


class GroupByTest(ReusedSQLTestCase, TestUtils):

    def test_groupby(self):
        pdf = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7],
                            'b': [4, 2, 7, 3, 3, 1, 1, 1, 2],
                            'c': [4, 2, 7, 3, None, 1, 1, 1, 2],
                            'd': list('abcdefght')},
                           index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
        kdf = koalas.from_pandas(pdf)

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

        self.assertRaises(TypeError, lambda: kdf.a.groupby(kdf.b, as_index=False))

    def test_split_apply_combine_on_series(self):
        pdf = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7],
                            'b': [4, 2, 7, 3, 3, 1, 1, 1, 2],
                            'c': [4, 2, 7, 3, None, 1, 1, 1, 2],
                            'd': list('abcdefght')},
                           index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
        kdf = koalas.from_pandas(pdf)

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
        kdf = koalas.from_pandas(pdf)

        for as_index in [True, False]:
            self.assert_eq(kdf.groupby('A', as_index=as_index).agg({'B': 'min', 'C': 'sum'}),
                           pdf.groupby('A', as_index=as_index).agg({'B': 'min', 'C': 'sum'}))

            self.assert_eq(kdf.groupby('A', as_index=as_index).agg({'B': ['min', 'max'],
                                                                    'C': 'sum'}),
                           pdf.groupby('A', as_index=as_index).agg({'B': ['min', 'max'],
                                                                    'C': 'sum'}))

    def test_all_any(self):
        pdf = pd.DataFrame({'A': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                            'B': [True, True, True, False, False, False, None, True, None, False]})
        kdf = koalas.from_pandas(pdf)

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

    def test_raises(self):
        kdf = koalas.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7],
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
        kdf = koalas.DataFrame(pdf)

        for as_index in [True, False]:
            self.assert_eq(kdf.groupby("a", as_index=as_index).agg({"b": "nunique"}),
                           pdf.groupby("a", as_index=as_index).agg({"b": "nunique"}))

    def test_value_counts(self):
        pdf = pd.DataFrame({'A': [1, 2, 2, 3, 3, 3],
                            'B': [1, 1, 2, 3, 3, 3]}, columns=['A', 'B'])
        kdf = koalas.DataFrame(pdf)
        self.assert_eq(repr(kdf.groupby("A")['B'].value_counts().sort_index()),
                       repr(pdf.groupby("A")['B'].value_counts().sort_index()))
        self.assert_eq(repr(kdf.groupby("A")['B']
                            .value_counts(sort=True, ascending=False).sort_index()),
                       repr(pdf.groupby("A")['B']
                            .value_counts(sort=True, ascending=False).sort_index()))

    def test_size(self):
        pdf = pd.DataFrame({'A': [1, 2, 2, 3, 3, 3],
                            'B': [1, 1, 2, 3, 3, 3]})
        kdf = koalas.DataFrame(pdf)
        self.assert_eq(kdf.groupby("A").size().sort_index(),
                       pdf.groupby("A").size().sort_index())
        self.assert_eq(kdf.groupby("A")['B'].size().sort_index(),
                       pdf.groupby("A")['B'].size().sort_index())
        self.assert_eq(kdf.groupby(['A', 'B']).size().sort_index(),
                       pdf.groupby(['A', 'B']).size().sort_index())

    def test_missing(self):
        kdf = koalas.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9]})

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
