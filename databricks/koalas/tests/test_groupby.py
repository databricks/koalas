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

        self.assert_eq(kdf.groupby('a').sum(), pdf.groupby('a').sum())
        self.assert_eq(kdf.groupby('a').a.sum(), pdf.groupby('a').a.sum())
        self.assert_eq(kdf.groupby('a').b.sum(), pdf.groupby('a').b.sum())
        self.assert_eq(kdf.groupby('a')['a'].sum(), pdf.groupby('a')['a'].sum())
        self.assert_eq(kdf.groupby('a')['b'].sum(), pdf.groupby('a')['b'].sum())
        self.assert_eq(kdf.groupby('a')[['a']].sum(), pdf.groupby('a')[['a']].sum())
        self.assert_eq(kdf.groupby('a')[['b', 'c']].sum(), pdf.groupby('a')[['b', 'c']].sum())
        self.assert_eq(kdf.groupby('a')[[]].sum(), pdf.groupby('a')[[]].sum())
        self.assert_eq(kdf.groupby('a')['c'].sum(), pdf.groupby('a')['c'].sum())
        self.assert_eq(kdf.groupby('a')[['a', 'c']].sum(), pdf.groupby('a')[['a', 'c']].sum())

        self.assert_eq(kdf.a.groupby(kdf.b).sum(), pdf.a.groupby(pdf.b).sum())

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

        self.assert_eq(kdf.groupby('A').agg({'B': 'min', 'C': 'sum'}),
                       pdf.groupby('A').agg({'B': 'min', 'C': 'sum'}))

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

    def test_missing(self):
        kdf = koalas.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9]})

        missing_functions = inspect.getmembers(_MissingPandasLikeDataFrameGroupBy,
                                               inspect.isfunction)
        for name, _ in missing_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*GroupBy.*{}.*not implemented"
                                        .format(name)):
                getattr(kdf.groupby('a'), name)()

        missing_functions = inspect.getmembers(_MissingPandasLikeSeriesGroupBy,
                                               inspect.isfunction)
        for name, _ in missing_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*GroupBy.*{}.*not implemented"
                                        .format(name)):
                getattr(kdf.a.groupby('a'), name)()

        missing_properties = inspect.getmembers(_MissingPandasLikeDataFrameGroupBy,
                                                lambda o: isinstance(o, property))
        for name, _ in missing_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*GroupBy.*{}.*not implemented"
                                        .format(name)):
                getattr(kdf.groupby('a'), name)

        missing_properties = inspect.getmembers(_MissingPandasLikeSeriesGroupBy,
                                                lambda o: isinstance(o, property))
        for name, _ in missing_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*GroupBy.*{}.*not implemented"
                                        .format(name)):
                getattr(kdf.a.groupby('a'), name)
