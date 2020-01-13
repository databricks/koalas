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

from distutils.version import LooseVersion
import inspect

import numpy as np
import pandas as pd
import pyspark

import databricks.koalas as ks
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.missing.indexes import _MissingPandasLikeIndex, _MissingPandasLikeMultiIndex
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


class IndexesTest(ReusedSQLTestCase, TestUtils):

    @property
    def pdf(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'b': [4, 5, 6, 3, 2, 1, 0, 0, 0],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])

    @property
    def kdf(self):
        return ks.from_pandas(self.pdf)

    def test_index(self):
        for pdf in [pd.DataFrame(np.random.randn(10, 5), index=list('abcdefghij')),
                    pd.DataFrame(np.random.randn(10, 5),
                                 index=pd.date_range('2011-01-01', freq='D', periods=10)),
                    pd.DataFrame(np.random.randn(10, 5),
                                 columns=list('abcde')).set_index(['a', 'b'])]:
            kdf = ks.from_pandas(pdf)
            self.assert_eq(kdf.index, pdf.index)

    def test_index_getattr(self):
        kidx = self.kdf.index
        item = 'databricks'

        expected_error_message = ("'Index' object has no attribute '{}'".format(item))
        with self.assertRaisesRegex(AttributeError, expected_error_message):
            kidx.__getattr__(item)

    def test_multi_index_getattr(self):
        arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        idx = pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        pdf = pd.DataFrame(np.random.randn(4, 5), idx)
        kdf = ks.from_pandas(pdf)
        kidx = kdf.index
        item = 'databricks'

        expected_error_message = ("'MultiIndex' object has no attribute '{}'".format(item))
        with self.assertRaisesRegex(AttributeError, expected_error_message):
            kidx.__getattr__(item)

    def test_to_series(self):
        pidx = self.pdf.index
        kidx = self.kdf.index

        self.assert_eq(kidx.to_series(), pidx.to_series())
        self.assert_eq(kidx.to_series(name='a'), pidx.to_series(name='a'))

        # FIXME: the index values are not addressed the change. (#1190)
        # self.assert_eq((kidx + 1).to_series(), (pidx + 1).to_series())

        pidx = self.pdf.set_index('b', append=True).index
        kidx = self.kdf.set_index('b', append=True).index

        if LooseVersion(pyspark.__version__) < LooseVersion('2.4'):
            # PySpark < 2.4 does not support struct type with arrow enabled.
            with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
                self.assert_eq(kidx.to_series(), pidx.to_series())
                self.assert_eq(kidx.to_series(name='a'), pidx.to_series(name='a'))
        else:
            self.assert_eq(kidx.to_series(), pidx.to_series())
            self.assert_eq(kidx.to_series(name='a'), pidx.to_series(name='a'))

    def test_to_frame(self):
        pidx = self.pdf.index
        kidx = self.kdf.index

        self.assert_eq(repr(kidx.to_frame()), repr(pidx.to_frame()))
        self.assert_eq(repr(kidx.to_frame(index=False)), repr(pidx.to_frame(index=False)))

        if LooseVersion(pd.__version__) >= LooseVersion('0.24'):
            # The `name` argument is added in pandas 0.24.
            self.assert_eq(repr(kidx.to_frame(name='x')), repr(pidx.to_frame(name='x')))
            self.assert_eq(repr(kidx.to_frame(index=False, name='x')),
                           repr(pidx.to_frame(index=False, name='x')))

        pidx = self.pdf.set_index('b', append=True).index
        kidx = self.kdf.set_index('b', append=True).index

        self.assert_eq(repr(kidx.to_frame()), repr(pidx.to_frame()))
        self.assert_eq(repr(kidx.to_frame(index=False)), repr(pidx.to_frame(index=False)))

        if LooseVersion(pd.__version__) >= LooseVersion('0.24'):
            # The `name` argument is added in pandas 0.24.
            self.assert_eq(repr(kidx.to_frame(name=['x', 'y'])),
                           repr(pidx.to_frame(name=['x', 'y'])))
            self.assert_eq(repr(kidx.to_frame(index=False, name=['x', 'y'])),
                           repr(pidx.to_frame(index=False, name=['x', 'y'])))

    def test_index_names(self):
        kdf = self.kdf
        self.assertIsNone(kdf.index.name)

        idx = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name='x')
        pdf = pd.DataFrame(np.random.randn(10, 5), idx)
        kdf = ks.from_pandas(pdf)

        self.assertEqual(kdf.index.name, pdf.index.name)
        self.assertEqual(kdf.index.names, pdf.index.names)

        pidx = pdf.index
        kidx = kdf.index
        pidx.name = 'renamed'
        kidx.name = 'renamed'
        self.assertEqual(kidx.name, pidx.name)
        self.assertEqual(kidx.names, pidx.names)
        self.assert_eq(kidx, pidx)

        pidx.name = None
        kidx.name = None
        self.assertEqual(kidx.name, pidx.name)
        self.assertEqual(kidx.names, pidx.names)
        self.assert_eq(kidx, pidx)

        with self.assertRaisesRegex(ValueError, "Names must be a list-like"):
            kidx.names = 'hi'

        expected_error_message = ("Length of new names must be {}, got {}"
                                  .format(len(kdf._internal.index_map), len(['0', '1'])))
        with self.assertRaisesRegex(ValueError, expected_error_message):
            kidx.names = ['0', '1']

    def test_multi_index_names(self):
        arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        idx = pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        pdf = pd.DataFrame(np.random.randn(4, 5), idx)
        kdf = ks.from_pandas(pdf)

        self.assertEqual(kdf.index.names, pdf.index.names)

        pidx = pdf.index
        kidx = kdf.index
        pidx.names = ['renamed_number', 'renamed_color']
        kidx.names = ['renamed_number', 'renamed_color']
        self.assertEqual(kidx.names, pidx.names)

        pidx.names = ['renamed_number', None]
        kidx.names = ['renamed_number', None]
        self.assertEqual(kidx.names, pidx.names)
        if LooseVersion(pyspark.__version__) < LooseVersion('2.4'):
            # PySpark < 2.4 does not support struct type with arrow enabled.
            with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
                self.assert_eq(kidx, pidx)
        else:
            self.assert_eq(kidx, pidx)

        with self.assertRaises(PandasNotImplementedError):
            kidx.name
        with self.assertRaises(PandasNotImplementedError):
            kidx.name = 'renamed'

    def test_multi_index_levshape(self):
        pidx = pd.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2)])
        kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2)])
        self.assertEqual(pidx.levshape, kidx.levshape)

    def test_index_unique(self):
        kidx = self.kdf.index

        # here the output is different than pandas in terms of order
        expected = [0, 1, 3, 5, 6, 8, 9]

        self.assert_eq(expected, sorted(kidx.unique().to_pandas()))
        self.assert_eq(expected, sorted(kidx.unique(level=0).to_pandas()))

        expected = [1, 2, 4, 6, 7, 9, 10]
        self.assert_eq(expected, sorted((kidx + 1).unique().to_pandas()))

        with self.assertRaisesRegexp(IndexError, "Too many levels*"):
            kidx.unique(level=1)

        with self.assertRaisesRegexp(KeyError, "Requested level (hi)*"):
            kidx.unique(level='hi')

    def test_multi_index_copy(self):
        arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        idx = pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        pdf = pd.DataFrame(np.random.randn(4, 5), idx)
        kdf = ks.from_pandas(pdf)

        self.assert_eq(kdf.index.copy(), pdf.index.copy())

    def test_index_symmetric_difference(self):
        idx = ks.Index(['a', 'b', 'c'])
        midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])

        with self.assertRaisesRegexp(NotImplementedError, "Doesn't support*"):
            idx.symmetric_difference(midx)

    def test_multi_index_symmetric_difference(self):
        idx = ks.Index(['a', 'b', 'c'])
        midx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])
        midx_ = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')])

        self.assert_eq(
            midx.symmetric_difference(midx_),
            midx.to_pandas().symmetric_difference(midx_.to_pandas()))

        with self.assertRaisesRegexp(NotImplementedError, "Doesn't support*"):
            midx.symmetric_difference(idx)

    def test_missing(self):
        kdf = ks.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

        # Index functions
        missing_functions = inspect.getmembers(_MissingPandasLikeIndex, inspect.isfunction)
        unsupported_functions = [name for (name, type_) in missing_functions
                                 if type_.__name__ == 'unsupported_function']
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*Index.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.set_index('a').index, name)()

        deprecated_functions = [name for (name, type_) in missing_functions
                                if type_.__name__ == 'deprecated_function']
        for name in deprecated_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Index.*{}.*is deprecated".format(name)):
                getattr(kdf.set_index('a').index, name)()

        # MultiIndex functions
        missing_functions = inspect.getmembers(_MissingPandasLikeMultiIndex, inspect.isfunction)
        unsupported_functions = [name for (name, type_) in missing_functions
                                 if type_.__name__ == 'unsupported_function']
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*Index.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.set_index(['a', 'b']).index, name)()

        deprecated_functions = [name for (name, type_) in missing_functions
                                if type_.__name__ == 'deprecated_function']
        for name in deprecated_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Index.*{}.*is deprecated".format(name)):
                getattr(kdf.set_index(['a', 'b']).index, name)()

        # Index properties
        missing_properties = inspect.getmembers(_MissingPandasLikeIndex,
                                                lambda o: isinstance(o, property))
        unsupported_properties = [name for (name, type_) in missing_properties
                                  if type_.fget.__name__ == 'unsupported_property']
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*Index.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.set_index('a').index, name)

        deprecated_properties = [name for (name, type_) in missing_properties
                                 if type_.fget.__name__ == 'deprecated_property']
        for name in deprecated_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Index.*{}.*is deprecated".format(name)):
                getattr(kdf.set_index('a').index, name)

        # MultiIndex properties
        missing_properties = inspect.getmembers(_MissingPandasLikeMultiIndex,
                                                lambda o: isinstance(o, property))
        unsupported_properties = [name for (name, type_) in missing_properties
                                  if type_.fget.__name__ == 'unsupported_property']
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*Index.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.set_index(['a', 'b']).index, name)

        deprecated_properties = [name for (name, type_) in missing_properties
                                 if type_.fget.__name__ == 'deprecated_property']
        for name in deprecated_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Index.*{}.*is deprecated".format(name)):
                getattr(kdf.set_index(['a', 'b']).index, name)

    def test_index_has_duplicates(self):
        indexes = [("a", "b", "c"), ("a", "a", "c"), (1, 3, 3), (1, 2, 3)]
        names = [None, 'ks', 'ks', None]
        has_dup = [False, True, True, False]

        for idx, name, expected in zip(indexes, names, has_dup):
            pdf = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index(idx, name=name))
            kdf = ks.from_pandas(pdf)

            self.assertEqual(kdf.index.has_duplicates, expected)

    def test_multiindex_has_duplicates(self):
        indexes = [[list("abc"), list("edf")], [list("aac"), list("edf")],
                   [list("aac"), list("eef")], [[1, 4, 4], [4, 6, 6]]]
        has_dup = [False, False, True, True]

        for idx, expected in zip(indexes, has_dup):
            pdf = pd.DataFrame({"a": [1, 2, 3]}, index=idx)
            kdf = ks.from_pandas(pdf)

            self.assertEqual(kdf.index.has_duplicates, expected)

    def test_multi_index_not_supported(self):
        kdf = ks.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

        with self.assertRaisesRegex(TypeError,
                                    "cannot perform any with this index type"):
            kdf.set_index(['a', 'b']).index.any()

        with self.assertRaisesRegex(TypeError,
                                    "cannot perform all with this index type"):
            kdf.set_index(['a', 'b']).index.all()

    def test_index_nlevels(self):
        pdf = pd.DataFrame({"a": [1, 2, 3]}, index=pd.Index(['a', 'b', 'c']))
        kdf = ks.from_pandas(pdf)

        self.assertEqual(kdf.index.nlevels, 1)

    def test_multiindex_nlevel(self):
        pdf = pd.DataFrame({'a': [1, 2, 3]}, index=[list('abc'), list('def')])
        kdf = ks.from_pandas(pdf)

        self.assertEqual(kdf.index.nlevels, 2)

    def test_multiindex_from_arrays(self):
        arrays = [['a', 'a', 'b', 'b'], ['red', 'blue', 'red', 'blue']]
        pidx = pd.MultiIndex.from_arrays(arrays)
        kidx = ks.MultiIndex.from_arrays(arrays)

        self.assert_eq(pidx, kidx)

    def test_index_fillna(self):
        pidx = pd.DataFrame({'a': ['a', 'b', 'c']}, index=[1, 2, None]).index
        kidx = ks.DataFrame({'a': ['a', 'b', 'c']}, index=[1, 2, None]).index

        self.assert_eq(pidx.fillna(0), kidx.fillna(0))
        self.assert_eq(pidx.rename('name').fillna(0), kidx.rename('name').fillna(0))

        with self.assertRaisesRegex(TypeError, "Unsupported type <class 'list'>"):
            kidx.fillna([1, 2])

    def test_index_drop(self):
        pidx = pd.DataFrame({'a': ['a', 'b', 'c']}, index=[1, 2, 3]).index
        kidx = ks.DataFrame({'a': ['a', 'b', 'c']}, index=[1, 2, 3]).index

        self.assert_eq(pidx.drop(1), kidx.drop(1))
        self.assert_eq(pidx.drop([1, 2]), kidx.drop([1, 2]))

    def test_multiindex_drop(self):
        pidx = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')],
                                         names=['level1', 'level2'])
        kidx = ks.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z')],
                                         names=['level1', 'level2'])
        self.assert_eq(pidx.drop('a'), kidx.drop('a'))
        self.assert_eq(pidx.drop(['a', 'b']), kidx.drop(['a', 'b']))
        self.assert_eq(pidx.drop(['x', 'y'], level='level2'),
                       kidx.drop(['x', 'y'], level='level2'))

    def test_sort_values(self):
        pidx = pd.Index([-10, -100, 200, 100])
        kidx = ks.Index([-10, -100, 200, 100])

        self.assert_eq(pidx.sort_values(), kidx.sort_values())
        self.assert_eq(pidx.sort_values(ascending=False), kidx.sort_values(ascending=False))

        pidx.name = 'koalas'
        kidx.name = 'koalas'

        self.assert_eq(pidx.sort_values(), kidx.sort_values())
        self.assert_eq(pidx.sort_values(ascending=False), kidx.sort_values(ascending=False))

        pidx = pd.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])

        pidx.names = ['hello', 'koalas', 'goodbye']
        kidx.names = ['hello', 'koalas', 'goodbye']

        self.assert_eq(pidx.sort_values(), kidx.sort_values())
        self.assert_eq(pidx.sort_values(ascending=False), kidx.sort_values(ascending=False))

    def test_index_drop_duplicates(self):
        pidx = pd.Index([1, 1, 2])
        kidx = ks.Index([1, 1, 2])
        self.assert_eq(pidx.drop_duplicates(), kidx.drop_duplicates())

        pidx = pd.MultiIndex.from_tuples([(1, 1), (1, 1), (2, 2)], names=['level1', 'level2'])
        kidx = ks.MultiIndex.from_tuples([(1, 1), (1, 1), (2, 2)], names=['level1', 'level2'])
        self.assert_eq(pidx.drop_duplicates(), kidx.drop_duplicates())

    def test_index_sort(self):
        idx = ks.Index([1, 2, 3, 4, 5])
        midx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2)])

        with self.assertRaisesRegex(
                TypeError,
                "cannot sort an Index object in-place, use sort_values instead"):
            idx.sort()
        with self.assertRaisesRegex(
                TypeError,
                "cannot sort an Index object in-place, use sort_values instead"):
            midx.sort()

    def test_multiindex_isna(self):
        kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])

        with self.assertRaisesRegex(
                NotImplementedError,
                "isna is not defined for MultiIndex"):
            kidx.isna()

        with self.assertRaisesRegex(
                NotImplementedError,
                "isna is not defined for MultiIndex"):
            kidx.isnull()

        with self.assertRaisesRegex(
                NotImplementedError,
                "notna is not defined for MultiIndex"):
            kidx.notna()

        with self.assertRaisesRegex(
                NotImplementedError,
                "notna is not defined for MultiIndex"):
            kidx.notnull()

    def test_index_nunique(self):
        pidx = pd.Index([1, 1, 2, None])
        kidx = ks.Index([1, 1, 2, None])

        self.assert_eq(pidx.nunique(), kidx.nunique())
        self.assert_eq(pidx.nunique(dropna=True), kidx.nunique(dropna=True))

    def test_multiindex_nunique(self):
        kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        with self.assertRaisesRegex(
                NotImplementedError,
                "notna is not defined for MultiIndex"):
            kidx.notnull()

    def test_multiindex_rename(self):
        pidx = pd.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])

        pidx = pidx.rename(list('ABC'))
        kidx = kidx.rename(list('ABC'))
        self.assert_eq(pidx, kidx)

        pidx = pidx.rename(['my', 'name', 'is'])
        kidx = kidx.rename(['my', 'name', 'is'])
        self.assert_eq(pidx, kidx)

    def test_multiindex_set_names(self):
        pidx = pd.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])

        pidx = pidx.set_names(['set', 'new', 'names'])
        kidx = kidx.set_names(['set', 'new', 'names'])
        self.assert_eq(pidx, kidx)

        pidx.set_names(['set', 'new', 'names'], inplace=True)
        kidx.set_names(['set', 'new', 'names'], inplace=True)
        self.assert_eq(pidx, kidx)

        pidx = pidx.set_names('first', level=0)
        kidx = kidx.set_names('first', level=0)
        self.assert_eq(pidx, kidx)

        pidx = pidx.set_names('second', level=1)
        kidx = kidx.set_names('second', level=1)
        self.assert_eq(pidx, kidx)

        pidx = pidx.set_names('third', level=2)
        kidx = kidx.set_names('third', level=2)
        self.assert_eq(pidx, kidx)

        pidx.set_names('first', level=0, inplace=True)
        kidx.set_names('first', level=0, inplace=True)
        self.assert_eq(pidx, kidx)

        pidx.set_names('second', level=1, inplace=True)
        kidx.set_names('second', level=1, inplace=True)
        self.assert_eq(pidx, kidx)

        pidx.set_names('third', level=2, inplace=True)
        kidx.set_names('third', level=2, inplace=True)
        self.assert_eq(pidx, kidx)

    def test_multiindex_from_product(self):
        iterables = [[0, 1, 2], ['green', 'purple']]
        pidx = pd.MultiIndex.from_product(iterables)
        kidx = ks.MultiIndex.from_product(iterables)

        self.assert_eq(pidx, kidx)

    def test_multiindex_tuple_column_name(self):
        column_index = pd.MultiIndex.from_tuples([('a', 'x'), ('a', 'y'), ('b', 'z')])
        pdf = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=column_index)
        pdf.set_index(('a', 'x'), append=True, inplace=True)
        kdf = ks.from_pandas(pdf)
        self.assert_eq(pdf, kdf)

    def test_len(self):
        pidx = pd.Index(range(10000))
        kidx = ks.Index(range(10000))

        self.assert_eq(len(pidx), len(kidx))

        pidx = pd.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])

        self.assert_eq(len(pidx), len(kidx))

    def test_append(self):
        # Index
        pidx = pd.Index(range(10000))
        kidx = ks.Index(range(10000))

        self.assert_eq(
            pidx.append(pidx),
            kidx.append(kidx))

        # Index with name
        pidx1 = pd.Index(range(10000), name='a')
        pidx2 = pd.Index(range(10000), name='b')
        kidx1 = ks.Index(range(10000), name='a')
        kidx2 = ks.Index(range(10000), name='b')

        self.assert_eq(
            pidx1.append(pidx2),
            kidx1.append(kidx2))

        self.assert_eq(
            pidx2.append(pidx1),
            kidx2.append(kidx1))

        # Index from DataFrame
        pdf1 = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]},
            index=['a', 'b', 'c'])
        pdf2 = pd.DataFrame({
            'a': [7, 8, 9],
            'd': [10, 11, 12]},
            index=['x', 'y', 'z'])
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        pidx1 = pdf1.set_index('a').index
        pidx2 = pdf2.set_index('d').index
        kidx1 = kdf1.set_index('a').index
        kidx2 = kdf2.set_index('d').index

        self.assert_eq(
            pidx1.append(pidx2),
            kidx1.append(kidx2))

        self.assert_eq(
            pidx2.append(pidx1),
            kidx2.append(kidx1))

        # Index from DataFrame with MultiIndex columns
        pdf1 = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]})
        pdf2 = pd.DataFrame({
            'a': [7, 8, 9],
            'd': [10, 11, 12]})
        pdf1.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y')])
        pdf2.columns = pd.MultiIndex.from_tuples([('a', 'x'), ('d', 'y')])
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        pidx1 = pdf1.set_index(('a', 'x')).index
        pidx2 = pdf2.set_index(('d', 'y')).index
        kidx1 = kdf1.set_index(('a', 'x')).index
        kidx2 = kdf2.set_index(('d', 'y')).index

        self.assert_eq(
            pidx1.append(pidx2),
            kidx1.append(kidx2))

        self.assert_eq(
            pidx2.append(pidx1),
            kidx2.append(kidx1))

        # MultiIndex
        pmidx = pd.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        kmidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])

        self.assert_eq(pmidx.append(pmidx), kmidx.append(kmidx))

        # MultiIndex with names
        pmidx1 = pd.MultiIndex.from_tuples(
            [('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)],
            names=['x', 'y', 'z'])
        pmidx2 = pd.MultiIndex.from_tuples(
            [('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)],
            names=['p', 'q', 'r'])
        kmidx1 = ks.MultiIndex.from_tuples(
            [('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)],
            names=['x', 'y', 'z'])
        kmidx2 = ks.MultiIndex.from_tuples(
            [('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)],
            names=['p', 'q', 'r'])

        self.assert_eq(
            pmidx1.append(pmidx2),
            kmidx1.append(kmidx2))

        self.assert_eq(
            pmidx2.append(pmidx1),
            kmidx2.append(kmidx1))

        self.assert_eq(
            pmidx1.append(pmidx2).names,
            kmidx1.append(kmidx2).names)

        self.assert_eq(
            pmidx1.append(pmidx2).names,
            kmidx1.append(kmidx2).names)

        # Index & MultiIndex currently is not supported
        expected_error_message = r"append\(\) between Index & MultiIndex currently is not supported"
        with self.assertRaisesRegex(NotImplementedError, expected_error_message):
            kidx.append(kmidx)
        with self.assertRaisesRegex(NotImplementedError, expected_error_message):
            kmidx.append(kidx)

    def test_argmin(self):
        pidx = pd.Index([100, 50, 10, 20, 30, 60, 0, 50, 0, 100, 100, 100, 20, 0, 0])
        kidx = ks.Index([100, 50, 10, 20, 30, 60, 0, 50, 0, 100, 100, 100, 20, 0, 0])

        self.assert_eq(pidx.argmin(), kidx.argmin())

        # MultiIndex
        kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        with self.assertRaisesRegex(
                TypeError,
                "reduction operation 'argmin' not allowed for this dtype"):
            kidx.argmin()

    def test_argmax(self):
        pidx = pd.Index([100, 50, 10, 20, 30, 60, 0, 50, 0, 100, 100, 100, 20, 0, 0])
        kidx = ks.Index([100, 50, 10, 20, 30, 60, 0, 50, 0, 100, 100, 100, 20, 0, 0])

        self.assert_eq(pidx.argmax(), kidx.argmax())

        # MultiIndex
        kidx = ks.MultiIndex.from_tuples([('a', 'x', 1), ('b', 'y', 2), ('c', 'z', 3)])
        with self.assertRaisesRegex(
                TypeError,
                "reduction operation 'argmax' not allowed for this dtype"):
            kidx.argmax()
