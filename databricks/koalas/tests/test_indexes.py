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

    def test_to_series(self):
        pind = self.pdf.index
        kind = self.kdf.index

        self.assert_eq(kind.to_series(), pind.to_series())
        self.assert_eq(kind.to_series(name='a'), pind.to_series(name='a'))

        pind = self.pdf.set_index('b', append=True).index
        kind = self.kdf.set_index('b', append=True).index

        if LooseVersion(pyspark.__version__) < LooseVersion('2.4'):
            # PySpark < 2.4 does not support struct type with arrow enabled.
            with self.sql_conf({'spark.sql.execution.arrow.enabled': False}):
                self.assert_eq(kind.to_series(), pind.to_series())
                self.assert_eq(kind.to_series(name='a'), pind.to_series(name='a'))
        else:
            self.assert_eq(kind.to_series(), pind.to_series())
            self.assert_eq(kind.to_series(name='a'), pind.to_series(name='a'))

    def test_index_names(self):
        kdf = self.kdf
        self.assertIsNone(kdf.index.name)

        idx = pd.Index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], name='x')
        pdf = pd.DataFrame(np.random.randn(10, 5), idx)
        kdf = ks.from_pandas(pdf)

        self.assertEqual(kdf.index.name, pdf.index.name)
        self.assertEqual(kdf.index.names, pdf.index.names)

        pind = pdf.index
        kind = kdf.index
        pind.name = 'renamed'
        kind.name = 'renamed'
        self.assertEqual(kind.name, pind.name)
        self.assertEqual(kind.names, pind.names)
        self.assert_eq(kind, pind)

    def test_multi_index_names(self):
        arrays = [[1, 1, 2, 2], ['red', 'blue', 'red', 'blue']]
        idx = pd.MultiIndex.from_arrays(arrays, names=('number', 'color'))
        pdf = pd.DataFrame(np.random.randn(4, 5), idx)
        kdf = ks.from_pandas(pdf)

        self.assertEqual(kdf.index.names, pdf.index.names)

        pind = pdf.index
        kind = kdf.index
        pind.names = ['renamed_number', 'renamed_color']
        kind.names = ['renamed_number', 'renamed_color']
        self.assertEqual(kind.names, pind.names)
        self.assert_eq(kind, pind)

        with self.assertRaises(PandasNotImplementedError):
            kind.name
        with self.assertRaises(PandasNotImplementedError):
            kind.name = 'renamed'

    def test_missing(self):
        kdf = ks.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})

        missing_functions = inspect.getmembers(_MissingPandasLikeIndex, inspect.isfunction)
        for name, _ in missing_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Index.*{}.*not implemented".format(name)):
                getattr(kdf.set_index('a').index, name)()

        missing_functions = inspect.getmembers(_MissingPandasLikeMultiIndex, inspect.isfunction)
        for name, _ in missing_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Index.*{}.*not implemented".format(name)):
                getattr(kdf.set_index(['a', 'b']).index, name)()

        missing_properties = inspect.getmembers(_MissingPandasLikeIndex,
                                                lambda o: isinstance(o, property))
        for name, _ in missing_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Index.*{}.*not implemented".format(name)):
                getattr(kdf.set_index('a').index, name)

        missing_properties = inspect.getmembers(_MissingPandasLikeMultiIndex,
                                                lambda o: isinstance(o, property))
        for name, _ in missing_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Index.*{}.*not implemented".format(name)):
                getattr(kdf.set_index(['a', 'b']).index, name)
