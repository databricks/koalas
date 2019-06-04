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


import pandas as pd
import pandas.testing as mt
import numpy as np

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class SeriesStringTest(ReusedSQLTestCase, SQLTestUtils):
    @property
    def pds1(self):
        return pd.Series(['apples', 'Bananas', 'carrots', '1', '100', '',
                          None, np.NaN])

    def check_func(self, func):
        ks1 = koalas.from_pandas(self.pds1)
        mt.assert_series_equal(
            func(ks1).toPandas(),
            func(self.pds1),
            check_names=False
        )

    def check_func_on_series(self, func, pds):
        ks = koalas.from_pandas(pds)
        mt.assert_series_equal(
            func(ks).toPandas(),
            func(pds),
            check_names=False
        )

    def test_string_add_str_num(self):
        pdf = pd.DataFrame(dict(col1=['a'], col2=[1]))
        ds = koalas.from_pandas(pdf)
        with self.assertRaises(TypeError):
            ds['col1'] + ds['col2']

    def test_string_add_assign(self):
        pdf = pd.DataFrame(dict(col1=['a', 'b', 'c'], col2=['1', '2', '3']))
        ds = koalas.from_pandas(pdf)
        ds['col1'] += ds['col2']
        pdf['col1'] += pdf['col2']
        self.assert_eq((ds['col1']).to_pandas(), pdf['col1'])

    def test_string_add_str_str(self):
        pdf = pd.DataFrame(dict(col1=['a', 'b', 'c'], col2=['1', '2', '3']))
        ds = koalas.from_pandas(pdf)
        self.assert_eq((ds['col1'] + ds['col2']).to_pandas(), pdf['col1'] + pdf['col2'])
        self.assert_eq((ds['col2'] + ds['col1']).to_pandas(), pdf['col2'] + pdf['col1'])

    def test_string_add_str_lit(self):
        pdf = pd.DataFrame(dict(col1=['a', 'b', 'c']))
        ds = koalas.from_pandas(pdf)
        self.assert_eq((ds['col1'] + '_lit').to_pandas(), pdf['col1'] + '_lit')
        self.assert_eq(('_lit' + ds['col1']).to_pandas(), '_lit' + pdf['col1'])

    def test_string_capitalize(self):
        self.check_func(lambda x:  x.str.capitalize())

    def test_string_lower(self):
        self.check_func(lambda x:  x.str.lower())

    def test_string_swapcase(self):
        self.check_func(lambda x: x.str.swapcase())

    def test_string_startswith(self):
        pattern = 'car'
        self.check_func(lambda x: x.str.startswith(pattern))
        self.check_func(lambda x: x.str.startswith(pattern, na=False))

    def test_string_endswith(self):
        pattern = 's'
        self.check_func(lambda x: x.str.endswith(pattern))
        self.check_func(lambda x: x.str.endswith(pattern, na=False))

    def test_string_get(self):
        self.check_func(lambda x: x.str.get(6))
        self.check_func(lambda x: x.str.get(-1))

    def test_string_encode(self):
        self.check_func(lambda x: x.str.encode(encoding='raw_unicode_escape'))

    def test_string_decode(self):
        series = pd.Series([b'a', b'b', b'c', None])
        self.check_func_on_series(lambda x: x.str.decode(encoding='raw_unicode_escape'),
                                  series)

    def test_string_isalnum(self):
        self.check_func(lambda x: x.str.isalnum())

    def test_string_isalpha(self):
        self.check_func(lambda x: x.str.isalpha())

    def test_string_isdigit(self):
        self.check_func(lambda x: x.str.isdigit())

    def test_string_isspace(self):
        self.check_func(lambda x: x.str.isspace())

    def test_string_islower(self):
        self.check_func(lambda x: x.str.islower())

    def test_string_isupper(self):
        self.check_func(lambda x: x.str.isupper())

    def test_string_istitle(self):
        self.check_func(lambda x: x.str.istitle())

    def test_string_isnumeric(self):
        self.check_func(lambda x: x.str.isnumeric())

    def test_string_isdecimal(self):
        self.check_func(lambda x: x.str.isdecimal())
