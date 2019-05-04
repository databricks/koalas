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

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class DataFrameTest(ReusedSQLTestCase, SQLTestUtils):
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
