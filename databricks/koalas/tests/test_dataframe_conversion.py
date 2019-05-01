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

import string

import pandas as pd

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class DataFrameConversionTest(ReusedSQLTestCase, SQLTestUtils):

    @property
    def pdf(self):
        return pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        }, index=[0, 1, 3])

    @property
    def kdf(self):
        return koalas.from_pandas(self.pdf)

    @staticmethod
    def strip_all_whitespace(str):
        """A helper function to remove all whitespace from a string."""
        return str.translate({ord(c): None for c in string.whitespace})

    def test_to_html(self):
        expected = self.strip_all_whitespace("""
            <table border="1" class="dataframe">
              <thead>
                <tr style="text-align: right;"><th></th><th>a</th><th>b</th></tr>
              </thead>
              <tbody>
                <tr><th>0</th><td>1</td><td>4</td></tr>
                <tr><th>1</th><td>2</td><td>5</td></tr>
                <tr><th>3</th><td>3</td><td>6</td></tr>
              </tbody>
            </table>
            """)
        got = self.strip_all_whitespace(self.kdf.to_html())
        self.assert_eq(got, expected)

        # with max_rows set
        expected = self.strip_all_whitespace("""
            <table border="1" class="dataframe">
              <thead>
                <tr style="text-align: right;"><th></th><th>a</th><th>b</th></tr>
              </thead>
              <tbody>
                <tr><th>0</th><td>1</td><td>4</td></tr>
                <tr><th>1</th><td>2</td><td>5</td></tr>
              </tbody>
            </table>
            """)
        got = self.strip_all_whitespace(self.kdf.to_html(max_rows=2))
        self.assert_eq(got, expected)

    def test_to_string(self):
        self.assert_eq(self.kdf.to_string(),
                       '   a  b\n0  1  4\n1  2  5\n3  3  6')

        self.assert_eq(self.kdf.to_string(max_rows=2),
                       '   a  b\n0  1  4\n1  2  5')
