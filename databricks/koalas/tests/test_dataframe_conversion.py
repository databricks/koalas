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

import os
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

    @staticmethod
    def setup_location(directory):
        """Helper function to set up a temporary directory within test folder."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def teardown_location(location1, location2, directory):
        """Helper function to remove the temporary directory and it's contents."""
        if os.path.isfile(location1):
            os.remove(location1)
        if os.path.isfile(location2):
            os.remove(location2)
        if os.path.exists(directory):
            os.rmdir(directory)

    @staticmethod
    def get_pandas_dataframes(koalas_location, pandas_location):
        return {
            'got': pd.read_excel(koalas_location, index_col=0),
            'expected': pd.read_excel(pandas_location, index_col=0)
        }

    def test_to_excel(self):
        pdf = self.pdf
        kdf = self.kdf

        directory = "./databricks/koalas/tests/temp/"
        pandas_location = directory + "output1.xlsx"
        koalas_location = directory + "output2.xlsx"
        self.setup_location(directory)

        kdf.to_excel(koalas_location)
        pdf.to_excel(pandas_location)
        dataframes = self.get_pandas_dataframes(koalas_location, pandas_location)
        self.assert_eq(dataframes['got'], dataframes['expected'])

        pdf = pd.DataFrame({
            'a': [1, None, 3],
            'b': ["one", "two", None],
        }, index=[0, 1, 3])

        kdf = koalas.from_pandas(pdf)

        kdf.to_excel(koalas_location, na_rep='null')
        pdf.to_excel(pandas_location, na_rep='null')
        dataframes = self.get_pandas_dataframes(koalas_location, pandas_location)
        self.assert_eq(dataframes['got'], dataframes['expected'])

        pdf = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [4.0, 5.0, 6.0],
        }, index=[0, 1, 3])

        kdf = koalas.from_pandas(pdf)

        kdf.to_excel(koalas_location, float_format='%.1f')
        pdf.to_excel(pandas_location, float_format='%.1f')
        dataframes = self.get_pandas_dataframes(koalas_location, pandas_location)
        self.assert_eq(dataframes['got'], dataframes['expected'])

        kdf.to_excel(koalas_location, header=False)
        pdf.to_excel(pandas_location, header=False)
        dataframes = self.get_pandas_dataframes(koalas_location, pandas_location)
        self.assert_eq(dataframes['got'], dataframes['expected'])

        kdf.to_excel(koalas_location, index=False)
        pdf.to_excel(pandas_location, index=False)
        dataframes = self.get_pandas_dataframes(koalas_location, pandas_location)
        self.assert_eq(dataframes['got'], dataframes['expected'])

        self.teardown_location(pandas_location, koalas_location, directory)
