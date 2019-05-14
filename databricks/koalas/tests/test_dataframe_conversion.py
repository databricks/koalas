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

import numpy as np
import pandas as pd

from databricks import koalas
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils, TestUtils


class DataFrameConversionTest(ReusedSQLTestCase, SQLTestUtils, TestUtils):

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

    def test_csv(self):
        pdf = self.pdf
        kdf = self.kdf

        self.assert_eq(kdf.to_csv(), pdf.to_csv())

        pdf = pd.DataFrame({
            'a': [1, np.nan, 3],
            'b': ["one", "two", None],
        }, index=[0, 1, 3])

        kdf = koalas.from_pandas(pdf)

        self.assert_eq(kdf.to_csv(na_rep='null'), pdf.to_csv(na_rep='null'))

        pdf = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [4.0, 5.0, 6.0],
        }, index=[0, 1, 3])

        kdf = koalas.from_pandas(pdf)

        self.assert_eq(kdf.to_csv(float_format='%.1f'), pdf.to_csv(float_format='%.1f'))
        self.assert_eq(kdf.to_csv(header=False), pdf.to_csv(header=False))
        self.assert_eq(kdf.to_csv(index=False), pdf.to_csv(index=False))

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
    def get_excel_dfs(koalas_location, pandas_location):
        return {
            'got': pd.read_excel(koalas_location, index_col=0),
            'expected': pd.read_excel(pandas_location, index_col=0)
        }

    def test_to_excel(self):
        with self.temp_dir() as dirpath:
            pandas_location = dirpath + "/" + "output1.xlsx"
            koalas_location = dirpath + "/" + "output2.xlsx"

            pdf = self.pdf
            kdf = self.kdf
            kdf.to_excel(koalas_location)
            pdf.to_excel(pandas_location)
            dataframes = self.get_excel_dfs(koalas_location, pandas_location)
            self.assert_eq(dataframes['got'], dataframes['expected'])

            pdf = pd.DataFrame({
                'a': [1, None, 3],
                'b': ["one", "two", None],
            }, index=[0, 1, 3])

            kdf = koalas.from_pandas(pdf)

            kdf.to_excel(koalas_location, na_rep='null')
            pdf.to_excel(pandas_location, na_rep='null')
            dataframes = self.get_excel_dfs(koalas_location, pandas_location)
            self.assert_eq(dataframes['got'], dataframes['expected'])

            pdf = pd.DataFrame({
                'a': [1.0, 2.0, 3.0],
                'b': [4.0, 5.0, 6.0],
            }, index=[0, 1, 3])

            kdf = koalas.from_pandas(pdf)

            kdf.to_excel(koalas_location, float_format='%.1f')
            pdf.to_excel(pandas_location, float_format='%.1f')
            dataframes = self.get_excel_dfs(koalas_location, pandas_location)
            self.assert_eq(dataframes['got'], dataframes['expected'])

            kdf.to_excel(koalas_location, header=False)
            pdf.to_excel(pandas_location, header=False)
            dataframes = self.get_excel_dfs(koalas_location, pandas_location)
            self.assert_eq(dataframes['got'], dataframes['expected'])

            kdf.to_excel(koalas_location, index=False)
            pdf.to_excel(pandas_location, index=False)
            dataframes = self.get_excel_dfs(koalas_location, pandas_location)
            self.assert_eq(dataframes['got'], dataframes['expected'])

    def test_to_json(self):
        pdf = self.pdf
        kdf = koalas.from_pandas(pdf)

        self.assert_eq(kdf.to_json(), pdf.to_json())
        self.assert_eq(kdf.to_json(orient='split'), pdf.to_json(orient='split'))
        self.assert_eq(kdf.to_json(orient='records'), pdf.to_json(orient='records'))
        self.assert_eq(kdf.to_json(orient='index'), pdf.to_json(orient='index'))
        self.assert_eq(kdf.to_json(orient='values'), pdf.to_json(orient='values'))
        self.assert_eq(kdf.to_json(orient='table'), pdf.to_json(orient='table'))
        self.assert_eq(kdf.to_json(orient='records', lines=True),
                       pdf.to_json(orient='records', lines=True))
        self.assert_eq(kdf.to_json(orient='split', index=False),
                       pdf.to_json(orient='split', index=False))

    def test_to_clipboard(self):
        pdf = self.pdf
        kdf = self.kdf

        self.assert_eq(kdf.to_clipboard(), pdf.to_clipboard())
        self.assert_eq(kdf.to_clipboard(excel=False),
                       pdf.to_clipboard(excel=False))
        self.assert_eq(kdf.to_clipboard(sep=";", index=False),
                       pdf.to_clipboard(sep=";", index=False))

    def test_to_latex(self):
        pdf = self.pdf
        kdf = self.kdf

        self.assert_eq(kdf.to_latex(), pdf.to_latex())
        self.assert_eq(kdf.to_latex(col_space=2), pdf.to_latex(col_space=2))
        self.assert_eq(kdf.to_latex(header=True), pdf.to_latex(header=True))
        self.assert_eq(kdf.to_latex(index=False), pdf.to_latex(index=False))
        self.assert_eq(kdf.to_latex(na_rep='-'), pdf.to_latex(na_rep='-'))
        self.assert_eq(kdf.to_latex(float_format='%.1f'), pdf.to_latex(float_format='%.1f'))
        self.assert_eq(kdf.to_latex(sparsify=False), pdf.to_latex(sparsify=False))
        self.assert_eq(kdf.to_latex(index_names=False), pdf.to_latex(index_names=False))
        self.assert_eq(kdf.to_latex(bold_rows=True), pdf.to_latex(bold_rows=True))
        self.assert_eq(kdf.to_latex(encoding='ascii'), pdf.to_latex(encoding='ascii'))
        self.assert_eq(kdf.to_latex(decimal=','), pdf.to_latex(decimal=','))
