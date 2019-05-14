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


class SeriesConversionTest(ReusedSQLTestCase, SQLTestUtils):

    @property
    def ps(self):
        return pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    @property
    def ks(self):
        return koalas.from_pandas(self.ps)

    def test_to_clipboard(self):
        ps = self.ps
        ks = self.ks

        self.assert_eq(ks.to_clipboard(), ps.to_clipboard())
        self.assert_eq(ks.to_clipboard(excel=False),
                       ps.to_clipboard(excel=False))
        self.assert_eq(ks.to_clipboard(sep=',', index=False),
                       ps.to_clipboard(sep=',', index=False))

    def test_to_latex(self):
        ps = self.ps
        ks = self.ks

        self.assert_eq(ks.to_latex(), ps.to_latex())
        self.assert_eq(ks.to_latex(col_space=2), ps.to_latex(col_space=2))
        self.assert_eq(ks.to_latex(header=True), ps.to_latex(header=True))
        self.assert_eq(ks.to_latex(index=False), ps.to_latex(index=False))
        self.assert_eq(ks.to_latex(na_rep='-'), ps.to_latex(na_rep='-'))
        self.assert_eq(ks.to_latex(float_format='%.1f'), ps.to_latex(float_format='%.1f'))
        self.assert_eq(ks.to_latex(sparsify=False), ps.to_latex(sparsify=False))
        self.assert_eq(ks.to_latex(index_names=False), ps.to_latex(index_names=False))
        self.assert_eq(ks.to_latex(bold_rows=True), ps.to_latex(bold_rows=True))
        self.assert_eq(ks.to_latex(encoding='ascii'), ps.to_latex(encoding='ascii'))
        self.assert_eq(ks.to_latex(decimal=','), ps.to_latex(decimal=','))
