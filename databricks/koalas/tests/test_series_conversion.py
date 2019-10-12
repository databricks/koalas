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

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class SeriesConversionTest(ReusedSQLTestCase, SQLTestUtils):

    @property
    def pser(self):
        return pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')

    @property
    def kser(self):
        return ks.from_pandas(self.pser)

    def test_to_clipboard(self):
        pser = self.pser
        kser = self.kser

        self.assert_eq(kser.to_clipboard(), pser.to_clipboard())
        self.assert_eq(kser.to_clipboard(excel=False),
                       pser.to_clipboard(excel=False))
        self.assert_eq(kser.to_clipboard(sep=',', index=False),
                       pser.to_clipboard(sep=',', index=False))

    def test_to_latex(self):
        pser = self.pser
        kser = self.kser

        self.assert_eq(kser.to_latex(), pser.to_latex())
        self.assert_eq(kser.to_latex(col_space=2), pser.to_latex(col_space=2))
        self.assert_eq(kser.to_latex(header=True), pser.to_latex(header=True))
        self.assert_eq(kser.to_latex(index=False), pser.to_latex(index=False))
        self.assert_eq(kser.to_latex(na_rep='-'), pser.to_latex(na_rep='-'))
        self.assert_eq(kser.to_latex(float_format='%.1f'), pser.to_latex(float_format='%.1f'))
        self.assert_eq(kser.to_latex(sparsify=False), pser.to_latex(sparsify=False))
        self.assert_eq(kser.to_latex(index_names=False), pser.to_latex(index_names=False))
        self.assert_eq(kser.to_latex(bold_rows=True), pser.to_latex(bold_rows=True))
        self.assert_eq(kser.to_latex(encoding='ascii'), pser.to_latex(encoding='ascii'))
        self.assert_eq(kser.to_latex(decimal=','), pser.to_latex(decimal=','))
