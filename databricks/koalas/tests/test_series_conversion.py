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
        ps = pd.Series([1, 2, 3, 4, 5, 6, 7], name='x')
        ks = koalas.from_pandas(ps)

        result = ks.to_latex()
        expected = """\\begin{tabular}{lr}
\\toprule
{} &  x \\\\
\\midrule
0 &  1 \\\\
1 &  2 \\\\
2 &  3 \\\\
3 &  4 \\\\
4 &  5 \\\\
5 &  6 \\\\
6 &  7 \\\\
\\bottomrule
\\end{tabular}
"""
        self.assert_eq(expected, result)
