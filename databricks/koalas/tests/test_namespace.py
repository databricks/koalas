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


class NamespaceTest(ReusedSQLTestCase, SQLTestUtils):

    def test_to_datetime(self):
        pdf = pd.DataFrame({'year': [2015, 2016],
                            'month': [2, 3],
                            'day': [4, 5]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(pd.to_datetime(pdf), ks.to_datetime(kdf))

    def test_concat(self):
        pdf = pd.DataFrame({'A': [0, 2, 4], 'B': [1, 3, 5]})
        kdf = ks.from_pandas(pdf)

        self.assert_eq(
            ks.concat([kdf, kdf.reset_index()]),
            pd.concat([pdf, pdf.reset_index()]))

        self.assert_eq(
            ks.concat([kdf, kdf[['A']]], ignore_index=True),
            pd.concat([pdf, pdf[['A']]], ignore_index=True))

        self.assert_eq(
            ks.concat([kdf, kdf[['A']]], join="inner"),
            pd.concat([pdf, pdf[['A']]], join="inner"))

        self.assert_eq(repr(ks.concat([kdf, kdf.A, kdf.B])),
                       repr(pd.concat([pdf, pdf.A, pdf.B])))

        self.assertRaisesRegex(TypeError, "first argument must be", lambda: ks.concat(kdf))
        self.assertRaisesRegex(
            TypeError, "cannot concatenate object", lambda: ks.concat([kdf, 1]))

        kdf2 = kdf.set_index('B', append=True)
        self.assertRaisesRegex(
            ValueError, "Index type and names should be same", lambda: ks.concat([kdf, kdf2]))

        self.assertRaisesRegex(ValueError, "No objects to concatenate", lambda: ks.concat([]))

        self.assertRaisesRegex(
            ValueError, "All objects passed", lambda: ks.concat([None, None]))

        self.assertRaisesRegex(
            ValueError, 'axis should be either 0 or', lambda: ks.concat([kdf, kdf], axis=1))

        pdf3 = pdf.copy()
        kdf3 = kdf.copy()

        columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B')])
        pdf3.columns = columns
        kdf3.columns = columns

        self.assert_eq(ks.concat([kdf3, kdf3.reset_index()]),
                       pd.concat([pdf3, pdf3.reset_index()]))

        self.assert_eq(
            ks.concat([kdf3, kdf3[[('X', 'A')]]], ignore_index=True),
            pd.concat([pdf3, pdf3[[('X', 'A')]]], ignore_index=True))

        self.assert_eq(
            ks.concat([kdf3, kdf3[[('X', 'A')]]], join="inner"),
            pd.concat([pdf3, pdf3[[('X', 'A')]]], join="inner"))

        self.assertRaisesRegex(ValueError, "MultiIndex columns should have the same levels",
                               lambda: ks.concat([kdf, kdf3]))
