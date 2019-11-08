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

from databricks.koalas.internal import _InternalFrame, SPARK_INDEX_NAME_FORMAT
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils


class InternalFrameTest(ReusedSQLTestCase, SQLTestUtils):

    def test_from_pandas(self):
        pdf = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        internal = _InternalFrame.from_pandas(pdf)
        sdf = internal.sdf

        self.assert_eq(internal.index_map, [(SPARK_INDEX_NAME_FORMAT(0), None)])
        self.assert_eq(internal.column_index, [('a', ), ('b', )])
        self.assert_eq(internal.data_columns, ['a', 'b'])
        self.assertTrue(internal.scol_for(('a',))._jc.equals(sdf['a']._jc))
        self.assertTrue(internal.scol_for(('b',))._jc.equals(sdf['b']._jc))

        self.assert_eq(internal.pandas_df, pdf)

        # multi-index
        pdf.set_index('a', append=True, inplace=True)

        internal = _InternalFrame.from_pandas(pdf)
        sdf = internal.sdf

        self.assert_eq(internal.index_map, [(SPARK_INDEX_NAME_FORMAT(0), None), ('a', ('a',))])
        self.assert_eq(internal.column_index, [('b', )])
        self.assert_eq(internal.data_columns, ['b'])
        self.assertTrue(internal.scol_for(('b',))._jc.equals(sdf['b']._jc))

        self.assert_eq(internal.pandas_df, pdf)

        # multi-index columns
        pdf.columns = pd.MultiIndex.from_tuples([('x', 'b')])

        internal = _InternalFrame.from_pandas(pdf)
        sdf = internal.sdf

        self.assert_eq(internal.index_map, [(SPARK_INDEX_NAME_FORMAT(0), None), ('a', ('a',))])
        self.assert_eq(internal.column_index, [('x', 'b')])
        self.assert_eq(internal.data_columns, ['(x, b)'])
        self.assertTrue(internal.scol_for(('x', 'b'))._jc.equals(sdf['(x, b)']._jc))

        self.assert_eq(internal.pandas_df, pdf)
