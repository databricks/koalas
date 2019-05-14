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
from databricks import koalas as ks
from databricks.koalas.generic import max_display_count
from databricks.koalas.testing.utils import ReusedSQLTestCase


class ReprTests(ReusedSQLTestCase):

    def test_repr_dataframe(self):
        kdf = ks.range(max_display_count)
        self.assertTrue("Showing only the first" not in repr(kdf))
        self.assert_eq(repr(kdf), repr(kdf.to_pandas()))

        kdf = ks.range(max_display_count + 1)
        self.assertTrue("Showing only the first" in repr(kdf))

    def test_repr_series(self):
        kser = ks.range(max_display_count).id
        self.assertTrue("Showing only the first" not in repr(kser))
        self.assert_eq(repr(kser), repr(kser.to_pandas()))

        kser = ks.range(max_display_count + 1).id
        self.assertTrue("Showing only the first" in repr(kser))

    def test_html_repr(self):
        kdf = ks.range(max_display_count)
        self.assertTrue("Showing only the first" not in kdf._repr_html_())
        self.assertEqual(kdf._repr_html_(), kdf.to_pandas()._repr_html_())

        kdf = ks.range(max_display_count + 1)
        self.assertTrue("Showing only the first" in kdf._repr_html_())
