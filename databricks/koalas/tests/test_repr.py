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
import numpy as np

from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option, option_context
from databricks.koalas.testing.utils import ReusedSQLTestCase


class ReprTest(ReusedSQLTestCase):
    max_display_count = 23

    @classmethod
    def setUpClass(cls):
        super(ReprTest, cls).setUpClass()
        set_option("display.max_rows", ReprTest.max_display_count)

    @classmethod
    def tearDownClass(cls):
        reset_option("display.max_rows")
        super(ReprTest, cls).tearDownClass()

    def test_repr_dataframe(self):
        kdf = ks.range(ReprTest.max_display_count)
        self.assertTrue("Showing only the first" not in repr(kdf))
        self.assert_eq(repr(kdf), repr(kdf.to_pandas()))

        kdf = ks.range(ReprTest.max_display_count + 1)
        self.assertTrue("Showing only the first" in repr(kdf))

        with option_context("display.max_rows", None):
            kdf = ks.range(ReprTest.max_display_count + 1)
            self.assert_eq(repr(kdf), repr(kdf.to_pandas()))

    def test_repr_series(self):
        kser = ks.range(ReprTest.max_display_count).id
        self.assertTrue("Showing only the first" not in repr(kser))
        self.assert_eq(repr(kser), repr(kser.to_pandas()))

        kser = ks.range(ReprTest.max_display_count + 1).id
        self.assertTrue("Showing only the first" in repr(kser))

        with option_context("display.max_rows", None):
            kser = ks.range(ReprTest.max_display_count + 1).id
            self.assert_eq(repr(kser), repr(kser.to_pandas()))

    def test_repr_indexes(self):
        kdf = ks.range(ReprTest.max_display_count)
        kidx = kdf.index
        self.assertTrue("Showing only the first" not in repr(kidx))
        self.assert_eq(repr(kidx), repr(kidx.to_pandas()))

        kdf = ks.range(ReprTest.max_display_count + 1)
        kidx = kdf.index
        self.assertTrue("Showing only the first" in repr(kidx))

        with option_context("display.max_rows", None):
            kdf = ks.range(ReprTest.max_display_count + 1)
            kidx = kdf.index
            self.assert_eq(repr(kidx), repr(kidx.to_pandas()))

    def test_html_repr(self):
        kdf = ks.range(ReprTest.max_display_count)
        self.assertTrue("Showing only the first" not in kdf._repr_html_())
        self.assertEqual(kdf._repr_html_(), kdf.to_pandas()._repr_html_())

        kdf = ks.range(ReprTest.max_display_count + 1)
        self.assertTrue("Showing only the first" in kdf._repr_html_())

        with option_context("display.max_rows", None):
            kdf = ks.range(ReprTest.max_display_count + 1)
            self.assertEqual(kdf._repr_html_(), kdf.to_pandas()._repr_html_())

    def test_repr_float_index(self):
        kdf = ks.DataFrame(
            {"a": np.random.rand(ReprTest.max_display_count)},
            index=np.random.rand(ReprTest.max_display_count),
        )
        self.assertTrue("Showing only the first" not in repr(kdf))
        self.assert_eq(repr(kdf), repr(kdf.to_pandas()))
        self.assertTrue("Showing only the first" not in repr(kdf.a))
        self.assert_eq(repr(kdf.a), repr(kdf.a.to_pandas()))
        self.assertTrue("Showing only the first" not in repr(kdf.index))
        self.assert_eq(repr(kdf.index), repr(kdf.index.to_pandas()))

        self.assertTrue("Showing only the first" not in kdf._repr_html_())
        self.assertEqual(kdf._repr_html_(), kdf.to_pandas()._repr_html_())

        kdf = ks.DataFrame(
            {"a": np.random.rand(ReprTest.max_display_count + 1)},
            index=np.random.rand(ReprTest.max_display_count + 1),
        )
        self.assertTrue("Showing only the first" in repr(kdf))
        self.assertTrue("Showing only the first" in repr(kdf.a))
        self.assertTrue("Showing only the first" in repr(kdf.index))
        self.assertTrue("Showing only the first" in kdf._repr_html_())
