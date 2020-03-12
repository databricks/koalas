#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from distutils.version import LooseVersion

import numpy as np
import pandas as pd

import databricks.koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils
from databricks.koalas.window import Expanding


class ExpandingTest(ReusedSQLTestCase, TestUtils):
    def _test_expanding_func(self, f):
        kser = ks.Series([1, 2, 3], index=np.random.rand(3))
        pser = kser.to_pandas()
        self.assert_eq(repr(getattr(kser.expanding(2), f)()), repr(getattr(pser.expanding(2), f)()))

        # Multiindex
        kser = ks.Series(
            [1, 2, 3], index=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y"), ("b", "z")])
        )
        pser = kser.to_pandas()
        self.assert_eq(repr(getattr(kser.expanding(2), f)()), repr(getattr(pser.expanding(2), f)()))

        kdf = ks.DataFrame({"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0]})
        pdf = kdf.to_pandas()
        self.assert_eq(repr(getattr(kdf.expanding(2), f)()), repr(getattr(pdf.expanding(2), f)()))

        # Multiindex column
        kdf = ks.DataFrame({"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0]}, index=np.random.rand(4))
        kdf.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
        pdf = kdf.to_pandas()
        self.assert_eq(repr(getattr(kdf.expanding(2), f)()), repr(getattr(pdf.expanding(2), f)()))

    def test_expanding_error(self):
        with self.assertRaisesRegex(ValueError, "min_periods must be >= 0"):
            ks.range(10).expanding(-1)

        with self.assertRaisesRegex(
            TypeError, "kdf_or_kser must be a series or dataframe; however, got:.*int"
        ):
            Expanding(1, 2)

    def test_expanding_repr(self):
        self.assertEqual(repr(ks.range(10).expanding(5)), "Expanding [min_periods=5]")

    def test_expanding_count(self):
        # The behaviour of Expanding.count are different between pandas>=1.0.0 and lower,
        # and we're following the behaviour of latest version of pandas.
        if LooseVersion(pd.__version__) >= LooseVersion("1.0.0"):
            self._test_expanding_func("count")
        else:
            # Series
            kser = ks.Series([1, 2, 3], index=np.random.rand(3))
            expected_result = ks.Series([None, 2.0, 3.0], index=kser.index.to_pandas())
            self.assert_eq(
                repr(kser.expanding(2).count().sort_index()), repr(expected_result.sort_index())
            )
            # MultiIndex
            kser = ks.Series(
                [1, 2, 3], index=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y"), ("b", "z")])
            )
            expected_result = ks.Series([None, 2.0, 3.0], index=kser.index.to_pandas())
            self.assert_eq(
                repr(kser.expanding(2).count().sort_index()), repr(expected_result.sort_index())
            )

            # DataFrame
            kdf = ks.DataFrame({"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0]})
            expected_result = ks.DataFrame({"a": [None, 2.0, 3.0, 4.0], "b": [None, 2.0, 3.0, 4.0]})
            self.assert_eq(
                repr(kdf.expanding(2).count().sort_index()), repr(expected_result.sort_index())
            )

            # MultiIndex columns
            kdf = ks.DataFrame(
                {"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0]}, index=np.random.rand(4)
            )
            kdf.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
            expected_result = ks.DataFrame(
                {"a": [None, 2.0, 3.0, 4.0], "b": [None, 2.0, 3.0, 4.0]},
                index=kdf.index.to_pandas(),
            )
            expected_result.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
            self.assert_eq(
                repr(kdf.expanding(2).count().sort_index()), repr(expected_result.sort_index())
            )

    def test_expanding_min(self):
        self._test_expanding_func("min")

    def test_expanding_max(self):
        self._test_expanding_func("max")

    def test_expanding_mean(self):
        self._test_expanding_func("mean")

    def test_expanding_sum(self):
        self._test_expanding_func("sum")

    def test_expanding_std(self):
        self._test_expanding_func("std")

    def test_expanding_var(self):
        self._test_expanding_func("var")

    def _test_groupby_expanding_func(self, f):
        kser = ks.Series([1, 2, 3], index=np.random.rand(3))
        pser = kser.to_pandas()
        self.assert_eq(
            repr(getattr(kser.groupby(kser).expanding(2), f)().sort_index()),
            repr(getattr(pser.groupby(pser).expanding(2), f)().sort_index()),
        )

        # Multiindex
        kser = ks.Series(
            [1, 2, 3], index=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y"), ("b", "z")])
        )
        pser = kser.to_pandas()
        self.assert_eq(
            repr(getattr(kser.groupby(kser).expanding(2), f)().sort_index()),
            repr(getattr(pser.groupby(pser).expanding(2), f)().sort_index()),
        )

        kdf = ks.DataFrame({"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0]})
        pdf = kdf.to_pandas()
        self.assert_eq(
            repr(getattr(kdf.groupby(kdf.a).expanding(2), f)().sort_index()),
            repr(getattr(pdf.groupby(pdf.a).expanding(2), f)().sort_index()),
        )

        # Multiindex column
        kdf = ks.DataFrame({"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0]})
        kdf.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
        pdf = kdf.to_pandas()
        self.assert_eq(
            repr(getattr(kdf.groupby(("a", "x")).expanding(2), f)().sort_index()),
            repr(getattr(pdf.groupby(("a", "x")).expanding(2), f)().sort_index()),
        )

        self.assert_eq(
            repr(getattr(kdf.groupby([("a", "x"), ("a", "y")]).expanding(2), f)().sort_index()),
            repr(getattr(pdf.groupby([("a", "x"), ("a", "y")]).expanding(2), f)().sort_index()),
        )

    def test_groupby_expanding_count(self):
        # The behaviour of ExpandingGroupby.count are different between pandas>=1.0.0 and lower,
        # and we're following the behaviour of latest version of pandas.
        if LooseVersion(pd.__version__) >= LooseVersion("1.0.0"):
            self._test_groupby_expanding_func("count")
        else:
            # Series
            kser = ks.Series([1, 2, 3], index=np.random.rand(3))
            midx = pd.MultiIndex.from_tuples(
                list(zip(kser.to_pandas().values, kser.index.to_pandas().values))
            )
            expected_result = ks.Series([np.nan, np.nan, np.nan], index=midx)
            self.assert_eq(
                kser.groupby(kser).expanding(2).count().sort_index(),
                expected_result.sort_index(),
                almost=True,
            )
            # MultiIndex
            kser = ks.Series(
                [1, 2, 3], index=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y"), ("b", "z")])
            )
            midx = pd.MultiIndex.from_tuples([(1, "a", "x"), (2, "a", "y"), (3, "b", "z")])
            expected_result = ks.Series([np.nan, np.nan, np.nan], index=midx)
            self.assert_eq(
                kser.groupby(kser).expanding(2).count().sort_index(),
                expected_result.sort_index(),
                almost=True,
            )
            # DataFrame
            kdf = ks.DataFrame({"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0]})
            midx = pd.MultiIndex.from_tuples([(1, 0), (2, 1), (2, 3), (3, 2)])
            expected_result = ks.DataFrame(
                {"a": [None, None, 2.0, None], "b": [None, None, 2.0, None]}, index=midx
            )
            self.assert_eq(
                kdf.groupby(kdf.a).expanding(2).count().sort_index(),
                expected_result.sort_index(),
                almost=True,
            )
            # MultiIndex column
            kdf = ks.DataFrame({"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0]})
            kdf.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
            midx = pd.MultiIndex.from_tuples([(1, 0), (2, 1), (2, 3), (3, 2)])
            expected_result = ks.DataFrame(
                {"a": [None, None, 2.0, None], "b": [None, None, 2.0, None]}, index=midx
            )
            expected_result.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
            self.assert_eq(
                kdf.groupby(("a", "x")).expanding(2).count().sort_index(),
                expected_result.sort_index(),
                almost=True,
            )
            midx = pd.MultiIndex.from_tuples([(1, 4.0, 0), (2, 1.0, 3), (2, 2.0, 1), (3, 3.0, 2)])
            expected_result = ks.DataFrame(
                {"a": [np.nan, np.nan, np.nan, np.nan], "b": [np.nan, np.nan, np.nan, np.nan]},
                index=midx,
            )
            expected_result.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
            self.assert_eq(
                kdf.groupby([("a", "x"), ("a", "y")]).expanding(2).count().sort_index(),
                expected_result.sort_index(),
                almost=True,
            )

    def test_groupby_expanding_min(self):
        self._test_groupby_expanding_func("min")

    def test_groupby_expanding_max(self):
        self._test_groupby_expanding_func("max")

    def test_groupby_expanding_mean(self):
        self._test_groupby_expanding_func("mean")

    def test_groupby_expanding_sum(self):
        self._test_groupby_expanding_func("sum")

    def test_groupby_expanding_std(self):
        self._test_groupby_expanding_func("std")

    def test_groupby_expanding_var(self):
        self._test_groupby_expanding_func("var")
