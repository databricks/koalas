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

from datetime import datetime

import pandas as pd
from pandas.tseries.offsets import DateOffset

import databricks.koalas as ks
from databricks.koalas.indexes.extension import MapExtension
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


class MapExtensionTest(ReusedSQLTestCase, TestUtils):
    @property
    def kidx(self):
        return ks.Index([1, 2, 3])

    def test_na_action(self):
        with self.assertRaisesRegex(
            NotImplementedError, "Currently do not support na_action functionality"
        ):
            MapExtension(self.kidx, "ignore")

    def test_map_dict(self):
        self.assert_eq(
            MapExtension(self.kidx, None)._map_dict({1: "one", 2: "two", 3: "three"}),
            ks.Index(["one", "two", "three"]),
        )
        self.assert_eq(
            MapExtension(self.kidx, None)._map_dict({1: "one", 2: "two"}),
            ks.Index(["one", "two", "3"]),
        )
        self.assert_eq(
            MapExtension(self.kidx, None)._map_dict({1: 10, 2: 20}), ks.Index([10, 20, 3])
        )

    def test_map_series(self):
        self.assert_eq(
            MapExtension(self.kidx, None)._map_series(
                pd.Series(["one", "2", "three"], index=[1, 2, 3])
            ),
            ks.Index(["one", "2", "three"]),
        )
        self.assert_eq(
            MapExtension(self.kidx, None)._map_series(pd.Series(["one", "2"], index=[1, 2])),
            ks.Index(["one", "2", "3"]),
        )

    def test_map_lambda(self):
        self.assert_eq(
            MapExtension(self.kidx, None)._map_lambda(lambda id: id + 1), ks.Index([2, 3, 4])
        )
        self.assert_eq(
            MapExtension(self.kidx, None)._map_lambda(lambda id: id + 1.1),
            ks.Index([2.1, 3.1, 4.1]),
        )
        self.assert_eq(
            MapExtension(self.kidx, None)._map_lambda(lambda id: "{id} + 1".format(id=id)),
            ks.Index(["1 + 1", "2 + 1", "3 + 1"]),
        )
        kser = ks.Series([1, 2, 3, 4], index=pd.date_range("2018-04-09", periods=4, freq="2D"))
        self.assert_eq(
            MapExtension(kser.index, None)._map_lambda(lambda id: id + DateOffset(days=1)),
            ks.Series([1, 2, 3, 4], index=pd.date_range("2018-04-10", periods=4, freq="2D")).index,
        )

    def test_map(self):
        with self.assertRaisesRegex(
            NotImplementedError, "Currently do not support input of ks.Series in Index.map"
        ):
            MapExtension(self.kidx, None).map(ks.Series(["one", "two", "three"], index=[1, 2, 3]))
