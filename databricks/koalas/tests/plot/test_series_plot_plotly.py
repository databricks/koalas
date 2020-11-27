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
import unittest
from distutils.version import LooseVersion

import pandas as pd

from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


@unittest.skipIf(
    LooseVersion(pd.__version__) < "1.0.0",
    "pandas<1.0 does not support latest plotly and/or 'plotting.backend' option.",
)
class SeriesPlotPlotlyTest(ReusedSQLTestCase, TestUtils):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        pd.set_option("plotting.backend", "plotly")
        set_option("plotting.backend", "plotly")
        set_option("plotting.max_rows", 1000)
        set_option("plotting.sample_ratio", None)

    @classmethod
    def tearDownClass(cls):
        pd.reset_option("plotting.backend")
        reset_option("plotting.backend")
        reset_option("plotting.max_rows")
        reset_option("plotting.sample_ratio")
        super().tearDownClass()

    @property
    def pdf1(self):
        return pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50],}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10]
        )

    @property
    def kdf1(self):
        return ks.from_pandas(self.pdf1)

    @property
    def kdf2(self):
        return ks.range(1002)

    @property
    def pdf2(self):
        return self.kdf2.to_pandas()

    def test_bar_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        self.assertEqual(pdf["a"].plot(kind="bar"), kdf["a"].plot(kind="bar"))
        self.assertEqual(pdf["a"].plot.bar(), kdf["a"].plot.bar())

    def test_line_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        self.assertEqual(pdf["a"].plot(kind="line"), kdf["a"].plot(kind="line"))
        self.assertEqual(pdf["a"].plot.line(), kdf["a"].plot.line())

    def test_barh_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        self.assertEqual(pdf["a"].plot(kind="barh"), kdf["a"].plot(kind="barh"))

    def test_area_plot(self):
        pdf = pd.DataFrame(
            {
                "sales": [3, 2, 3, 9, 10, 6],
                "signups": [5, 5, 6, 12, 14, 13],
                "visits": [20, 42, 28, 62, 81, 50],
            },
            index=pd.date_range(start="2018/01/01", end="2018/07/01", freq="M"),
        )
        kdf = ks.from_pandas(pdf)

        self.assertEqual(pdf["sales"].plot(kind="area"), kdf["sales"].plot(kind="area"))
        self.assertEqual(pdf["sales"].plot.area(), kdf["sales"].plot.area())

        # just a sanity check for df.col type
        self.assertEqual(pdf.sales.plot(kind="area"), kdf.sales.plot(kind="area"))
