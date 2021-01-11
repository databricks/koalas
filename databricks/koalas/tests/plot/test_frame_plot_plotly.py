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
import numpy as np
from plotly import express

from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


@unittest.skipIf(
    LooseVersion(pd.__version__) < "1.0.0",
    "pandas<1.0 does not support latest plotly and/or 'plotting.backend' option.",
)
class DataFramePlotPlotlyTest(ReusedSQLTestCase, TestUtils):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        pd.set_option("plotting.backend", "plotly")
        set_option("plotting.backend", "plotly")
        set_option("plotting.max_rows", 2000)
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
            {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50], "b": [2, 3, 4, 5, 7, 9, 10, 15, 34, 45, 49]},
            index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10],
        )

    @property
    def kdf1(self):
        return ks.from_pandas(self.pdf1)

    def test_line_plot(self):
        def check_line_plot(pdf, kdf):
            self.assertEqual(pdf.plot(kind="line"), kdf.plot(kind="line"))
            self.assertEqual(pdf.plot.line(), kdf.plot.line())

        pdf1 = self.pdf1
        kdf1 = self.kdf1
        check_line_plot(pdf1, kdf1)

    def test_area_plot(self):
        def check_area_plot(pdf, kdf):
            self.assertEqual(pdf.plot(kind="area"), kdf.plot(kind="area"))
            self.assertEqual(pdf.plot.area(), kdf.plot.area())

        pdf = self.pdf1
        kdf = self.kdf1
        check_area_plot(pdf, kdf)

    def test_area_plot_y(self):
        def check_area_plot_y(pdf, kdf, y):
            self.assertEqual(pdf.plot.area(y=y), kdf.plot.area(y=y))

        # test if frame area plot is correct when y is specified
        pdf = pd.DataFrame(
            {
                "sales": [3, 2, 3, 9, 10, 6],
                "signups": [5, 5, 6, 12, 14, 13],
                "visits": [20, 42, 28, 62, 81, 50],
            },
            index=pd.date_range(start="2018/01/01", end="2018/07/01", freq="M"),
        )
        kdf = ks.from_pandas(pdf)
        check_area_plot_y(pdf, kdf, y="sales")

    def test_barh_plot_with_x_y(self):
        def check_barh_plot_with_x_y(pdf, kdf, x, y):
            self.assertEqual(pdf.plot(kind="barh", x=x, y=y), kdf.plot(kind="barh", x=x, y=y))
            self.assertEqual(pdf.plot.barh(x=x, y=y), kdf.plot.barh(x=x, y=y))

        # this is testing plot with specified x and y
        pdf1 = pd.DataFrame({"lab": ["A", "B", "C"], "val": [10, 30, 20]})
        kdf1 = ks.from_pandas(pdf1)
        check_barh_plot_with_x_y(pdf1, kdf1, x="lab", y="val")

    def test_barh_plot(self):
        def check_barh_plot(pdf, kdf):
            self.assertEqual(pdf.plot(kind="barh"), kdf.plot(kind="barh"))
            self.assertEqual(pdf.plot.barh(), kdf.plot.barh())

        # this is testing when x or y is not assigned
        pdf1 = pd.DataFrame({"lab": [20.1, 40.5, 60.6], "val": [10, 30, 20]})
        kdf1 = ks.from_pandas(pdf1)
        check_barh_plot(pdf1, kdf1)

    def test_bar_plot(self):
        def check_bar_plot(pdf, kdf):
            self.assertEqual(pdf.plot(kind="bar"), kdf.plot(kind="bar"))
            self.assertEqual(pdf.plot.bar(), kdf.plot.bar())

        pdf1 = self.pdf1
        kdf1 = self.kdf1
        check_bar_plot(pdf1, kdf1)

    def test_bar_with_x_y(self):
        # this is testing plot with specified x and y
        pdf = pd.DataFrame({"lab": ["A", "B", "C"], "val": [10, 30, 20]})
        kdf = ks.from_pandas(pdf)

        self.assertEqual(
            pdf.plot(kind="bar", x="lab", y="val"), kdf.plot(kind="bar", x="lab", y="val")
        )
        self.assertEqual(pdf.plot.bar(x="lab", y="val"), kdf.plot.bar(x="lab", y="val"))

    def test_scatter_plot(self):
        def check_scatter_plot(pdf, kdf, x, y, c):
            self.assertEqual(pdf.plot.scatter(x=x, y=y), kdf.plot.scatter(x=x, y=y))
            self.assertEqual(pdf.plot(kind="scatter", x=x, y=y), kdf.plot(kind="scatter", x=x, y=y))

            # check when keyword c is given as name of a column
            self.assertEqual(
                pdf.plot.scatter(x=x, y=y, c=c, s=50), kdf.plot.scatter(x=x, y=y, c=c, s=50)
            )

        # Use pandas scatter plot example
        pdf1 = pd.DataFrame(np.random.rand(50, 4), columns=["a", "b", "c", "d"])
        kdf1 = ks.from_pandas(pdf1)
        check_scatter_plot(pdf1, kdf1, x="a", y="b", c="c")

    def test_pie_plot(self):
        def check_pie_plot(kdf):
            pdf = kdf.to_pandas()
            self.assertEqual(
                kdf.plot(kind="pie", y=kdf.columns[0]),
                express.pie(pdf, values="a", names=pdf.index),
            )

            self.assertEqual(
                kdf.plot(kind="pie", values="a"), express.pie(pdf, values="a"),
            )

        kdf1 = self.kdf1
        check_pie_plot(kdf1)

        # TODO: support multi-index columns
        # columns = pd.MultiIndex.from_tuples([("x", "y"), ("y", "z")])
        # kdf1.columns = columns
        # check_pie_plot(kdf1)

        # TODO: support multi-index
        # kdf1 = ks.DataFrame(
        #     {
        #         "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50],
        #         "b": [2, 3, 4, 5, 7, 9, 10, 15, 34, 45, 49]
        #     },
        #     index=pd.MultiIndex.from_tuples([("x", "y")] * 11),
        # )
        # check_pie_plot(kdf1)
