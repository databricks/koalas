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
import pprint

import pandas as pd
import numpy as np
from plotly import express
import plotly.graph_objs as go

from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils
from databricks.koalas.utils import name_like_string


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

    def test_pie_plot(self):
        kdf = self.kdf1
        pdf = kdf.to_pandas()
        self.assertEqual(
            kdf["a"].plot(kind="pie"), express.pie(pdf, values=pdf.columns[0], names=pdf.index),
        )

        # TODO: support multi-index columns
        # columns = pd.MultiIndex.from_tuples([("x", "y")])
        # kdf.columns = columns
        # pdf.columns = columns
        # self.assertEqual(
        #     kdf[("x", "y")].plot(kind="pie"),
        #     express.pie(pdf, values=pdf.iloc[:, 0].to_numpy(), names=pdf.index.to_numpy()),
        # )

        # TODO: support multi-index
        # kdf = ks.DataFrame(
        #     {
        #         "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50],
        #         "b": [2, 3, 4, 5, 7, 9, 10, 15, 34, 45, 49]
        #     },
        #     index=pd.MultiIndex.from_tuples([("x", "y")] * 11),
        # )
        # pdf = kdf.to_pandas()
        # self.assertEqual(
        #     kdf["a"].plot(kind="pie"), express.pie(pdf, values=pdf.columns[0], names=pdf.index),
        # )

    def test_hist_plot(self):
        def check_hist_plot(kser):
            bins = np.array([1.0, 5.9, 10.8, 15.7, 20.6, 25.5, 30.4, 35.3, 40.2, 45.1, 50.0])
            data = np.array([5.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,])
            prev = bins[0]
            text_bins = []
            for b in bins[1:]:
                text_bins.append("[%s, %s)" % (prev, b))
                prev = b
            text_bins[-1] = text_bins[-1][:-1] + "]"
            bins = 0.5 * (bins[:-1] + bins[1:])
            name_a = name_like_string(kser.name)
            bars = [
                go.Bar(
                    x=bins,
                    y=data,
                    name=name_a,
                    text=text_bins,
                    hovertemplate=("variable=" + name_a + "<br>value=%{text}<br>count=%{y}"),
                ),
            ]
            fig = go.Figure(data=bars, layout=go.Layout(barmode="stack"))
            fig["layout"]["xaxis"]["title"] = "value"
            fig["layout"]["yaxis"]["title"] = "count"

            self.assertEqual(
                pprint.pformat(kser.plot(kind="hist").to_dict()), pprint.pformat(fig.to_dict())
            )

        kdf1 = self.kdf1
        check_hist_plot(kdf1["a"])

        columns = pd.MultiIndex.from_tuples([("x", "y")])
        kdf1.columns = columns
        check_hist_plot(kdf1[("x", "y")])

    def test_pox_plot(self):
        def check_pox_plot(kser):
            fig = go.Figure()
            fig.add_trace(
                go.Box(
                    name=name_like_string(kser.name),
                    q1=[3],
                    median=[6],
                    q3=[9],
                    mean=[10.0],
                    lowerfence=[1],
                    upperfence=[15],
                    y=[[50]],
                    boxpoints="suspectedoutliers",
                    notched=False,
                )
            )
            fig["layout"]["xaxis"]["title"] = name_like_string(kser.name)
            fig["layout"]["yaxis"]["title"] = "value"

            self.assertEqual(
                pprint.pformat(kser.plot(kind="box").to_dict()), pprint.pformat(fig.to_dict())
            )

        kdf1 = self.kdf1
        check_pox_plot(kdf1["a"])

        columns = pd.MultiIndex.from_tuples([("x", "y")])
        kdf1.columns = columns
        check_pox_plot(kdf1[("x", "y")])

    def test_pox_plot_arguments(self):
        with self.assertRaisesRegex(ValueError, "does not support"):
            self.kdf1.a.plot.box(boxpoints="all")
        with self.assertRaisesRegex(ValueError, "does not support"):
            self.kdf1.a.plot.box(notched=True)
        self.kdf1.a.plot.box(hovertext="abc")  # other arguments should not throw an exception

    def test_kde_plot(self):
        kdf = ks.DataFrame({"a": [1, 2, 3, 4, 5]})
        pdf = pd.DataFrame(
            {
                "Density": [0.05709372, 0.07670272, 0.05709372],
                "names": ["a", "a", "a"],
                "index": [-1.0, 3.0, 7.0],
            }
        )

        actual = kdf.a.plot.kde(bw_method=5, ind=3)

        expected = express.line(pdf, x="index", y="Density")
        expected["layout"]["xaxis"]["title"] = None

        self.assertEqual(pprint.pformat(actual.to_dict()), pprint.pformat(expected.to_dict()))
