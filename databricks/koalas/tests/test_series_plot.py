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

import base64
from io import BytesIO

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils
from databricks.koalas.plot import KoalasBoxPlot, KoalasHistPlot

matplotlib.use("agg")


class SeriesPlotTest(ReusedSQLTestCase, TestUtils):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        set_option("plotting.max_rows", 1000)

    @classmethod
    def tearDownClass(cls):
        reset_option("plotting.max_rows")
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

    @staticmethod
    def plot_to_base64(ax):
        bytes_data = BytesIO()
        ax.figure.savefig(bytes_data, format="png")
        bytes_data.seek(0)
        b64_data = base64.b64encode(bytes_data.read())
        plt.close(ax.figure)
        return b64_data

    def test_plot_backends(self):
        plot_backend = "plotly"

        with ks.option_context("plotting.backend", plot_backend):
            self.assertEqual(ks.options.plotting.backend, plot_backend)

            module = ks.plot._get_plot_backend(plot_backend)
            self.assertEqual(module.__name__, plot_backend)

    def test_plot_backends_incorrect(self):
        fake_plot_backend = "none_plotting_module"

        with ks.option_context("plotting.backend", fake_plot_backend):
            self.assertEqual(ks.options.plotting.backend, fake_plot_backend)

            with self.assertRaises(ValueError):
                ks.plot._get_plot_backend(fake_plot_backend)

    def test_bar_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf["a"].plot(kind="bar", colormap="Paired")
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["a"].plot(kind="bar", colormap="Paired")
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

        ax1 = pdf["a"].plot(kind="bar", colormap="Paired")
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["a"].plot(kind="bar", colormap="Paired")
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

    def test_bar_plot_limited(self):
        pdf = self.pdf2
        kdf = self.kdf2

        _, ax1 = plt.subplots(1, 1)
        ax1 = pdf["id"][:1000].plot.bar(colormap="Paired")
        ax1.text(
            1,
            1,
            "showing top 1000 elements only",
            size=6,
            ha="right",
            va="bottom",
            transform=ax1.transAxes,
        )
        bin1 = self.plot_to_base64(ax1)

        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf["id"].plot.bar(colormap="Paired")
        bin2 = self.plot_to_base64(ax2)

        self.assertEqual(bin1, bin2)

    def test_pie_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf["a"].plot.pie(colormap="Paired")
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["a"].plot.pie(colormap="Paired")
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

        ax1 = pdf["a"].plot(kind="pie", colormap="Paired")
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["a"].plot(kind="pie", colormap="Paired")
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

    def test_pie_plot_limited(self):
        pdf = self.pdf2
        kdf = self.kdf2

        _, ax1 = plt.subplots(1, 1)
        ax1 = pdf["id"][:1000].plot.pie(colormap="Paired")
        ax1.text(
            1,
            1,
            "showing top 1000 elements only",
            size=6,
            ha="right",
            va="bottom",
            transform=ax1.transAxes,
        )
        bin1 = self.plot_to_base64(ax1)

        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf["id"].plot.pie(colormap="Paired")
        bin2 = self.plot_to_base64(ax2)

        self.assertEqual(bin1, bin2)

    def test_line_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf["a"].plot(kind="line", colormap="Paired")
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["a"].plot(kind="line", colormap="Paired")
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

        ax1 = pdf["a"].plot.line(colormap="Paired")
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["a"].plot.line(colormap="Paired")
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

    def test_barh_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf["a"].plot(kind="barh", colormap="Paired")
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["a"].plot(kind="barh", colormap="Paired")
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

    def test_barh_plot_limited(self):
        pdf = self.pdf2
        kdf = self.kdf2

        _, ax1 = plt.subplots(1, 1)
        ax1 = pdf["id"][:1000].plot.barh(colormap="Paired")
        ax1.text(
            1,
            1,
            "showing top 1000 elements only",
            size=6,
            ha="right",
            va="bottom",
            transform=ax1.transAxes,
        )
        bin1 = self.plot_to_base64(ax1)

        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf["id"].plot.barh(colormap="Paired")
        bin2 = self.plot_to_base64(ax2)

        self.assertEqual(bin1, bin2)

    def test_hist_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        _, ax1 = plt.subplots(1, 1)
        ax1 = pdf["a"].plot.hist()
        bin1 = self.plot_to_base64(ax1)
        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf["a"].plot.hist()
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

        ax1 = pdf["a"].plot.hist(bins=15)
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["a"].plot.hist(bins=15)
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

        ax1 = pdf["a"].plot(kind="hist", bins=15)
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["a"].plot(kind="hist", bins=15)
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

        ax1 = pdf["a"].plot.hist(bins=3, bottom=[2, 1, 3])
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["a"].plot.hist(bins=3, bottom=[2, 1, 3])
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

    def test_compute_hist(self):
        kdf = self.kdf1
        expected_bins = np.linspace(1, 50, 11)
        bins = KoalasHistPlot._get_bins(kdf[["a"]].to_spark(), 10)

        expected_histogram = np.array([5, 4, 1, 0, 0, 0, 0, 0, 0, 1])
        histogram = KoalasHistPlot._compute_hist(kdf[["a"]].to_spark(), bins)
        self.assert_eq(pd.Series(expected_bins), pd.Series(bins))
        self.assert_eq(pd.Series(expected_histogram, name="__a_bucket"), histogram, almost=True)

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

        ax1 = pdf["sales"].plot(kind="area", colormap="Paired")
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["sales"].plot(kind="area", colormap="Paired")
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

        ax1 = pdf["sales"].plot.area(colormap="Paired")
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf["sales"].plot.area(colormap="Paired")
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

        # just a sanity check for df.col type
        ax1 = pdf.sales.plot(kind="area", colormap="Paired")
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf.sales.plot(kind="area", colormap="Paired")
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)

    def test_box_plot(self):
        def check_box_plot(pser, kser, *args, **kwargs):
            _, ax1 = plt.subplots(1, 1)
            ax1 = pser.plot.box(*args, **kwargs)
            _, ax2 = plt.subplots(1, 1)
            ax2 = kser.plot.box(*args, **kwargs)

            diffs = [
                np.array([0, 0.5, 0, 0.5, 0, -0.5, 0, -0.5, 0, 0.5]),
                np.array([0, 0.5, 0, 0]),
                np.array([0, -0.5, 0, 0]),
            ]

            try:
                for i, (line1, line2) in enumerate(zip(ax1.get_lines(), ax2.get_lines())):
                    expected = line1.get_xydata().ravel()
                    actual = line2.get_xydata().ravel()
                    if i < 3:
                        actual += diffs[i]
                    self.assert_eq(pd.Series(expected), pd.Series(actual))
            finally:
                ax1.cla()
                ax2.cla()

        # Non-named Series
        pser = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50], [0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10])
        kser = ks.from_pandas(pser)

        spec = [(self.pdf1.a, self.kdf1.a), (pser, kser)]

        for p, k in spec:
            check_box_plot(p, k)
            check_box_plot(p, k, showfliers=True)
            check_box_plot(p, k, sym="")
            check_box_plot(p, k, sym=".", color="r")
            check_box_plot(p, k, use_index=False, labels=["Test"])
            check_box_plot(p, k, usermedians=[2.0])
            check_box_plot(p, k, conf_intervals=[(1.0, 3.0)])

        val = (1, 3)
        self.assertRaises(
            ValueError, lambda: check_box_plot(self.pdf1, self.kdf1, usermedians=[2.0, 3.0])
        )
        self.assertRaises(
            ValueError, lambda: check_box_plot(self.pdf1, self.kdf1, conf_intervals=[val, val])
        )
        self.assertRaises(
            ValueError, lambda: check_box_plot(self.pdf1, self.kdf1, conf_intervals=[(1,)])
        )

    def test_box_summary(self):
        def check_box_summary(kdf, pdf):
            k = 1.5
            stats, fences = KoalasBoxPlot._compute_stats(kdf["a"], "a", whis=k, precision=0.01)
            outliers = KoalasBoxPlot._outliers(kdf["a"], "a", *fences)
            whiskers = KoalasBoxPlot._calc_whiskers("a", outliers)
            fliers = KoalasBoxPlot._get_fliers("a", outliers, whiskers[0])

            expected_mean = pdf["a"].mean()
            expected_median = pdf["a"].median()
            expected_q1 = np.percentile(pdf["a"], 25)
            expected_q3 = np.percentile(pdf["a"], 75)
            iqr = expected_q3 - expected_q1
            expected_fences = (expected_q1 - k * iqr, expected_q3 + k * iqr)
            pdf["outlier"] = ~pdf["a"].between(fences[0], fences[1])
            expected_whiskers = (
                pdf.query("not outlier")["a"].min(),
                pdf.query("not outlier")["a"].max(),
            )
            expected_fliers = pdf.query("outlier")["a"].values

            self.assertEqual(expected_mean, stats["mean"])
            self.assertEqual(expected_median, stats["med"])
            self.assertEqual(expected_q1, stats["q1"] + 0.5)
            self.assertEqual(expected_q3, stats["q3"] - 0.5)
            self.assertEqual(expected_fences[0], fences[0] + 2.0)
            self.assertEqual(expected_fences[1], fences[1] - 2.0)
            self.assertEqual(expected_whiskers[0], whiskers[0])
            self.assertEqual(expected_whiskers[1], whiskers[1])
            self.assertEqual(expected_fliers, fliers)

        check_box_summary(self.kdf1, self.pdf1)
        check_box_summary(-self.kdf1, -self.pdf1)

    def test_kde_plot(self):
        def moving_average(a, n=10):
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1 :] / n

        def check_kde_plot(pdf, kdf, *args, **kwargs):
            _, ax1 = plt.subplots(1, 1)
            ax1 = pdf["a"].plot.kde(*args, **kwargs)
            _, ax2 = plt.subplots(1, 1)
            ax2 = kdf["a"].plot.kde(*args, **kwargs)

            try:
                for i, (line1, line2) in enumerate(zip(ax1.get_lines(), ax2.get_lines())):
                    expected = line1.get_xydata().ravel()
                    actual = line2.get_xydata().ravel()
                    # TODO: Due to implementation difference, the output is different comparing
                    # to pandas'. We should identify the root cause of difference, and reduce
                    # the diff.

                    # Note: Data is from 1 to 50. So, it smooths them by moving average and compares
                    # both.
                    self.assertTrue(
                        np.allclose(moving_average(actual), moving_average(expected), rtol=3)
                    )
            finally:
                ax1.cla()
                ax2.cla()

        check_kde_plot(self.pdf1, self.kdf1, bw_method=0.3)
        check_kde_plot(self.pdf1, self.kdf1, ind=[1, 2, 3, 4, 5], bw_method=3.0)

    def test_empty_hist(self):
        pdf = self.pdf1.assign(categorical="A")
        kdf = ks.from_pandas(pdf)
        kser = kdf["categorical"]

        with self.assertRaisesRegex(TypeError, "Empty 'DataFrame': no numeric data to plot"):
            kser.plot.hist()

    def test_single_value_hist(self):
        pdf = self.pdf1.assign(single=2)
        kdf = ks.from_pandas(pdf)

        _, ax1 = plt.subplots(1, 1)
        ax1 = pdf["single"].plot.hist()
        bin1 = self.plot_to_base64(ax1)
        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf["single"].plot.hist()
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)
