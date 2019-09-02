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

from databricks import koalas
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils
from databricks.koalas.plot import KoalasHistPlotSummary, KoalasBoxPlotSummary


matplotlib.use('agg')


class SeriesPlotTest(ReusedSQLTestCase, TestUtils):

    @classmethod
    def setUpClass(cls):
        super(SeriesPlotTest, cls).setUpClass()
        set_option('plotting.max_rows', 1000)

    @classmethod
    def tearDownClass(cls):
        super(SeriesPlotTest, cls).tearDownClass()
        reset_option('plotting.max_rows')

    @property
    def pdf1(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50],
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10])

    @property
    def kdf1(self):
        return koalas.from_pandas(self.pdf1)

    @property
    def kdf2(self):
        return koalas.range(1002)

    @property
    def pdf2(self):
        return self.kdf2.to_pandas()

    @staticmethod
    def plot_to_base64(ax):
        bytes_data = BytesIO()
        ax.figure.savefig(bytes_data, format='png')
        bytes_data.seek(0)
        b64_data = base64.b64encode(bytes_data.read())
        plt.close(ax.figure)
        return b64_data

    def compare_plots(self, ax1, ax2):
        self.assert_eq(self.plot_to_base64(ax1), self.plot_to_base64(ax2))

    def test_bar_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf['a'].plot("bar", colormap='Paired')
        ax2 = kdf['a'].plot("bar", colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax1 = pdf['a'].plot(kind='bar', colormap='Paired')
        ax2 = kdf['a'].plot(kind='bar', colormap='Paired')
        self.compare_plots(ax1, ax2)

    def test_bar_plot_limited(self):
        pdf = self.pdf2
        kdf = self.kdf2

        _, ax1 = plt.subplots(1, 1)
        ax1 = pdf['id'][:1000].plot.bar(colormap='Paired')
        ax1.text(1, 1, 'showing top 1,000 elements only', size=6, ha='right', va='bottom',
                 transform=ax1.transAxes)
        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf['id'].plot.bar(colormap='Paired')

        self.compare_plots(ax1, ax2)

    def test_pie_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf['a'].plot.pie(colormap='Paired')
        ax2 = kdf['a'].plot.pie(colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax1 = pdf['a'].plot(kind='pie', colormap='Paired')
        ax2 = kdf['a'].plot(kind='pie', colormap='Paired')
        self.compare_plots(ax1, ax2)

    def test_pie_plot_limited(self):
        pdf = self.pdf2
        kdf = self.kdf2

        _, ax1 = plt.subplots(1, 1)
        ax1 = pdf['id'][:1000].plot.pie(colormap='Paired')
        ax1.text(1, 1, 'showing top 1000 elements only', size=6, ha='right', va='bottom',
                 transform=ax1.transAxes)
        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf['id'].plot.pie(colormap='Paired')
        self.compare_plots(ax1, ax2)

    def test_line_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf['a'].plot("line", colormap='Paired')
        ax2 = kdf['a'].plot("line", colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax1 = pdf['a'].plot.line(colormap='Paired')
        ax2 = kdf['a'].plot.line(colormap='Paired')
        self.compare_plots(ax1, ax2)

    def test_barh_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf['a'].plot("barh", colormap='Paired')
        ax2 = kdf['a'].plot("barh", colormap='Paired')
        self.compare_plots(ax1, ax2)

    def test_barh_plot_limited(self):
        pdf = self.pdf2
        kdf = self.kdf2

        _, ax1 = plt.subplots(1, 1)
        ax1 = pdf['id'][:1000].plot.barh(colormap='Paired')
        ax1.text(1, 1, 'showing top 1,000 elements only', size=6, ha='right', va='bottom',
                 transform=ax1.transAxes)
        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf['id'].plot.barh(colormap='Paired')

        self.compare_plots(ax1, ax2)

    def test_hist_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        _, ax1 = plt.subplots(1, 1)
        ax1 = pdf['a'].plot.hist()
        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf['a'].plot.hist()
        self.compare_plots(ax1, ax2)

        ax1 = pdf['a'].plot.hist(bins=15)
        ax2 = kdf['a'].plot.hist(bins=15)
        self.compare_plots(ax1, ax2)

        ax1 = pdf['a'].plot(kind='hist', bins=15)
        ax2 = kdf['a'].plot(kind='hist', bins=15)
        self.compare_plots(ax1, ax2)

        ax1 = pdf['a'].plot.hist(bins=3, bottom=[2, 1, 3])
        ax2 = kdf['a'].plot.hist(bins=3, bottom=[2, 1, 3])
        self.compare_plots(ax1, ax2)

    def test_hist_summary(self):
        kdf = self.kdf1
        summary = KoalasHistPlotSummary(kdf['a'], 'a')

        expected_bins = np.linspace(1, 50, 11)
        bins = summary.get_bins(10)

        expected_histogram = np.array([5, 4, 1, 0, 0, 0, 0, 0, 0, 1])
        histogram = summary.calc_histogram(bins)['__a_bucket']
        self.assert_eq(pd.Series(expected_bins), pd.Series(bins))
        self.assert_eq(pd.Series(expected_histogram), histogram)

    def test_area_plot(self):
        pdf = pd.DataFrame({
            'sales': [3, 2, 3, 9, 10, 6],
            'signups': [5, 5, 6, 12, 14, 13],
            'visits': [20, 42, 28, 62, 81, 50],
        }, index=pd.date_range(start='2018/01/01', end='2018/07/01', freq='M'))
        kdf = koalas.from_pandas(pdf)

        ax1 = pdf['sales'].plot("area", colormap='Paired')
        ax2 = kdf['sales'].plot("area", colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax1 = pdf['sales'].plot.area(colormap='Paired')
        ax2 = kdf['sales'].plot.area(colormap='Paired')
        self.compare_plots(ax1, ax2)

        # just a sanity check for df.col type
        ax1 = pdf.sales.plot("area", colormap='Paired')
        ax2 = kdf.sales.plot("area", colormap='Paired')
        self.compare_plots(ax1, ax2)

    def boxplot_comparison(self, *args, **kwargs):
        pdf = self.pdf1
        kdf = self.kdf1

        _, ax1 = plt.subplots(1, 1)
        ax1 = pdf['a'].plot.box(*args, **kwargs)
        _, ax2 = plt.subplots(1, 1)
        ax2 = kdf['a'].plot.box(*args, **kwargs)

        diffs = [np.array([0, .5, 0, .5, 0, -.5, 0, -.5, 0, .5]),
                 np.array([0, .5, 0, 0]),
                 np.array([0, -.5, 0, 0])]

        for i, (line1, line2) in enumerate(zip(ax1.get_lines(), ax2.get_lines())):
            expected = line1.get_xydata().ravel()
            actual = line2.get_xydata().ravel()
            if i < 3:
                actual += diffs[i]
            self.assert_eq(pd.Series(expected), pd.Series(actual))

    def test_box_plot(self):
        self.boxplot_comparison()
        self.boxplot_comparison(showfliers=True)
        self.boxplot_comparison(sym='')
        self.boxplot_comparison(sym='.', color='r')
        self.boxplot_comparison(use_index=False, labels=['Test'])
        self.boxplot_comparison(usermedians=[2.0])
        self.boxplot_comparison(conf_intervals=[(1.0, 3.0)])

        val = (1, 3)
        self.assertRaises(ValueError, lambda: self.boxplot_comparison(usermedians=[2.0, 3.0]))
        self.assertRaises(ValueError, lambda: self.boxplot_comparison(conf_intervals=[val, val]))
        self.assertRaises(ValueError, lambda: self.boxplot_comparison(conf_intervals=[(1,)]))

    def test_box_summary(self):
        kdf = self.kdf1
        pdf = self.pdf1
        k = 1.5

        summary = KoalasBoxPlotSummary(kdf['a'], 'a')
        stats, fences = summary.compute_stats(whis=k, precision=0.01)
        outliers = summary.outliers(*fences)
        whiskers = summary.calc_whiskers(outliers)
        fliers = summary.get_fliers(outliers)

        expected_mean = pdf['a'].mean()
        expected_median = pdf['a'].median()
        expected_q1 = np.percentile(pdf['a'], 25)
        expected_q3 = np.percentile(pdf['a'], 75)
        iqr = (expected_q3 - expected_q1)
        expected_fences = (expected_q1 - k * iqr, expected_q3 + k * iqr)
        pdf['outlier'] = ~pdf['a'].between(fences[0], fences[1])
        expected_whiskers = pdf.query('not outlier')['a'].min(), pdf.query('not outlier')['a'].max()
        expected_fliers = pdf.query('outlier')['a'].values

        self.assert_eq(expected_mean, stats['mean'])
        self.assert_eq(expected_median, stats['med'])
        self.assert_eq(expected_q1, stats['q1'] + .5)
        self.assert_eq(expected_q3, stats['q3'] - .5)
        self.assert_eq(expected_fences[0], fences[0] + 2.0)
        self.assert_eq(expected_fences[1], fences[1] - 2.0)
        self.assert_eq(expected_whiskers[0], whiskers[0])
        self.assert_eq(expected_whiskers[1], whiskers[1])
        self.assert_eq(expected_fliers, fliers)

    def test_missing(self):
        ks = self.kdf1['a']

        unsupported_functions = ['kde']
        for name in unsupported_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Series.*{}.*not implemented".format(name)):
                getattr(ks.plot, name)()
