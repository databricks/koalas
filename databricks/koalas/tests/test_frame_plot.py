import base64
from io import BytesIO

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from databricks import koalas
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.plot import TopNPlot, SampledPlot
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


matplotlib.use('agg')


class DataFramePlotTest(ReusedSQLTestCase, TestUtils):
    sample_ratio_default = None

    @classmethod
    def setUpClass(cls):
        super(DataFramePlotTest, cls).setUpClass()
        set_option('plotting.max_rows', 2000)
        set_option('plotting.sample_ratio', None)

    @classmethod
    def tearDownClass(cls):
        reset_option('plotting.max_rows')
        reset_option('plotting.sample_ratio')
        super(DataFramePlotTest, cls).tearDownClass()

    @property
    def pdf1(self):
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50],
            'b': [2, 3, 4, 5, 7, 9, 10, 15, 34, 45, 49]
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10])

    @property
    def kdf1(self):
        return koalas.from_pandas(self.pdf1)

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

    def test_line_plot(self):

        def _test_line_plot(pdf, kdf):
            ax1 = pdf.plot(kind="line", colormap='Paired')
            ax2 = kdf.plot(kind="line", colormap='Paired')
            self.compare_plots(ax1, ax2)

            ax3 = pdf.plot.line(colormap='Paired')
            ax4 = kdf.plot.line(colormap='Paired')
            self.compare_plots(ax3, ax4)

        pdf = self.pdf1
        kdf = self.kdf1
        _test_line_plot(pdf, kdf)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('y', 'b')])
        pdf.columns = columns
        kdf.columns = columns
        _test_line_plot(pdf, kdf)

    def test_area_plot(self):

        def _test_are_plot(pdf, kdf):

            ax1 = pdf.plot(kind="area", colormap='Paired')
            ax2 = kdf.plot(kind="area", colormap='Paired')
            self.compare_plots(ax1, ax2)

            ax3 = pdf.plot.area(colormap='Paired')
            ax4 = kdf.plot.area(colormap='Paired')
            self.compare_plots(ax3, ax4)

        pdf = self.pdf1
        kdf = self.kdf1
        _test_are_plot(pdf, kdf)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('y', 'b')])
        pdf.columns = columns
        kdf.columns = columns
        _test_are_plot(pdf, kdf)

    def test_area_plot_stacked_false(self):

        def _test_area_plot_stacked_false(pdf, kdf):
            ax1 = pdf.plot.area(stacked=False)
            ax2 = kdf.plot.area(stacked=False)
            self.compare_plots(ax1, ax2)

            # test if frame area plot is correct when stacked=False because default is True
        pdf = pd.DataFrame({
            'sales': [3, 2, 3, 9, 10, 6],
            'signups': [5, 5, 6, 12, 14, 13],
            'visits': [20, 42, 28, 62, 81, 50],
        }, index=pd.date_range(start='2018/01/01', end='2018/07/01', freq='M'))
        kdf = koalas.from_pandas(pdf)
        _test_area_plot_stacked_false(pdf, kdf)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'sales'), ('x', 'signups'), ('y', 'visits')])
        pdf.columns = columns
        kdf.columns = columns
        _test_area_plot_stacked_false(pdf, kdf)

    def test_area_plot_y(self):

        def _test_area_plot_y(pdf, kdf, y):
            ax1 = pdf.plot.area(y=y)
            ax2 = kdf.plot.area(y=y)
            self.compare_plots(ax1, ax2)

        # test if frame area plot is correct when y is specified
        pdf = pd.DataFrame({
            'sales': [3, 2, 3, 9, 10, 6],
            'signups': [5, 5, 6, 12, 14, 13],
            'visits': [20, 42, 28, 62, 81, 50],
        }, index=pd.date_range(start='2018/01/01', end='2018/07/01', freq='M'))
        kdf = koalas.from_pandas(pdf)
        _test_area_plot_y(pdf, kdf, y='sales')

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'sales'), ('x', 'signups'), ('y', 'visits')])
        pdf.columns = columns
        kdf.columns = columns
        _test_area_plot_y(pdf, kdf, y=('x', 'sales'))

    def test_barh_plot_with_x_y(self):

        def _test_barh_plot_with_x_y(pdf, kdf, x, y):
            ax1 = pdf.plot(kind="barh", x=x, y=y, colormap='Paired')
            ax2 = kdf.plot(kind="barh", x=x, y=y, colormap='Paired')
            self.compare_plots(ax1, ax2)

            ax3 = pdf.plot.barh(x=x, y=y, colormap='Paired')
            ax4 = kdf.plot.barh(x=x, y=y, colormap='Paired')
            self.compare_plots(ax3, ax4)

        # this is testing plot with specified x and y
        pdf = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
        kdf = koalas.from_pandas(pdf)
        _test_barh_plot_with_x_y(pdf, kdf, x='lab', y='val')

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'lab'), ('y', 'val')])
        pdf.columns = columns
        kdf.columns = columns
        _test_barh_plot_with_x_y(pdf, kdf, x=('x', 'lab'), y=('y', 'val'))

    def test_barh_plot(self):

        def _test_barh_plot(pdf, kdf):
            ax1 = pdf.plot(kind="barh", colormap='Paired')
            ax2 = kdf.plot(kind="barh", colormap='Paired')
            self.compare_plots(ax1, ax2)

            ax3 = pdf.plot.barh(colormap='Paired')
            ax4 = kdf.plot.barh(colormap='Paired')
            self.compare_plots(ax3, ax4)

        # this is testing when x or y is not assigned
        pdf = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
        kdf = koalas.from_pandas(pdf)
        _test_barh_plot(pdf, kdf)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'lab'), ('y', 'val')])
        pdf.columns = columns
        kdf.columns = columns
        _test_barh_plot(pdf, kdf)

    def test_bar_plot(self):

        def _test_bar_plot(pdf, kdf):
            ax1 = pdf.plot(kind='bar', colormap='Paired')
            ax2 = kdf.plot(kind='bar', colormap='Paired')
            self.compare_plots(ax1, ax2)

            ax3 = pdf.plot.bar(colormap='Paired')
            ax4 = kdf.plot.bar(colormap='Paired')
            self.compare_plots(ax3, ax4)

        pdf = self.pdf1
        kdf = self.kdf1
        _test_bar_plot(pdf, kdf)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'lab'), ('y', 'val')])
        pdf.columns = columns
        kdf.columns = columns
        _test_bar_plot(pdf, kdf)

    def test_bar_with_x_y(self):
        # this is testing plot with specified x and y
        pdf = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
        kdf = koalas.from_pandas(pdf)

        ax1 = pdf.plot(kind="bar", x='lab', y='val', colormap='Paired')
        ax2 = kdf.plot(kind="bar", x='lab', y='val', colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax3 = pdf.plot.bar(x='lab', y='val', colormap='Paired')
        ax4 = kdf.plot.bar(x='lab', y='val', colormap='Paired')
        self.compare_plots(ax3, ax4)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'lab'), ('y', 'val')])
        pdf.columns = columns
        kdf.columns = columns

        ax5 = pdf.plot(kind="bar", x=('x', 'lab'), y=('y', 'val'), colormap='Paired')
        ax6 = kdf.plot(kind="bar", x=('x', 'lab'), y=('y', 'val'), colormap='Paired')
        self.compare_plots(ax5, ax6)

        ax7 = pdf.plot.bar(x=('x', 'lab'), y=('y', 'val'), colormap='Paired')
        ax8 = kdf.plot.bar(x=('x', 'lab'), y=('y', 'val'), colormap='Paired')
        self.compare_plots(ax7, ax8)

    def test_pie_plot(self):

        def _test_pie_plot(pdf, kdf, y):

            ax1 = pdf.plot.pie(y=y, figsize=(5, 5), colormap='Paired')
            ax2 = kdf.plot.pie(y=y, figsize=(5, 5), colormap='Paired')
            self.compare_plots(ax1, ax2)

            ax1 = pdf.plot(kind="pie", y=y, figsize=(5, 5), colormap='Paired')
            ax2 = kdf.plot(kind="pie", y=y, figsize=(5, 5), colormap='Paired')
            self.compare_plots(ax1, ax2)

            ax11, ax12 = pdf.plot.pie(figsize=(5, 5), subplots=True, colormap='Paired')
            ax21, ax22 = kdf.plot.pie(figsize=(5, 5), subplots=True, colormap='Paired')
            self.compare_plots(ax11, ax21)
            self.compare_plots(ax12, ax22)

            ax11, ax12 = pdf.plot(kind="pie", figsize=(5, 5), subplots=True, colormap='Paired')
            ax21, ax22 = kdf.plot(kind="pie", figsize=(5, 5), subplots=True, colormap='Paired')
            self.compare_plots(ax11, ax21)
            self.compare_plots(ax12, ax22)

        pdf = pd.DataFrame({'mass': [0.330, 4.87, 5.97], 'radius': [2439.7, 6051.8, 6378.1]},
                           index=['Mercury', 'Venus', 'Earth'])
        kdf = koalas.from_pandas(pdf)
        _test_pie_plot(pdf, kdf, y='mass')

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'mass'), ('y', 'radius')])
        pdf.columns = columns
        kdf.columns = columns
        _test_pie_plot(pdf, kdf, y=('x', 'mass'))

    def test_pie_plot_error_message(self):
        # this is to test if error is correctly raising when y is not specified
        # and subplots is not set to True
        pdf = pd.DataFrame({'mass': [0.330, 4.87, 5.97], 'radius': [2439.7, 6051.8, 6378.1]},
                           index=['Mercury', 'Venus', 'Earth'])
        kdf = koalas.from_pandas(pdf)

        with self.assertRaises(ValueError) as context:
            kdf.plot.pie(figsize=(5, 5), colormap='Paired')
        error_message = "pie requires either y column or 'subplots=True'"
        self.assertTrue(error_message in str(context.exception))

    def test_scatter_plot(self):

        def _test_scatter_plot(pdf, kdf, x, y, c):
            ax1 = pdf.plot.scatter(x=x, y=y)
            ax2 = kdf.plot.scatter(x=x, y=y)
            self.compare_plots(ax1, ax2)

            ax1 = pdf.plot(kind='scatter', x=x, y=y)
            ax2 = kdf.plot(kind='scatter', x=x, y=y)
            self.compare_plots(ax1, ax2)

            # check when keyword c is given as name of a column
            ax1 = pdf.plot.scatter(x=x, y=y, c=c, s=50)
            ax2 = kdf.plot.scatter(x=x, y=y, c=c, s=50)
            self.compare_plots(ax1, ax2)

        # Use pandas scatter plot example
        pdf = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
        kdf = koalas.from_pandas(pdf)
        _test_scatter_plot(pdf, kdf, x='a', y='b', c='c')

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c'), ('z', 'd')])
        pdf.columns = columns
        kdf.columns = columns
        _test_scatter_plot(pdf, kdf, x=('x', 'a'), y=('x', 'b'), c=('y', 'c'))

    def test_hist_plot(self):

        def _test_hist_plot(pdf, kdf):
            _, ax1 = plt.subplots(1, 1)
            ax1 = pdf.plot.hist()
            _, ax2 = plt.subplots(1, 1)
            ax2 = kdf.plot.hist()
            self.compare_plots(ax1, ax2)

            ax1 = pdf.plot.hist(bins=15)
            ax2 = kdf.plot.hist(bins=15)
            self.compare_plots(ax1, ax2)

            ax1 = pdf.plot(kind='hist', bins=15)
            ax2 = kdf.plot(kind='hist', bins=15)
            self.compare_plots(ax1, ax2)

            ax1 = pdf.plot.hist(bins=3, bottom=[2, 1, 3])
            ax2 = kdf.plot.hist(bins=3, bottom=[2, 1, 3])
            self.compare_plots(ax1, ax2)

        pdf = self.pdf1
        kdf = self.kdf1
        _test_hist_plot(pdf, kdf)

        # multi-index columns
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('y', 'b')])
        pdf.columns = columns
        kdf.columns = columns
        _test_hist_plot(pdf, kdf)

    def test_missing(self):
        ks = self.kdf1

        unsupported_functions = ['box', 'density', 'hexbin', 'kde']

        for name in unsupported_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*DataFrame.*{}.*not implemented".format(name)):
                getattr(ks.plot, name)()

    def test_topn_max_rows(self):

        pdf = pd.DataFrame(np.random.rand(2500, 4), columns=['a', 'b', 'c', 'd'])
        kdf = koalas.from_pandas(pdf)

        data = TopNPlot().get_top_n(kdf)
        self.assertEqual(len(data), 2000)

    def test_sampled_plot_with_ratio(self):
        set_option('plotting.sample_ratio', 0.5)
        try:
            pdf = pd.DataFrame(np.random.rand(2500, 4), columns=['a', 'b', 'c', 'd'])
            kdf = koalas.from_pandas(pdf)
            data = SampledPlot().get_sampled(kdf)
            self.assertEqual(round(len(data) / 2500, 1), 0.5)
        finally:
            set_option('plotting.sample_ratio', DataFramePlotTest.sample_ratio_default)

    def test_sampled_plot_with_max_rows(self):
        # 'plotting.max_rows' is 2000
        pdf = pd.DataFrame(np.random.rand(2000, 4), columns=['a', 'b', 'c', 'd'])
        kdf = koalas.from_pandas(pdf)
        data = SampledPlot().get_sampled(kdf)
        self.assertEqual(round(len(data) / 2000, 1), 1)
