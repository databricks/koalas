import base64
from io import BytesIO

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from databricks import koalas
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


matplotlib.use('agg')


class DataFramePlotTest(ReusedSQLTestCase, TestUtils):
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
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf.plot(kind="line", colormap='Paired')
        ax2 = kdf.plot(kind="line", colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax3 = pdf.plot.line(colormap='Paired')
        ax4 = kdf.plot.line(colormap='Paired')
        self.compare_plots(ax3, ax4)

    def test_area_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf.plot(kind="area", colormap='Paired')
        ax2 = kdf.plot(kind="area", colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax3 = pdf.plot.area(colormap='Paired')
        ax4 = kdf.plot.area(colormap='Paired')
        self.compare_plots(ax3, ax4)

    def test_area_plot_stacked_false(self):
        # test if frame area plot is correct when stacked=False because default is True
        pdf = pd.DataFrame({
            'sales': [3, 2, 3, 9, 10, 6],
            'signups': [5, 5, 6, 12, 14, 13],
            'visits': [20, 42, 28, 62, 81, 50],
        }, index=pd.date_range(start='2018/01/01', end='2018/07/01', freq='M'))
        kdf = koalas.from_pandas(pdf)

        ax1 = pdf.plot.area(stacked=False)
        ax2 = kdf.plot.area(stacked=False)
        self.compare_plots(ax1, ax2)

    def test_area_plot_y(self):
        # test if frame area plot is correct when y is specified
        pdf = pd.DataFrame({
            'sales': [3, 2, 3, 9, 10, 6],
            'signups': [5, 5, 6, 12, 14, 13],
            'visits': [20, 42, 28, 62, 81, 50],
        }, index=pd.date_range(start='2018/01/01', end='2018/07/01', freq='M'))
        kdf = koalas.from_pandas(pdf)

        ax1 = pdf.plot.area(y='sales')
        ax2 = kdf.plot.area(y='sales')
        self.compare_plots(ax1, ax2)

    def test_barh_plot_with_x_y(self):
        # this is testing plot with specified x and y
        pdf = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
        kdf = koalas.from_pandas(pdf)

        ax1 = pdf.plot(kind="barh", x='lab', y='val', colormap='Paired')
        ax2 = kdf.plot(kind="barh", x='lab', y='val', colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax3 = pdf.plot.barh(x='lab', y='val', colormap='Paired')
        ax4 = kdf.plot.barh(x='lab', y='val', colormap='Paired')
        self.compare_plots(ax3, ax4)

    def test_barh_plot(self):
        # this is testing when x or y is not assigned
        pdf = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
        kdf = koalas.from_pandas(pdf)

        ax1 = pdf.plot(kind="barh", colormap='Paired')
        ax2 = kdf.plot(kind="barh", colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax3 = pdf.plot.barh(colormap='Paired')
        ax4 = kdf.plot.barh(colormap='Paired')
        self.compare_plots(ax3, ax4)

    def test_bar_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf.plot(kind='bar', colormap='Paired')
        ax2 = kdf.plot(kind='bar', colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax3 = pdf.plot.bar(colormap='Paired')
        ax4 = kdf.plot.bar(colormap='Paired')
        self.compare_plots(ax3, ax4)

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

    def test_pie_plot(self):
        pdf = pd.DataFrame({'mass': [0.330, 4.87, 5.97], 'radius': [2439.7, 6051.8, 6378.1]},
                           index=['Mercury', 'Venus', 'Earth'])
        kdf = koalas.from_pandas(pdf)

        ax1 = pdf.plot.pie(y='mass', figsize=(5, 5), colormap='Paired')
        ax2 = kdf.plot.pie(y='mass', figsize=(5, 5), colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax1 = pdf.plot(kind="pie", y='mass', figsize=(5, 5), colormap='Paired')
        ax2 = kdf.plot(kind="pie", y='mass', figsize=(5, 5), colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax11, ax12 = pdf.plot.pie(figsize=(5, 5), subplots=True, colormap='Paired')
        ax21, ax22 = kdf.plot.pie(figsize=(5, 5), subplots=True, colormap='Paired')
        self.compare_plots(ax11, ax21)
        self.compare_plots(ax12, ax22)

        ax11, ax12 = pdf.plot(kind="pie", figsize=(5, 5), subplots=True, colormap='Paired')
        ax21, ax22 = kdf.plot(kind="pie", figsize=(5, 5), subplots=True, colormap='Paired')
        self.compare_plots(ax11, ax21)
        self.compare_plots(ax12, ax22)

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
        # Use pandas scatter plot example
        pdf = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
        kdf = koalas.from_pandas(pdf)

        ax1 = pdf.plot.scatter(x='a', y='b')
        ax2 = kdf.plot.scatter(x='a', y='b')
        self.compare_plots(ax1, ax2)

        ax1 = pdf.plot(kind='scatter', x='a', y='b')
        ax2 = kdf.plot(kind='scatter', x='a', y='b')
        self.compare_plots(ax1, ax2)

        # check when keyword c is given as name of a column
        ax1 = pdf.plot.scatter(x='a', y='b', c='c', s=50)
        ax2 = kdf.plot.scatter(x='a', y='b', c='c', s=50)
        self.compare_plots(ax1, ax2)

    def test_missing(self):
        ks = self.kdf1

        unsupported_functions = ['box', 'density', 'hexbin', 'hist', 'kde']

        for name in unsupported_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*DataFrame.*{}.*not implemented".format(name)):
                getattr(ks.plot, name)()
