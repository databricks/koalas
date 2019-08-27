import base64
from io import BytesIO

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

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

    def test_bar_plot(self):
        pdf = self.pdf1
        kdf = self.kdf1

        ax1 = pdf.plot(kind="bar", colormap='Paired')
        ax2 = kdf.plot(kind="bar", colormap='Paired')
        self.compare_plots(ax1, ax2)

        ax3 = pdf.plot.bar(colormap='Paired')
        ax4 = kdf.plot.bar(colormap='Paired')
        self.compare_plots(ax3, ax4)

    def test_missing(self):
        ks = self.kdf1

        unsupported_functions = ['area', 'barh', 'box', 'density', 'hexbin', 'hist',
                                 'kde', 'pie', 'scatter']
        for name in unsupported_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*DataFrame.*{}.*not implemented".format(name)):
                getattr(ks.plot, name)()
