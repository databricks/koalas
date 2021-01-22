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

import pandas as pd
import numpy as np

from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option, option_context
from databricks.koalas.plot import TopNPlotBase, SampledPlotBase, HistogramPlotBase
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.testing.utils import ReusedSQLTestCase


class DataFramePlotTest(ReusedSQLTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        set_option("plotting.max_rows", 2000)
        set_option("plotting.sample_ratio", None)

    @classmethod
    def tearDownClass(cls):
        reset_option("plotting.max_rows")
        reset_option("plotting.sample_ratio")
        super().tearDownClass()

    def test_missing(self):
        kdf = ks.DataFrame(np.random.rand(2500, 4), columns=["a", "b", "c", "d"])

        unsupported_functions = ["box", "hexbin"]

        for name in unsupported_functions:
            with self.assertRaisesRegex(
                PandasNotImplementedError, "method.*DataFrame.*{}.*not implemented".format(name)
            ):
                getattr(kdf.plot, name)()

    def test_topn_max_rows(self):

        pdf = pd.DataFrame(np.random.rand(2500, 4), columns=["a", "b", "c", "d"])
        kdf = ks.from_pandas(pdf)

        data = TopNPlotBase().get_top_n(kdf)
        self.assertEqual(len(data), 2000)

    def test_sampled_plot_with_ratio(self):
        with option_context("plotting.sample_ratio", 0.5):
            pdf = pd.DataFrame(np.random.rand(2500, 4), columns=["a", "b", "c", "d"])
            kdf = ks.from_pandas(pdf)
            data = SampledPlotBase().get_sampled(kdf)
            self.assertEqual(round(len(data) / 2500, 1), 0.5)

    def test_sampled_plot_with_max_rows(self):
        # 'plotting.max_rows' is 2000
        pdf = pd.DataFrame(np.random.rand(2000, 4), columns=["a", "b", "c", "d"])
        kdf = ks.from_pandas(pdf)
        data = SampledPlotBase().get_sampled(kdf)
        self.assertEqual(round(len(data) / 2000, 1), 1)

    def test_compute_hist_single_column(self):
        kdf = ks.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50],}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10]
        )

        expected_bins = np.linspace(1, 50, 11)
        bins = HistogramPlotBase.get_bins(kdf[["a"]].to_spark(), 10)

        expected_histogram = np.array([5, 4, 1, 0, 0, 0, 0, 0, 0, 1])
        histogram = HistogramPlotBase.compute_hist(kdf[["a"]], bins)[0]
        self.assert_eq(pd.Series(expected_bins), pd.Series(bins))
        self.assert_eq(pd.Series(expected_histogram, name="a"), histogram, almost=True)

    def test_compute_hist_multi_columns(self):
        expected_bins = np.linspace(1, 50, 11)
        kdf = ks.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50],
                "b": [50, 50, 30, 30, 30, 24, 10, 5, 4, 3, 1],
            }
        )

        bins = HistogramPlotBase.get_bins(kdf.to_spark(), 10)
        self.assert_eq(pd.Series(expected_bins), pd.Series(bins))

        expected_histograms = [
            np.array([5, 4, 1, 0, 0, 0, 0, 0, 0, 1]),
            np.array([4, 1, 0, 0, 1, 3, 0, 0, 0, 2]),
        ]
        histograms = HistogramPlotBase.compute_hist(kdf, bins)
        expected_names = ["a", "b"]

        for histogram, expected_histogram, expected_name in zip(
            histograms, expected_histograms, expected_names
        ):
            self.assert_eq(
                pd.Series(expected_histogram, name=expected_name), histogram, almost=True
            )
