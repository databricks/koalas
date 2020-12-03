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
from databricks.koalas.plot import TopNPlot, SampledPlot
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

        data = TopNPlot().get_top_n(kdf)
        self.assertEqual(len(data), 2000)

    def test_sampled_plot_with_ratio(self):
        with option_context("plotting.sample_ratio", 0.5):
            pdf = pd.DataFrame(np.random.rand(2500, 4), columns=["a", "b", "c", "d"])
            kdf = ks.from_pandas(pdf)
            data = SampledPlot().get_sampled(kdf)
            self.assertEqual(round(len(data) / 2500, 1), 0.5)

    def test_sampled_plot_with_max_rows(self):
        # 'plotting.max_rows' is 2000
        pdf = pd.DataFrame(np.random.rand(2000, 4), columns=["a", "b", "c", "d"])
        kdf = ks.from_pandas(pdf)
        data = SampledPlot().get_sampled(kdf)
        self.assertEqual(round(len(data) / 2000, 1), 1)
