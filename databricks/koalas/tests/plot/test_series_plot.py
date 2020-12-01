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

from databricks import koalas as ks


class SeriesPlotTest(unittest.TestCase):
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
