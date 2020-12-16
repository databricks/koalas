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


def plot_plotly(origin_plot):
    def plot(data, kind, **kwargs):
        # Koalas specific plots
        if kind == "pie":
            return plot_pie(data, **kwargs)

        # Other plots.
        return origin_plot(data, kind, **kwargs)

    return plot


def plot_pie(data, **kwargs):
    from plotly import express

    if isinstance(data, pd.Series):
        pdf = data.to_frame()
        return express.pie(pdf, values=pdf.columns[0], names=pdf.index, **kwargs)
    elif isinstance(data, pd.DataFrame):
        # DataFrame
        values = kwargs.pop("y", None)
        default_names = None
        if values is not None:
            default_names = data.index

        return express.pie(
            data,
            values=kwargs.pop("values", values),
            names=kwargs.pop("names", default_names),
            **kwargs
        )
    else:
        raise RuntimeError("Unexpected type: [%s]" % type(data))
