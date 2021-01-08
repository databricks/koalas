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

from databricks.koalas.plot import HistogramPlotBase


def plot_plotly(origin_plot):
    def plot(data, kind, **kwargs):
        # Koalas specific plots
        if kind == "pie":
            return plot_pie(data, **kwargs)
        if kind == "hist":
            # Note that here data is a Koalas DataFrame or Series unlike other type of plots.
            return plot_histogram(data, **kwargs)

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


def plot_histogram(data, **kwargs):
    from plotly import express
    from plotly.subplots import make_subplots
    import plotly.graph_objs as go
    from databricks import koalas as ks

    assert "bins" in kwargs
    bins = kwargs["bins"]
    data, bins = HistogramPlotBase.prepare_hist_data(data, bins)

    is_single_column = False
    if isinstance(data, ks.Series):
        is_single_column = True
        kdf = data.to_frame()
    elif isinstance(data, ks.DataFrame):
        kdf = data
    else:
        raise RuntimeError("Unexpected type: [%s]" % type(data))

    output_series = HistogramPlotBase.compute_hist(kdf, bins)
    bins = 0.5 * (bins[:-1] + bins[1:])
    if is_single_column:
        output_series = list(output_series)
        assert len(output_series) == 1
        output_series = output_series[0]
        return express.bar(
            x=bins, y=output_series, labels={"x": str(output_series.name), "y": "count"}
        )
    else:
        output_series = list(output_series)
        fig = make_subplots(rows=1, cols=len(output_series))

        for i, series in enumerate(output_series):
            fig.add_trace(go.Bar(x=bins, y=series, name=series.name,), row=1, col=i + 1)

        for i, series in enumerate(output_series):
            if i == 0:
                xaxis = "xaxis"
                yaxis = "yaxis"
            else:
                xaxis = "xaxis%s" % (i + 1)
                yaxis = "yaxis%s" % (i + 1)
            fig["layout"][xaxis]["title"] = str(series.name)
            fig["layout"][yaxis]["title"] = "count"
        return fig
