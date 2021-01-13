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

from databricks.koalas.plot import HistogramPlotBase, name_like_string


def plot(data, kind, **kwargs):
    import plotly

    # Koalas specific plots
    if kind == "pie":
        return plot_pie(data, **kwargs)
    if kind == "hist":
        # Note that here data is a Koalas DataFrame or Series unlike other type of plots.
        return plot_histogram(data, **kwargs)

    # Other plots.
    return plotly.plot(data, kind, **kwargs)


def plot_pie(data, **kwargs):
    from plotly import express

    if isinstance(data, pd.Series):
        pdf = data.to_frame()
        return express.pie(pdf, values=pdf.columns[0], names=pdf.index, **kwargs)
    elif isinstance(data, pd.DataFrame):
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
    import plotly.graph_objs as go

    bins = kwargs.get("bins", 10)
    kdf, bins = HistogramPlotBase.prepare_hist_data(data, bins)
    assert len(bins) > 2, "the number of buckets must be higher than 2."
    output_series = HistogramPlotBase.compute_hist(kdf, bins)
    prev = float("%.9f" % bins[0])  # to make it prettier, truncate.
    text_bins = []
    for b in bins[1:]:
        norm_b = float("%.9f" % b)
        text_bins.append("[%s, %s)" % (prev, norm_b))
        prev = norm_b
    text_bins[-1] = text_bins[-1][:-1] + "]"  # replace ) to ] for the last bucket.

    bins = 0.5 * (bins[:-1] + bins[1:])

    output_series = list(output_series)
    bars = []
    for series in output_series:
        bars.append(
            go.Bar(
                x=bins,
                y=series,
                name=name_like_string(series.name),
                text=text_bins,
                hovertemplate=(
                    "variable=" + name_like_string(series.name) + "<br>value=%{text}<br>count=%{y}"
                ),
            )
        )

    fig = go.Figure(data=bars, layout=go.Layout(barmode="stack"))
    fig["layout"]["xaxis"]["title"] = "value"
    fig["layout"]["yaxis"]["title"] = "count"
    return fig
