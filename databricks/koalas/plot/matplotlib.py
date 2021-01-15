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

from distutils.version import LooseVersion

import matplotlib as mat
import numpy as np
import pandas as pd
from matplotlib.axes._base import _process_plot_format
from pandas.core.dtypes.inference import is_integer, is_list_like
from pandas.io.formats.printing import pprint_thing

from databricks.koalas.plot import (
    TopNPlotBase,
    SampledPlotBase,
    HistogramPlotBase,
    unsupported_function,
)
from pyspark.mllib.stat import KernelDensity
from pyspark.sql import functions as F


if LooseVersion(pd.__version__) < LooseVersion("0.25"):
    from pandas.plotting._core import (
        _all_kinds,
        BarPlot as PandasBarPlot,
        BoxPlot as PandasBoxPlot,
        HistPlot as PandasHistPlot,
        MPLPlot as PandasMPLPlot,
        PiePlot as PandasPiePlot,
        AreaPlot as PandasAreaPlot,
        LinePlot as PandasLinePlot,
        BarhPlot as PandasBarhPlot,
        ScatterPlot as PandasScatterPlot,
        KdePlot as PandasKdePlot,
    )
else:
    from pandas.plotting._matplotlib import (
        BarPlot as PandasBarPlot,
        BoxPlot as PandasBoxPlot,
        HistPlot as PandasHistPlot,
        PiePlot as PandasPiePlot,
        AreaPlot as PandasAreaPlot,
        LinePlot as PandasLinePlot,
        BarhPlot as PandasBarhPlot,
        ScatterPlot as PandasScatterPlot,
        KdePlot as PandasKdePlot,
    )
    from pandas.plotting._core import PlotAccessor
    from pandas.plotting._matplotlib.core import MPLPlot as PandasMPLPlot

    _all_kinds = PlotAccessor._all_kinds


class KoalasBarPlot(PandasBarPlot, TopNPlotBase):
    def __init__(self, data, **kwargs):
        super().__init__(self.get_top_n(data), **kwargs)

    def _plot(self, ax, x, y, w, start=0, log=False, **kwds):
        self.set_result_text(ax)
        return ax.bar(x, y, w, bottom=start, log=log, **kwds)


class KoalasBoxPlot(PandasBoxPlot):
    def boxplot(
        self,
        ax,
        bxpstats,
        notch=None,
        sym=None,
        vert=None,
        whis=None,
        positions=None,
        widths=None,
        patch_artist=None,
        bootstrap=None,
        usermedians=None,
        conf_intervals=None,
        meanline=None,
        showmeans=None,
        showcaps=None,
        showbox=None,
        showfliers=None,
        boxprops=None,
        labels=None,
        flierprops=None,
        medianprops=None,
        meanprops=None,
        capprops=None,
        whiskerprops=None,
        manage_ticks=None,
        # manage_xticks is for compatibility of matplotlib < 3.1.0.
        # Remove this when minimum version is 3.0.0
        manage_xticks=None,
        autorange=False,
        zorder=None,
        precision=None,
    ):
        def update_dict(dictionary, rc_name, properties):
            """ Loads properties in the dictionary from rc file if not already
            in the dictionary"""
            rc_str = "boxplot.{0}.{1}"
            if dictionary is None:
                dictionary = dict()
            for prop_dict in properties:
                dictionary.setdefault(prop_dict, mat.rcParams[rc_str.format(rc_name, prop_dict)])
            return dictionary

        # Common property dictionaries loading from rc
        flier_props = [
            "color",
            "marker",
            "markerfacecolor",
            "markeredgecolor",
            "markersize",
            "linestyle",
            "linewidth",
        ]
        default_props = ["color", "linewidth", "linestyle"]

        boxprops = update_dict(boxprops, "boxprops", default_props)
        whiskerprops = update_dict(whiskerprops, "whiskerprops", default_props)
        capprops = update_dict(capprops, "capprops", default_props)
        medianprops = update_dict(medianprops, "medianprops", default_props)
        meanprops = update_dict(meanprops, "meanprops", default_props)
        flierprops = update_dict(flierprops, "flierprops", flier_props)

        if patch_artist:
            boxprops["linestyle"] = "solid"
            boxprops["edgecolor"] = boxprops.pop("color")

        # if non-default sym value, put it into the flier dictionary
        # the logic for providing the default symbol ('b+') now lives
        # in bxp in the initial value of final_flierprops
        # handle all of the `sym` related logic here so we only have to pass
        # on the flierprops dict.
        if sym is not None:
            # no-flier case, which should really be done with
            # 'showfliers=False' but none-the-less deal with it to keep back
            # compatibility
            if sym == "":
                # blow away existing dict and make one for invisible markers
                flierprops = dict(linestyle="none", marker="", color="none")
                # turn the fliers off just to be safe
                showfliers = False
            # now process the symbol string
            else:
                # process the symbol string
                # discarded linestyle
                _, marker, color = _process_plot_format(sym)
                # if we have a marker, use it
                if marker is not None:
                    flierprops["marker"] = marker
                # if we have a color, use it
                if color is not None:
                    # assume that if color is passed in the user want
                    # filled symbol, if the users want more control use
                    # flierprops
                    flierprops["color"] = color
                    flierprops["markerfacecolor"] = color
                    flierprops["markeredgecolor"] = color

        # replace medians if necessary:
        if usermedians is not None:
            if len(np.ravel(usermedians)) != len(bxpstats) or np.shape(usermedians)[0] != len(
                bxpstats
            ):
                raise ValueError("usermedians length not compatible with x")
            else:
                # reassign medians as necessary
                for stats, med in zip(bxpstats, usermedians):
                    if med is not None:
                        stats["med"] = med

        if conf_intervals is not None:
            if np.shape(conf_intervals)[0] != len(bxpstats):
                err_mess = "conf_intervals length not compatible with x"
                raise ValueError(err_mess)
            else:
                for stats, ci in zip(bxpstats, conf_intervals):
                    if ci is not None:
                        if len(ci) != 2:
                            raise ValueError("each confidence interval must " "have two values")
                        else:
                            if ci[0] is not None:
                                stats["cilo"] = ci[0]
                            if ci[1] is not None:
                                stats["cihi"] = ci[1]

        should_manage_ticks = True
        if manage_xticks is not None:
            should_manage_ticks = manage_xticks
        if manage_ticks is not None:
            should_manage_ticks = manage_ticks

        if LooseVersion(mat.__version__) < LooseVersion("3.1.0"):
            extra_args = {"manage_xticks": should_manage_ticks}
        else:
            extra_args = {"manage_ticks": should_manage_ticks}

        artists = ax.bxp(
            bxpstats,
            positions=positions,
            widths=widths,
            vert=vert,
            patch_artist=patch_artist,
            shownotches=notch,
            showmeans=showmeans,
            showcaps=showcaps,
            showbox=showbox,
            boxprops=boxprops,
            flierprops=flierprops,
            medianprops=medianprops,
            meanprops=meanprops,
            meanline=meanline,
            showfliers=showfliers,
            capprops=capprops,
            whiskerprops=whiskerprops,
            zorder=zorder,
            **extra_args,
        )
        return artists

    def _plot(self, ax, bxpstats, column_num=None, return_type="axes", **kwds):
        bp = self.boxplot(ax, bxpstats, **kwds)

        if return_type == "dict":
            return bp, bp
        elif return_type == "both":
            return self.BP(ax=ax, lines=bp), bp
        else:
            return ax, bp

    def _compute_plot_data(self):
        colname = self.data.name
        spark_column_name = self.data._internal.spark_column_name_for(self.data._column_label)
        data = self.data

        # Updates all props with the rc defaults from matplotlib
        self.kwds.update(KoalasBoxPlot.rc_defaults(**self.kwds))

        # Gets some important kwds
        showfliers = self.kwds.get("showfliers", False)
        whis = self.kwds.get("whis", 1.5)
        labels = self.kwds.get("labels", [colname])

        # This one is Koalas specific to control precision for approx_percentile
        precision = self.kwds.get("precision", 0.01)

        # # Computes mean, median, Q1 and Q3 with approx_percentile and precision
        col_stats, col_fences = KoalasBoxPlot._compute_stats(
            data, spark_column_name, whis, precision
        )

        # # Creates a column to flag rows as outliers or not
        outliers = KoalasBoxPlot._outliers(data, spark_column_name, *col_fences)

        # # Computes min and max values of non-outliers - the whiskers
        whiskers = KoalasBoxPlot._calc_whiskers(spark_column_name, outliers)

        if showfliers:
            fliers = KoalasBoxPlot._get_fliers(spark_column_name, outliers, whiskers[0])
        else:
            fliers = []

        # Builds bxpstats dict
        stats = []
        item = {
            "mean": col_stats["mean"],
            "med": col_stats["med"],
            "q1": col_stats["q1"],
            "q3": col_stats["q3"],
            "whislo": whiskers[0],
            "whishi": whiskers[1],
            "fliers": fliers,
            "label": labels[0],
        }
        stats.append(item)

        self.data = {labels[0]: stats}

    def _make_plot(self):
        bxpstats = list(self.data.values())[0]
        ax = self._get_ax(0)
        kwds = self.kwds.copy()

        for stats in bxpstats:
            if len(stats["fliers"]) > 1000:
                stats["fliers"] = stats["fliers"][:1000]
                ax.text(
                    1,
                    1,
                    "showing top 1,000 fliers only",
                    size=6,
                    ha="right",
                    va="bottom",
                    transform=ax.transAxes,
                )

        ret, bp = self._plot(ax, bxpstats, column_num=0, return_type=self.return_type, **kwds)
        self.maybe_color_bp(bp)
        self._return_obj = ret

        labels = [l for l, _ in self.data.items()]
        labels = [pprint_thing(l) for l in labels]
        if not self.use_index:
            labels = [pprint_thing(key) for key in range(len(labels))]
        self._set_ticklabels(ax, labels)

    @staticmethod
    def rc_defaults(
        notch=None,
        vert=None,
        whis=None,
        patch_artist=None,
        bootstrap=None,
        meanline=None,
        showmeans=None,
        showcaps=None,
        showbox=None,
        showfliers=None,
        **kwargs
    ):
        # Missing arguments default to rcParams.
        if whis is None:
            whis = mat.rcParams["boxplot.whiskers"]
        if bootstrap is None:
            bootstrap = mat.rcParams["boxplot.bootstrap"]

        if notch is None:
            notch = mat.rcParams["boxplot.notch"]
        if vert is None:
            vert = mat.rcParams["boxplot.vertical"]
        if patch_artist is None:
            patch_artist = mat.rcParams["boxplot.patchartist"]
        if meanline is None:
            meanline = mat.rcParams["boxplot.meanline"]
        if showmeans is None:
            showmeans = mat.rcParams["boxplot.showmeans"]
        if showcaps is None:
            showcaps = mat.rcParams["boxplot.showcaps"]
        if showbox is None:
            showbox = mat.rcParams["boxplot.showbox"]
        if showfliers is None:
            showfliers = mat.rcParams["boxplot.showfliers"]

        return dict(
            whis=whis,
            bootstrap=bootstrap,
            notch=notch,
            vert=vert,
            patch_artist=patch_artist,
            meanline=meanline,
            showmeans=showmeans,
            showcaps=showcaps,
            showbox=showbox,
            showfliers=showfliers,
        )

    @staticmethod
    def _compute_stats(data, colname, whis, precision):
        # Computes mean, median, Q1 and Q3 with approx_percentile and precision
        pdf = data._kdf._internal.resolved_copy.spark_frame.agg(
            *[
                F.expr(
                    "approx_percentile(`{}`, {}, {})".format(colname, q, int(1.0 / precision))
                ).alias("{}_{}%".format(colname, int(q * 100)))
                for q in [0.25, 0.50, 0.75]
            ],
            F.mean("`%s`" % colname).alias("{}_mean".format(colname)),
        ).toPandas()

        # Computes IQR and Tukey's fences
        iqr = "{}_iqr".format(colname)
        p75 = "{}_75%".format(colname)
        p25 = "{}_25%".format(colname)
        pdf.loc[:, iqr] = pdf.loc[:, p75] - pdf.loc[:, p25]
        pdf.loc[:, "{}_lfence".format(colname)] = pdf.loc[:, p25] - whis * pdf.loc[:, iqr]
        pdf.loc[:, "{}_ufence".format(colname)] = pdf.loc[:, p75] + whis * pdf.loc[:, iqr]

        qnames = ["25%", "50%", "75%", "mean", "lfence", "ufence"]
        col_summ = pdf[["{}_{}".format(colname, q) for q in qnames]]
        col_summ.columns = qnames
        lfence, ufence = col_summ["lfence"], col_summ["ufence"]

        stats = {
            "mean": col_summ["mean"].values[0],
            "med": col_summ["50%"].values[0],
            "q1": col_summ["25%"].values[0],
            "q3": col_summ["75%"].values[0],
        }

        return stats, (lfence.values[0], ufence.values[0])

    @staticmethod
    def _outliers(data, colname, lfence, ufence):
        # Builds expression to identify outliers
        expression = F.col("`%s`" % colname).between(lfence, ufence)
        # Creates a column to flag rows as outliers or not
        return data._kdf._internal.resolved_copy.spark_frame.withColumn(
            "__{}_outlier".format(colname), ~expression
        )

    @staticmethod
    def _calc_whiskers(colname, outliers):
        # Computes min and max values of non-outliers - the whiskers
        minmax = (
            outliers.filter("not `__{}_outlier`".format(colname))
            .agg(F.min("`%s`" % colname).alias("min"), F.max(colname).alias("max"))
            .toPandas()
        )
        return minmax.iloc[0][["min", "max"]].values

    @staticmethod
    def _get_fliers(colname, outliers, min_val):
        # Filters only the outliers, should "showfliers" be True
        fliers_df = outliers.filter("`__{}_outlier`".format(colname))

        # If shows fliers, takes the top 1k with highest absolute values
        # Here we normalize the values by subtracting the minimum value from
        # each, and use absolute values.
        order_col = F.abs(F.col("`{}`".format(colname)) - min_val.item())
        fliers = (
            fliers_df.select(F.col("`{}`".format(colname)))
            .orderBy(order_col)
            .limit(1001)
            .toPandas()[colname]
            .values
        )

        return fliers


class KoalasHistPlot(PandasHistPlot, HistogramPlotBase):
    def _args_adjust(self):
        if is_list_like(self.bottom):
            self.bottom = np.array(self.bottom)

    def _compute_plot_data(self):
        self.data, self.bins = HistogramPlotBase.prepare_hist_data(self.data, self.bins)

    def _make_plot(self):
        # TODO: this logic is similar with KdePlot. Might have to deduplicate it.
        # 'num_colors' requires to calculate `shape` which has to count all.
        # Use 1 for now to save the computation.
        colors = self._get_colors(num_colors=1)
        stacking_id = self._get_stacking_id()
        output_series = HistogramPlotBase.compute_hist(self.data, self.bins)

        for (i, label), y in zip(enumerate(self.data._internal.column_labels), output_series):
            ax = self._get_ax(i)

            kwds = self.kwds.copy()

            label = pprint_thing(label if len(label) > 1 else label[0])
            kwds["label"] = label

            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            if style is not None:
                kwds["style"] = style

            kwds = self._make_plot_keywords(kwds, y)
            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)
            self._add_legend_handle(artists[0], label, index=i)

    @classmethod
    def _plot(cls, ax, y, style=None, bins=None, bottom=0, column_num=0, stacking_id=None, **kwds):
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(bins) - 1)

        base = np.zeros(len(bins) - 1)
        bottom = bottom + cls._get_stacked_values(ax, stacking_id, base, kwds["label"])

        # Since the counts were computed already, we use them as weights and just generate
        # one entry for each bin
        n, bins, patches = ax.hist(bins[:-1], bins=bins, bottom=bottom, weights=y, **kwds)

        cls._update_stacker(ax, stacking_id, n)
        return patches


class KoalasPiePlot(PandasPiePlot, TopNPlotBase):
    def __init__(self, data, **kwargs):
        super().__init__(self.get_top_n(data), **kwargs)

    def _make_plot(self):
        self.set_result_text(self._get_ax(0))
        super()._make_plot()


class KoalasAreaPlot(PandasAreaPlot, SampledPlotBase):
    def __init__(self, data, **kwargs):
        super().__init__(self.get_sampled(data), **kwargs)

    def _make_plot(self):
        self.set_result_text(self._get_ax(0))
        super()._make_plot()


class KoalasLinePlot(PandasLinePlot, SampledPlotBase):
    def __init__(self, data, **kwargs):
        super().__init__(self.get_sampled(data), **kwargs)

    def _make_plot(self):
        self.set_result_text(self._get_ax(0))
        super()._make_plot()


class KoalasBarhPlot(PandasBarhPlot, TopNPlotBase):
    def __init__(self, data, **kwargs):
        super().__init__(self.get_top_n(data), **kwargs)

    def _make_plot(self):
        self.set_result_text(self._get_ax(0))
        super()._make_plot()


class KoalasScatterPlot(PandasScatterPlot, TopNPlotBase):
    def __init__(self, data, x, y, **kwargs):
        super().__init__(self.get_top_n(data), x, y, **kwargs)

    def _make_plot(self):
        self.set_result_text(self._get_ax(0))
        super()._make_plot()


class KoalasKdePlot(PandasKdePlot):
    def _compute_plot_data(self):
        from databricks.koalas.series import Series

        data = self.data
        if isinstance(data, Series):
            data = data.to_frame()

        numeric_data = data.select_dtypes(
            include=["byte", "decimal", "integer", "float", "long", "double", np.datetime64]
        )

        # no empty frames or series allowed
        if len(numeric_data.columns) == 0:
            raise TypeError(
                "Empty {0!r}: no numeric data to " "plot".format(numeric_data.__class__.__name__)
            )

        self.data = numeric_data

    def _make_plot(self):
        # 'num_colors' requires to calculate `shape` which has to count all.
        # Use 1 for now to save the computation.
        colors = self._get_colors(num_colors=1)
        stacking_id = self._get_stacking_id()

        sdf = self.data._internal.spark_frame

        for i, label in enumerate(self.data._internal.column_labels):
            # 'y' is a Spark DataFrame that selects one column.
            y = sdf.select(self.data._internal.spark_column_for(label))
            ax = self._get_ax(i)

            kwds = self.kwds.copy()

            label = pprint_thing(label if len(label) > 1 else label[0])
            kwds["label"] = label

            style, kwds = self._apply_style_colors(colors, kwds, i, label)
            if style is not None:
                kwds["style"] = style

            kwds = self._make_plot_keywords(kwds, y)
            artists = self._plot(ax, y, column_num=i, stacking_id=stacking_id, **kwds)
            self._add_legend_handle(artists[0], label, index=i)

    def _get_ind(self, y):
        # 'y' is a Spark DataFrame that selects one column.
        if self.ind is None:
            min_val, max_val = y.select(F.min(y.columns[-1]), F.max(y.columns[-1])).first()

            sample_range = max_val - min_val
            ind = np.linspace(min_val - 0.5 * sample_range, max_val + 0.5 * sample_range, 1000,)
        elif is_integer(self.ind):
            min_val, max_val = y.select(F.min(y.columns[-1]), F.max(y.columns[-1])).first()

            sample_range = np.nanmax(y) - np.nanmin(y)
            ind = np.linspace(min_val - 0.5 * sample_range, max_val + 0.5 * sample_range, self.ind,)
        else:
            ind = self.ind
        return ind

    @classmethod
    def _plot(
        cls, ax, y, style=None, bw_method=None, ind=None, column_num=None, stacking_id=None, **kwds
    ):
        # 'y' is a Spark DataFrame that selects one column.

        # Using RDD is slow so we might have to change it to Dataset based implementation
        # once Spark has that implementation.
        sample = y.rdd.map(lambda x: float(x[0]))
        kd = KernelDensity()
        kd.setSample(sample)

        assert isinstance(bw_method, (int, float)), "'bw_method' must be set as a scalar number."

        if bw_method is not None:
            # Match the bandwidth with Spark.
            kd.setBandwidth(float(bw_method))
        y = kd.estimate(list(map(float, ind)))
        lines = PandasMPLPlot._plot(ax, ind, y, style=style, **kwds)
        return lines


_klasses = [
    KoalasHistPlot,
    KoalasBarPlot,
    KoalasBoxPlot,
    KoalasPiePlot,
    KoalasAreaPlot,
    KoalasLinePlot,
    KoalasBarhPlot,
    KoalasScatterPlot,
    KoalasKdePlot,
]
_plot_klass = {getattr(klass, "_kind"): klass for klass in _klasses}
_common_kinds = {"area", "bar", "barh", "box", "hist", "kde", "line", "pie"}
_series_kinds = _common_kinds.union(set())
_dataframe_kinds = _common_kinds.union({"scatter", "hexbin"})
_koalas_all_kinds = _common_kinds.union(_series_kinds).union(_dataframe_kinds)


def plot_koalas(data, kind, **kwargs):
    if kind not in _koalas_all_kinds:
        raise ValueError("{} is not a valid plot kind".format(kind))

    from databricks.koalas import DataFrame, Series

    if isinstance(data, Series):
        if kind not in _series_kinds:
            return unsupported_function(class_name="pd.Series", method_name=kind)()
        return plot_series(data=data, kind=kind, **kwargs)
    elif isinstance(data, DataFrame):
        if kind not in _dataframe_kinds:
            return unsupported_function(class_name="pd.DataFrame", method_name=kind)()
        return plot_frame(data=data, kind=kind, **kwargs)


def plot_series(
    data,
    kind="line",
    ax=None,  # Series unique
    figsize=None,
    use_index=True,
    title=None,
    grid=None,
    legend=False,
    style=None,
    logx=False,
    logy=False,
    loglog=False,
    xticks=None,
    yticks=None,
    xlim=None,
    ylim=None,
    rot=None,
    fontsize=None,
    colormap=None,
    table=False,
    yerr=None,
    xerr=None,
    label=None,
    secondary_y=False,  # Series unique
    **kwds
):
    """
    Make plots of Series using matplotlib / pylab.

    Each plot kind has a corresponding method on the
    ``Series.plot`` accessor:
    ``s.plot(kind='line')`` is equivalent to
    ``s.plot.line()``.

    Parameters
    ----------
    data : Series

    kind : str
        - 'line' : line plot (default)
        - 'bar' : vertical bar plot
        - 'barh' : horizontal bar plot
        - 'hist' : histogram
        - 'box' : boxplot
        - 'kde' : Kernel Density Estimation plot
        - 'density' : same as 'kde'
        - 'area' : area plot
        - 'pie' : pie plot

    ax : matplotlib axes object
        If not passed, uses gca()
    figsize : a tuple (width, height) in inches
    use_index : boolean, default True
        Use index as ticks for x axis
    title : string or list
        Title to use for the plot. If a string is passed, print the string at
        the top of the figure. If a list is passed and `subplots` is True,
        print each item in the list above the corresponding subplot.
    grid : boolean, default None (matlab style default)
        Axis grid lines
    legend : False/True/'reverse'
        Place legend on axis subplots
    style : list or dict
        matplotlib line style per column
    logx : boolean, default False
        Use log scaling on x axis
    logy : boolean, default False
        Use log scaling on y axis
    loglog : boolean, default False
        Use log scaling on both x and y axes
    xticks : sequence
        Values to use for the xticks
    yticks : sequence
        Values to use for the yticks
    xlim : 2-tuple/list
    ylim : 2-tuple/list
    rot : int, default None
        Rotation for ticks (xticks for vertical, yticks for horizontal plots)
    fontsize : int, default None
        Font size for xticks and yticks
    colormap : str or matplotlib colormap object, default None
        Colormap to select colors from. If string, load colormap with that name
        from matplotlib.
    colorbar : boolean, optional
        If True, plot colorbar (only relevant for 'scatter' and 'hexbin' plots)
    position : float
        Specify relative alignments for bar plot layout.
        From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5 (center)
    table : boolean, Series or DataFrame, default False
        If True, draw a table using the data in the DataFrame and the data will
        be transposed to meet matplotlib's default layout.
        If a Series or DataFrame is passed, use passed data to draw a table.
    yerr : DataFrame, Series, array-like, dict and str
        See :ref:`Plotting with Error Bars <visualization.errorbars>` for
        detail.
    xerr : same types as yerr.
    label : label argument to provide to plot
    secondary_y : boolean or sequence of ints, default False
        If True then y-axis will be on the right
    mark_right : boolean, default True
        When using a secondary_y axis, automatically mark the column
        labels with "(right)" in the legend
    **kwds : keywords
        Options to pass to matplotlib plotting method

    Returns
    -------
    axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them

    Notes
    -----

    - See matplotlib documentation online for more on this subject
    - If `kind` = 'bar' or 'barh', you can specify relative alignments
      for bar plot layout by `position` keyword.
      From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5 (center)
    """

    # function copied from pandas.plotting._core
    # so it calls modified _plot below

    import matplotlib.pyplot as plt

    if ax is None and len(plt.get_fignums()) > 0:
        with plt.rc_context():
            ax = plt.gca()
        ax = PandasMPLPlot._get_ax_layer(ax)
    return _plot(
        data,
        kind=kind,
        ax=ax,
        figsize=figsize,
        use_index=use_index,
        title=title,
        grid=grid,
        legend=legend,
        style=style,
        logx=logx,
        logy=logy,
        loglog=loglog,
        xticks=xticks,
        yticks=yticks,
        xlim=xlim,
        ylim=ylim,
        rot=rot,
        fontsize=fontsize,
        colormap=colormap,
        table=table,
        yerr=yerr,
        xerr=xerr,
        label=label,
        secondary_y=secondary_y,
        **kwds,
    )


def plot_frame(
    data,
    x=None,
    y=None,
    kind="line",
    ax=None,
    subplots=None,
    sharex=None,
    sharey=False,
    layout=None,
    figsize=None,
    use_index=True,
    title=None,
    grid=None,
    legend=True,
    style=None,
    logx=False,
    logy=False,
    loglog=False,
    xticks=None,
    yticks=None,
    xlim=None,
    ylim=None,
    rot=None,
    fontsize=None,
    colormap=None,
    table=False,
    yerr=None,
    xerr=None,
    secondary_y=False,
    sort_columns=False,
    **kwds
):
    """
    Make plots of DataFrames using matplotlib / pylab.

    Each plot kind has a corresponding method on the
    ``DataFrame.plot`` accessor:
    ``kdf.plot(kind='line')`` is equivalent to
    ``kdf.plot.line()``.

    Parameters
    ----------
    data : DataFrame

    kind : str
        - 'line' : line plot (default)
        - 'bar' : vertical bar plot
        - 'barh' : horizontal bar plot
        - 'hist' : histogram
        - 'box' : boxplot
        - 'kde' : Kernel Density Estimation plot
        - 'density' : same as 'kde'
        - 'area' : area plot
        - 'pie' : pie plot
        - 'scatter' : scatter plot
    ax : matplotlib axes object
        If not passed, uses gca()
    x : label or position, default None
    y : label, position or list of label, positions, default None
        Allows plotting of one column versus another.
    figsize : a tuple (width, height) in inches
    use_index : boolean, default True
        Use index as ticks for x axis
    title : string or list
        Title to use for the plot. If a string is passed, print the string at
        the top of the figure. If a list is passed and `subplots` is True,
        print each item in the list above the corresponding subplot.
    grid : boolean, default None (matlab style default)
        Axis grid lines
    legend : False/True/'reverse'
        Place legend on axis subplots
    style : list or dict
        matplotlib line style per column
    logx : boolean, default False
        Use log scaling on x axis
    logy : boolean, default False
        Use log scaling on y axis
    loglog : boolean, default False
        Use log scaling on both x and y axes
    xticks : sequence
        Values to use for the xticks
    yticks : sequence
        Values to use for the yticks
    xlim : 2-tuple/list
    ylim : 2-tuple/list
    sharex: bool or None, default is None
        Whether to share x axis or not.
    sharey: bool, default is False
        Whether to share y axis or not.
    rot : int, default None
        Rotation for ticks (xticks for vertical, yticks for horizontal plots)
    fontsize : int, default None
        Font size for xticks and yticks
    colormap : str or matplotlib colormap object, default None
        Colormap to select colors from. If string, load colormap with that name
        from matplotlib.
    colorbar : boolean, optional
        If True, plot colorbar (only relevant for 'scatter' and 'hexbin' plots)
    position : float
        Specify relative alignments for bar plot layout.
        From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5 (center)
    table : boolean, Series or DataFrame, default False
        If True, draw a table using the data in the DataFrame and the data will
        be transposed to meet matplotlib's default layout.
        If a Series or DataFrame is passed, use passed data to draw a table.
    yerr : DataFrame, Series, array-like, dict and str
        See :ref:`Plotting with Error Bars <visualization.errorbars>` for
        detail.
    xerr : same types as yerr.
    label : label argument to provide to plot
    secondary_y : boolean or sequence of ints, default False
        If True then y-axis will be on the right
    mark_right : boolean, default True
        When using a secondary_y axis, automatically mark the column
        labels with "(right)" in the legend
    sort_columns: bool, default is False
        When True, will sort values on plots.
    **kwds : keywords
        Options to pass to matplotlib plotting method

    Returns
    -------
    axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them

    Notes
    -----

    - See matplotlib documentation online for more on this subject
    - If `kind` = 'bar' or 'barh', you can specify relative alignments
      for bar plot layout by `position` keyword.
      From 0 (left/bottom-end) to 1 (right/top-end). Default is 0.5 (center)
    """

    return _plot(
        data,
        kind=kind,
        x=x,
        y=y,
        ax=ax,
        figsize=figsize,
        use_index=use_index,
        title=title,
        grid=grid,
        legend=legend,
        subplots=subplots,
        style=style,
        logx=logx,
        logy=logy,
        loglog=loglog,
        xticks=xticks,
        yticks=yticks,
        xlim=xlim,
        ylim=ylim,
        rot=rot,
        fontsize=fontsize,
        colormap=colormap,
        table=table,
        yerr=yerr,
        xerr=xerr,
        sharex=sharex,
        sharey=sharey,
        secondary_y=secondary_y,
        layout=layout,
        sort_columns=sort_columns,
        **kwds,
    )


def _plot(data, x=None, y=None, subplots=False, ax=None, kind="line", **kwds):
    from databricks.koalas import DataFrame

    # function copied from pandas.plotting._core
    # and adapted to handle Koalas DataFrame and Series

    kind = kind.lower().strip()
    kind = {"density": "kde"}.get(kind, kind)
    if kind in _all_kinds:
        klass = _plot_klass[kind]
    else:
        raise ValueError("%r is not a valid plot kind" % kind)

    # scatter and hexbin are inherited from PlanePlot which require x and y
    if kind in ("scatter", "hexbin"):
        plot_obj = klass(data, x, y, subplots=subplots, ax=ax, kind=kind, **kwds)
    else:

        # check data type and do preprocess before applying plot
        if isinstance(data, DataFrame):
            if x is not None:
                data = data.set_index(x)
            # TODO: check if value of y is plottable
            if y is not None:
                data = data[y]

        plot_obj = klass(data, subplots=subplots, ax=ax, kind=kind, **kwds)
    plot_obj.generate()
    plot_obj.draw()
    return plot_obj.result
