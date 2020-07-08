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

import plotly
import matplotlib
import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
from matplotlib.axes._base import _process_plot_format
from pandas.core.dtypes.inference import is_integer, is_list_like
from pandas.io.formats.printing import pprint_thing
from pandas.core.base import PandasObject
from pyspark.ml.feature import Bucketizer
from pyspark.mllib.stat import KernelDensity
from pyspark.sql import functions as F

from databricks.koalas.missing import unsupported_function
from databricks.koalas.config import get_option



from pandas.plotting._core import PlotAccessor

# Overriding the list as all plots are not implemented by plotly
_all_kinds = [*PlotAccessor._all_kinds]



class TopNPlot:
    def get_top_n(self, data):
        from databricks.koalas import DataFrame, Series

        max_rows = get_option("plotting.max_rows")
        # Simply use the first 1k elements and make it into a pandas dataframe
        # For categorical variables, it is likely called from df.x.value_counts().plot.xxx().
        if isinstance(data, (Series, DataFrame)):
            data = data.head(max_rows + 1).to_pandas()
        else:
            raise ValueError(
                "Only DataFrame and Series are supported for plotting.")

        self.partial = False
        if len(data) > max_rows:
            self.partial = True
            data = data.iloc[:max_rows]
        return data

    def set_result_text(self, ax):
        max_rows = get_option("plotting.max_rows")
        assert hasattr(self, "partial")

        if self.partial:
            ax.update_layout(
                title="showing top {} elements only".format(max_rows),
                xaxis_title="x Axis Title",
                yaxis_title="y Axis Title",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )


class SampledPlot:
    def get_sampled(self, data):
        from databricks.koalas import DataFrame, Series

        fraction = get_option("plotting.sample_ratio")
        if fraction is None:
            fraction = 1 / (len(data) / get_option("plotting.max_rows"))
            fraction = min(1.0, fraction)
        self.fraction = fraction

        if isinstance(data, (DataFrame, Series)):
            if isinstance(data, Series):
                data = data.to_frame()
            sampled = data._internal.resolved_copy.spark_frame.sample(
                fraction=self.fraction)
            return DataFrame(data._internal.with_new_sdf(sampled)).to_pandas()
        else:
            raise ValueError(
                "Only DataFrame and Series are supported for plotting.")

    def set_result_text(self, ax):
        assert hasattr(self, "fraction")

        if self.fraction < 1:
            ax.update_layout(
                title="showing the sampled result by fraction %s" % self.fraction,
                xaxis_title="x Axis Title",
                yaxis_title="y Axis Title",
                font=dict(
                    family="Courier New, monospace",
                    size=18,
                    color="#7f7f7f"
                )
            )


class KoalasPiePlot(TopNPlot):
    _kind="pie"
    pass


class KoalasKdePlot(TopNPlot):
    _kind="kde"
    pass


class KoalasHistPlot(TopNPlot):
    _kind="hist"
    pass


class KoalasBoxPlot(TopNPlot):
    _kind="box"
    pass

# Used for all the plots natively supported by plotly
# Natively Supported kinds by Plotly: ["line", "scatter", "area", "bar", "barh", "box", "histogram"]
class KoalasBarPlot(PlotAccessor, TopNPlot):
    _kind="bar"

    def __init__(self, data, **kwargs):
        self._parent = super(KoalasBarPlot, self).get_top_n(data)

    def draw(self, **kwargs):
        return super(KoalasBarPlot, self).__call__(kind="bar", **kwargs)


class KoalasAreaPlot(PlotAccessor, SampledPlot):
    _kind="area"

    def __init__(self, data, **kwargs):
        self._parent = super(KoalasAreaPlot, self).get_sampled(data)

    def draw(self, **kwargs):
        return super(KoalasLinePlot, self).__call__(kind="area", **kwargs)


class KoalasLinePlot(PlotAccessor, SampledPlot):
    _kind="line"

    def __init__(self, data, **kwargs):
        self._parent = super(KoalasLinePlot, self).get_sampled(data)

    def draw(self, **kwargs):
        return super(KoalasLinePlot, self).__call__(kind="line", **kwargs)


class KoalasBarhPlot(PlotAccessor, TopNPlot):
    _kind="barh"

    def __init__(self, data, **kwargs):
        self._parent = super(KoalasBarhPlot, self).get_top_n(data)

    def draw(self, **kwargs):
        return super(KoalasBarhPlot, self).__call__(kind="barh", orientation="h", **kwargs)


class KoalasScatterPlot(PlotAccessor, TopNPlot):
    _kind="scatter"

    def __init__(self, data, x, y, **kwargs):
        self._parent = super(KoalasScatterPlot, self).get_top_n(data)

    def draw(self, **kwargs):
        return super(KoalasBarhPlot, self).__call__(kind="scatter", **kwargs)


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



def plotly_frame(
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
        # ax=ax,
        # figsize=figsize,
        # use_index=use_index,
        # title=title,
        # grid=grid,
        # legend=legend,
        # subplots=subplots,
        # style=style,
        # logx=logx,
        # logy=logy,
        # loglog=loglog,
        # xticks=xticks,
        # yticks=yticks,
        # xlim=xlim,
        # ylim=ylim,
        # rot=rot,
        # fontsize=fontsize,
        # colormap=colormap,
        # table=table,
        # yerr=yerr,
        # xerr=xerr,
        # sharex=sharex,
        # sharey=sharey,
        # secondary_y=secondary_y,
        # layout=layout,
        # sort_columns=sort_columns,
        # **kwds
    )


def _plot(data, x=None, y=None, ax=None, kind="line", **kwds):
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
        plot_obj = klass(data, x, y, ax=ax, kind=kind, **kwds)
    else:

        # check data type and do preprocess before applying plot
        if isinstance(data, DataFrame):
            if x is not None:
                data = data.set_index(x)
            # TODO: check if value of y is plottable
            if y is not None:
                data = data[y]

        plot_obj = klass(data, ax=ax, kind=kind, **kwds)
    result = plot_obj.draw()
    return result


class KoalasFramePlotlyMethods(PandasObject):
    # TODO: not sure if Koalas wants to combine plot method for Series and DataFrame
    """
    DataFrame plotting accessor and method.

    Plotting methods can also be accessed by calling the accessor as a method
    with the ``kind`` argument:
    ``df.plot(kind='hist')`` is equivalent to ``df.plot.hist()``
    """

    def __init__(self, data):
        self.data = data

    def __call__(
        self,
        x=None,
        y=None,
        kind="line",
        # data_frame=None,
        # line_group=None,
        # color=None,
        # line_dash=None,
        # hover_name=None,
        # hover_data=None,
        # custom_data=None,
        # text=None,
        # facet_row=None,
        # facet_col=None,
        # facet_col_wrap=0,
        # error_x=None,
        # error_x_minus=None,
        # error_y=None,
        # error_y_minus=None,
        # animation_frame=None,
        # animation_group=None,
        # category_orders={},
        # labels={},
        # orientation=None,
        # color_discrete_sequence=None,
        # color_discrete_map={},
        # line_dash_sequence=None,
        # line_dash_map={},
        # log_x=False,
        # log_y=False,
        # range_x=None,
        # range_y=None,
        # line_shape=None,
        # render_mode="auto",
        # title=None,
        # template=None,
        # width=None,  # figsize=None,
        # height=None,
        
        # fontsize=None,
        # xticks=None,
        # yticks=None,
        # subplots=None, # What is this used for
        # sharex=None,
        # sharey=False,
        # layout=None,
        
        # use_index=True,
        # grid=None,
        # legend=True,
        # style=None,
        # loglog=False,
        # xlim=None,
        # ylim=None,
        # rot=None,
        # table=False,
        # yerr=None,
        # xerr=None,
        # secondary_y=False,
        **kwds
    ):
        return plotly_frame(
            self.data,
            x=x,
            y=y,
            kind=kind,
            # ax=ax,
            # subplots=subplots,
            # sharex=sharex,
            # sharey=sharey,
            # layout=layout,
            # # figsize=figsize,
            # use_index=use_index,
            # title=title,
            # grid=grid,
            # legend=legend,
            # style=style,
            # logx=logx,
            # logy=logy,
            # loglog=loglog,
            # xticks=xticks,
            # yticks=yticks,
            # xlim=xlim,
            # ylim=ylim,
            # rot=rot,
            # fontsize=fontsize,
            # colormap=colormap,
            # table=table,
            # yerr=yerr,
            # xerr=xerr,
            # secondary_y=secondary_y,
            # sort_columns=sort_columns,
            **kwds
        )

    def line(self, x=None, y=None, **kwargs):
        """
        Plot DataFrame as lines.

        Parameters
        ----------
        x: int or str, optional
            Columns to use for the horizontal axis.
        y : int, str, or list of them, optional
            The values to be plotted.
        **kwargs
            Keyword arguments to pass on to :meth:`DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or :class:`numpy.ndarray`
            Return an ndarray when ``subplots=True``.

        See Also
        --------
        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.

        Examples
        --------

        .. plot::
            :context: close-figs

            The following example shows the populations for some animals
            over the years.

            >>> df = ks.DataFrame({'pig': [20, 18, 489, 675, 1776],
            ...                    'horse': [4, 25, 281, 600, 1900]},
            ...                   index=[1990, 1997, 2003, 2009, 2014])
            >>> lines = df.plot.line()

        .. plot::
            :context: close-figs

            An example with subplots, so an array of axes is returned.

            >>> axes = df.plot.line(subplots=True)
            >>> type(axes)
            <class 'numpy.ndarray'>

        .. plot::
            :context: close-figs

            The following example shows the relationship between both
            populations.

            >>> lines = df.plot.line(x='pig', y='horse')
        """
        return self(kind="line", x=x, y=y, **kwargs)

    def kde(self, bw_method=None, ind=None, **kwargs):
        """
        Generate Kernel Density Estimate plot using Gaussian kernels.

        Parameters
        ----------
        bw_method : scalar
            The method used to calculate the estimator bandwidth.
            See KernelDensity in PySpark for more information.
        ind : NumPy array or integer, optional
            Evaluation points for the estimated PDF. If None (default),
            1000 equally spaced points are used. If `ind` is a NumPy array, the
            KDE is evaluated at the points passed. If `ind` is an integer,
            `ind` number of equally spaced points are used.
        **kwargs : optional
            Keyword arguments to pass on to :meth:`Koalas.DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        Examples
        --------
        For DataFrame, it works in the same way as Series:

        .. plot::
            :context: close-figs

            >>> df = ks.DataFrame({
            ...     'x': [1, 2, 2.5, 3, 3.5, 4, 5],
            ...     'y': [4, 4, 4.5, 5, 5.5, 6, 6],
            ... })
            >>> ax = df.plot.kde(bw_method=0.3)

        .. plot::
            :context: close-figs

            >>> ax = df.plot.kde(bw_method=3)

        .. plot::
            :context: close-figs

            >>> ax = df.plot.kde(ind=[1, 2, 3, 4, 5, 6], bw_method=0.3)
        """
        # return self(kind="kde", bw_method=bw_method, ind=ind, **kwargs)
        raise NotImplementedError
    # density = kde

    def pie(self, y=None, **kwds):
        """
        Generate a pie plot.
        A pie plot is a proportional representation of the numerical data in a
        column. This function wraps :meth:`matplotlib.pyplot.pie` for the
        specified column. If no column reference is passed and
        ``subplots=True`` a pie plot is drawn for each numerical column
        independently.

        Parameters
        ----------
        y : int or label, optional
            Label or position of the column to plot.
            If not provided, ``subplots=True`` argument must be passed.
        **kwds
            Keyword arguments to pass on to :meth:`Koalas.DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them
            A NumPy array is returned when `subplots` is True.

        Examples
        --------
        In the example below we have a DataFrame with the information about
        planet's mass and radius. We pass the the 'mass' column to the
        pie function to get a pie plot.

        .. plot::
            :context: close-figs

            >>> df = ks.DataFrame({'mass': [0.330, 4.87, 5.97],
            ...                    'radius': [2439.7, 6051.8, 6378.1]},
            ...                   index=['Mercury', 'Venus', 'Earth'])
            >>> plot = df.plot.pie(y='mass', figsize=(5, 5))

        .. plot::
            :context: close-figs

            >>> plot = df.plot.pie(subplots=True, figsize=(6, 3))
        """
        raise NotImplementedError

    def area(self, x=None, y=None, stacked=True, **kwds):
        """
        Draw a stacked area plot.

        An area plot displays quantitative data visually.
        This function wraps the matplotlib area function.

        Parameters
        ----------
        x : label or position, optional
            Coordinates for the X axis. By default uses the index.
        y : label or position, optional
            Column to plot. By default uses all columns.
        stacked : bool, default True
            Area plots are stacked by default. Set to False to create a
            unstacked plot.
        **kwds : optional
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray
            Area plot, or array of area plots if subplots is True.

        Examples
        --------

        .. plot::
            :context: close-figs

            >>> df = ks.DataFrame({
            ...     'sales': [3, 2, 3, 9, 10, 6],
            ...     'signups': [5, 5, 6, 12, 14, 13],
            ...     'visits': [20, 42, 28, 62, 81, 50],
            ... }, index=pd.date_range(start='2018/01/01', end='2018/07/01',
            ...                        freq='M'))
            >>> plot = df.plot.area()
        """
        return self(kind="area", x=x, y=y, stacked=stacked, **kwds)

    def bar(self, x=None, y=None, **kwds):
        """
        Vertical bar plot.

        Parameters
        ----------
        x : label or position, optional
            Allows plotting of one column versus another.
            If not specified, the index of the DataFrame is used.
        y : label or position, optional
            Allows plotting of one column versus another.
            If not specified, all numerical columns are used.
        **kwds : optional
            Additional keyword arguments are documented in
            :meth:`Koalas.DataFrame.plot`.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        Examples
        --------
        Basic plot.

        .. plot::
            :context: close-figs

            >>> df = ks.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> ax = df.plot.bar(x='lab', y='val', rot=0)

        Plot a whole dataframe to a bar plot. Each column is assigned a
        distinct color, and each row is nested in a group along the
        horizontal axis.

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = ks.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.bar(rot=0)

        Instead of nesting, the figure can be split by column with
        ``subplots=True``. In this case, a :class:`numpy.ndarray` of
        :class:`matplotlib.axes.Axes` are returned.

        .. plot::
            :context: close-figs

            >>> axes = df.plot.bar(rot=0, subplots=True)
            >>> axes[1].legend(loc=2)  # doctest: +SKIP

        Plot a single column.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(y='speed', rot=0)

        Plot only selected categories for the DataFrame.

        .. plot::
            :context: close-figs

            >>> ax = df.plot.bar(x='lifespan', rot=0)
        """
        return self(kind="bar", x=x, y=y, **kwds)

    def barh(self, x=None, y=None, **kwargs):
        """
        Make a horizontal bar plot.

        A horizontal bar plot is a plot that presents quantitative data with rectangular
        bars with lengths proportional to the values that they represent. A bar plot shows
        comparisons among discrete categories. One axis of the plot shows the specific
        categories being compared, and the other axis represents a measured value.

        Parameters
        ----------
        x : label or position, default DataFrame.index
            Column to be used for categories.
        y : label or position, default All numeric columns in dataframe
            Columns to be plotted from the DataFrame.
        **kwds:
            Keyword arguments to pass on to :meth:`databricks.koalas.DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        See Also
        --------
        matplotlib.axes.Axes.bar : Plot a vertical bar plot using matplotlib.

        Examples
        --------
        Basic example

        .. plot::
            :context: close-figs

            >>> df = ks.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> ax = df.plot.barh(x='lab', y='val')

        Plot a whole DataFrame to a horizontal bar plot

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = ks.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh()

        Plot a column of the DataFrame to a horizontal bar plot

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = ks.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh(y='speed')

        Plot DataFrame versus the desired column

        .. plot::
            :context: close-figs

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = ks.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> ax = df.plot.barh(x='lifespan')
        """
        return self(kind="barh", x=x, y=y, **kwargs)

    def hexbin(self, **kwds):
        return unsupported_function(class_name="pd.DataFrame", method_name="hexbin")()

    def box(self, **kwds):
        return unsupported_function(class_name="pd.DataFrame", method_name="box")()

    def hist(self, bins=10, **kwds):
        """
        Make a histogram of the DataFrame's.
        A `histogram`_ is a representation of the distribution of data.
        This function calls :meth:`matplotlib.pyplot.hist`, on each series in
        the DataFrame, resulting in one histogram per column.

        .. _histogram: https://en.wikipedia.org/wiki/Histogram

        Parameters
        ----------
        bins : integer or sequence, default 10
            Number of histogram bins to be used. If an integer is given, bins + 1
            bin edges are calculated and returned. If bins is a sequence, gives
            bin edges, including left edge of first bin and right edge of last
            bin. In this case, bins is returned unmodified.
        **kwds
            All other plotting keyword arguments to be passed to
            :meth:`matplotlib.pyplot.hist`.

        Returns
        -------
        matplotlib.AxesSubplot or numpy.ndarray of them

        See Also
        --------
        matplotlib.pyplot.hist : Plot a histogram using matplotlib.

        Examples
        --------
        When we draw a dice 6000 times, we expect to get each value around 1000
        times. But when we draw two dices and sum the result, the distribution
        is going to be quite different. A histogram illustrates those
        distributions.

        .. plot::
            :context: close-figs

            >>> df = pd.DataFrame(
            ...     np.random.randint(1, 7, 6000),
            ...     columns=['one'])
            >>> df['two'] = df['one'] + np.random.randint(1, 7, 6000)
            >>> df = ks.from_pandas(df)
            >>> ax = df.plot.hist(bins=12, alpha=0.5)
        """
        return self(kind="hist", bins=bins, **kwds)

    def scatter(self, x, y, s=None, c=None, **kwds):
        """
        Create a scatter plot with varying marker point size and color.

        The coordinates of each point are defined by two dataframe columns and
        filled circles are used to represent each point. This kind of plot is
        useful to see complex correlations between two variables. Points could
        be for instance natural 2D coordinates like longitude and latitude in
        a map or, in general, any pair of metrics that can be plotted against
        each other.

        Parameters
        ----------
        x : int or str
            The column name or column position to be used as horizontal
            coordinates for each point.
        y : int or str
            The column name or column position to be used as vertical
            coordinates for each point.
        s : scalar or array_like, optional
        c : str, int or array_like, optional

        **kwds: Optional
            Keyword arguments to pass on to :meth:`databricks.koalas.DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        See Also
        --------
        matplotlib.pyplot.scatter : Scatter plot using multiple input data
            formats.

        Examples
        --------
        Let's see how to draw a scatter plot using coordinates from the values
        in a DataFrame's columns.

        .. plot::
            :context: close-figs

            >>> df = ks.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
            ...                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
            ...                   columns=['length', 'width', 'species'])
            >>> ax1 = df.plot.scatter(x='length',
            ...                       y='width',
            ...                       c='DarkBlue')

        And now with the color determined by a column as well.

        .. plot::
            :context: close-figs

            >>> ax2 = df.plot.scatter(x='length',
            ...                       y='width',
            ...                       c='species',
            ...                       colormap='viridis')
        """
        return self(kind="scatter", x=x, y=y, s=s, c=c, **kwds)
