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
import matplotlib
import numpy as np
import pandas as pd

from matplotlib.axes._base import _process_plot_format
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._core import (
    _all_kinds, _get_standard_kind, _gca, BarPlot, BasePlotMethods, BoxPlot, HistPlot,
    is_list_like,
    is_integer, MPLPlot)

from databricks.koalas.missing import _unsupported_function
from pyspark.ml.feature import Bucketizer
from pyspark.sql import functions as F


class KoalasBarPlot(BarPlot):
    max_rows = 1000

    def __init__(self, data, **kwargs):
        # Simply use the first 1k elements and make it into a pandas dataframe
        # For categorical variables, it is likely called from df.x.value_counts().plot.bar()
        data = data.head(KoalasBarPlot.max_rows + 1).to_pandas().to_frame()
        self.partial = False
        if len(data) > KoalasBarPlot.max_rows:
            self.partial = True
            data = data.iloc[:KoalasBarPlot.max_rows]
        super().__init__(data, **kwargs)

    def _plot(self, ax, x, y, w, start=0, log=False, **kwds):
        if self.partial:
            ax.text(1, 1, 'showing top 1,000 elements only', size=6, ha='right', va='bottom',
                    transform=ax.transAxes)
            self.data = self.data.iloc[:KoalasBarPlot.max_rows]

        return ax.bar(x, y, w, bottom=start, log=log, **kwds)


class KoalasBoxPlotSummary:
    def __init__(self, data, colname):
        self.data = data
        self.colname = colname

    def compute_stats(self, whis, precision):
        # Computes mean, median, Q1 and Q3 with approx_percentile and precision
        pdf = (self.data._kdf._sdf
               .agg(*[F.expr('approx_percentile({}, {}, {})'.format(self.colname, q,
                                                                    1. / precision))
                      .alias('{}_{}%'.format(self.colname, int(q * 100)))
                      for q in [.25, .50, .75]],
                    F.mean(self.colname).alias('{}_mean'.format(self.colname))).toPandas())

        # Computes IQR and Tukey's fences
        iqr = '{}_iqr'.format(self.colname)
        p75 = '{}_75%'.format(self.colname)
        p25 = '{}_25%'.format(self.colname)
        pdf.loc[:, iqr] = pdf.loc[:, p75] - pdf.loc[:, p25]
        pdf.loc[:, '{}_lfence'.format(self.colname)] = pdf.loc[:, p25] - whis * pdf.loc[:, iqr]
        pdf.loc[:, '{}_ufence'.format(self.colname)] = pdf.loc[:, p75] + whis * pdf.loc[:, iqr]

        qnames = ['25%', '50%', '75%', 'mean', 'lfence', 'ufence']
        col_summ = pdf[['{}_{}'.format(self.colname, q) for q in qnames]]
        col_summ.columns = qnames
        lfence, ufence = col_summ['lfence'], col_summ['ufence']

        stats = {'mean': col_summ['mean'].values[0],
                 'med': col_summ['50%'].values[0],
                 'q1': col_summ['25%'].values[0],
                 'q3': col_summ['75%'].values[0]}

        return stats, (lfence.values[0], ufence.values[0])

    def outliers(self, lfence, ufence):
        # Builds expression to identify outliers
        expression = F.col(self.colname).between(lfence, ufence)
        # Creates a column to flag rows as outliers or not
        return self.data._kdf._sdf.withColumn('__{}_outlier'.format(self.colname), ~expression)

    def calc_whiskers(self, outliers):
        # Computes min and max values of non-outliers - the whiskers
        minmax = (outliers
                  .filter('not __{}_outlier'.format(self.colname))
                  .agg(F.min(self.colname).alias('min'),
                       F.max(self.colname).alias('max'))
                  .toPandas())
        return minmax.iloc[0][['min', 'max']].values

    def get_fliers(self, outliers):
        # Filters only the outliers, should "showfliers" be True
        fliers_df = outliers.filter('__{}_outlier'.format(self.colname))

        # If shows fliers, takes the top 1k with highest absolute values
        fliers = (fliers_df
                  .select(F.abs(F.col(self.colname)).alias(self.colname))
                  .orderBy(F.desc(self.colname))
                  .limit(1001)
                  .toPandas()[self.colname].values)

        return fliers


class KoalasBoxPlot(BoxPlot):
    @staticmethod
    def rc_defaults(notch=None, vert=None, whis=None,
                    patch_artist=None, bootstrap=None, meanline=None,
                    showmeans=None, showcaps=None, showbox=None,
                    showfliers=None, **kwargs):
        # Missing arguments default to rcParams.
        if whis is None:
            whis = matplotlib.rcParams['boxplot.whiskers']
        if bootstrap is None:
            bootstrap = matplotlib.rcParams['boxplot.bootstrap']

        if notch is None:
            notch = matplotlib.rcParams['boxplot.notch']
        if vert is None:
            vert = matplotlib.rcParams['boxplot.vertical']
        if patch_artist is None:
            patch_artist = matplotlib.rcParams['boxplot.patchartist']
        if meanline is None:
            meanline = matplotlib.rcParams['boxplot.meanline']
        if showmeans is None:
            showmeans = matplotlib.rcParams['boxplot.showmeans']
        if showcaps is None:
            showcaps = matplotlib.rcParams['boxplot.showcaps']
        if showbox is None:
            showbox = matplotlib.rcParams['boxplot.showbox']
        if showfliers is None:
            showfliers = matplotlib.rcParams['boxplot.showfliers']

        return dict(whis=whis, bootstrap=bootstrap, notch=notch, vert=vert,
                    patch_artist=patch_artist, meanline=meanline, showmeans=showmeans,
                    showcaps=showcaps, showbox=showbox, showfliers=showfliers)

    def boxplot(self, ax, bxpstats, notch=None, sym=None, vert=None,
                whis=None, positions=None, widths=None, patch_artist=None,
                bootstrap=None, usermedians=None, conf_intervals=None,
                meanline=None, showmeans=None, showcaps=None,
                showbox=None, showfliers=None, boxprops=None,
                labels=None, flierprops=None, medianprops=None,
                meanprops=None, capprops=None, whiskerprops=None,
                manage_xticks=True, autorange=False, zorder=None,
                precision=None):

        def _update_dict(dictionary, rc_name, properties):
            """ Loads properties in the dictionary from rc file if not already
            in the dictionary"""
            rc_str = 'boxplot.{0}.{1}'
            if dictionary is None:
                dictionary = dict()
            for prop_dict in properties:
                dictionary.setdefault(prop_dict,
                                      matplotlib.rcParams[rc_str.format(rc_name, prop_dict)])
            return dictionary

        # Common property dictionnaries loading from rc
        flier_props = ['color', 'marker', 'markerfacecolor', 'markeredgecolor',
                       'markersize', 'linestyle', 'linewidth']
        default_props = ['color', 'linewidth', 'linestyle']

        boxprops = _update_dict(boxprops, 'boxprops', default_props)
        whiskerprops = _update_dict(whiskerprops, 'whiskerprops',
                                    default_props)
        capprops = _update_dict(capprops, 'capprops', default_props)
        medianprops = _update_dict(medianprops, 'medianprops', default_props)
        meanprops = _update_dict(meanprops, 'meanprops', default_props)
        flierprops = _update_dict(flierprops, 'flierprops', flier_props)

        if patch_artist:
            boxprops['linestyle'] = 'solid'
            boxprops['edgecolor'] = boxprops.pop('color')

        # if non-default sym value, put it into the flier dictionary
        # the logic for providing the default symbol ('b+') now lives
        # in bxp in the initial value of final_flierprops
        # handle all of the `sym` related logic here so we only have to pass
        # on the flierprops dict.
        if sym is not None:
            # no-flier case, which should really be done with
            # 'showfliers=False' but none-the-less deal with it to keep back
            # compatibility
            if sym == '':
                # blow away existing dict and make one for invisible markers
                flierprops = dict(linestyle='none', marker='', color='none')
                # turn the fliers off just to be safe
                showfliers = False
            # now process the symbol string
            else:
                # process the symbol string
                # discarded linestyle
                _, marker, color = _process_plot_format(sym)
                # if we have a marker, use it
                if marker is not None:
                    flierprops['marker'] = marker
                # if we have a color, use it
                if color is not None:
                    # assume that if color is passed in the user want
                    # filled symbol, if the users want more control use
                    # flierprops
                    flierprops['color'] = color
                    flierprops['markerfacecolor'] = color
                    flierprops['markeredgecolor'] = color

        # replace medians if necessary:
        if usermedians is not None:
            if (len(np.ravel(usermedians)) != len(bxpstats) or
                    np.shape(usermedians)[0] != len(bxpstats)):
                raise ValueError('usermedians length not compatible with x')
            else:
                # reassign medians as necessary
                for stats, med in zip(bxpstats, usermedians):
                    if med is not None:
                        stats['med'] = med

        if conf_intervals is not None:
            if np.shape(conf_intervals)[0] != len(bxpstats):
                err_mess = 'conf_intervals length not compatible with x'
                raise ValueError(err_mess)
            else:
                for stats, ci in zip(bxpstats, conf_intervals):
                    if ci is not None:
                        if len(ci) != 2:
                            raise ValueError('each confidence interval must '
                                             'have two values')
                        else:
                            if ci[0] is not None:
                                stats['cilo'] = ci[0]
                            if ci[1] is not None:
                                stats['cihi'] = ci[1]

        artists = ax.bxp(bxpstats, positions=positions, widths=widths,
                         vert=vert, patch_artist=patch_artist,
                         shownotches=notch, showmeans=showmeans,
                         showcaps=showcaps, showbox=showbox,
                         boxprops=boxprops, flierprops=flierprops,
                         medianprops=medianprops, meanprops=meanprops,
                         meanline=meanline, showfliers=showfliers,
                         capprops=capprops, whiskerprops=whiskerprops,
                         manage_xticks=manage_xticks, zorder=zorder)
        return artists

    def _plot(self, ax, bxpstats, column_num=None, return_type='axes', **kwds):
        bp = self.boxplot(ax, bxpstats, **kwds)

        if return_type == 'dict':
            return bp, bp
        elif return_type == 'both':
            return self.BP(ax=ax, lines=bp), bp
        else:
            return ax, bp

    def _compute_plot_data(self):
        colname = self.data.name
        summary = KoalasBoxPlotSummary(self.data, colname)

        # Updates all props with the rc defaults from matplotlib
        self.kwds.update(KoalasBoxPlot.rc_defaults(**self.kwds))

        # Gets some important kwds
        showfliers = self.kwds.get('showfliers', False)
        whis = self.kwds.get('whis', 1.5)
        labels = self.kwds.get('labels', [colname])

        # This one is Koalas specific to control precision for approx_percentile
        precision = self.kwds.get('precision', 0.01)

        # # Computes mean, median, Q1 and Q3 with approx_percentile and precision
        col_stats, col_fences = summary.compute_stats(whis, precision)

        # # Creates a column to flag rows as outliers or not
        outliers = summary.outliers(*col_fences)

        # # Computes min and max values of non-outliers - the whiskers
        whiskers = summary.calc_whiskers(outliers)

        if showfliers:
            fliers = summary.get_fliers(outliers)
        else:
            fliers = []

        # Builds bxpstats dict
        stats = []
        item = {'mean': col_stats['mean'],
                'med': col_stats['med'],
                'q1': col_stats['q1'],
                'q3': col_stats['q3'],
                'whislo': whiskers[0],
                'whishi': whiskers[1],
                'fliers': fliers,
                'label': labels[0]}
        stats.append(item)

        self.data = {labels[0]: stats}

    def _make_plot(self):
        bxpstats = list(self.data.values())[0]
        ax = self._get_ax(0)
        kwds = self.kwds.copy()

        for stats in bxpstats:
            if len(stats['fliers']) > 1000:
                stats['fliers'] = stats['fliers'][:1000]
                ax.text(1, 1, 'showing top 1,000 fliers only', size=6, ha='right', va='bottom',
                        transform=ax.transAxes)

        ret, bp = self._plot(ax, bxpstats, column_num=0,
                             return_type=self.return_type, **kwds)
        self.maybe_color_bp(bp)
        self._return_obj = ret

        labels = [l for l, _ in self.data.items()]
        labels = [pprint_thing(l) for l in labels]
        if not self.use_index:
            labels = [pprint_thing(key) for key in range(len(labels))]
        self._set_ticklabels(ax, labels)


class KoalasHistPlotSummary:
    def __init__(self, data, colname):
        self.data = data
        self.colname = colname

    def get_bins(self, n_bins):
        boundaries = (self.data._kdf._sdf
                      .agg(F.min(self.colname),
                           F.max(self.colname))
                      .rdd
                      .map(tuple)
                      .collect()[0])
        # divides the boundaries into bins
        return np.linspace(boundaries[0], boundaries[1], n_bins + 1)

    def calc_histogram(self, bins):
        bucket_name = '__{}_bucket'.format(self.colname)
        # creates a Bucketizer to get corresponding bin of each value
        bucketizer = Bucketizer(splits=bins,
                                inputCol=self.colname,
                                outputCol=bucket_name,
                                handleInvalid="skip")
        # after bucketing values, groups and counts them
        result = (bucketizer
                  .transform(self.data._kdf._sdf)
                  .select(bucket_name)
                  .groupby(bucket_name)
                  .agg(F.count('*').alias('count'))
                  .toPandas()
                  .sort_values(by=bucket_name))

        # generates a pandas DF with one row for each bin
        # we need this as some of the bins may be empty
        indexes = pd.DataFrame({bucket_name: np.arange(0, len(bins) - 1),
                                'bucket': bins[:-1]})
        # merges the bins with counts on it and fills remaining ones with zeros
        data = indexes.merge(result, how='left', on=[bucket_name]).fillna(0)[['count']]
        data.columns = [bucket_name]

        return data


class KoalasHistPlot(HistPlot):
    def _args_adjust(self):
        if is_integer(self.bins):
            summary = KoalasHistPlotSummary(self.data, self.data.name)
            # computes boundaries for the column
            self.bins = summary.get_bins(self.bins)

        if is_list_like(self.bottom):
            self.bottom = np.array(self.bottom)

    @classmethod
    def _plot(cls, ax, y, style=None, bins=None, bottom=0, column_num=0,
              stacking_id=None, **kwds):
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(bins) - 1)

        base = np.zeros(len(bins) - 1)
        bottom = bottom + \
            cls._get_stacked_values(ax, stacking_id, base, kwds['label'])

        # Since the counts were computed already, we use them as weights and just generate
        # one entry for each bin
        n, bins, patches = ax.hist(bins[:-1], bins=bins, bottom=bottom, weights=y, **kwds)

        cls._update_stacker(ax, stacking_id, n)
        return patches

    def _compute_plot_data(self):
        summary = KoalasHistPlotSummary(self.data, self.data.name)
        # generates a pandas DF with one row for each bin
        self.data = summary.calc_histogram(self.bins)


_klasses = [KoalasHistPlot, KoalasBarPlot, KoalasBoxPlot]
_plot_klass = {getattr(klass, '_kind'): klass for klass in _klasses}


def plot_series(data, kind='line', ax=None,                    # Series unique
                figsize=None, use_index=True, title=None, grid=None,
                legend=False, style=None, logx=False, logy=False, loglog=False,
                xticks=None, yticks=None, xlim=None, ylim=None,
                rot=None, fontsize=None, colormap=None, table=False,
                yerr=None, xerr=None,
                label=None, secondary_y=False,                 # Series unique
                **kwds):
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
    `**kwds` : keywords
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
        ax = _gca()
        ax = MPLPlot._get_ax_layer(ax)
    return _plot(data, kind=kind, ax=ax,
                 figsize=figsize, use_index=use_index, title=title,
                 grid=grid, legend=legend,
                 style=style, logx=logx, logy=logy, loglog=loglog,
                 xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim,
                 rot=rot, fontsize=fontsize, colormap=colormap, table=table,
                 yerr=yerr, xerr=xerr,
                 label=label, secondary_y=secondary_y,
                 **kwds)


def _plot(data, x=None, y=None, subplots=False,
          ax=None, kind='line', **kwds):

    # function copied from pandas.plotting._core
    # and adapted to handle Koalas DataFrame and Series

    kind = _get_standard_kind(kind.lower().strip())
    if kind in _all_kinds:
        klass = _plot_klass[kind]
    else:
        raise ValueError("%r is not a valid plot kind" % kind)

    plot_obj = klass(data, subplots=subplots, ax=ax, kind=kind, **kwds)
    plot_obj.generate()
    plot_obj.draw()
    return plot_obj.result


class KoalasSeriesPlotMethods(BasePlotMethods):
    """
    Series plotting accessor and method.

    Plotting methods can also be accessed by calling the accessor as a method
    with the ``kind`` argument:
    ``s.plot(kind='hist')`` is equivalent to ``s.plot.hist()``
    """

    def __call__(self, kind='line', ax=None,
                 figsize=None, use_index=True, title=None, grid=None,
                 legend=False, style=None, logx=False, logy=False,
                 loglog=False, xticks=None, yticks=None,
                 xlim=None, ylim=None,
                 rot=None, fontsize=None, colormap=None, table=False,
                 yerr=None, xerr=None,
                 label=None, secondary_y=False, **kwds):
        if LooseVersion(pd.__version__) < LooseVersion('0.24'):
            data = self._data
        else:
            data = self._parent
        return plot_series(data, kind=kind, ax=ax, figsize=figsize,
                           use_index=use_index, title=title, grid=grid,
                           legend=legend, style=style, logx=logx, logy=logy,
                           loglog=loglog, xticks=xticks, yticks=yticks,
                           xlim=xlim, ylim=ylim, rot=rot, fontsize=fontsize,
                           colormap=colormap, table=table, yerr=yerr,
                           xerr=xerr, label=label, secondary_y=secondary_y,
                           **kwds)
    __call__.__doc__ = plot_series.__doc__

    def line(self, **kwds):
        return _unsupported_function(class_name='pd.Series', method_name='line')()

    def bar(self, **kwds):
        """
        Vertical bar plot.

        Parameters
        ----------
        `**kwds` : optional
            Additional keyword arguments are documented in
            :meth:`Koalas.Series.plot`.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them
        """
        return self(kind='bar', **kwds)

    def barh(self, **kwds):
        return _unsupported_function(class_name='pd.Series', method_name='barh')()

    def box(self, **kwds):
        """
        Boxplot.

        Parameters
        ----------
        `**kwds` : optional
            Additional keyword arguments are documented in
            :meth:`Koalas.Series.plot`.

        `precision`: scalar, default = 0.01
            This argument is used by Koalas to compute approximate statistics
            for building a boxplot. Use *smaller* values to get more precise
            statistics.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        Notes
        -----
        There are behavior differences between Koalas and pandas.

        * Koalas computes approximate statistics - expect differences between
          pandas and Koalas boxplots, especially regarding 1st and 3rd quartiles.
        * The `whis` argument is only supported as a single number.
        * Koalas doesn't support the following argument(s).

          * `bootstrap` argument is not supported
          * `autorange` argument is not supported
        """
        return self(kind='box', **kwds)

    def hist(self, bins=10, **kwds):
        """
        Histogram.

        Parameters
        ----------
        bins : integer, default 10
            Number of histogram bins to be used
        `**kwds` : optional
            Additional keyword arguments are documented in
            :meth:`Koalas.Series.plot`.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them
        """
        return self(kind='hist', bins=bins, **kwds)

    def kde(self, bw_method=None, ind=None, **kwds):
        return _unsupported_function(class_name='pd.Series', method_name='kde')()

    density = kde

    def area(self, **kwds):
        return _unsupported_function(class_name='pd.Series', method_name='area')()

    def pie(self, **kwds):
        return _unsupported_function(class_name='pd.Series', method_name='pie')()
