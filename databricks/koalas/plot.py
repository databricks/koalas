from databricks.koalas.dask.utils import derived_from
from databricks.koalas.missing import _unsupported_function
import matplotlib
from matplotlib.axes._base import _process_plot_format
import numpy as np
import pandas as pd
from pandas.core.generic import _shared_docs
from pandas.io.formats.printing import pprint_thing
from pandas.plotting._core import (
    _all_kinds, _dataframe_kinds, _get_standard_kind, _gca, _series_kinds,
    _shared_doc_series_kwargs, BarPlot, BasePlotMethods, BoxPlot, HistPlot, is_list_like,
    is_integer, MPLPlot, string_types
)
from pandas.util._decorators import Appender
from pyspark.ml.feature import Bucketizer
from pyspark.sql import functions as F


class KoalasBarPlot(BarPlot):
    def _compute_plot_data(self):
        # Simply use the first 1k elements and make it into a pandas dataframe
        # For categorical variables, it is likely called from df.x.value_counts().plot.bar()
        self.data = self.data.head(1000).to_pandas().to_frame()


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
                manage_xticks=True, autorange=False, zorder=None):

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
        # Updates all props with the rc defaults from matplotlib
        self.kwds.update(KoalasBoxPlot.rc_defaults(**self.kwds))

        # Gets some important kwds
        showfliers = self.kwds.get('showfliers', False)
        whis = self.kwds.get('whis', 1.5)
        labels = self.kwds.get('labels', colname)

        # This one is Koalas specific to control precision for approx_percentile
        precision = self.kwds.get('precision', 0.01)

        # The code below was originally built to handle multiple columns
        # It is currently handling a single column, though

        # Computes mean, median, Q1 and Q3 with approx_percentile and precision
        pdf = (self.data._kdf._sdf
               .agg(*[F.expr('approx_percentile({}, {}, {})'.format(colname, q, 1. / precision))
                      .alias('{}_{}%'.format(colname, int(q * 100)))
                      for q in [.25, .50, .75]],
                    F.mean(colname).alias('{}_mean'.format(colname))).toPandas())

        # Computes IQR and Tukey's fences
        iqr = '{}_iqr'.format(colname)
        p75 = '{}_75%'.format(colname)
        p25 = '{}_25%'.format(colname)
        pdf.loc[:, iqr] = pdf.loc[:, p75] - pdf.loc[:, p25]
        pdf.loc[:, '{}_lfence'.format(colname)] = pdf.loc[:, p25] - whis * pdf.loc[:, iqr]
        pdf.loc[:, '{}_ufence'.format(colname)] = pdf.loc[:, p75] + whis * pdf.loc[:, iqr]

        qnames = ['25%', '50%', '75%', 'mean', 'lfence', 'ufence']
        col_summ = pdf[['{}_{}'.format(colname, q) for q in qnames]]
        col_summ.columns = qnames
        lfence, ufence = col_summ[['lfence']], col_summ[['ufence']]

        # Builds expression to identify outliers
        expression = F.col(colname).between(lfence.iloc[0, 0], ufence.iloc[0, 0])
        # Creates a column to flag rows as outliers or not
        outlier = self.data._kdf._sdf.withColumn('__{}_outlier'.format(colname), ~expression)

        # Computes min and max values of non-outliers - the whiskers
        minmax = (outlier
                  .filter('not __{}_outlier'.format(colname))
                  .agg(F.min(colname).alias('min'),
                       F.max(colname).alias('max'))
                  .toPandas())
        whiskers = minmax.iloc[0][['min', 'max']].values

        # Filters only the outliers, should "showfliers" be True
        fliers_df = outlier.filter('__{}_outlier'.format(colname))

        if showfliers:
            # If shows fliers, takes the top 1k with highest absolute values
            fliers = (fliers_df
                      .select(F.abs(F.col(colname)).alias(colname))
                      .orderBy(F.desc(colname))
                      .limit(1000)
                      .toPandas()[colname].values)
        else:
            fliers = []

        # Builds bxpstats dict
        stats = []
        item = {'mean': col_summ['mean'].values[0],
                'med': col_summ['50%'].values[0],
                'q1': col_summ['25%'].values[0],
                'q3': col_summ['75%'].values[0],
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

        ret, bp = self._plot(ax, bxpstats, column_num=0,
                             return_type=self.return_type, **kwds)
        self.maybe_color_bp(bp)
        self._return_obj = ret

        labels = [l for l, _ in self.data.items()]
        labels = [pprint_thing(l) for l in labels]
        if not self.use_index:
            labels = [pprint_thing(key) for key in range(len(labels))]
        self._set_ticklabels(ax, labels)


class KoalasHistPlot(HistPlot):
    def _args_adjust(self):
        if is_integer(self.bins):
            # computes boundaries for the column
            boundaries = (self.data._kdf._sdf
                          .agg(F.min(self.data._scol),
                               F.max(self.data._scol))
                          .rdd
                          .map(tuple)
                          .collect()[0])
            # divides the boundaries into bins
            self.bins = np.linspace(boundaries[0], boundaries[1], self.bins + 1)

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
        data = self.data
        colname = data.name

        bucket_name = '__{}_bucket'.format(colname)
        # creates a Bucketizer to get corresponding bin of each value
        bucketizer = Bucketizer(splits=self.bins,
                                inputCol=colname,
                                outputCol=bucket_name,
                                handleInvalid="skip")
        # after bucketing values, groups and counts them
        result = (bucketizer
                  .transform(data._kdf._sdf)
                  .select(bucket_name)
                  .groupby(bucket_name)
                  .agg(F.count('*').alias('count'))
                  .toPandas()
                  .sort_values(by=bucket_name))

        # generates a pandas DF with one row for each bin
        # we need this as some of the bins may be empty
        indexes = pd.DataFrame({bucket_name: np.arange(0, len(self.bins) - 1),
                                'bucket': self.bins[:-1]})
        # merges the bins with counts on it and fills remaining ones with zeros
        self.data = indexes.merge(result, how='left', on=[bucket_name]).fillna(0)[['count']]
        self.data.columns = [bucket_name]


_klasses = [KoalasHistPlot, KoalasBarPlot, KoalasBoxPlot]
_plot_klass = {klass._kind: klass for klass in _klasses}


@Appender(_shared_docs['plot'] % _shared_doc_series_kwargs)
def plot_series(data, kind='line', ax=None,                    # Series unique
                figsize=None, use_index=True, title=None, grid=None,
                legend=False, style=None, logx=False, logy=False, loglog=False,
                xticks=None, yticks=None, xlim=None, ylim=None,
                rot=None, fontsize=None, colormap=None, table=False,
                yerr=None, xerr=None,
                label=None, secondary_y=False,                 # Series unique
                **kwds):

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

    from databricks.koalas.frame import DataFrame
    from databricks.koalas.series import Series

    kind = _get_standard_kind(kind.lower().strip())
    if kind in _all_kinds:
        klass = _plot_klass[kind]
    else:
        raise ValueError("%r is not a valid plot kind" % kind)

    if kind in _dataframe_kinds:
        if isinstance(data, DataFrame):
            plot_obj = klass(data, x=x, y=y, subplots=subplots, ax=ax,
                             kind=kind, **kwds)
        else:
            raise ValueError("plot kind %r can only be used for data frames"
                             % kind)

    elif kind in _series_kinds:
        if isinstance(data, DataFrame):
            if y is None and subplots is False:
                msg = "{0} requires either y column or 'subplots=True'"
                raise ValueError(msg.format(kind))
            elif y is not None:
                if is_integer(y) and not data.columns.holds_integer():
                    y = data.columns[y]
                # converted to series
                data = data[y]

        plot_obj = klass(data, subplots=subplots, ax=ax, kind=kind, **kwds)
    else:
        if isinstance(data, DataFrame):
            data_cols = data.columns
            if x is not None:
                if is_integer(x) and not data.columns.holds_integer():
                    x = data_cols[x]
                elif not isinstance(data[x], Series):
                    raise ValueError("x must be a label or position")
                data = data.set_index(x)

            if y is not None:
                # check if we have y as int or list of ints
                int_ylist = is_list_like(y) and all(is_integer(c) for c in y)
                int_y_arg = is_integer(y) or int_ylist
                if int_y_arg and not data.columns.holds_integer():
                    y = data_cols[y]

                label_kw = kwds['label'] if 'label' in kwds else False
                for kw in ['xerr', 'yerr']:
                    if (kw in kwds) and \
                        (isinstance(kwds[kw], string_types) or
                            is_integer(kwds[kw])):
                        try:
                            kwds[kw] = data[kwds[kw]]
                        except (IndexError, KeyError, TypeError):
                            pass

                data = data[y]

                if isinstance(data, Series):
                    label_name = label_kw or y
                    data.name = label_name
                else:
                    match = is_list_like(label_kw) and len(label_kw) == len(y)
                    if label_kw and not match:
                        raise ValueError(
                            "label should be list-like and same length as y"
                        )
                    label_name = label_kw or data.columns
                    data.columns = label_name

        plot_obj = klass(data, subplots=subplots, ax=ax, kind=kind, **kwds)

    plot_obj.generate()
    plot_obj.draw()
    return plot_obj.result


class KoalasSeriesPlotMethods(BasePlotMethods):
    """
    Series plotting accessor and method.

    Examples
    --------
    >>> s.plot.line()
    >>> s.plot.bar()
    >>> s.plot.hist()

    Plotting methods can also be accessed by calling the accessor as a method
    with the ``kind`` argument:
    ``s.plot(kind='line')`` is equivalent to ``s.plot.line()``
    """

    def __call__(self, kind='line', ax=None,
                 figsize=None, use_index=True, title=None, grid=None,
                 legend=False, style=None, logx=False, logy=False,
                 loglog=False, xticks=None, yticks=None,
                 xlim=None, ylim=None,
                 rot=None, fontsize=None, colormap=None, table=False,
                 yerr=None, xerr=None,
                 label=None, secondary_y=False, **kwds):
        return plot_series(self._parent, kind=kind, ax=ax, figsize=figsize,
                           use_index=use_index, title=title, grid=grid,
                           legend=legend, style=style, logx=logx, logy=logy,
                           loglog=loglog, xticks=xticks, yticks=yticks,
                           xlim=xlim, ylim=ylim, rot=rot, fontsize=fontsize,
                           colormap=colormap, table=table, yerr=yerr,
                           xerr=xerr, label=label, secondary_y=secondary_y,
                           **kwds)
    __call__.__doc__ = plot_series.__doc__

    @derived_from(pd.plotting._core.SeriesPlotMethods)
    def line(self, **kwds):
        return _unsupported_function(class_name='pd.Series', method_name='line')

    @derived_from(pd.plotting._core.SeriesPlotMethods)
    def bar(self, **kwds):
        return self(kind='bar', **kwds)

    @derived_from(pd.plotting._core.SeriesPlotMethods)
    def barh(self, **kwds):
        return _unsupported_function(class_name='pd.Series', method_name='barh')

    @derived_from(pd.plotting._core.SeriesPlotMethods)
    def box(self, **kwds):
        return self(kind='box', **kwds)

    @derived_from(pd.plotting._core.SeriesPlotMethods)
    def hist(self, bins=10, **kwds):
        return self(kind='hist', bins=bins, **kwds)

    @derived_from(pd.plotting._core.SeriesPlotMethods)
    def kde(self, bw_method=None, ind=None, **kwds):
        return _unsupported_function(class_name='pd.Series', method_name='kde')

    density = kde

    @derived_from(pd.plotting._core.SeriesPlotMethods)
    def area(self, **kwds):
        return _unsupported_function(class_name='pd.Series', method_name='area')

    @derived_from(pd.plotting._core.SeriesPlotMethods)
    def pie(self, **kwds):
        return _unsupported_function(class_name='pd.Series', method_name='pie')
