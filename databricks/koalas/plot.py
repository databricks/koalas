import numpy as np
import pandas as pd
from pandas.core.generic import _shared_docs
from pandas.plotting._core import (
    _gca, MPLPlot, BasePlotMethods, _get_standard_kind, _all_kinds, _dataframe_kinds, _series_kinds,
    is_list_like, is_integer, string_types, HistPlot, _shared_doc_series_kwargs
)
from pandas.util._decorators import Appender
from pyspark.ml.feature import Bucketizer
from pyspark.sql import functions as F

class KoalasHistPlot(HistPlot):
    def _args_adjust(self):
        if is_integer(self.bins):
            # create common bin edge
            self.bins = np.linspace(*self.data._kdf._sdf
                                    .agg(F.min(self.data._scol),
                                         F.max(self.data._scol))
                                    .rdd
                                    .map(tuple)
                                    .collect()[0],
                                    self.bins + 1)

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
        # ignore style

        # Since the counts were computed already, we use them as weights and just generate
        # one entry for each bin
        n, bins, patches = ax.hist(bins[:-1], bins=bins, bottom=bottom, weights=y, **kwds)

        cls._update_stacker(ax, stacking_id, n)
        return patches

    def _compute_plot_data(self):
        data = self.data
        colname = data.name

        bucket_name = '__{}_bucket'.format(colname)
        bucketizer = Bucketizer(splits=self.bins, inputCol=colname, outputCol=bucket_name, handleInvalid="skip")
        result = (bucketizer
                  .transform(data._kdf._sdf)
                  .select(bucket_name)
                  .groupby(bucket_name)
                  .agg(F.count('*').alias('count'))
                  .toPandas()
                  .sort_values(by=bucket_name))

        indexes = pd.DataFrame({bucket_name: np.arange(0, len(self.bins) - 1), 'bucket': self.bins[:-1]})
        self.data = indexes.merge(result, how='left', on=[bucket_name]).fillna(0)[['count']]
        self.data.columns = [bucket_name]

_klasses = [KoalasHistPlot]
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
                # converted to series actually. copy to not modify
                data = data[y].copy()
                data.index.name = y
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

                # don't overwrite
                data = data[y].copy()

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

    def line(self, **kwds):
        """
        Line plot.

        Parameters
        ----------
        `**kwds` : optional
            Additional keyword arguments are documented in
            :meth:`pandas.Series.plot`.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        Examples
        --------

        .. plot::
            :context: close-figs

            >>> s = pd.Series([1, 3, 2])
            >>> s.plot.line()
        """
        return self(kind='line', **kwds)

    def bar(self, **kwds):
        """
        Vertical bar plot.

        Parameters
        ----------
        `**kwds` : optional
            Additional keyword arguments are documented in
            :meth:`pandas.Series.plot`.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them
        """
        return self(kind='bar', **kwds)

    def barh(self, **kwds):
        """
        Horizontal bar plot.

        Parameters
        ----------
        `**kwds` : optional
            Additional keyword arguments are documented in
            :meth:`pandas.Series.plot`.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them
        """
        return self(kind='barh', **kwds)

    def box(self, **kwds):
        """
        Boxplot.

        Parameters
        ----------
        `**kwds` : optional
            Additional keyword arguments are documented in
            :meth:`pandas.Series.plot`.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them
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
            :meth:`pandas.Series.plot`.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them
        """
        return self(kind='hist', bins=bins, **kwds)
