"""
TODO:
1. Check the documentation and examples
"""
from databricks.koalas import plot
from databricks.koalas.plot import *
from databricks.koalas.missing import unsupported_function

class PlotAccessor(PandasObject):
    """
    Series/Frames plotting accessor and method.

    Uses the backend specified by the
    option ``plotting.backend``. By default, matplotlib is used.

    Plotting methods can also be accessed by calling the accessor as a method
    with the ``kind`` argument:
    ``s.plot(kind='hist')`` is equivalent to ``s.plot.hist()``
    """
    # from databricks.koalas import DataFrame, Series

    _common_kinds = ('area', 'bar', 'barh',
                     'box', 'hist', 'kde', 'line', 'pie')
    _series_kinds = () + _common_kinds
    _dataframe_kinds = ("scatter", "hexbin") + _common_kinds
    _kind_aliases = {"density": "kde"}
    _all_kinds = _common_kinds + _series_kinds + _dataframe_kinds

    def __init__(self, data):
        self.data = data

    @staticmethod
    def _format_args(backend_name, data, kind, kwargs):
        """
        TODO: Add something here
        """
        from databricks.koalas import DataFrame, Series

        data_preprocessor_map = {
            "pie": TopNPlot().get_top_n,
            "bar": TopNPlot().get_top_n,
            "barh": TopNPlot().get_top_n,
            "scatter": TopNPlot().get_top_n,
            "area": SampledPlot().get_sampled,
            "line": SampledPlot().get_sampled,
        }
        # make the arguments values of matplotlib compatible with that of plotting backend
        args_map = {
            "plotly": {
                "logx": "log_x",
                "logy": "log_y",
                "xlim": "range_x",
                "ylim": "range_y",
                "yerr": "error_y",
                "xerr": "error_x",
            }
        }

        if isinstance(data, Series):
            positional_args = {
                ("ax", None),
                ("figsize", None),
                ("use_index", True),
                ("title", None),
                ("grid", None),
                ("legend", False),
                ("style", None),
                ("logx", False),
                ("logy", False),
                ("loglog", False),
                ("xticks", None),
                ("yticks", None),
                ("xlim", None),
                ("ylim", None),
                ("rot", None),
                ("fontsize", None),
                ("colormap", None),
                ("table", False),
                ("yerr", None),
                ("xerr", None),
                ("label", None),
                ("secondary_y", False),
            }
        elif isinstance(data, DataFrame):
            positional_args = {
                ("x", None),
                ("y", None),
                ("ax", None),
                ("subplots", None),
                ("sharex", None),
                ("sharey", False),
                ("layout", None),
                ("figsize", None),
                ("use_index", True),
                ("title", None),
                ("grid", None),
                ("legend", True),
                ("style", None),
                ("logx", False),
                ("logy", False),
                ("loglog", False),
                ("xticks", None),
                ("yticks", None),
                ("xlim", None),
                ("ylim", None),
                ("rot", None),
                ("fontsize", None),
                ("colormap", None),
                ("table", False),
                ("yerr", None),
                ("xerr", None),
                ("secondary_y", False),
                ("sort_columns", False),
            }
        # removing keys that are not required
        things_to_remove = ["self","kind","data","data","kwargs", "backend"]
        # no need to map and remove anything  if the backedn is the default
        for temp in things_to_remove:
            kwargs.pop(temp, None)

        for arg, def_val in positional_args:
            # map the argument if possible
            if backend_name != "databricks.koalas.plot":
                if backend_name in args_map:
                    if arg in kwargs and arg in args_map[backend_name]:
                        kwargs[args_map[backend_name][arg]] = kwargs.pop(arg)
                # remove argument is default and not mapped
                if arg in kwargs and kwargs[arg] == def_val:
                    kwargs.pop(arg, None)
            else:
                if arg not in kwargs:
                    kwargs[arg] = def_val
        return data_preprocessor_map[kind](data), kwargs

    def __call__(
        self,
        kind="line",
        backend="matplotlib",
        **kwargs
    ):

        positional_args = locals()
        plot_backend = plot._get_plot_backend(backend)
        args = {**positional_args, **kwargs}
        # when using another backend, let the backend take the charge
        plot_data, kwds = self._format_args(
            plot_backend.__name__, self.data, kind, args)

        if plot_backend.__name__ != "databricks.koalas.plot":
            return plot_backend.plot(plot_data, kind=kind, **kwds)

        if kind not in self._all_kinds:
            raise ValueError(f"{kind} is not a valid plot kind")

        from databricks.koalas import DataFrame, Series

        if isinstance(self.data, Series):
            if kind not in self._series_kinds:
                return  unsupported_function(class_name="pd.Series", method_name=kind)()
            return plot_series(data=self.data, kind=kind, **kwds)
        elif isinstance(self.data, DataFrame):
            if kind not in self._dataframe_kinds:
                return unsupported_function(class_name="pd.DataFrame", method_name=kind)()

            return plot_frame(data=self.data, kind=kind, **kwds)

    # added this, as it was present before
    __call__.__doc__ = plot_frame.__doc__
    __call__.__doc__ = plot_series.__doc__

    def line(self, x=None, y=None, **kwargs):
        """
        Plot DataFrame/Series as lines.

        This function is useful to plot lines using Series's values
        as coordinates.

        Parameters
        ----------
        x : int or str, optional
            Columns to use for the horizontal axis.
            Either the location or the label of the columns to be used.
            By default, it will use the DataFrame indices.
        y : int, str, or list of them, optional
            The values to be plotted.
            Either the location or the label of the columns to be used.
            By default, it will use the remaining DataFrame numeric columns.
        **kwds
            Keyword arguments to pass on to :meth:`Series.plot` or :meth:`DataFrame.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or :class:`numpy.ndarray`
            Return an ndarray when ``subplots=True``.
            Return an custom object when ``backend!=matplotlib``.

        See Also
        --------
        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers.

        Examples
        --------
        Basic plot.

        .. plot::
            :context: close-figs

            >>> s = ks.Series([1, 3, 2])
            >>> ax = s.plot.line()

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
            :meth:`Koalas.Series.plot` or :meth:`Koalas.DataFrame.plot`.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        Examples
        --------
        Basic plot.

        .. plot::
            :context: close-figs

            >>> s = ks.Series([1, 3, 2])
            >>> ax = s.plot.bar()

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
        if isinstance(self.data, Series):
            return self(kind="bar", **kwds)
        elif isinstance(self.data, DataFrame):
            return self(kind="bar", x=x, y=y, **kwds)

    def barh(self, x=None, y=None, **kwargs):
        """
        Make a horizontal bar plot.

        A horizontal bar plot is a plot that presents quantitative data with
        rectangular bars with lengths proportional to the values that they
        represent. A bar plot shows comparisons among discrete categories. One
        axis of the plot shows the specific categories being compared, and the
        other axis represents a measured value.

        Parameters
        ----------
        x : label or position, default DataFrame.index
            Column to be used for categories.
        y : label or position, default All numeric columns in dataframe
            Columns to be plotted from the DataFrame.
        **kwds
            Keyword arguments to pass on to
            :meth:`databricks.koalas.DataFrame.plot` or :meth:`databricks.koalas.Series.plot`.

        Returns
        -------
        :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        See Also
        --------
        matplotlib.axes.Axes.bar : Plot a vertical bar plot using matplotlib.

        Examples
        --------
        .. plot::
            :context: close-figs

            >>> df = ks.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> plot = df.val.plot.barh()

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
        if isinstance(self.data, Series):
            return self(kind="barh", **kwds)
        elif isinstance(self.data, DataFrame):
            return self(kind="barh", x=x, y=y, **kwargs)

    def box(self, **kwds):
        """
        Make a box plot of the Series columns.

        Parameters
        ----------
        **kwds : optional
            Additional keyword arguments are documented in
            :meth:`Koalas.Series.plot`.

        precision: scalar, default = 0.01
            This argument is used by Koalas to compute approximate statistics
            for building a boxplot. Use *smaller* values to get more precise
            statistics.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them
            Return an custom object when ``backend!=matplotlib``.

        Notes
        -----
        There are behavior differences between Koalas and pandas.

        * Koalas computes approximate statistics - expect differences between
          pandas and Koalas boxplots, especially regarding 1st and 3rd quartiles.
        * The `whis` argument is only supported as a single number.
        * Koalas doesn't support the following argument(s).

          * `bootstrap` argument is not supported
          * `autorange` argument is not supported

        Examples
        --------
        Draw a box plot from a DataFrame with four columns of randomly
        generated data.

        .. plot::
            :context: close-figs

            >>> data = np.random.randn(25, 4)
            >>> df = ks.DataFrame(data, columns=list('ABCD'))
            >>> ax = df['A'].plot.box()
        """
        if isinstance(self.data, Series):
            return self(kind="box", **kwds)
        elif isinstance(self.data, DataFrame):
            return unsupported_function(class_name="pd.DataFrame", method_name="box")()

    def hist(self, bins=10, **kwds):
        """
        Draw one histogram of the DataFrame’s columns.
        A `histogram`_ is a representation of the distribution of data.
        This function calls :meth:`matplotlib.pyplot.hist` or :meth:`plotting.backend.plot`, on each series in
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
            :meth:`matplotlib.pyplot.hist`
            Koalas.Series.plot.

        Returns
        -------
        axes : :class:`matplotlib.axes.Axes` or numpy.ndarray of them

        Examples
        --------
        Basic plot.

        .. plot::
            :context: close-figs

            >>> s = ks.Series([1, 3, 2])
            >>> ax = s.plot.hist()
        """
        return self(kind="hist", bins=bins, **kwds)

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
            Keyword arguments to pass on to :meth:`Koalas.Series.plot`.

        Returns
        -------
        matplotlib.axes.Axes or numpy.ndarray of them

        Examples
        --------
        A scalar bandwidth should be specified. Using a small bandwidth value can
        lead to over-fitting, while using a large bandwidth value may result
        in under-fitting:

        .. plot::
            :context: close-figs

            >>> s = ks.Series([1, 2, 2.5, 3, 3.5, 4, 5])
            >>> ax = s.plot.kde(bw_method=0.3)

        .. plot::
            :context: close-figs

            >>> ax = s.plot.kde(bw_method=3)

        The `ind` parameter determines the evaluation points for the
        plot of the estimated KDF:

        .. plot::
            :context: close-figs

            >>> ax = s.plot.kde(ind=[1, 2, 3, 4, 5], bw_method=0.3)

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
        return self(kind="kde", bw_method=bw_method, ind=ind, **kwargs)

    density = kde

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
            >>> plot = df.sales.plot.area()

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
        if isinstance(self.data, Series):
            return self(kind="area", **kwds)
        elif isinstance(self.data, DataFrame):
            return self(kind="area", x=x, y=y, stacked=stacked, **kwds)

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
            Keyword arguments to pass on to :meth:`Koalas.Series.plot`.

        Returns
        -------
        matplotlib.axes.Axes or np.ndarray of them
            A NumPy array is returned when `subplots` is True.

        Examples
        --------

        .. plot::
            :context: close-figs

            >>> df = ks.DataFrame({'mass': [0.330, 4.87, 5.97],
            ...                    'radius': [2439.7, 6051.8, 6378.1]},
            ...                   index=['Mercury', 'Venus', 'Earth'])
            >>> plot = df.mass.plot.pie(figsize=(5, 5))

        .. plot::
            :context: close-figs

            >>> plot = df.mass.plot.pie(subplots=True, figsize=(6, 3))

        .. plot::
            :context: close-figs

            >>> plot = df.plot.pie(y='mass', figsize=(5, 5))
        """
        if isinstance(self.data, Series):
            return self(kind="pie", **kwds)
        else:
            # pandas will raise an error if y is None and subplots if not True
            if isinstance(self.data, DataFrame) and y is None and not kwds.get("subplots", False):
                raise ValueError("pie requires either y column or 'subplots=True'")
            return self(kind="pie", y=y, **kwds)

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

    def hexbin(self, **kwds):
        return unsupported_function(class_name="pd.DataFrame", method_name="hexbin")()
