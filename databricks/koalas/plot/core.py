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

import importlib

import pandas as pd
import numpy as np
from pyspark.ml.feature import Bucketizer
from pyspark.mllib.stat import KernelDensity
from pyspark.sql import functions as F
from pandas.core.base import PandasObject
from pandas.core.dtypes.inference import is_integer

from databricks.koalas.missing import unsupported_function
from databricks.koalas.config import get_option
from databricks.koalas.utils import name_like_string


class TopNPlotBase:
    def get_top_n(self, data):
        from databricks.koalas import DataFrame, Series

        max_rows = get_option("plotting.max_rows")
        # Simply use the first 1k elements and make it into a pandas dataframe
        # For categorical variables, it is likely called from df.x.value_counts().plot.xxx().
        if isinstance(data, (Series, DataFrame)):
            data = data.head(max_rows + 1).to_pandas()
        else:
            raise ValueError("Only DataFrame and Series are supported for plotting.")

        self.partial = False
        if len(data) > max_rows:
            self.partial = True
            data = data.iloc[:max_rows]
        return data

    def set_result_text(self, ax):
        max_rows = get_option("plotting.max_rows")
        assert hasattr(self, "partial")

        if self.partial:
            ax.text(
                1,
                1,
                "showing top {} elements only".format(max_rows),
                size=6,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
            )


class SampledPlotBase:
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
            sampled = data._internal.resolved_copy.spark_frame.sample(fraction=self.fraction)
            return DataFrame(data._internal.with_new_sdf(sampled)).to_pandas()
        else:
            raise ValueError("Only DataFrame and Series are supported for plotting.")

    def set_result_text(self, ax):
        assert hasattr(self, "fraction")

        if self.fraction < 1:
            ax.text(
                1,
                1,
                "showing the sampled result by fraction %s" % self.fraction,
                size=6,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
            )


class HistogramPlotBase:
    @staticmethod
    def prepare_hist_data(data, bins):
        # TODO: this logic is similar with KdePlotBase. Might have to deduplicate it.
        from databricks.koalas.series import Series

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

        if is_integer(bins):
            # computes boundaries for the column
            bins = HistogramPlotBase.get_bins(data.to_spark(), bins)

        return numeric_data, bins

    @staticmethod
    def get_bins(sdf, bins):
        # 'data' is a Spark DataFrame that selects all columns.
        if len(sdf.columns) > 1:
            min_col = F.least(*map(F.min, sdf))
            max_col = F.greatest(*map(F.max, sdf))
        else:
            min_col = F.min(sdf.columns[-1])
            max_col = F.max(sdf.columns[-1])
        boundaries = sdf.select(min_col, max_col).first()

        # divides the boundaries into bins
        if boundaries[0] == boundaries[1]:
            boundaries = (boundaries[0] - 0.5, boundaries[1] + 0.5)

        return np.linspace(boundaries[0], boundaries[1], bins + 1)

    @staticmethod
    def compute_hist(kdf, bins):
        # 'data' is a Spark DataFrame that selects one column.
        assert isinstance(bins, (np.ndarray, np.generic))

        sdf = kdf._internal.spark_frame
        scols = []
        input_column_names = []
        for label in kdf._internal.column_labels:
            input_column_name = name_like_string(label)
            input_column_names.append(input_column_name)
            scols.append(kdf._internal.spark_column_for(label).alias(input_column_name))
        sdf = sdf.select(*scols)

        # 1. Make the bucket output flat to:
        #     +----------+-------+
        #     |__group_id|buckets|
        #     +----------+-------+
        #     |0         |0.0    |
        #     |0         |0.0    |
        #     |0         |1.0    |
        #     |0         |2.0    |
        #     |0         |3.0    |
        #     |0         |3.0    |
        #     |1         |0.0    |
        #     |1         |1.0    |
        #     |1         |1.0    |
        #     |1         |2.0    |
        #     |1         |1.0    |
        #     |1         |0.0    |
        #     +----------+-------+
        colnames = sdf.columns
        bucket_names = ["__{}_bucket".format(colname) for colname in colnames]

        output_df = None
        for group_id, (colname, bucket_name) in enumerate(zip(colnames, bucket_names)):
            # creates a Bucketizer to get corresponding bin of each value
            bucketizer = Bucketizer(
                splits=bins, inputCol=colname, outputCol=bucket_name, handleInvalid="skip"
            )

            bucket_df = bucketizer.transform(sdf)

            if output_df is None:
                output_df = bucket_df.select(
                    F.lit(group_id).alias("__group_id"), F.col(bucket_name).alias("__bucket")
                )
            else:
                output_df = output_df.union(
                    bucket_df.select(
                        F.lit(group_id).alias("__group_id"), F.col(bucket_name).alias("__bucket")
                    )
                )

        # 2. Calculate the count based on each group and bucket.
        #     +----------+-------+------+
        #     |__group_id|buckets| count|
        #     +----------+-------+------+
        #     |0         |0.0    |2     |
        #     |0         |1.0    |1     |
        #     |0         |2.0    |1     |
        #     |0         |3.0    |2     |
        #     |1         |0.0    |2     |
        #     |1         |1.0    |3     |
        #     |1         |2.0    |1     |
        #     +----------+-------+------+
        result = (
            output_df.groupby("__group_id", "__bucket")
            .agg(F.count("*").alias("count"))
            .toPandas()
            .sort_values(by=["__group_id", "__bucket"])
        )

        # 3. Fill empty bins and calculate based on each group id. From:
        #     +----------+--------+------+
        #     |__group_id|__bucket| count|
        #     +----------+--------+------+
        #     |0         |0.0     |2     |
        #     |0         |1.0     |1     |
        #     |0         |2.0     |1     |
        #     |0         |3.0     |2     |
        #     +----------+--------+------+
        #     +----------+--------+------+
        #     |__group_id|__bucket| count|
        #     +----------+--------+------+
        #     |1         |0.0     |2     |
        #     |1         |1.0     |3     |
        #     |1         |2.0     |1     |
        #     +----------+--------+------+
        #
        # to:
        #     +-----------------+
        #     |__values1__bucket|
        #     +-----------------+
        #     |2                |
        #     |1                |
        #     |1                |
        #     |2                |
        #     |0                |
        #     +-----------------+
        #     +-----------------+
        #     |__values2__bucket|
        #     +-----------------+
        #     |2                |
        #     |3                |
        #     |1                |
        #     |0                |
        #     |0                |
        #     +-----------------+
        output_series = []
        for i, (input_column_name, bucket_name) in enumerate(zip(input_column_names, bucket_names)):
            current_bucket_result = result[result["__group_id"] == i]
            # generates a pandas DF with one row for each bin
            # we need this as some of the bins may be empty
            indexes = pd.DataFrame({"__bucket": np.arange(0, len(bins) - 1)})
            # merges the bins with counts on it and fills remaining ones with zeros
            pdf = indexes.merge(current_bucket_result, how="left", on=["__bucket"]).fillna(0)[
                ["count"]
            ]
            pdf.columns = [input_column_name]
            output_series.append(pdf[input_column_name])

        return output_series


class BoxPlotBase:
    @staticmethod
    def compute_stats(data, colname, whis, precision):
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
    def outliers(data, colname, lfence, ufence):
        # Builds expression to identify outliers
        expression = F.col("`%s`" % colname).between(lfence, ufence)
        # Creates a column to flag rows as outliers or not
        return data._kdf._internal.resolved_copy.spark_frame.withColumn(
            "__{}_outlier".format(colname), ~expression
        )

    @staticmethod
    def calc_whiskers(colname, outliers):
        # Computes min and max values of non-outliers - the whiskers
        minmax = (
            outliers.filter("not `__{}_outlier`".format(colname))
            .agg(F.min("`%s`" % colname).alias("min"), F.max(colname).alias("max"))
            .toPandas()
        )
        return minmax.iloc[0][["min", "max"]].values

    @staticmethod
    def get_fliers(colname, outliers, min_val):
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


class KdePlotBase:
    @staticmethod
    def prepare_kde_data(data):
        # TODO: this logic is similar with HistogramPlotBase. Might have to deduplicate it.
        from databricks.koalas.series import Series

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

        return numeric_data

    @staticmethod
    def get_ind(sdf, ind):
        def calc_min_max():
            if len(sdf.columns) > 1:
                min_col = F.least(*map(F.min, sdf))
                max_col = F.greatest(*map(F.max, sdf))
            else:
                min_col = F.min(sdf.columns[-1])
                max_col = F.max(sdf.columns[-1])
            return sdf.select(min_col, max_col).first()

        if ind is None:
            min_val, max_val = calc_min_max()
            sample_range = max_val - min_val
            ind = np.linspace(min_val - 0.5 * sample_range, max_val + 0.5 * sample_range, 1000,)
        elif is_integer(ind):
            min_val, max_val = calc_min_max()
            sample_range = max_val - min_val
            ind = np.linspace(min_val - 0.5 * sample_range, max_val + 0.5 * sample_range, ind,)
        return ind

    @staticmethod
    def compute_kde(sdf, bw_method=None, ind=None):
        # 'sdf' is a Spark DataFrame that selects one column.

        # Using RDD is slow so we might have to change it to Dataset based implementation
        # once Spark has that implementation.
        sample = sdf.rdd.map(lambda x: float(x[0]))
        kd = KernelDensity()
        kd.setSample(sample)

        assert isinstance(bw_method, (int, float)), "'bw_method' must be set as a scalar number."

        if bw_method is not None:
            # Match the bandwidth with Spark.
            kd.setBandwidth(float(bw_method))
        return kd.estimate(list(map(float, ind)))


class KoalasPlotAccessor(PandasObject):
    """
    Series/Frames plotting accessor and method.

    Uses the backend specified by the
    option ``plotting.backend``. By default, plotly is used.

    Plotting methods can also be accessed by calling the accessor as a method
    with the ``kind`` argument:
    ``s.plot(kind='hist')`` is equivalent to ``s.plot.hist()``
    """

    pandas_plot_data_map = {
        "pie": TopNPlotBase().get_top_n,
        "bar": TopNPlotBase().get_top_n,
        "barh": TopNPlotBase().get_top_n,
        "scatter": TopNPlotBase().get_top_n,
        "area": SampledPlotBase().get_sampled,
        "line": SampledPlotBase().get_sampled,
    }
    _backends = {}  # type: ignore

    def __init__(self, data):
        self.data = data

    @staticmethod
    def _find_backend(backend):
        """
        Find a Koalas plotting backend
        """
        try:
            return KoalasPlotAccessor._backends[backend]
        except KeyError:
            try:
                module = importlib.import_module(backend)
            except ImportError:
                # We re-raise later on.
                pass
            else:
                if hasattr(module, "plot") or hasattr(module, "plot_koalas"):
                    # Validate that the interface is implemented when the option
                    # is set, rather than at plot time.
                    KoalasPlotAccessor._backends[backend] = module
                    return module

        raise ValueError(
            "Could not find plotting backend '{backend}'. Ensure that you've installed "
            "the package providing the '{backend}' entrypoint, or that the package has a "
            "top-level `.plot` method.".format(backend=backend)
        )

    @staticmethod
    def _get_plot_backend(backend=None):
        backend = backend or get_option("plotting.backend")
        # Shortcut
        if backend in KoalasPlotAccessor._backends:
            return KoalasPlotAccessor._backends[backend]

        if backend == "matplotlib":
            # Because matplotlib is an optional dependency,
            # we need to attempt an import here to raise an ImportError if needed.
            try:
                # test if matplotlib can be imported
                import matplotlib  # noqa: F401
                from databricks.koalas.plot import matplotlib as module
            except ImportError:
                raise ImportError(
                    "matplotlib is required for plotting when the "
                    "default backend 'matplotlib' is selected."
                ) from None

            KoalasPlotAccessor._backends["matplotlib"] = module
        elif backend == "plotly":
            try:
                # test if plotly can be imported
                import plotly  # noqa: F401
                from databricks.koalas.plot import plotly as module
            except ImportError:
                raise ImportError(
                    "plotly is required for plotting when the "
                    "default backend 'plotly' is selected."
                ) from None

            KoalasPlotAccessor._backends["plotly"] = module
        else:
            module = KoalasPlotAccessor._find_backend(backend)
            KoalasPlotAccessor._backends[backend] = module
        return module

    def __call__(self, kind="line", backend=None, **kwargs):
        plot_backend = KoalasPlotAccessor._get_plot_backend(backend)
        plot_data = self.data

        kind = {"density": "kde"}.get(kind, kind)
        if hasattr(plot_backend, "plot_koalas"):
            # use if there's koalas specific method.
            return plot_backend.plot_koalas(plot_data, kind=kind, **kwargs)
        else:
            # fallback to use pandas'
            if not KoalasPlotAccessor.pandas_plot_data_map[kind]:
                raise NotImplementedError(
                    "'%s' plot is not supported with '%s' plot "
                    "backend yet." % (kind, plot_backend.__name__)
                )
            plot_data = KoalasPlotAccessor.pandas_plot_data_map[kind](plot_data)
            return plot_backend.plot(plot_data, kind=kind, **kwargs)

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
        :class:`plotly.graph_objs.Figure`
            Return an custom object when ``backend!=plotly``.
            Return an ndarray when ``subplots=True`` (matplotlib-only).

        See Also
        --------
        plotly.express.line : Plot y versus x as lines and/or markers (plotly).
        matplotlib.pyplot.plot : Plot y versus x as lines and/or markers (matplotlib).

        Examples
        --------
        Basic plot.

        For Series:

        .. plotly::

            >>> s = ks.Series([1, 3, 2])
            >>> s.plot.line()  # doctest: +SKIP

        For DataFrame:

        .. plotly::

            The following example shows the populations for some animals
            over the years.

            >>> df = ks.DataFrame({'pig': [20, 18, 489, 675, 1776],
            ...                    'horse': [4, 25, 281, 600, 1900]},
            ...                   index=[1990, 1997, 2003, 2009, 2014])
            >>> df.plot.line()  # doctest: +SKIP

        .. plotly::

            The following example shows the relationship between both
            populations.

            >>> df = ks.DataFrame({'pig': [20, 18, 489, 675, 1776],
            ...                    'horse': [4, 25, 281, 600, 1900]},
            ...                   index=[1990, 1997, 2003, 2009, 2014])
            >>> df.plot.line(x='pig', y='horse')  # doctest: +SKIP
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
        :class:`plotly.graph_objs.Figure`
            Return an custom object when ``backend!=plotly``.
            Return an ndarray when ``subplots=True`` (matplotlib-only).

        Examples
        --------
        Basic plot.

        For Series:

        .. plotly::

            >>> s = ks.Series([1, 3, 2])
            >>> s.plot.bar()  # doctest: +SKIP

        For DataFrame:

        .. plotly::

            >>> df = ks.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> df.plot.bar(x='lab', y='val')  # doctest: +SKIP

        Plot a whole dataframe to a bar plot. Each column is stacked with a
        distinct color along the horizontal axis.

        .. plotly::

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = ks.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> df.plot.bar()  # doctest: +SKIP

        Instead of stacking, the figure can be split by column with plotly
        APIs.

        .. plotly::

            >>> from plotly.subplots import make_subplots
            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = ks.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> fig = (make_subplots(rows=2, cols=1)
            ...        .add_trace(df.plot.bar(y='speed').data[0], row=1, col=1)
            ...        .add_trace(df.plot.bar(y='speed').data[0], row=1, col=1)
            ...        .add_trace(df.plot.bar(y='lifespan').data[0], row=2, col=1))
            >>> fig  # doctest: +SKIP

        Plot a single column.

        .. plotly::

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = ks.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> df.plot.bar(y='speed')  # doctest: +SKIP

        Plot only selected categories for the DataFrame.

        .. plotly::

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = ks.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> df.plot.bar(x='lifespan')  # doctest: +SKIP
        """
        from databricks.koalas import DataFrame, Series

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
        :class:`plotly.graph_objs.Figure`
            Return an custom object when ``backend!=plotly``.
            Return an ndarray when ``subplots=True`` (matplotlib-only).

        See Also
        --------
        plotly.express.bar : Plot a vertical bar plot using plotly.
        matplotlib.axes.Axes.bar : Plot a vertical bar plot using matplotlib.

        Examples
        --------
        For Series:

        .. plotly::

            >>> df = ks.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> df.val.plot.barh()  # doctest: +SKIP

        For DataFrame:

        .. plotly::

            >>> df = ks.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
            >>> df.plot.barh(x='lab', y='val')  # doctest: +SKIP

        Plot a whole DataFrame to a horizontal bar plot

        .. plotly::

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = ks.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> df.plot.barh()  # doctest: +SKIP

        Plot a column of the DataFrame to a horizontal bar plot

        .. plotly::

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = ks.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> df.plot.barh(y='speed')  # doctest: +SKIP

        Plot DataFrame versus the desired column

        .. plotly::

            >>> speed = [0.1, 17.5, 40, 48, 52, 69, 88]
            >>> lifespan = [2, 8, 70, 1.5, 25, 12, 28]
            >>> index = ['snail', 'pig', 'elephant',
            ...          'rabbit', 'giraffe', 'coyote', 'horse']
            >>> df = ks.DataFrame({'speed': speed,
            ...                    'lifespan': lifespan}, index=index)
            >>> df.plot.barh(x='lifespan')  # doctest: +SKIP
        """
        from databricks.koalas import DataFrame, Series

        if isinstance(self.data, Series):
            return self(kind="barh", **kwargs)
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
            statistics (matplotlib-only).

        Returns
        -------
        :class:`plotly.graph_objs.Figure`
            Return an custom object when ``backend!=plotly``.
            Return an ndarray when ``subplots=True`` (matplotlib-only).

        Notes
        -----
        There are behavior differences between Koalas and pandas.

        * Koalas computes approximate statistics - expect differences between
          pandas and Koalas boxplots, especially regarding 1st and 3rd quartiles.
        * The `whis` argument is only supported as a single number.
        * Koalas doesn't support the following argument(s) (matplotlib-only).

          * `bootstrap` argument is not supported
          * `autorange` argument is not supported

        Examples
        --------
        Draw a box plot from a DataFrame with four columns of randomly
        generated data.

        For Series:

        .. plotly::

            >>> data = np.random.randn(25, 4)
            >>> df = ks.DataFrame(data, columns=list('ABCD'))
            >>> df['A'].plot.box()  # doctest: +SKIP

        This is an unsupported function for DataFrame type
        """
        from databricks.koalas import DataFrame, Series

        if isinstance(self.data, Series):
            return self(kind="box", **kwds)
        elif isinstance(self.data, DataFrame):
            return unsupported_function(class_name="pd.DataFrame", method_name="box")()

    def hist(self, bins=10, **kwds):
        """
        Draw one histogram of the DataFrame’s columns.
        A `histogram`_ is a representation of the distribution of data.
        This function calls :meth:`plotting.backend.plot`,
        on each series in the DataFrame, resulting in one histogram per column.

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
            plotting backend.

        Returns
        -------
        :class:`plotly.graph_objs.Figure`
            Return an custom object when ``backend!=plotly``.
            Return an ndarray when ``subplots=True`` (matplotlib-only).

        Examples
        --------
        Basic plot.

        For Series:

        .. plotly::

            >>> s = ks.Series([1, 3, 2])
            >>> s.plot.hist()  # doctest: +SKIP

        For DataFrame:

        .. plotly::

            >>> df = pd.DataFrame(
            ...     np.random.randint(1, 7, 6000),
            ...     columns=['one'])
            >>> df['two'] = df['one'] + np.random.randint(1, 7, 6000)
            >>> df = ks.from_pandas(df)
            >>> df.plot.hist(bins=12, alpha=0.5)  # doctest: +SKIP
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
        :class:`plotly.graph_objs.Figure`
            Return an custom object when ``backend!=plotly``.
            Return an ndarray when ``subplots=True`` (matplotlib-only).

        Examples
        --------
        A scalar bandwidth should be specified. Using a small bandwidth value can
        lead to over-fitting, while using a large bandwidth value may result
        in under-fitting:

        .. plotly::

            >>> s = ks.Series([1, 2, 2.5, 3, 3.5, 4, 5])
            >>> s.plot.kde(bw_method=0.3)  # doctest: +SKIP

        .. plotly::

            >>> s = ks.Series([1, 2, 2.5, 3, 3.5, 4, 5])
            >>> s.plot.kde(bw_method=3)  # doctest: +SKIP

        The `ind` parameter determines the evaluation points for the
        plot of the estimated KDF:

        .. plotly::

            >>> s = ks.Series([1, 2, 2.5, 3, 3.5, 4, 5])
            >>> s.plot.kde(ind=[1, 2, 3, 4, 5], bw_method=0.3)  # doctest: +SKIP

        For DataFrame, it works in the same way as Series:

        .. plotly::

            >>> df = ks.DataFrame({
            ...     'x': [1, 2, 2.5, 3, 3.5, 4, 5],
            ...     'y': [4, 4, 4.5, 5, 5.5, 6, 6],
            ... })
            >>> df.plot.kde(bw_method=0.3)  # doctest: +SKIP

        .. plotly::

            >>> df = ks.DataFrame({
            ...     'x': [1, 2, 2.5, 3, 3.5, 4, 5],
            ...     'y': [4, 4, 4.5, 5, 5.5, 6, 6],
            ... })
            >>> df.plot.kde(bw_method=3)  # doctest: +SKIP

        .. plotly::

            >>> df = ks.DataFrame({
            ...     'x': [1, 2, 2.5, 3, 3.5, 4, 5],
            ...     'y': [4, 4, 4.5, 5, 5.5, 6, 6],
            ... })
            >>> df.plot.kde(ind=[1, 2, 3, 4, 5, 6], bw_method=0.3)  # doctest: +SKIP
        """
        return self(kind="kde", bw_method=bw_method, ind=ind, **kwargs)

    density = kde

    def area(self, x=None, y=None, **kwds):
        """
        Draw a stacked area plot.

        An area plot displays quantitative data visually.
        This function wraps the plotly area function.

        Parameters
        ----------
        x : label or position, optional
            Coordinates for the X axis. By default uses the index.
        y : label or position, optional
            Column to plot. By default uses all columns.
        stacked : bool, default True
            Area plots are stacked by default. Set to False to create a
            unstacked plot (matplotlib-only).
        **kwds : optional
            Additional keyword arguments are documented in
            :meth:`DataFrame.plot`.

        Returns
        -------
        :class:`plotly.graph_objs.Figure`
            Return an custom object when ``backend!=plotly``.
            Return an ndarray when ``subplots=True`` (matplotlib-only).

        Examples
        --------

        For Series

        .. plotly::

            >>> df = ks.DataFrame({
            ...     'sales': [3, 2, 3, 9, 10, 6],
            ...     'signups': [5, 5, 6, 12, 14, 13],
            ...     'visits': [20, 42, 28, 62, 81, 50],
            ... }, index=pd.date_range(start='2018/01/01', end='2018/07/01',
            ...                        freq='M'))
            >>> df.sales.plot.area()  # doctest: +SKIP

        For DataFrame

        .. plotly::

            >>> df = ks.DataFrame({
            ...     'sales': [3, 2, 3, 9, 10, 6],
            ...     'signups': [5, 5, 6, 12, 14, 13],
            ...     'visits': [20, 42, 28, 62, 81, 50],
            ... }, index=pd.date_range(start='2018/01/01', end='2018/07/01',
            ...                        freq='M'))
            >>> df.plot.area()  # doctest: +SKIP
        """
        from databricks.koalas import DataFrame, Series

        if isinstance(self.data, Series):
            return self(kind="area", **kwds)
        elif isinstance(self.data, DataFrame):
            return self(kind="area", x=x, y=y, **kwds)

    def pie(self, **kwds):
        """
        Generate a pie plot.

        A pie plot is a proportional representation of the numerical data in a
        column. This function wraps :meth:`plotly.express.pie` for the
        specified column.

        Parameters
        ----------
        y : int or label, optional
            Label or position of the column to plot.
            If not provided, ``subplots=True`` argument must be passed (matplotlib-only).
        **kwds
            Keyword arguments to pass on to :meth:`Koalas.Series.plot`.

        Returns
        -------
        :class:`plotly.graph_objs.Figure`
            Return an custom object when ``backend!=plotly``.
            Return an ndarray when ``subplots=True`` (matplotlib-only).

        Examples
        --------

        For Series:

        .. plotly::

            >>> df = ks.DataFrame({'mass': [0.330, 4.87, 5.97],
            ...                    'radius': [2439.7, 6051.8, 6378.1]},
            ...                   index=['Mercury', 'Venus', 'Earth'])
            >>> df.mass.plot.pie()  # doctest: +SKIP


        For DataFrame:

        .. plotly::

            >>> df = ks.DataFrame({'mass': [0.330, 4.87, 5.97],
            ...                    'radius': [2439.7, 6051.8, 6378.1]},
            ...                   index=['Mercury', 'Venus', 'Earth'])
            >>> df.plot.pie(y='mass')  # doctest: +SKIP
        """
        from databricks.koalas import DataFrame, Series

        if isinstance(self.data, Series):
            return self(kind="pie", **kwds)
        else:
            # pandas will raise an error if y is None and subplots if not True
            if (
                isinstance(self.data, DataFrame)
                and kwds.get("y", None) is None
                and not kwds.get("subplots", False)
            ):
                raise ValueError(
                    "pie requires either y column or 'subplots=True' (matplotlib-only)"
                )
            return self(kind="pie", **kwds)

    def scatter(self, x, y, **kwds):
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
            (matplotlib-only).
        c : str, int or array_like, optional
            (matplotlib-only).

        **kwds: Optional
            Keyword arguments to pass on to :meth:`databricks.koalas.DataFrame.plot`.

        Returns
        -------
        :class:`plotly.graph_objs.Figure`
            Return an custom object when ``backend!=plotly``.
            Return an ndarray when ``subplots=True`` (matplotlib-only).

        See Also
        --------
        plotly.express.scatter : Scatter plot using multiple input data
            formats (plotly).
        matplotlib.pyplot.scatter : Scatter plot using multiple input data
            formats (matplotlib).

        Examples
        --------
        Let's see how to draw a scatter plot using coordinates from the values
        in a DataFrame's columns.

        .. plotly::

            >>> df = ks.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
            ...                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
            ...                   columns=['length', 'width', 'species'])
            >>> df.plot.scatter(x='length', y='width')  # doctest: +SKIP

        And now with dark scheme:

        .. plotly::

            >>> df = ks.DataFrame([[5.1, 3.5, 0], [4.9, 3.0, 0], [7.0, 3.2, 1],
            ...                    [6.4, 3.2, 1], [5.9, 3.0, 2]],
            ...                   columns=['length', 'width', 'species'])
            >>> fig = df.plot.scatter(x='length', y='width')
            >>> fig.update_layout(template="plotly_dark")  # doctest: +SKIP
        """
        return self(kind="scatter", x=x, y=y, **kwds)

    def hexbin(self, **kwds):
        return unsupported_function(class_name="pd.DataFrame", method_name="hexbin")()
