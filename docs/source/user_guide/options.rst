====================
Options and settings
====================
.. currentmodule:: databricks.koalas

Koalas has an options system that lets you customize some aspects of its behaviour,
display-related options being those the user is most likely to adjust.

Options have a full "dotted-style", case-insensitive name (e.g. ``display.max_rows``).
You can get/set options directly as attributes of the top-level ``options`` attribute:


.. code-block:: python

   >>> import databricks.koalas as ks
   >>> ks.options.display.max_rows
   1000
   >>> ks.options.display.max_rows = 10
   >>> ks.options.display.max_rows
   10

The API is composed of 3 relevant functions, available directly from the ``koalas``
namespace:

* :func:`get_option` / :func:`set_option` - get/set the value of a single option.
* :func:`reset_option` - reset one or more options to their default value.

**Note:** Developers can check out `databricks/koalas/config.py <https://github.com/databricks/koalas/blob/master/databricks/koalas/config.py>`_ for more information.

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> ks.get_option("display.max_rows")
   1000
   >>> ks.set_option("display.max_rows", 101)
   >>> ks.get_option("display.max_rows")
   101


Getting and setting options
---------------------------

As described above, :func:`get_option` and :func:`set_option`
are available from the koalas namespace.  To change an option, call
``set_option('option name', new_value)``.

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> ks.get_option('compute.max_rows')
   1000
   >>> ks.set_option('compute.max_rows', 2000)
   >>> ks.get_option('compute.max_rows')
   2000

All options also have a default value, and you can use ``reset_option`` to do just that:

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> ks.reset_option("display.max_rows")

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> ks.get_option("display.max_rows")
   1000
   >>> ks.set_option("display.max_rows", 999)
   >>> ks.get_option("display.max_rows")
   999
   >>> ks.reset_option("display.max_rows")
   >>> ks.get_option("display.max_rows")
   1000

``option_context`` context manager has been exposed through
the top-level API, allowing you to execute code with given option values. Option values
are restored automatically when you exit the `with` block:

.. code-block:: python

   >>> with ks.option_context("display.max_rows", 10, "compute.max_rows", 5):
   ...    print(ks.get_option("display.max_rows"))
   ...    print(ks.get_option("compute.max_rows"))
   10
   5
   >>> print(ks.get_option("display.max_rows"))
   >>> print(ks.get_option("compute.max_rows"))
   1000
   1000


Operations on different DataFrames
----------------------------------

Koalas disallows the operations on different DataFrames (or Series) by default to prevent expensive
operations. It internally performs a join operation which can be expensive in general.

This can be enabled by setting `compute.ops_on_diff_frames` to `True` to allow such cases.
See the examples below.

.. code-block:: python

    >>> import databricks.koalas as ks
    >>> ks.set_option('compute.ops_on_diff_frames', True)
    >>> kdf1 = ks.range(5)
    >>> kdf2 = ks.DataFrame({'id': [5, 4, 3]})
    >>> (kdf1 - kdf2).sort_index()
        id
    0 -5.0
    1 -3.0
    2 -1.0
    3  NaN
    4  NaN
    >>> ks.reset_option('compute.ops_on_diff_frames')

.. code-block:: python

    >>> import databricks.koalas as ks
    >>> ks.set_option('compute.ops_on_diff_frames', True)
    >>> kdf = ks.range(5)
    >>> kser_a = ks.Series([1, 2, 3, 4])
    >>> # 'kser_a' is not from 'kdf' DataFrame. So it is considered as a Series not from 'kdf'.
    >>> kdf['new_col'] = kser_a
    >>> kdf
       id  new_col
    0   0      1.0
    1   1      2.0
    3   3      4.0
    2   2      3.0
    4   4      NaN
    >>> ks.reset_option('compute.ops_on_diff_frames')


Default Index type
------------------

In Koalas, the default index is used in several cases, for instance,
when Spark DataFrame is converted into Koalas DataFrame. In this case, internally Koalas attaches a
default index into Koalas DataFrame.

There are several types of the default index that can be configured by `compute.default_index_type` as below:

**sequence**: It implements a sequence that increases one by one, by PySpark's Window function without
specifying partition. Therefore, it can end up with whole partition in single node.
This index type should be avoided when the data is large. This is default. See the example below:

.. code-block:: python

    >>> import databricks.koalas as ks
    >>> ks.set_option('compute.default_index_type', 'sequence')
    >>> kdf = ks.range(3)
    >>> ks.reset_option('compute.default_index_type')
    >>> kdf.index
    Int64Index([0, 1, 2], dtype='int64')

This is conceptually equivalent to the PySpark example as below:

.. code-block:: python

    >>> from pyspark.sql import functions as F, Window
    >>> import databricks.koalas as ks
    >>> spark_df = ks.range(3).to_spark()
    >>> sequential_index = F.row_number().over(
    ...    Window.orderBy(F.monotonically_increasing_id().asc())) - 1
    >>> spark_df.select(sequential_index).rdd.map(lambda r: r[0]).collect()
    [0, 1, 2]

**distributed-sequence**: It implements a sequence that increases one by one, by group-by and
group-map approach in a distributed manner. It still generates the sequential index globally.
If the default index must be the sequence in a large dataset, this
index has to be used.
Note that if more data are added to the data source after creating this index,
then it does not guarantee the sequential index. See the example below:

.. code-block:: python

    >>> import databricks.koalas as ks
    >>> ks.set_option('compute.default_index_type', 'distributed-sequence')
    >>> kdf = ks.range(3)
    >>> ks.reset_option('compute.default_index_type')
    >>> kdf.index
    Int64Index([0, 1, 2], dtype='int64')

This is conceptually equivalent to the PySpark example as below:

.. code-block:: python

    >>> import databricks.koalas as ks
    >>> spark_df = ks.range(3).to_spark()
    >>> spark_df.rdd.zipWithIndex().map(lambda p: p[1]).collect()
    [0, 1, 2]

**distributed**: It implements a monotonically increasing sequence simply by using
PySpark's `monotonically_increasing_id` function in a fully distributed manner. The
values are indeterministic. If the index does not have to be a sequence that increases
one by one, this index should be used. Performance-wise, this index almost does not
have any penalty comparing to other index types. See the example below:

.. code-block:: python

    >>> import databricks.koalas as ks
    >>> ks.set_option('compute.default_index_type', 'distributed')
    >>> kdf = ks.range(3)
    >>> ks.reset_option('compute.default_index_type')
    >>> kdf.index
    Int64Index([25769803776, 60129542144, 94489280512], dtype='int64')

This is conceptually equivalent to the PySpark example as below:

.. code-block:: python

    >>> from pyspark.sql import functions as F
    >>> import databricks.koalas as ks
    >>> spark_df = ks.range(3).to_spark()
    >>> spark_df.select(F.monotonically_increasing_id()) \
    ...     .rdd.map(lambda r: r[0]).collect()
    [25769803776, 60129542144, 94489280512]

.. warning::
    It is very unlikely for this type of index to be used for computing two
    different dataframes because it is not guaranteed to have the same indexes in two dataframes.
    If you use this default index and turn on `compute.ops_on_diff_frames`, the result
    from the operations between two different DataFrames will likely be an unexpected
    output due to the indeterministic index values.


Available options
-----------------

=============================== ============== =====================================================
Option                          Default        Description
=============================== ============== =====================================================
display.max_rows                1000           This sets the maximum number of rows Koalas should
                                               output when printing out various output. For example,
                                               this value determines the number of rows to be shown
                                               at the repr() in a dataframe. Set `None` to unlimit
                                               the input length. Default is 1000.
compute.max_rows                1000           'compute.max_rows' sets the limit of the current
                                               DataFrame. Set `None` to unlimit the input length.
                                               When the limit is set, it is executed by the shortcut
                                               by collecting the data into driver side, and then
                                               using pandas API. If the limit is unset, the
                                               operation is executed by PySpark. Default is 1000.
compute.shortcut_limit          1000           'compute.shortcut_limit' sets the limit for a
                                               shortcut. It computes specified number of rows and
                                               use its schema. When the dataframe length is larger
                                               than this limit, Koalas uses PySpark to compute.
compute.ops_on_diff_frames      False          This determines whether or not to operate between two
                                               different dataframes. For example, 'combine_frames'
                                               function internally performs a join operation which
                                               can be expensive in general. So, if
                                               `compute.ops_on_diff_frames` variable is not True,
                                               that method throws an exception.
compute.default_index_type      'sequence'     This sets the default index type: sequence,
                                               distributed and distributed-sequence.
compute.ordered_head            False          'compute.ordered_head' sets whether or not to operate
                                               head with natural ordering. Koalas does not guarantee
                                               the row ordering so `head` could return some rows
                                               from distributed partitions. If
                                               'compute.ordered_head' is set to True, Koalas
                                               performs natural ordering beforehand, but it will
                                               cause a performance overhead.
plotting.max_rows               1000           'plotting.max_rows' sets the visual limit on top-n-
                                               based plots such as `plot.bar` and `plot.pie`. If it
                                               is set to 1000, the first 1000 data points will be
                                               used for plotting. Default is 1000.
plotting.sample_ratio           None           'plotting.sample_ratio' sets the proportion of data
                                               that will be plotted for sample-based plots such as
                                               `plot.line` and `plot.area`. This option defaults to
                                               'plotting.max_rows' option.
plotting.backend                'matplotlib'   Backend to use for plotting. Default is matplotlib.
                                               Supports any package that has a top-level `.plot`
                                               method. Some options are: [matplotlib, plotly,
                                               pandas_bokeh, pandas_altair].
=============================== ============== =====================================================
