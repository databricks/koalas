==============
Best Practices
==============

Leverage PySpark APIs
---------------------

Koalas uses Spark under the hood; therefore, many features and performance optimization are available
in Koalas as well. Leverage and combine those cutting-edge features with Koalas.

Existing Spark context and Spark sessions are used out of the box in Koalas. If you already have your own
configured Spark context or sessions running, Koalas uses them.

If there is no Spark context or session running in your environment (e.g., ordinary Python interpreter),
such configurations can be set to ``SparkContext`` and/or ``SparkSession``.
Once Spark context and/or session is created, Koalas can use this context and/or session automatically.
For example, if you want to configure the executor memory in Spark, you can do as below:

.. code-block:: python

   from pyspark import SparkConf, SparkContext
   conf = SparkConf()
   conf.set('spark.executor.memory', '2g')
   # Koalas automatically uses this Spark context with the configurations set.
   SparkContext(conf=conf)

   import databricks.koalas as ks
   ...

Another common configuration might be Arrow optimization in PySpark. In case of SQL configuration,
it can be set into Spark session as below:

.. code-block:: python

   from pyspark.sql import SparkSession
   builder = SparkSession.builder.appName("Koalas")
   builder = builder.config("spark.sql.execution.arrow.enabled", "true")
   # Koalas automatically uses this Spark session with the configurations set.
   builder.getOrCreate()

   import databricks.koalas as ks
   ...

All Spark features such as history server, web UI and deployment modes can be used as are with Koalas.
If you are interested in performance tuning, please see also `Tuning Spark <https://spark.apache.org/docs/latest/tuning.html>`_.


Check execution plans
---------------------

Expensive operations can be predicted by leveraging PySpark API `DataFrame.spark.explain()`
before the actual computation since Koalas is based on lazy execution. For example, see below.

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'id': range(10)})
   >>> kdf = kdf[kdf.id > 5]
   >>> kdf.spark.explain()
   == Physical Plan ==
   *(1) Filter (id#1L > 5)
   +- *(1) Scan ExistingRDD[__index_level_0__#0L,id#1L]


Whenever you are not sure about such cases, you can check the actual execution plans and
foresee the expensive cases.

Even though Koalas tries its best to optimize and reduce such shuffle operations by leveraging Spark
optimizers, it is best to avoid shuffling in the application side whenever possible.


Avoid shuffling
---------------

Some operations such as ``sort_values`` are more difficult to do in a parallel or distributed
environment than in in-memory on a single machine because it needs to send data to other nodes,
and exchange the data across multiple nodes via networks. See the example below.

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'id': range(10)}).sort_values(by="id")
   >>> kdf.spark.explain()
   == Physical Plan ==
   *(2) Sort [id#9L ASC NULLS LAST], true, 0
   +- Exchange rangepartitioning(id#9L ASC NULLS LAST, 200), true, [id=#18]
      +- *(1) Scan ExistingRDD[__index_level_0__#8L,id#9L]

As you can see, it requires ``Exchange`` which requires a shuffle and it is likely expensive.


Avoid computation on single partition
-------------------------------------

Another common case is the computation on a single partition. Currently, some APIs such as
`DataFrame.rank <https://koalas.readthedocs.io/en/latest/reference/api/databricks.koalas.DataFrame.rank.html>`_
uses PySpark’s Window without specifying partition specification. This leads to move all data into a single
partition in single machine and could cause serious performance degradation.
Such APIs should be avoided very large dataset.

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'id': range(10)})
   >>> kdf.rank().spark.explain()
   == Physical Plan ==
   *(4) Project [__index_level_0__#16L, id#24]
   +- Window [avg(cast(_w0#26 as bigint)) windowspecdefinition(id#17L, specifiedwindowframe(RowFrame, unboundedpreceding$(), unboundedfollowing$())) AS id#24], [id#17L]
      +- *(3) Project [__index_level_0__#16L, _w0#26, id#17L]
         +- Window [row_number() windowspecdefinition(id#17L ASC NULLS FIRST, specifiedwindowframe(RowFrame, unboundedpreceding$(), currentrow$())) AS _w0#26], [id#17L ASC NULLS FIRST]
            +- *(2) Sort [id#17L ASC NULLS FIRST], false, 0
               +- Exchange SinglePartition, true, [id=#48]
                  +- *(1) Scan ExistingRDD[__index_level_0__#16L,id#17L]

Instead, use 
`GroupBy.rank <https://koalas.readthedocs.io/en/latest/reference/api/databricks.koalas.groupby.GroupBy.rank.html>`_
as it is less expensive because data can be distributed and computed for each group.


Avoid reserved column names
---------------------------

Columns with leading ``__`` and trailing ``__`` are reserved in Koalas. To handle internal behaviors for, such as, index,
Koalas uses some internal columns. Therefore, it is discouraged to use such column names and not guaranteed to work.


Do not use duplicated column names
----------------------------------

It is disallowed to use duplicated column names because Spark SQL does not allow this in general. Koalas inherits
this behavior. For instance, see below:

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'a': [1, 2], 'b':[3, 4]})
   >>> kdf.columns = ["a", "a"]
   ...
   Reference 'a' is ambiguous, could be: a, a.;

Additionally, it is strongly discouraged to use case sensitive column names. Koalas disallows it by default.

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'a': [1, 2], 'A':[3, 4]})
   ...
   Reference 'a' is ambiguous, could be: a, a.;

However, you can turn on ``spark.sql.caseSensitive`` in Spark configuration to enable it if you use on your own risk.

.. code-block:: python

   >>> from pyspark.sql import SparkSession
   >>> builder = SparkSession.builder.appName("Koalas")
   >>> builder = builder.config("spark.sql.caseSensitive", "true")
   >>> builder.getOrCreate()

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'a': [1, 2], 'A':[3, 4]})
   >>> kdf
      a  A
   0  1  3
   1  2  4


Specify the index column in conversion from Spark DataFrame to Koalas DataFrame
-------------------------------------------------------------------------------

When Koalas Dataframe is converted from Spark DataFrame, it loses the index information, which results in using
the default index in Koalas DataFrame. The default index is inefficient in general comparing to explicitly specifying
the index column. Specify the index column whenever possible.

See  `working with PySpark <pandas_pyspark.rst#pyspark>`_

Use ``distributed`` or ``distributed-sequence`` default index
-------------------------------------------------------------

One common issue when Koalas users face is the slow performance by default index. Koalas attaches
a default index when the index is unknown, for example, Spark DataFrame is directly converted to Koalas DataFrame.

This default index is ``sequence`` which requires the computation on single partition which is discouraged. If you plan
to handle large data in production, make it distributed by configuring the default index to ``distributed`` or
``distributed-sequence`` .

See `Default Index Type <options.rst#default-index-type>`_ for more details about configuring default index.


Reduce the operations on different DataFrame/Series
---------------------------------------------------

Koalas disallows the operations on different DataFrames (or Series) by default to prevent expensive operations.
It internally performs a join operation which can be expensive in general, which is discouraged. Whenever possible,
this operation should be avoided.

See `Operations on different DataFrames <options.rst#operations-on-different-dataframes>`_ for more details.


Use Koalas APIs directly whenever possible
------------------------------------------

Although Koalas has most of the pandas-equivalent APIs, there are several APIs not implemented yet or explicitly unsupported.

As an example, Koalas does not implement ``__iter__()`` to prevent users from collecting all data into the client (driver) side from the whole cluster.
Unfortunately, many external APIs such as Python built-in functions such as min, max, sum, etc. require the given argument to be iterable.
In case of pandas, it works properly out of the box as below:

.. code-block:: python

   >>> import pandas as pd
   >>> max(pd.Series([1, 2, 3]))
   3
   >>> min(pd.Series([1, 2, 3]))
   1
   >>> sum(pd.Series([1, 2, 3]))
   6

pandas dataset lives in the single machine, and is naturally iterable locally within the same machine.
However, Koalas dataset lives across multiple machines, and they are computed in a distributed manner.
It is difficult to be locally iterable and it is very likely users collect the entire data into the client side without knowing it.
Therefore, it is best to stick to using Koalas APIs.
The examples above can be converted as below:

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> ks.Series([1, 2, 3]).max()
   3
   >>> ks.Series([1, 2, 3]).min()
   1
   >>> ks.Series([1, 2, 3]).sum()
   6

Another common pattern from pandas users might be to rely on list comprehension or generator expression.
However, it also assumes the dataset is locally iterable under the hood.
Therefore, it works seamlessly in pandas as below:

.. code-block:: python

   >>> import pandas as pd
   >>> data = []
   >>> countries = ['London', 'New York', 'Helsinki']
   >>> pser = pd.Series([20., 21., 12.], index=countries)
   >>> for temperature in pser:
   ...     assert temperature > 0
   ...     if temperature > 1000:
   ...         temperature = None
   ...     data.append(temperature ** 2)
   ...
   >>> pd.Series(data, index=countries)
   London      400.0
   New York    441.0
   Helsinki    144.0
   dtype: float64

However, for Koalas it does not work as the same reason above.
The example above can be also changed to directly using Koalas APIs as below:

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> import numpy as np
   >>> countries = ['London', 'New York', 'Helsinki']
   >>> kser = ks.Series([20., 21., 12.], index=countries)
   >>> def square(temperature) -> np.float64:
   ...     assert temperature > 0
   ...     if temperature > 1000:
   ...         temperature = None
   ...     return temperature ** 2
   ...
   >>> kser.apply(square)
   London      400.0
   New York    441.0
   Helsinki    144.0
   dtype: float64
