==============
Best Practices
==============

Leverage PySpark APIs
---------------------

Koalas uses Spark under the hood; therefore, many features and performance optimization are available
in Koalas as well. Leverage and combine those cutting-edge features with Koalas.

Existing Spark context and Spark sessions are used out of the box in Koalas. If you already have your own
configured Spark context or sessions running, Koalas uses them.

If there is no Spark context or session running in your environment (e.g., ordinary Python interpretor),
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
If you are interested in performance turning, please see also `Tuning Spark <https://spark.apache.org/docs/latest/tuning.html>`_.


Check execution plans
---------------------

Expensive operations can be predicted by leveraging PySpark API `to_spark().explain()`
before the actual computation since Koalas is based on lazy execution. For example, see below.

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'id': range(10)})
   >>> kdf = kdf[kdf.id > 5]
   >>> kdf.explain()
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
and exchange the data across multile nodes via networks. See the example below.

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'id': range(10)}).sort_values(by="id")
   >>> kdf.explain()
   == Physical Plan ==
   *(2) Sort [id#9L ASC NULLS LAST], true, 0
   +- Exchange rangepartitioning(id#9L ASC NULLS LAST, 200), true, [id=#18]
      +- *(1) Scan ExistingRDD[__index_level_0__#8L,id#9L]

As you can see it requires ``Exchange`` which requires a shuffle and it is likely expensive.


Avoid computation on single partition
-------------------------------------

Another common case is the computation on single partition. Currently some APIs such as
`DataFrame.rank <https://koalas.readthedocs.io/en/latest/reference/api/databricks.koalas.DataFrame.rank.html>`_
uses PySpark’s Window without specifying partition specification. This leads to move all data into single
partition in single machine and could cause serious performance degradation.
Such APIs shoild be avoided very large dataset.

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'id': range(10)})
   >>> kdf.rank().explain()
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
as it is less expensive because data can be distributed and computed on each group.


Avoid reserved column names
---------------------------

Columns with leading ``__`` and trailing ``__`` are reserved in Koalas. To handle internal behaviors for, such as, index,
Koalas uses some internal columns. Therefore it is discouraged to use such column names and not guaranteed to work.


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


Specify the index column in conversion from Spark DataFrame to Koalas DataFrame
-------------------------------------------------------------------------------

When Koalas Dataframe is converted from Spark DataFrame, it loses the index information, which results in using
the default index in Koalas DataFrame. The default index is inefficient in general comparing to explicitly specifying
the index column. Specify the index column whenever possible.

See  `working with PySpark <pandas_pyspark.rst#pyspark>`_

Use ``distributed`` or ``distributed-sequence`` default index
-------------------------------------------------------------

One common issue when Koalas users face is the slow performance by default index. Koalas attaches
a default index when index is unknown, for example, Spark DataFrame is directly converted to Koalas DataFrame.

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

