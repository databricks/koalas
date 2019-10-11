===============================
Working with pandas and PySpark
===============================

.. currentmodule:: databricks.koalas

Users from pandas and/or PySpark face API compatibility issue sometimes when they
work with Koalas. Since Koalas does not target 100% compatibility of both pandas and
PySpark, users need to do some workaround to port their pandas and/or PySpark codes or
get familiar with Koalas in this case. This page aims to describe it.


pandas
------

pandas users can access to full pandas APIs by calling :func:`DataFrame.to_pandas`.
Koalas DataFrame and pandas DataFrame are similar. However, the former is distributed
and the latter is in a single machine. When converting to each other, the data is
transferred between multiple machines and the single client machine.

For example, if you need to call ``pandas_df.values`` of pandas DataFrame, you can do
as below:

.. code-block:: python

   >>> import databricks.koalas as ks
   >>>
   >>> kdf = ks.range(10)
   >>> pdf = kdf.to_pandas()
   >>> pdf.values
   array([[0],
          [1],
          [2],
          [3],
          [4],
          [5],
          [6],
          [7],
          [8],
          [9]])

pandas DataFrame can be a Koalas DataFrame easily as below:

.. code-block:: python

   >>> ks.from_pandas(pdf)
      id
   0   0
   1   1
   2   2
   3   3
   4   4
   5   5
   6   6
   7   7
   8   8
   9   9

Note that converting Koalas DataFrame to pandas requires to collect all the data into the client machine; therefore,
if possible, it is recommended to use Koalas or PySpark APIs instead.


PySpark
-------

PySpark users can access to full PySpark APIs by calling :func:`DataFrame.to_spark`.
Koalas DataFrame and Spark DataFrame are virtually interchangeable.

For example, if you need to call ``spark_df.filter(...)`` of Spark DataFrame, you can do
as below:

.. code-block:: python

   >>> import databricks.koalas as ks
   >>>
   >>> kdf = ks.range(10)
   >>> sdf = kdf.to_spark().filter("id > 5")
   >>> sdf.show()
   +---+
   | id|
   +---+
   |  6|
   |  7|
   |  8|
   |  9|
   +---+

Spark DataFrame can be a Koalas DataFrame easily as below:

.. code-block:: python

   >>> sdf.to_koalas()
      id
   0   6
   1   7
   2   8
   3   9

However, note that it requires to create new default index in case Koalas DataFrame is created from
Spark DataFrame. See `Default Index Type <options.rst#default-index-type>`__. In order to avoid this overhead, specify the column
to use as an index when possible.

.. code-block:: python

   >>> # Create a Koalas DataFrame with an explicit index.
   ... kdf = ks.DataFrame({'id': range(10)}, index=range(10))
   >>> # Keep the explcit index.
   ... sdf = kdf.to_spark(index_col='index')
   >>> # Call Spark APIs
   ... sdf = sdf.filter("id > 5")
   >>> # Uses the explicit index to avoid to create default index.
   ... sdf.to_koalas(index_col='index')
          id
   index
   6       6
   7       7
   8       8
   9       9
