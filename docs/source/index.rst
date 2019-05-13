.. Koalas' documentation master file

Koalas: pandas API on Apache Spark
============================================

Koalas makes data scientists more productive when interacting with big data, by augmenting the Apache Spark `Python DataFrame API <https://spark.apache.org/docs/latest/api/python/index.html>`_ to be compatible with the `pandas DataFrame API <https://pandas.pydata.org/>`_.

pandas is the de facto standard (single-node) DataFrame implementation in Python, while Spark is the de facto standard for big data processing. With Koalas package, you can:

* Be immediately productive with Spark, with no learning curve, if you are already familiar with pandas.
* Have a single codebase that works both with pandas (tests, smaller datasets) and with Spark (distributed datasets).

.. toctree::
    :maxdepth: 3

    quickstart
    reference/index
