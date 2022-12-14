.. Koalas' documentation master file

.. warning::
   DEPRECATED: Koalas supports Apache Spark 3.1 and below as it is `officially included to PySpark in Apache Spark 3.2 <https://issues.apache.org/jira/browse/SPARK-34849>`_. This repository is now in maintenance mode. For Apache Spark 3.2 and above, please use `PySpark <https://spark.apache.org/docs/latest/api/python/migration_guide/koalas_to_pyspark.html>`_ directly.

Koalas: pandas API on Apache Spark
============================================

The Koalas project makes data scientists more productive when interacting with big data, by implementing the pandas DataFrame API on top of Apache Spark.
pandas is the de facto standard (single-node) DataFrame implementation in Python, while Spark is the de facto standard for big data processing.
With this package, you can:

* Be immediately productive with Spark, with no learning curve, if you are already familiar with pandas.
* Have a single codebase that works both with pandas (tests, smaller datasets) and with Spark (distributed datasets).

We would love to have you try it and give us feedback,
through our `mailing lists <https://groups.google.com/forum/#!forum/koalas-dev>`_ or `GitHub issues <https://github.com/databricks/koalas/issues>`_.
Try the Koalas 10 minutes tutorial on a live Jupyter notebook `here <https://mybinder.org/v2/gh/databricks/koalas/master?filepath=docs%2Fsource%2Fgetting_started%2F10min.ipynb>`_.
The initial launch can take up to several minutes.

.. whatsnew/index is automatically generated (see conf.py for Sphinx and dev/gendoc.py).

.. toctree::
    :maxdepth: 3

    getting_started/index
    user_guide/index
    reference/index
    development/index
    whatsnew/index

