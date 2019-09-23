.. _api.io:

============
Input/Output
============
.. currentmodule:: databricks.koalas


Data Generator
--------------
.. autosummary::
   :toctree: api/

   range

Spark Metastore Table
---------------------
.. autosummary::
   :toctree: api/

   read_table
   DataFrame.to_table

Delta Lake
----------
.. autosummary::
   :toctree: api/

   read_delta
   DataFrame.to_delta

Parquet
-------
.. autosummary::
   :toctree: api/

   read_parquet
   DataFrame.to_parquet

Generic Spark I/O
-----------------
.. autosummary::
   :toctree: api/

   read_spark_io
   DataFrame.to_spark_io

Flat File / CSV
---------------
.. autosummary::
   :toctree: api/

   read_csv
   DataFrame.to_csv

Clipboard
---------
.. autosummary::
   :toctree: api/

   read_clipboard
   DataFrame.to_clipboard

Excel
-----
.. autosummary::
   :toctree: api/

   read_excel
   DataFrame.to_excel

JSON
----
.. autosummary::
   :toctree: api/

   read_json

HTML
----
.. autosummary::
   :toctree: api/

   read_html
   DataFrame.to_html

SQL
---
.. autosummary::
   :toctree: api/

   read_sql_table
   read_sql_query
   read_sql
