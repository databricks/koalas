.. _api.general_functions:

=================
General functions
=================
.. currentmodule:: databricks.koalas

Working with options
--------------------

.. autosummary::
   :toctree: api/

    reset_option
    get_option
    set_option

Data manipulations and SQL
--------------------------
.. autosummary::
   :toctree: api/

   melt
   merge
   get_dummies
   concat
   sql

Top-level missing data
----------------------

.. autosummary::
   :toctree: api/

   to_numeric
   isna
   isnull
   notna
   notnull

Top-level dealing with datetimelike
-----------------------------------
.. autosummary::
   :toctree: api/

   to_datetime


Integration with Spark and pandas
---------------------------------
.. autosummary::
   :toctree: api/

   pandas_wraps
