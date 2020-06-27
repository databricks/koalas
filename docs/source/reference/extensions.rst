.. _api.extensions:

==========
Extensions
==========
.. currentmodule:: databricks.koalas.extensions

Accessors
---------

Accessors can be written and registered with Koalas Dataframes, Series, and
Index objects. Accessors allow developers to extend the functionality of
Koalas objects seamlessly by writing arbitrary classes and methods which are
then wrapped in one of the following decorators.

.. autosummary::
   :toctree: api/

   register_dataframe_accessor
   register_series_accessor
   register_index_accessor