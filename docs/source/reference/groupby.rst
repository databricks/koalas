.. _api.groupby:

=======
GroupBy
=======
.. currentmodule:: databricks.koalas

GroupBy objects are returned by groupby calls: :func:`DataFrame.groupby`, :func:`Series.groupby`, etc.

.. currentmodule:: databricks.koalas.groupby

Function application
--------------------
.. autosummary::
   :toctree: api/

   GroupBy.apply
   GroupBy.agg
   GroupBy.aggregate
   GroupBy.transform

Computations / Descriptive Stats
--------------------------------
.. autosummary::
   :toctree: api/

   GroupBy.all
   GroupBy.any
   GroupBy.count
   GroupBy.first
   GroupBy.last
   GroupBy.max
   GroupBy.mean
   GroupBy.min
   GroupBy.std
   GroupBy.sum
   GroupBy.var
   GroupBy.size
