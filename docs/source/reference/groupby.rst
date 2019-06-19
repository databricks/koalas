.. _api.groupby:

=======
GroupBy
=======
.. currentmodule:: databricks.koalas.groupby

GroupBy objects are returned by groupby calls: :func:`koalas.DataFrame.groupby`, :func:`koalas.Series.groupby`, etc.

Function application
--------------------
.. autosummary::
   :toctree: api/

   GroupBy.agg
   GroupBy.aggregate

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
