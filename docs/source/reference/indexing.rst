.. _api.indexing:

========
Indexing
========

Index
-----
.. currentmodule:: databricks.koalas

.. autosummary::
   :toctree: api/

   Index

Properties
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.dtype
   Index.name
   Index.names
   Index.empty

Modifying and computations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Index.copy
   Index.is_boolean
   Index.is_categorical
   Index.is_floating
   Index.is_integer
   Index.is_interval
   Index.is_numeric
   Index.is_object

Missing Values
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.isna
   Index.notna

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.astype
   Index.to_series

.. _api.multiindex:

Selecting
~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.isin

MultiIndex
----------
.. autosummary::
   :toctree: api/

   MultiIndex

MultiIndex Properties
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.names

MultiIndex Modifying and computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.copy
