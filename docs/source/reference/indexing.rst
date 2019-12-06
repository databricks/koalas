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
   Index.shape
   Index.name
   Index.names
   Index.ndim
   Index.empty
   Index.T

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
   Index.value_counts

Missing Values
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.fillna
   Index.dropna
   Index.isna
   Index.notna

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.astype
   Index.to_series
   Index.to_numpy

.. _api.multiindex:

Combining / joining / set operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.symmetric_difference

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

MultiIndex Constructors
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.from_tuples

MultiIndex Properties
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.shape
   MultiIndex.names
   MultiIndex.ndim
   MultiIndex.T
   MultiIndex.nlevels
   MultiIndex.levshape

MultiIndex Missing Values
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.dropna

MultiIndex Modifying and computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.copy
   MultiIndex.value_counts

MultiIndex Combining / joining / set operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.symmetric_difference

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.to_numpy
