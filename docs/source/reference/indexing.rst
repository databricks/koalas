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

   Index.is_monotonic
   Index.is_monotonic_increasing
   Index.is_monotonic_decreasing
   Index.has_duplicates
   Index.hasnans
   Index.dtype
   Index.shape
   Index.name
   Index.names
   Index.ndim
   Index.size
   Index.nlevels
   Index.empty
   Index.T

Modifying and computations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Index.all
   Index.any
   Index.argmin
   Index.argmax
   Index.copy
   Index.is_boolean
   Index.is_categorical
   Index.is_floating
   Index.is_integer
   Index.is_interval
   Index.is_numeric
   Index.is_object
   Index.drop
   Index.drop_duplicates
   Index.min
   Index.max
   Index.rename
   Index.unique
   Index.nunique
   Index.value_counts

Compatibility with MultiIndex
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   Index.set_names

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
   Index.to_frame
   Index.to_numpy

Sorting
~~~~~~~
.. autosummary::
   :toctree: api/

   Index.sort_values

Time-specific operations
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.shift

Combining / joining / set operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.append
   Index.symmetric_difference

Selecting
~~~~~~~~~
.. autosummary::
   :toctree: api/

   Index.isin

.. _api.multiindex:

MultiIndex
----------
.. autosummary::
   :toctree: api/

   MultiIndex

MultiIndex Constructors
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.from_arrays
   MultiIndex.from_tuples
   MultiIndex.from_product

MultiIndex Properties
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.has_duplicates
   MultiIndex.hasnans
   MultiIndex.shape
   MultiIndex.names
   MultiIndex.ndim
   MultiIndex.empty
   MultiIndex.T
   MultiIndex.size
   MultiIndex.nlevels
   MultiIndex.levshape

MultiIndex Missing Values
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.fillna
   MultiIndex.dropna

MultiIndex Modifying and computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.drop
   MultiIndex.copy
   MultiIndex.rename
   MultiIndex.min
   MultiIndex.max
   MultiIndex.value_counts

MultiIndex Combining / joining / set operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.append
   MultiIndex.symmetric_difference

MultiIndex Conversion
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.astype
   MultiIndex.to_series
   MultiIndex.to_frame
   MultiIndex.to_numpy

MultiIndex Sorting
~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MultiIndex.sort_values
