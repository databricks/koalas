.. _api.series:

======
Series
======
.. currentmodule:: databricks.koalas

Constructor
-----------
.. autosummary::
   :toctree: api/

   Series

Attributes
----------

.. autosummary::
   :toctree: api/

   Series.index

.. autosummary::
   :toctree: api/

   Series.dtype
   Series.name
   Series.schema
   Series.shape

Conversion
----------
.. autosummary::
   :toctree: api/

   Series.astype

Indexing, iteration
-------------------
.. autosummary::
   :toctree: api/

   Series.loc

Function application, GroupBy & Window
--------------------------------------
.. autosummary::
   :toctree: api/

   Series.groupby

.. _api.series.stats:

Computations / Descriptive Stats
--------------------------------
.. autosummary::
   :toctree: api/

   Series.abs
   Series.clip
   Series.corr
   Series.count
   Series.kurt
   Series.max
   Series.mean
   Series.min
   Series.skew
   Series.std
   Series.sum
   Series.var
   Series.kurtosis
   Series.unique
   Series.value_counts

Reindexing / Selection / Label manipulation
-------------------------------------------
.. autosummary::
   :toctree: api/

   Series.head
   Series.isin
   Series.rename
   Series.reset_index
   Series.sample

Missing data handling
---------------------
.. autosummary::
   :toctree: api/

   Series.isna
   Series.isnull
   Series.notna
   Series.notnull
   Series.dropna
   Series.fillna

Serialization / IO / Conversion
-------------------------------
.. autosummary::
   :toctree: api/

   Series.to_pandas
   Series.to_numpy
   Series.to_string
   Series.to_dict
   Series.to_clipboard
   Series.to_latex

Plotting
-------------------------------
``Series.plot`` is both a callable method and a namespace attribute for
specific plotting methods of the form ``Series.plot.<kind>``.

.. autosummary::
   :toctree: api/

   Series.plot.bar
   Series.plot.box
   Series.plot.hist

.. autosummary::
   :toctree: api/

   Series.hist
