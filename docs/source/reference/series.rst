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
   Series.dtypes
   Series.name
   Series.schema
   Series.shape
   Series.size

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
   Series.iloc

Binary operator functions
-------------------------

.. autosummary::
   :toctree: api/

   Series.add
   Series.div
   Series.divide
   Series.mul
   Series.multiply
   Series.radd
   Series.rdiv
   Series.rmul
   Series.rsub
   Series.rtruediv
   Series.sub
   Series.subtract
   Series.truediv

Function application, GroupBy & Window
--------------------------------------
.. autosummary::
   :toctree: api/

   Series.apply
   Series.map
   Series.groupby
   Series.pipe

.. _api.series.stats:

Computations / Descriptive Stats
--------------------------------
.. autosummary::
   :toctree: api/

   Series.abs
   Series.all
   Series.any
   Series.clip
   Series.corr
   Series.count
   Series.describe
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

Reshaping, sorting, transposing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   Series.sort_index
   Series.sort_values

Serialization / IO / Conversion
-------------------------------
.. autosummary::
   :toctree: api/

   Series.to_pandas
   Series.to_numpy
   Series.to_list
   Series.to_string
   Series.to_dict
   Series.to_clipboard
   Series.to_latex
   Series.to_json
   Series.to_csv
   Series.to_excel

Datetime Methods
----------------
Methods accessible through `Series.dt`

.. currentmodule:: databricks.koalas.datetimes
.. autosummary::
   :toctree: api/

   DatetimeMethods.date
   DatetimeMethods.year
   DatetimeMethods.month
   DatetimeMethods.week
   DatetimeMethods.weekofyear
   DatetimeMethods.day
   DatetimeMethods.dayofweek
   DatetimeMethods.weekday
   DatetimeMethods.dayofyear
   DatetimeMethods.hour
   DatetimeMethods.minute
   DatetimeMethods.second
   DatetimeMethods.millisecond
   DatetimeMethods.microsecond

   DatetimeMethods.strftime
