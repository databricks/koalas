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
   Series.empty

Conversion
----------
.. autosummary::
   :toctree: api/

   Series.astype

Indexing, iteration
-------------------
.. autosummary::
   :toctree: api/

   Series.at
   Series.loc
   Series.iloc

Binary operator functions
-------------------------

.. autosummary::
   :toctree: api/

   Series.add
   Series.div
   Series.mul
   Series.radd
   Series.rdiv
   Series.rmul
   Series.rsub
   Series.rtruediv
   Series.sub
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
   Series.nunique
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

   Series.add_prefix
   Series.add_suffix
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
-------------------------------
.. autosummary::
   :toctree: api/

   Series.sort_index
   Series.sort_values

Combining / joining / merging
-----------------------------
.. autosummary::
   :toctree: api/

   Series.append

Accessors
---------

Koalas provides dtype-specific methods under various accessors.
These are separate namespaces within :class:`Series` that only apply
to specific data types.

========= ===========================
Data Type                    Accessor
========= ===========================
Datetime  :ref:`dt <api.series.dt>`
String    :ref:`str <api.series.str>`
Plot      :ref:`plot <api.series.plot>`
========= ===========================

.. _api.series.dt:

Date Time Handling
------------------

``Series.dt`` can be used to access the values of the series as
datetimelike and return several properties.
These can be accessed like ``Series.dt.<property>``.

Datetime Properties
~~~~~~~~~~~~~~~~~~~

.. currentmodule:: databricks.koalas.series
.. autosummary::
   :toctree: api/

   Series.dt.date
   Series.dt.year
   Series.dt.month
   Series.dt.week
   Series.dt.weekofyear
   Series.dt.day
   Series.dt.dayofweek
   Series.dt.weekday
   Series.dt.dayofyear
   Series.dt.hour
   Series.dt.minute
   Series.dt.second
   Series.dt.millisecond
   Series.dt.microsecond

Datetime Methods
~~~~~~~~~~~~~~~~

.. currentmodule:: databricks.koalas.series
.. autosummary::
   :toctree: api/

   Series.dt.strftime

.. _api.series.str:

String Handling
---------------

``Series.str`` can be used to access the values of the series as
strings and apply several methods to it. These can be accessed
like ``Series.str.<function/property>``.

.. currentmodule:: databricks.koalas.series
.. autosummary::
   :toctree: api/

   Series.str.capitalize
   Series.str.endswith
   Series.str.get
   Series.str.isalnum
   Series.str.isalpha
   Series.str.isdigit
   Series.str.isspace
   Series.str.islower
   Series.str.isupper
   Series.str.istitle
   Series.str.isnumeric
   Series.str.isdecimal
   Series.str.lower
   Series.str.lstrip
   Series.str.rstrip
   Series.str.startswith
   Series.str.strip
   Series.str.swapcase
   Series.str.upper

.. _api.series.plot:

Plotting
-------------------------------
``Series.plot`` is both a callable method and a namespace attribute for
specific plotting methods of the form ``Series.plot.<kind>``.

.. currentmodule:: databricks.koalas.series
.. autosummary::
   :toctree: api/

   Series.plot.bar
   Series.plot.box
   Series.plot.hist

.. currentmodule:: databricks.koalas
.. autosummary::
   :toctree: api/

   Series.hist

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
