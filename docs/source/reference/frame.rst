.. _api.dataframe:

=========
DataFrame
=========
.. currentmodule:: databricks.koalas

Constructor
-----------
.. autosummary::
   :toctree: api/

   DataFrame

Attributes and underlying data
------------------------------

.. autosummary::
   :toctree: api/

   DataFrame.index
   DataFrame.columns
   DataFrame.empty

.. autosummary::
   :toctree: api/

   DataFrame.dtypes
   DataFrame.shape
   DataFrame.size

Conversion
----------
.. autosummary::
   :toctree: api/

   DataFrame.copy
   DataFrame.isna
   DataFrame.astype
   DataFrame.isnull
   DataFrame.notna
   DataFrame.notnull

Indexing, iteration
-------------------
.. autosummary::
   :toctree: api/

   DataFrame.at
   DataFrame.head
   DataFrame.loc
   DataFrame.iloc
   DataFrame.iteritems
   DataFrame.get

Binary operator functions
-------------------------
.. autosummary::
   :toctree: api/

   DataFrame.add
   DataFrame.radd
   DataFrame.div
   DataFrame.rdiv
   DataFrame.truediv
   DataFrame.rtruediv
   DataFrame.mul
   DataFrame.rmul
   DataFrame.sub
   DataFrame.rsub
   DataFrame.lt
   DataFrame.gt
   DataFrame.le
   DataFrame.ge
   DataFrame.ne
   DataFrame.eq

Function application, GroupBy & Window
--------------------------------------
.. autosummary::
   :toctree: api/

   DataFrame.applymap
   DataFrame.pipe
   DataFrame.groupby

.. _api.dataframe.stats:

Computations / Descriptive Stats
--------------------------------
.. autosummary::
   :toctree: api/

   DataFrame.abs
   DataFrame.clip
   DataFrame.corr
   DataFrame.count
   DataFrame.describe
   DataFrame.kurt
   DataFrame.kurtosis
   DataFrame.max
   DataFrame.mean
   DataFrame.min
   DataFrame.nunique
   DataFrame.skew
   DataFrame.sum
   DataFrame.std
   DataFrame.var

Reindexing / Selection / Label manipulation
-------------------------------------------
.. autosummary::
   :toctree: api/

   DataFrame.add_prefix
   DataFrame.add_suffix
   DataFrame.drop
   DataFrame.head
   DataFrame.reset_index
   DataFrame.set_index
   DataFrame.isin
   DataFrame.sample

.. _api.dataframe.missing:

Missing data handling
---------------------
.. autosummary::
   :toctree: api/

   DataFrame.dropna
   DataFrame.fillna

Reshaping, sorting, transposing
-------------------------------
.. autosummary::
   :toctree: api/

   DataFrame.sort_index
   DataFrame.sort_values
   DataFrame.nlargest
   DataFrame.nsmallest
   DataFrame.melt

Combining / joining / merging
-----------------------------
.. autosummary::
   :toctree: api/

   DataFrame.append
   DataFrame.assign
   DataFrame.merge
   DataFrame.join

Cache
-------------------------------
.. autosummary::
   :toctree: api/

   DataFrame.cache

Serialization / IO / Conversion
-------------------------------
.. autosummary::
   :toctree: api/

   DataFrame.from_records
   DataFrame.to_table
   DataFrame.to_delta
   DataFrame.to_parquet
   DataFrame.to_spark_io
   DataFrame.to_csv
   DataFrame.to_pandas
   DataFrame.to_html
   DataFrame.to_numpy
   DataFrame.to_koalas
   DataFrame.to_spark
   DataFrame.to_string
   DataFrame.to_json
   DataFrame.to_dict
   DataFrame.to_excel
   DataFrame.to_clipboard
   DataFrame.to_records
   DataFrame.to_latex
