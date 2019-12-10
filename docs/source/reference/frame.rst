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
   DataFrame.ndim
   DataFrame.size
   DataFrame.select_dtypes

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
   DataFrame.bool

Indexing, iteration
-------------------
.. autosummary::
   :toctree: api/

   DataFrame.at
   DataFrame.head
   DataFrame.idxmax
   DataFrame.idxmin
   DataFrame.loc
   DataFrame.iloc
   DataFrame.items
   DataFrame.iteritems
   DataFrame.iterrows
   DataFrame.keys
   DataFrame.xs
   DataFrame.get
   DataFrame.where
   DataFrame.mask

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
   DataFrame.pow
   DataFrame.rpow
   DataFrame.mod
   DataFrame.rmod
   DataFrame.floordiv
   DataFrame.rfloordiv
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
   DataFrame.agg
   DataFrame.aggregate
   DataFrame.groupby
   DataFrame.transform

.. _api.dataframe.stats:

Computations / Descriptive Stats
--------------------------------
.. autosummary::
   :toctree: api/

   DataFrame.abs
   DataFrame.all
   DataFrame.any
   DataFrame.clip
   DataFrame.corr
   DataFrame.count
   DataFrame.describe
   DataFrame.kurt
   DataFrame.kurtosis
   DataFrame.max
   DataFrame.mean
   DataFrame.min
   DataFrame.median
   DataFrame.pct_change
   DataFrame.quantile
   DataFrame.nunique
   DataFrame.skew
   DataFrame.sum
   DataFrame.std
   DataFrame.var
   DataFrame.cummin
   DataFrame.cummax
   DataFrame.cumsum
   DataFrame.cumprod
   DataFrame.round
   DataFrame.diff

Reindexing / Selection / Label manipulation
-------------------------------------------
.. autosummary::
   :toctree: api/

   DataFrame.add_prefix
   DataFrame.add_suffix
   DataFrame.drop
   DataFrame.drop_duplicates
   DataFrame.duplicated
   DataFrame.filter
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
   DataFrame.bfill
   DataFrame.ffill

Reshaping, sorting, transposing
-------------------------------
.. autosummary::
   :toctree: api/

   DataFrame.pivot_table
   DataFrame.pivot
   DataFrame.sort_index
   DataFrame.sort_values
   DataFrame.nlargest
   DataFrame.nsmallest
   DataFrame.melt
   DataFrame.T
   DataFrame.transpose
   DataFrame.reindex
   DataFrame.rank

Combining / joining / merging
-----------------------------
.. autosummary::
   :toctree: api/

   DataFrame.append
   DataFrame.assign
   DataFrame.merge
   DataFrame.join
   DataFrame.update

Time series-related
-------------------
.. autosummary::
   :toctree: api/

   DataFrame.shift
   DataFrame.first_valid_index

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
   DataFrame.style

.. _api.dataframe.plot:

Plotting
-------------------------------
``DataFrame.plot`` is both a callable method and a namespace attribute for
specific plotting methods of the form ``DataFrame.plot.<kind>``.

.. currentmodule:: databricks.koalas.frame
.. autosummary::
   :toctree: api/

   DataFrame.plot
   DataFrame.plot.area
   DataFrame.plot.barh
   DataFrame.plot.bar
   DataFrame.plot.hist
   DataFrame.plot.line
   DataFrame.plot.pie
   DataFrame.plot.scatter
   DataFrame.plot.kde
   DataFrame.plot.density
   DataFrame.hist
