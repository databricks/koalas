.. _api.dataframe:

=========
DataFrame
=========
.. currentmodule:: databricks.koalas

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame

Attributes and underlying data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   DataFrame.index
   DataFrame.columns

.. autosummary::
   :toctree: api/

   DataFrame.dtypes
   DataFrame.shape

Conversion
~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.copy
   DataFrame.isna
   DataFrame.isnull
   DataFrame.notna
   DataFrame.notnull

Indexing, iteration
~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.head
   DataFrame.loc
   DataFrame.__iter__
   DataFrame.iteritems
   DataFrame.get

Function application, GroupBy & Window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.pipe
   DataFrame.groupby

.. _api.dataframe.stats:

Computations / Descriptive Stats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.abs
   DataFrame.corr
   DataFrame.count
   DataFrame.kurt
   DataFrame.kurtosis
   DataFrame.max
   DataFrame.mean
   DataFrame.min
   DataFrame.skew
   DataFrame.sum
   DataFrame.std
   DataFrame.var

Reindexing / Selection / Label manipulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.drop
   DataFrame.head
   DataFrame.reset_index
   DataFrame.set_index
   DataFrame.isin

.. _api.dataframe.missing:

Missing data handling
~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.dropna
   DataFrame.fillna

Reshaping, sorting, transposing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.sort_values

Combining / joining / merging
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   DataFrame.assign

Serialization / IO / Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

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
   DataFrame.to_latex
