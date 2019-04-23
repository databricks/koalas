.. currentmodule:: databricks.koalas

API Reference
=============

Intput/Output
-------------

.. autosummary::
    :toctree: functions

    read_csv
    read_parquet
    to_datetime
    from_pandas
    get_dummies

DataFrame
---------

.. automodapi::

.. autosummary::
    :toctree: functions

    DataFrame
    DataFrame.count
    DataFrame.iteritems
    DataFrame.to_html
    DataFrame.index
    DataFrame.set_index
    DataFrame.reset_index
    DataFrame.isnull
    DataFrame.notnull
    DataFrame.isna
    DataFrame.notna
    DataFrame.toPandas
    DataFrame.assign
    DataFrame.loc
    DataFrame.copy
    DataFrame.dropna
    DataFrame.head
    DataFrame.columns
    DataFrame.columns
    DataFrame.count
    DataFrame.unique
    DataFrame.drop
    DataFrame.get
    DataFrame.sort_values
    DataFrame.groupby
    DataFrame.pipe
    DataFrame.shape
    DataFrame.mean
    DataFrame.sum
    DataFrame.skew
    DataFrame.kurtosis
    DataFrame.kurt
    DataFrame.min
    DataFrame.max
    DataFrame.std
    DataFrame.var
    DataFrame.abs
    DataFrame.compute

Series
---------

.. autosummary::
    :toctree: functions

    Series
    Series.dtype
    Series.spark_type
    Series.astype
    Series.alias
    Series.getField
    Series.schema
    Series.shape
    Series.name
    Series.rename
    Series.index
    Series.reset_index
    Series.loc
    Series.to_dataframe
    Series.toPandas
    Series.isnull
    Series.isna
    Series.notnull
    Series.notna
    Series.dropna
    Series.head
    Series.unique
    Series.value_counts
    Series.count
    Series.mean
    Series.sum
    Series.skew
    Series.kurtosis
    Series.kurt
    Series.min
    Series.max
    Series.std
    Series.var
    Series.abs
    Series.compute

