# Contributing to Koalas - design and principles

This document gives guidance to developers if they plan to contribute to Koalas.
In particular, it answers the questions:
 - what is in the scope of the Koalas project? What should go into PySpark or Pandas instead?
 - What is expected for code contributions

Koalas helps developers familiar with the Pandas API to scale their data science pipelines with Spark.
As such, it focuses on existing users of pandas.
 - it focuses on the high level, public APIs that users are usually working with
 - it respects to the largest extent the conventions of the Python numerical ecosystem, and allows the use of numpy types, etc. that are supported by Spark.
 - it should clearly document when results may differ between Spark and pandas (which may happen because Spark rows are not ordered)


### A unified API for data science

The Koalas dataframe is meant to provide the best of pandas and Spark under a single API, with easy and clear conversions 
between each API when necessary. It aims at incorporating:
 - most of the data transform tools from pandas
 - the SQL and streaming capabilities of Spark
 - a solid numerical foundation to integrate ML models and algorithms


There are 4 different classes of functions:
 - functions that are only found in Spark (`select`, `selectExpr`). These functions are also available in Koalas
 - functions that are found in Spark but that have a clear equivalent in pandas (`alias` and `rename` for example). These 
   functions will be implemented as the alias of the pandas function, but should be marked either for deprecation or that
   they are strictly aliases of the same functions. They are provided so that existing users of PySpark can get the benefits
   of Koalas without having to adapt their code.
 - functions that are found in both Spark and pandas under the same name (`count`, `dtypes`, `head`). The return value
   is the same as the return type in pandas (and not Spark).
 - functions that are only found in pandas. When these functions are appropriate for distributed datasets, they are available in Koalas.

Since Spark and Pandas have the similar API's with slight differences, the choice is to honor the contract of the pandas API first.

The `pandas.Series` object is much more versatile and universal than the `PySpark.Row` object. In particular, it can be used for most 
practical purposes as a replacement, so it is the preferred way of returning single results, when they are not scalars.

### Pandas functions that are not considered for inclusion in Koalas

A few categories of functions are not considered for now to be part of the API, for different reasons:

*Low level and deprecated functions* (Frame, user types, DataFrame.ix)

*Functions that have an unexpected performance impact* 
These functions (and the caller of theses functions) assume that the data is represented in a compact format (numpy in the case of pandas).
Because these functions would force the full collection of the data and because there is a well-documented workaround, it is recommended that 
they are not included. 

The workaround is to force the materialization of the pandas DataFrame, either by calling:
 - `.to_pandas()` (koalas only)
 - `.to_numpy()` (works with both pandas and koalas)

Here is a list of such functions:
 - DataFrame.values
 - `DataFrame.__iter__` and the array protocol `__array__`

Other frameworks like Dask or Molin have a low-level block representation of a multidimensional array that Spark lacks. Until such representation is available, these functions should not be considered.

### Spark functions that are temporarily included in Koalas

- Column.alias
- DataFrame.collect? `.to_pandas` is more complete
- DataFrame.describe
- DataFrame.distinct alias for `unique`
- DataFrame.join alias for `.merge`
- DataFrame.limit alias for `.head`

### Spark functions that should be included in Koalas

- pyspark.range

Streaming functions

Functions to control the partitioning, and in-memory representation of the data.

Functions that add SQL-like functionalities to DataFrame.

Functions that present algorithms specific to distributed datasets
(approx quantiles for example)

- DataFrame.approxQuantile, cache, checkpoint, coalesce, colRegex, createGlobalTempView, createOrReplaceGlobalTempView, createOrReplaceTempView, createTempView, crossJoin, crosstab, dropDuplicates / drop_duplicates, explain, hint, intersect, intersectAll, isLocal, isStreaming, localCheckpoint, persist, printSchema, registerTempTable, repartition, repartitionByRange, rollup, schema, select, selectExpr, subtract, take, toDF, unionByName, withWatermark


### Reading and writing files.

TODO: Koalas methods for reading and writing should work for both local and distributed files.


When writing, the protocol to use is the default Spark protocol. This means in particular that
dataframes written with Koalas will have multiple partitions, like in Spark. The current workaround is 
to call `.to_pandas()` and then save it.

## How to contribute

The largest amount of work consists simply in implementing the pandas API in Spark terms, which is usually straightforward.
Because this project is aimed at users who may not be familiar with the intimate technical details of pandas or Spark, a
few points have to be respected:

*Testing* For pandas functions, the testing coverage should be as good as in pandas. This is easily done by copying the 
relevant tests from pandas or dask into koalas.

*Documentation* For the implemented parameters, the documentation should be as comprehensive as in the corresponding parameter 
in PySpark or Pandas. At the very least, use the `@derived_from` decorator that automatically lifts the documentation from
the relevant project.

*Exposing details* Do not add internal fields if possible. Nearly all the state should be already encapsulated in the
Spark dataframe. Similarly, do not replicate the abstract interfaces found in Pandas. They are meant to be a protocol to exchange
data at high performance with numpy, which we cannot do anyway.

*Monkey patching, field introspection and other advanced python techniques* The current design should not require any of 
these.