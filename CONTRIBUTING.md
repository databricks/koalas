# Contributing Guide and Design Principles

This document gives guidance to developers if they plan to contribute to Koalas.
In particular, it answers the questions:
 - What is in the scope of the Koalas project? What should go into PySpark or pandas instead?
 - What is expected for code contributions?

Koalas focuses on making data scientists productive when analyzing big data. The initial goal is to remove as much friction as possible for data scientists when they transition from using pandas against small datasets to Spark on large datasets. As such, it focuses on existing users of pandas.
 - It focuses on the high level, public APIs that users are usually working with.
 - It respects to the largest extent the conventions of the Python numerical ecosystem, and allows the use of numpy types, etc. that are supported by Spark.
 - It should clearly document when results may differ between Spark and pandas (which may happen because Spark rows are not ordered).

Over time, the project will expand to include functionalities specific to big data analytics, e.g. plotting, data profiling, but for now we are focusing on the above.


## Design Principles

#### Pythonic

Koalas targets Python data scientists. We want to stick to the convention that users are already familiar with as much as possible. Here are some examples:

- Function names and parameters use snake_case, rather than CamelCase. This is different from PySpark's design. For example, Koalas has `to_pandas()`, whereas PySpark has `toPandas()` for converting a DataFrame into a pandas DataFrame. In limited cases, to maintain compatibility with Spark, we also provide Spark's variant as an alias.

- Koalas respects to the largest extent the conventions of the Python numerical ecosystem, and allows the use of numpy types, etc. that are supported by Spark.

#### Koalas data structure for big data, and pandas data structure for small data.

Often developers face the question whether a particular function should return a Koalas DataFrame/Series, or a pandas DataFrame/Series. The principle is: if the returned object can be large, use a Koalas DataFrame/Series. If the data is bound to be small, use a pandas DataFrame/Series. For example, `DataFrame.dtypes` return a pandas Series, because the number of columns in a DataFrame is bounded and small, whereas `DataFrame.head()` or `Series.unique()` returns a Koalas DataFrame, because the resulting object can be large.

#### Well documented APIs, with examples

Every single function and parameter should be documented. Most functions are documented with examples, because those are the easiest to understand than a blob of text explaining what the function does.

A recommended way to add documentation is to start with the docstring of the corresponding function in PySpark or pandas, and adapt it for Koalas. If you are adding a new function, also add it to the API reference doc index page in `docs/source/reference` directory. The examples in docstring also improve our test coverage.


#### Minimize the chances for users to shoot themselves in the foot

   


## Unifying pandas API and Spark API

The Koalas DataFrame is meant to provide the best of pandas and Spark under a single API, with easy and clear conversions between each API when necessary. It aims at incorporating:
 - most of the data transform tools from pandas
 - the SQL capabilities of Spark
 - a solid numerical foundation to integrate ML models and algorithms

There are 4 different classes of functions:
 1. Functions that are only found in Spark (`select`, `selectExpr`). These functions should also beavailable in Koalas.
 2. Functions that are found in Spark but that have a clear equivalent in pandas, e.g. `alias` and `rename`. These 
   functions will be implemented as the alias of the pandas function, but should be marked that they are aliases of the same functions. They are provided so that existing users of PySpark can get the benefits
   of Koalas without having to adapt their code.
 3. Functions that are found in both Spark and pandas under the same name (`count`, `dtypes`, `head`). The return value is the same as the return type in pandas (and not Spark's).
 4. Functions that are only found in pandas. When these functions are appropriate for distributed datasets, they are available in Koalas.

Since Spark and Pandas have the similar API's with slight differences, the choice is to honor the contract of the pandas API first.


### Return Type

Often developers face the question whether a particular function should return a Koalas DataFrame/Series, or a pandas DataFrame/Series.

The principle is: if the returned object can be large, use a Koalas DataFrame/Series. If the data is bound to be small, use a pandas DataFrame/Series. For example, `DataFrame.dtypes` return a pandas Series, because the number of columns in a DataFrame is bounded and small, whereas `DataFrame.head(n)` returns a Koalas DataFrame, because the number n can be very large.


### Pandas functions that are NOT considered for inclusion in Koalas

A few categories of functions are not considered for now to be part of the API, for different reasons:

1. *Low level and deprecated functions* (Frame, user types, DataFrame.ix)

2. *Functions that have an unexpected performance impact*

    These functions (and the caller of theses functions) assume that the data is represented in a compact format (numpy in the case of pandas). Because these functions would force the full collection of the data and because there is a well-documented workaround, it is recommended that  they are not included. 

    The workaround is to force the materialization of the pandas DataFrame, either by calling:
      - [`.to_pandas()`](https://koalas.readthedocs.io/en/stable/reference/api/databricks.koalas.DataFrame.to_pandas.html) : returns a pandas DataFrame, koalas only
      - [`.to_numpy()`](https://koalas.readthedocs.io/en/stable/reference/api/databricks.koalas.DataFrame.to_numpy.html): returns a numpy array, works with both pandas and Koalas

    Here is a list of such functions:
    - DataFrame.values
    - `DataFrame.__iter__` and the array protocol `__array__`

3. *Low-level functions for multidimensional arrays*: Other frameworks like Dask or Molin have a low-level block representation of a multidimensional array that Spark lacks. This includes for example all the array representations in `pandas.array`. Until such representation is available, these functions should not be considered.


### Spark functions that should be included in Koalas

- pyspark.range
- Functions to control the partitioning, and in-memory representation of the data.
- Functions that add SQL-like functionalities to DataFrame.
- Functions that present algorithms specific to distributed datasets (approx quantiles for example)


## How to contribute

The largest amount of work consists simply in implementing the pandas API in Spark terms, which is usually straightforward. Because this project is aimed at users who may not be familiar with the intimate technical details of pandas or Spark, a few points have to be respected:

- *Signaling your work*: If you are working on something, comment on the relevant ticket that are you doing so to avoid multiple people taking on the same work at the same time. It is also a good practice to signal that your work has stalled or you have moved on and want somebody else to take over.

- *Testing*: For pandas functions, the testing coverage should be as good as in pandas. This is easily done by copying the relevant tests from pandas or dask into Koalas.

- *Documentation*: For the implemented parameters, the documentation should be as comprehensive as in the corresponding parameter in PySpark or pandas. A recommended way to add documentation is to start with the docstring of the corresponding function in PySpark or pandas, and adapt it for Koalas. If you are adding a new function, also add it to the API reference doc index page in `docs/source/reference` directory.

- *Exposing details*: Do not add internal fields if possible. Nearly all the state should be already encapsulated in the Spark dataframe. Similarly, do not replicate the abstract interfaces found in Pandas. They are meant to be a protocol to exchange data at high performance with numpy, which we cannot do anyway.

- *Monkey patching, field introspection and other advanced python techniques*: The current design should not require any of these. Avoid using these advanced techniques as much as possible.
