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

#### Be Pythonic

Koalas targets Python data scientists. We want to stick to the convention that users are already familiar with as much as possible. Here are some examples:

- Function names and parameters use snake_case, rather than CamelCase. This is different from PySpark's design. For example, Koalas has `to_pandas()`, whereas PySpark has `toPandas()` for converting a DataFrame into a pandas DataFrame. In limited cases, to maintain compatibility with Spark, we also provide Spark's variant as an alias.

- Koalas respects to the largest extent the conventions of the Python numerical ecosystem, and allows the use of numpy types, etc. that are supported by Spark.

- Koalas docs follow rest of the PyData project docs.

#### Unify small data (pandas) API and big data (Spark) API, but pandas first

The Koalas DataFrame is meant to provide the best of pandas and Spark under a single API, with easy and clear conversions between each API when necessary. When Spark and pandas have similar APIs with subtle differences, the principle is to honor the contract of the pandas API first.

There are 4 different classes of functions:

 1. Functions that are only found in Spark (`select`, `selectExpr`). These functions should also beavailable in Koalas.

 2. Functions that are found in Spark but that have a clear equivalent in pandas, e.g. `alias` and `rename`. These functions will be implemented as the alias of the pandas function, but should be marked that they are aliases of the same functions. They are provided so that existing users of PySpark can get the benefits of Koalas without having to adapt their code.

 3. Functions that are found in both Spark and pandas under the same name (`count`, `dtypes`, `head`). The return value is the same as the return type in pandas (and not Spark's).

 4. Functions that are only found in pandas. When these functions are appropriate for distributed datasets, they should become available in Koalas.


#### Return Koalas data structure for big data, and pandas data structure for small data

Often developers face the question whether a particular function should return a Koalas DataFrame/Series, or a pandas DataFrame/Series. The principle is: if the returned object can be large, use a Koalas DataFrame/Series. If the data is bound to be small, use a pandas DataFrame/Series. For example, `DataFrame.dtypes` return a pandas Series, because the number of columns in a DataFrame is bounded and small, whereas `DataFrame.head()` or `Series.unique()` returns a Koalas DataFrame, because the resulting object can be large.


#### Provide well documented APIs, with examples

All functions and parameters should be documented. Most functions should be documented with examples, because those are the easiest to understand than a blob of text explaining what the function does.

A recommended way to add documentation is to start with the docstring of the corresponding function in PySpark or pandas, and adapt it for Koalas. If you are adding a new function, also add it to the API reference doc index page in `docs/source/reference` directory. The examples in docstring also improve our test coverage.


#### Guardrails to prevent users from shooting themselves in the foot

Certain operations in pandas are prohibitively expensive as data scales, and we don't want to give users the illusion that they can rely on such operations in Koalas. That is to say, methods implemented in Koalas should be safe to perform by default on large datasets. As a result, the following capabilities are not implemented in Koalas:

1. Capabilities that are fundamentally not parallelizable: e.g. imperatively looping over each element
2. Capabilities that require materializing the entire working set in a single node's memory. This is why we do not implement [`pandas.DataFrame.values`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.values.html#pandas.DataFrame.values). Another example is the `_repr_html_` call caps the total number of records shown to a maximum of 1000, to prevent users from blowing up their driver node simply by typing the name of the DataFrame in a notebook.

A few exceptions, however, exist. One common pattern with "big data science" is that while the initial dataset is large, the working set becomes smaller as the analysis goes deeper. For example, data scientists often perform aggregation on datasets and want to then convert the aggregated dataset to some local data structure. To help data scientists, we offer the following:

- [`DataFrame.to_pandas()`](https://koalas.readthedocs.io/en/stable/reference/api/databricks.koalas.DataFrame.to_pandas.html) : returns a pandas DataFrame, koalas only
- [`DataFrame.to_numpy()`](https://koalas.readthedocs.io/en/stable/reference/api/databricks.koalas.DataFrame.to_numpy.html): returns a numpy array, works with both pandas and Koalas

Note that it is clear from the names that these functions return some local data structure that would require materializing data in a single node's memory. For these functions, we also explicitly document them with a warning note that the resulting data structure must be small.


#### Be a lean API layer and move fast

Koalas is designed as an API overlay layer on top of Spark. The project should be lightweight, and most functions should be implemented as wrappers around Spark or pandas. Koalas does not accept heavyweight implementations, e.g. execution engine changes.

This approach enables us to move fast. For the considerable future, we aim to be making weekly releases.


#### High test coverage

Koalas should be well tested. The project tracks its test coverage with over 90% across the entire codebase, and close to 100% for critical parts. Pull requests will not be accepted unless they have close to 100% statement coverage from the codecov report.


## How to contribute

The largest amount of work consists simply in implementing the pandas API in Spark terms, which is usually straightforward. Because this project is aimed at users who may not be familiar with the intimate technical details of pandas or Spark, a few points have to be respected:

- *Signaling your work*: If you are working on something, comment on the relevant ticket that are you doing so to avoid multiple people taking on the same work at the same time. It is also a good practice to signal that your work has stalled or you have moved on and want somebody else to take over.

- *Testing*: For pandas functions, the testing coverage should be as good as in pandas. This is easily done by copying the relevant tests from pandas or dask into Koalas.

- *Documentation*: For the implemented parameters, the documentation should be as comprehensive as in the corresponding parameter in PySpark or pandas. A recommended way to add documentation is to start with the docstring of the corresponding function in PySpark or pandas, and adapt it for Koalas. If you are adding a new function, also add it to the API reference doc index page in `docs/source/reference` directory.

- *Exposing details*: Do not add internal fields if possible. Nearly all the state should be already encapsulated in the Spark dataframe. Similarly, do not replicate the abstract interfaces found in Pandas. They are meant to be a protocol to exchange data at high performance with numpy, which we cannot do anyway.

- *Monkey patching, field introspection and other advanced python techniques*: The current design should not require any of these. Avoid using these advanced techniques as much as possible.
