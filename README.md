

# Koalas: pandas API on Apache Spark <!-- omit in toc -->

The Koalas project makes data scientists more productive when interacting with big data, by implementing the pandas DataFrame API on top of Apache Spark.

pandas is the de facto standard (single-node) DataFrame implementation in Python, while Spark is the de facto standard for big data processing. With this package, you can:
 - Be immediately productive with Spark, with no learning curve, if you are already familiar with pandas.
 - Have a single codebase that works both with pandas (tests, smaller datasets) and with Spark (distributed datasets).

This project is currently in beta and is rapidly evolving, with a weekly release cadence. We would love to have you try it and give us feedback, through our [mailing lists](https://groups.google.com/forum/#!forum/koalas-dev) or [GitHub issues](https://github.com/databricks/koalas/issues).

[![Build Status](https://travis-ci.com/databricks/koalas.svg?token=Rzzgd1itxsPZRuhKGnhD&branch=master)](https://travis-ci.com/databricks/koalas)
[![codecov](https://codecov.io/gh/databricks/koalas/branch/master/graph/badge.svg)](https://codecov.io/gh/databricks/koalas)
[![Documentation Status](https://readthedocs.org/projects/koalas/badge/?version=latest)](https://koalas.readthedocs.io/en/latest/?badge=latest)
[![Latest Release](https://img.shields.io/pypi/v/koalas.svg)](https://pypi.org/project/koalas/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/koalas.svg)](https://anaconda.org/conda-forge/koalas)


## Table of Contents <!-- omit in toc -->
- [Dependencies](#dependencies)
- [Get Started](#get-started)
- [Documentation](#documentation)
- [Mailing List](#mailing-list)
- [Development Guide](#development-guide)
- [Design Principles](#design-principles)
  - [Be Pythonic](#be-pythonic)
  - [Unify small data (pandas) API and big data (Spark) API, but pandas first](#unify-small-data-pandas-api-and-big-data-spark-api-but-pandas-first)
  - [Return Koalas data structure for big data, and pandas data structure for small data](#return-koalas-data-structure-for-big-data-and-pandas-data-structure-for-small-data)
  - [Provide discoverable APIs for common data science tasks](#provide-discoverable-apis-for-common-data-science-tasks)
  - [Provide well documented APIs, with examples](#provide-well-documented-apis-with-examples)
  - [Guardrails to prevent users from shooting themselves in the foot](#guardrails-to-prevent-users-from-shooting-themselves-in-the-foot)
  - [Be a lean API layer and move fast](#be-a-lean-api-layer-and-move-fast)
  - [High test coverage](#high-test-coverage)
- [FAQ](#faq)
  - [What's the project's status?](#whats-the-projects-status)
  - [Is it Koalas or koalas?](#is-it-koalas-or-koalas)
  - [Should I use PySpark's DataFrame API or Koalas?](#should-i-use-pysparks-dataframe-api-or-koalas)
  - [How can I request support for a method?](#how-can-i-request-support-for-a-method)
  - [How is Koalas different from Dask?](#how-is-koalas-different-from-dask)
  - [How can I contribute to Koalas?](#how-can-i-contribute-to-koalas)
  - [Why a new project (instead of putting this in Apache Spark itself)?](#why-a-new-project-instead-of-putting-this-in-apache-spark-itself)
  - [How do I use this on Databricks?](#how-do-i-use-this-on-databricks)


## Dependencies

 - [cmake](https://cmake.org/) for building pyarrow
 - Spark 2.4. Some older versions of Spark may work too but they are not officially supported.
 - A recent version of pandas. It is officially developed against 0.23+ but some other versions may work too.
 - Python 3.5+.


## Get Started

Koalas is available at the Python package index:
```bash
pip install koalas
```

or with the conda package manager:
```bash
conda install koalas -c conda-forge
```

If this fails to install the pyarrow dependency, you may want to try installing with Python 3.6.x, as `pip install arrow` does not work out of the box for 3.7 https://github.com/apache/arrow/issues/1125.

If you don't have Spark environment, you should also install `pyspark` package by:
```bash
pip install 'pyspark>=2.4'
```

or
```bash
conda install 'pyspark>=2.4' -c conda-forge
```

or downloading the release.

After installing the packages, you can import the package:
```py
import databricks.koalas as ks
```

Now you can turn a pandas DataFrame into a Koalas DataFrame that is API-compliant with the former:
```py
import pandas as pd
pdf = pd.DataFrame({'x':range(3), 'y':['a','b','b'], 'z':['a','b','b']})

# Create a Koalas DataFrame from pandas DataFrame
df = ks.from_pandas(pdf)

# Rename the columns
df.columns = ['x', 'y', 'z1']

# Do some operations in place:
df['x2'] = df.x * df.x
```


## Documentation

Project docs are published here: https://koalas.readthedocs.io


## Mailing List

We use Google Groups for mailling list: https://groups.google.com/forum/#!forum/koalas-dev


## Development Guide

See [CONTRIBUTING.md](https://github.com/databricks/koalas/blob/master/CONTRIBUTING.md).


## Design Principles

This section outlines design princples guiding the Koalas project.

### Be Pythonic

Koalas targets Python data scientists. We want to stick to the convention that users are already familiar with as much as possible. Here are some examples:

- Function names and parameters use snake_case, rather than CamelCase. This is different from PySpark's design. For example, Koalas has `to_pandas()`, whereas PySpark has `toPandas()` for converting a DataFrame into a pandas DataFrame. In limited cases, to maintain compatibility with Spark, we also provide Spark's variant as an alias.

- Koalas respects to the largest extent the conventions of the Python numerical ecosystem, and allows the use of NumPy types, etc. that can be supported by Spark.

- Koalas docs' style and infrastructure simply follow rest of the PyData projects'.

### Unify small data (pandas) API and big data (Spark) API, but pandas first

The Koalas DataFrame is meant to provide the best of pandas and Spark under a single API, with easy and clear conversions between each API when necessary. When Spark and pandas have similar APIs with subtle differences, the principle is to honor the contract of the pandas API first.

There are different classes of functions:

 1. Functions that are found in both Spark and pandas under the same name (`count`, `dtypes`, `head`). The return value is the same as the return type in pandas (and not Spark's).
    
 2. Functions that are found in Spark but that have a clear equivalent in pandas, e.g. `alias` and `rename`. These functions will be implemented as the alias of the pandas function, but should be marked that they are aliases of the same functions. They are provided so that existing users of PySpark can get the benefits of Koalas without having to adapt their code.
 
 3. Functions that are only found in pandas. When these functions are appropriate for distributed datasets, they should become available in Koalas.
 
 4. Functions that are only found in Spark that are essential to controlling the distributed nature of the computations, e.g. `cache`. These functions should be available in Koalas.

We are still debating whether data transformation functions only available in Spark should be added to Koalas, e.g. `select`. We would love to hear your feedback on that.


### Return Koalas data structure for big data, and pandas data structure for small data

Often developers face the question whether a particular function should return a Koalas DataFrame/Series, or a pandas DataFrame/Series. The principle is: if the returned object can be large, use a Koalas DataFrame/Series. If the data is bound to be small, use a pandas DataFrame/Series. For example, `DataFrame.dtypes` return a pandas Series, because the number of columns in a DataFrame is bounded and small, whereas `DataFrame.head()` or `Series.unique()` returns a Koalas DataFrame/Series, because the resulting object can be large.

### Provide discoverable APIs for common data science tasks

At the risk of overgeneralization, there are two API design approaches: the first focuses on providing APIs for common tasks; the second starts with abstractions, and enable users to accomplish their tasks by composing primitives. While the world is not black and white, pandas takes more of the former approach, while Spark has taken more of the later.

One example is value count (count by some key column), one of the most common operations in data science. pandas `DataFrame.value_count` returns the result in sorted order, which in 90% of the cases is what users prefer when exploring data, whereas Spark's does not sort, which is more desirable when building data pipelines, as users can accomplish the pandas behavior by adding an explicit `orderBy`.

Similar to pandas, Koalas should also lean more towards the former, providing discoverable APIs for common data science tasks. In most cases, this principle is well taken care of by simply implementing pandas' APIs. However, there will be circumstances in which pandas' APIs don't address a specific need, e.g. plotting for big data.

### Provide well documented APIs, with examples

All functions and parameters should be documented. Most functions should be documented with examples, because those are the easiest to understand than a blob of text explaining what the function does.

A recommended way to add documentation is to start with the docstring of the corresponding function in PySpark or pandas, and adapt it for Koalas. If you are adding a new function, also add it to the API reference doc index page in `docs/source/reference` directory. The examples in docstring also improve our test coverage.

### Guardrails to prevent users from shooting themselves in the foot

Certain operations in pandas are prohibitively expensive as data scales, and we don't want to give users the illusion that they can rely on such operations in Koalas. That is to say, methods implemented in Koalas should be safe to perform by default on large datasets. As a result, the following capabilities are not implemented in Koalas:

1. Capabilities that are fundamentally not parallelizable: e.g. imperatively looping over each element
2. Capabilities that require materializing the entire working set in a single node's memory. This is why we do not implement [`pandas.DataFrame.values`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.values.html#pandas.DataFrame.values). Another example is the `_repr_html_` call caps the total number of records shown to a maximum of 1000, to prevent users from blowing up their driver node simply by typing the name of the DataFrame in a notebook.

A few exceptions, however, exist. One common pattern with "big data science" is that while the initial dataset is large, the working set becomes smaller as the analysis goes deeper. For example, data scientists often perform aggregation on datasets and want to then convert the aggregated dataset to some local data structure. To help data scientists, we offer the following:

- [`DataFrame.to_pandas()`](https://koalas.readthedocs.io/en/stable/reference/api/databricks.koalas.DataFrame.to_pandas.html) : returns a pandas DataFrame, koalas only
- [`DataFrame.to_numpy()`](https://koalas.readthedocs.io/en/stable/reference/api/databricks.koalas.DataFrame.to_numpy.html): returns a numpy array, works with both pandas and Koalas

Note that it is clear from the names that these functions return some local data structure that would require materializing data in a single node's memory. For these functions, we also explicitly document them with a warning note that the resulting data structure must be small.

### Be a lean API layer and move fast

Koalas is designed as an API overlay layer on top of Spark. The project should be lightweight, and most functions should be implemented as wrappers around Spark or pandas. Koalas does not accept heavyweight implementations, e.g. execution engine changes.

This approach enables us to move fast. For the considerable future, we aim to be making weekly releases. If we find a critical bug, we will be making a new release as soon as the bug fix is available.

### High test coverage

Koalas should be well tested. The project tracks its test coverage with over 90% across the entire codebase, and close to 100% for critical parts. Pull requests will not be accepted unless they have close to 100% statement coverage from the codecov report.



## FAQ

### What's the project's status?
This project is currently in beta and is rapidly evolving.
We plan to do weekly releases at this stage.
You should expect the following differences:

 - some functions may be missing (see the [Contributions](#Contributions) section)

 - some behavior may be different, in particular in the treatment of nulls: Pandas uses
   Not a Number (NaN) special constants to indicate missing values, while Spark has a
   special flag on each value to indicate missing values. We would love to hear from you
   if you come across any discrepancies

 - because Spark is lazy in nature, some operations like creating new columns only get 
   performed when Spark needs to print or write the dataframe.

### Is it Koalas or koalas?

It's Koalas. Unlike pandas, we use upper case here.

### Should I use PySpark's DataFrame API or Koalas?

If you are already familiar with pandas and want to leverage Spark for big data, we recommend
using Koalas. If you are learning Spark from ground up, we recommend you start with PySpark's API.

### How can I request support for a method?

File a GitHub issue: https://github.com/databricks/koalas/issues

Databricks customers are also welcome to file a support ticket to request a new feature.

### How is Koalas different from Dask?

Different projects have different focuses. Spark is already deployed in virtually every
organization, and often is the primary interface to the massive amount of data stored in data lakes.
Koalas was inspired by Dask, and aims to make the transition from pandas to Spark easy for data
scientists.

### How can I contribute to Koalas?

See [CONTRIBUTING.md](https://github.com/databricks/koalas/blob/master/CONTRIBUTING.md).

### Why a new project (instead of putting this in Apache Spark itself)?

Two reasons:

1. We want a venue in which we can rapidly iterate and make new releases. The overhead of making a
release as a separate project is minuscule (in the order of minutes). A release on Spark takes a
lot longer (in the order of days)

2. Koalas takes a different approach that might contradict Spark's API design principles, and those
principles cannot be changed lightly given the large user base of Spark. A new, separate project
provides an opportunity for us to experiment with new design principles.

### How do I use this on Databricks?

Koalas requires Databricks Runtime 5.x or above. For the regular Databricks Runtime, you can install Koalas using the Libraries tab on the cluster UI, or using dbutils in a notebook:

```python
dbutils.library.installPyPI("koalas")
dbutils.library.restartPython()
```

In the future, we will package Koalas out-of-the-box in both the regular Databricks Runtime and
Databricks Runtime for Machine Learning.
