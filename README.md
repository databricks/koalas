

# Koalas: pandas APIs on Apache Spark <!-- omit in toc -->

The Koalas project makes data scientists more productive when interacting with big data, by augmenting Apache Spark's Python DataFrame API to be compatible with pandas'.

pandas is the de facto standard (single-node) dataframe implementation in Python, while Spark is the de facto standard for big data processing. With this package, data scientists can:
 - Be immediately productive with Spark, with no learning curve, if one is already familiar with pandas.
 - Have a single codebase that works both with pandas (tests, smaller datasets) and with Spark (distributed datasets).

[![Build Status](https://travis-ci.com/databricks/koalas.svg?token=Rzzgd1itxsPZRuhKGnhD&branch=master)](https://travis-ci.com/databricks/koalas)
[![Latest release](https://img.shields.io/pypi/v/koalas.svg)](https://pypi.org/project/koalas/)


## Table of Contents <!-- omit in toc -->
- [Dependencies](#dependencies)
- [Get Started](#get-started)
- [Documentation](#documentation)
- [Mailing List](#mailing-list)
- [Development Guide](#development-guide)
  - [Environment Setup](#environment-setup)
  - [Running Tests](#running-tests)
  - [Building Documentation](#building-documentation)
  - [Coding Conventions](#coding-conventions)
  - [Release Instructions](#release-instructions)
- [FAQ](#faq)
  - [What's the project's status?](#whats-the-projects-status)
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
 - Python 3.5+ if you want to use type hints in UDFs. Work is ongoing to also support Python 2.


## Get Started

Koalas is available at the Python package index:
```bash
pip install koalas
```

If this fails to install the pyarrow dependency, you may want to try installing with Python 3.6.x, as `pip install arrow` does not work out of the box for 3.7 https://github.com/apache/arrow/issues/1125.

After installing the package, you can import the package:
```py
from databricks import koalas
```

Now you can turn a pandas DataFrame into a Koalas DataFrame that is API-compliant with the former:
```py
import pandas as pd
pdf = pd.DataFrame({'x':range(3), 'y':['a','b','b'], 'z':['a','b','b']})

# Create a Koalas DataFrame from pandas DataFrame
df = koalas.from_pandas(pdf)

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

### Environment Setup

We recommend setting up a Conda environment for development:
```bash
conda create --name koalas-dev-env python=3.6
conda activate koalas-dev-env
conda install -c conda-forge pyspark=2.4
conda install -c conda-forge --yes --file requirements-dev.txt
pip install -e .  # installs koalas from current checkout
```

Once setup, make sure you switch to `koalas-dev-env` before development:
```bash
conda activate koalas-dev-env
```

### Running Tests

To run all the tests, similar to our CI pipeline:
```bash
./dev/run-tests.sh
```

To run a specific test file:
```bash
python databricks/koalas/tests/test_dataframe.py
```

To run a specific test method:
```bash
python databricks/koalas/tests/test_dataframe.py DataFrameTest.test_Dataframe
```

### Building Documentation

To build documentation via Sphinx:

```bash
cd docs && make clean html
```

It generates HTMLs under `docs/_build/html` directory. Open `docs/_build/html/index.html` to check if documentation is built properly.

### Coding Conventions
We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with one exception: lines can be up to 100 characters in length, not 79.

### Release Instructions
Only project maintainers can do the following.

Step 1. Make sure the build is green.

Step 2. Create a new release on GitHub. Tag it as the same version as the setup.py.
If the version is "0.1.0", tag the commit as "v0.1.0".

Step 3. Upload the package to PyPi:
```bash
rm -rf dist/koalas*
python setup.py bdist_wheel
export package_version=$(python setup.py --version)
echo $package_version

python3 -m pip install --user --upgrade twine
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/koalas-$package_version-py3-none-any.whl
python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/koalas-$package_version-py3-none-any.whl
```


## FAQ

### What's the project's status?
This project is currently in beta and is rapidly evolving.
You should expect the following differences:

 - some functions may be missing (see the [Contributions](#Contributions) section)

 - some behavior may be different, in particular in the treatment of nulls: Pandas uses
   Not a Number (NaN) special constants to indicate missing values, while Spark has a
   special flag on each value to indicate missing values. We would love to hear from you
   if you come across any discrepancies
   
 - because Spark is lazy in nature, some operations like creating new columns only get 
   performed when Spark needs to print or write the dataframe.
   
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

Please create a GitHub issue if your favorite function is not yet supported.

Make sure the name also reflects precisely which function you want to implement, such as
`DataFrame.fillna` or `Series.dropna`. If an open issue already exists and you want do add
missing parameters, consider contributing to that issue instead.

We also document all the functions that are not yet supported in the
[missing directory](https://github.com/databricks/koalas/tree/master/databricks/koalas/missing).
In most cases, it is very easy to add new functions by simply wrapping the existing pandas or
Spark functions. Pull requests welcome!

### Why a new project (instead of putting this in Apache Spark itself)?

Two reasons:

1. We want a venue in which we can rapidly iterate and make new releases. The overhead of making a
release as a separate project is minuscule (in the order of minutes). A release on Spark takes a
lot longer (in the order of days)

2. Koalas takes a different approach that might contradict Spark's API design principles, and those
principles cannot be changed lightly given the large user base of Spark. A new, separate project
provides an opportunity for us to experiment with new design principles.

### How do I use this on Databricks?

Databricks Runtime for Machine Learning has the right versions of dependencies setup already, so
you just need to install Koalas from pip when creating a cluster.

For the regular Databricks Runtime, you will need to upgrade pandas and NumPy versions to the
required list. 

In the future, we will package Koalas out-of-the-box in both the regular Databricks Runtime and
Databricks Runtime for Machine Learning.
