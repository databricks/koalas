<p align="center">
  <img src="https://raw.githubusercontent.com/databricks/koalas/master/Koalas-logo.png" width="140"/>
</p>

<p align="center">
  pandas API on Apache Spark
  <br/>
  <a href="https://koalas.readthedocs.io/en/latest/?badge=latest"><strong>Explore Koalas docs »</strong></a>
  <br/>
  <br/>
  <a href="https://mybinder.org/v2/gh/databricks/koalas/master?filepath=docs%2Fsource%2Fgetting_started%2F10min.ipynb">Live notebook</a>
  ·
  <a href="https://github.com/databricks/koalas/issues">Issues</a>
  ·
  <a href="https://groups.google.com/forum/#!forum/koalas-dev">Mailing list</a>
  <br/>
  <strong><a href="https://www.gofundme.com/f/help-thirsty-koalas-devastated-by-recent-fires">Help Thirsty Koalas Devasted by Recent Fires</a></strong>
</p>

The Koalas project makes data scientists more productive when interacting with big data, by implementing the pandas DataFrame API on top of Apache Spark.

pandas is the de facto standard (single-node) DataFrame implementation in Python, while Spark is the de facto standard for big data processing. With this package, you can:
 - Be immediately productive with Spark, with no learning curve, if you are already familiar with pandas.
 - Have a single codebase that works both with pandas (tests, smaller datasets) and with Spark (distributed datasets).

This project is currently in beta and is rapidly evolving, with a bi-weekly release cadence. We would love to have you try it and give us feedback, through our [mailing lists](https://groups.google.com/forum/#!forum/koalas-dev) or [GitHub issues](https://github.com/databricks/koalas/issues).

Try the Koalas 10 minutes tutorial on a live Jupyter notebook [here](https://mybinder.org/v2/gh/databricks/koalas/master?filepath=docs%2Fsource%2Fgetting_started%2F10min.ipynb). The initial launch can take up to several minutes.

[![Build Status](https://travis-ci.com/databricks/koalas.svg?token=Rzzgd1itxsPZRuhKGnhD&branch=master)](https://travis-ci.com/databricks/koalas)
[![codecov](https://codecov.io/gh/databricks/koalas/branch/master/graph/badge.svg)](https://codecov.io/gh/databricks/koalas)
[![Documentation Status](https://readthedocs.org/projects/koalas/badge/?version=latest)](https://koalas.readthedocs.io/en/latest/?badge=latest)
[![Latest Release](https://img.shields.io/pypi/v/koalas.svg)](https://pypi.org/project/koalas/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/koalas.svg)](https://anaconda.org/conda-forge/koalas)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/databricks/koalas/master?filepath=docs%2Fsource%2Fgetting_started%2F10min.ipynb)


## Getting Started

The recommended way of installing Koalas is Conda as below.

```bash
conda install koalas -c conda-forge
```

You can use not only Conda but also multiple ways to install Koalas. See [Installation](https://koalas.readthedocs.io/en/latest/getting_started/install.html) for full instructions to install Koalas.

If you are a Databricks Runtime user, you can install Koalas using the Libraries tab on the cluster UI, or using `dbutils` in a notebook as below, for the regular Databricks Runtime.

```python
dbutils.library.installPyPI("koalas")
dbutils.library.restartPython()
```

Note that Koalas requires Databricks Runtime 5.x or above. In the future, we will package Koalas out-of-the-box in both the regular Databricks Runtime and Databricks Runtime for Machine Learning.

Now you can turn a pandas DataFrame into a Koalas DataFrame that is API-compliant with the former:

```python
import databricks.koalas as ks
import pandas as pd

pdf = pd.DataFrame({'x':range(3), 'y':['a','b','b'], 'z':['a','b','b']})

# Create a Koalas DataFrame from pandas DataFrame
df = ks.from_pandas(pdf)

# Rename the columns
df.columns = ['x', 'y', 'z1']

# Do some operations in place:
df['x2'] = df.x * df.x
```

For more details, see [Getting Started](https://koalas.readthedocs.io/en/latest/getting_started/index.html) and [Dependencies](https://koalas.readthedocs.io/en/latest/getting_started/install.html#dependencies) in the official documentation.


## Contributing Guide

See [Contributing Guide](https://koalas.readthedocs.io/en/latest/development/contributing.html) and [Design Principles](https://koalas.readthedocs.io/en/latest/development/design.html) in the official documentation.


## FAQ

See [FAQ](https://koalas.readthedocs.io/en/latest/user_guide/faq.html) in the official documentation.


## Best Practices

See [Best Practices](https://koalas.readthedocs.io/en/latest/user_guide/best_practices.html) in the official documentation.

