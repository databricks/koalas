<p align="center">
  <img src="https://raw.githubusercontent.com/databricks/koalas/master/icons/koalas-logo.png" width="140"/>
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
  <strong><a href="https://www.gofundme.com/f/help-thirsty-koalas-devastated-by-recent-fires">Help Thirsty Koalas Devastated by Recent Fires</a></strong>
</p>

The Koalas project makes data scientists more productive when interacting with big data, by implementing the pandas DataFrame API on top of Apache Spark.

pandas is the de facto standard (single-node) DataFrame implementation in Python, while Spark is the de facto standard for big data processing. With this package, you can:
 - Be immediately productive with Spark, with no learning curve, if you are already familiar with pandas.
 - Have a single codebase that works both with pandas (tests, smaller datasets) and with Spark (distributed datasets).

We would love to have you try it and give us feedback, through our [mailing lists](https://groups.google.com/forum/#!forum/koalas-dev) or [GitHub issues](https://github.com/databricks/koalas/issues).

Try the Koalas 10 minutes tutorial on a live Jupyter notebook [here](https://mybinder.org/v2/gh/databricks/koalas/master?filepath=docs%2Fsource%2Fgetting_started%2F10min.ipynb). The initial launch can take up to several minutes.

[![Github Actions](https://github.com/databricks/koalas/workflows/master/badge.svg)](https://github.com/databricks/koalas/actions)
[![codecov](https://codecov.io/gh/databricks/koalas/branch/master/graph/badge.svg)](https://codecov.io/gh/databricks/koalas)
[![Documentation Status](https://readthedocs.org/projects/koalas/badge/?version=latest)](https://koalas.readthedocs.io/en/latest/?badge=latest)
[![Latest Release](https://img.shields.io/pypi/v/koalas.svg)](https://pypi.org/project/koalas/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/koalas.svg)](https://anaconda.org/conda-forge/koalas)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/databricks/koalas/master?filepath=docs%2Fsource%2Fgetting_started%2F10min.ipynb)
[![Downloads](https://pepy.tech/badge/koalas)](https://pepy.tech/project/koalas)


## Getting Started

Koalas can be installed in many ways such as Conda and pip.

```bash
# Conda
conda install koalas -c conda-forge
```

```bash
# pip
pip install koalas
```

See [Installation](https://koalas.readthedocs.io/en/latest/getting_started/install.html) for more details.

For Databricks Runtime, Koalas is pre-installed in Databricks Runtime 7.1 and above. Try [Databricks Community Edition](https://community.cloud.databricks.com/) for free. You can also follow these [steps](https://docs.databricks.com/libraries/index.html) to manually install a library on Databricks.

Lastly, if your PyArrow version is 0.15+ and your PySpark version is lower than 3.0, it is best for you to set `ARROW_PRE_0_15_IPC_FORMAT` environment variable to `1` manually.
Koalas will try its best to set it for you but it is impossible to set it if there is a Spark context already launched.

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


## Koalas Talks and Blogs

See [Koalas Talks and Blogs](https://koalas.readthedocs.io/en/latest/getting_started/videos_blogs.html) in the official documentation.
