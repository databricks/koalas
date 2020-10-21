============
Installation
============

Koalas requires PySpark so please make sure your PySpark is available.

To install Koalas, you can use:

- `Conda <https://anaconda.org/conda-forge/koalas>`__
- `PyPI <https://pypi.org/project/koalas>`__
- `Installation from source <../development/contributing.rst#environment-setup>`__

To install PySpark, you can use:

- `Installation with the official release channel <https://spark.apache.org/downloads.html>`__
- `Conda <https://anaconda.org/conda-forge/pyspark>`__
- `PyPI <https://pypi.org/project/pyspark>`__
- `Installation from source <https://github.com/apache/spark#building-spark>`__


Python version support
----------------------

Officially Python 3.5 to 3.8.


Installing Koalas
-----------------

Installing with Conda
~~~~~~~~~~~~~~~~~~~~~~

First you will need `Conda <http://conda.pydata.org/docs/>`__ to be installed.
After that, we should create a new conda environment. A conda environment is similar with a
virtualenv that allows you to specify a specific version of Python and set of libraries.
Run the following commands from a terminal window::

    conda create --name koalas-dev-env

This will create a minimal environment with only Python installed in it.
To put your self inside this environment run::

    conda activate koalas-dev-env

The final step required is to install Koalas. This can be done with the
following command::

    conda install -c conda-forge koalas

To install a specific Koalas version::

    conda install -c conda-forge koalas=1.3.0


Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

Koalas can be installed via pip from
`PyPI <https://pypi.org/project/koalas>`__::

    pip install koalas


Installing from source
~~~~~~~~~~~~~~~~~~~~~~

See the `Contribution Guide <../development/contributing.rst#environment-setup>`__ for complete instructions.


Installing PySpark
------------------

Installing with the official release channel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install PySpark by downloading a release in `the official release channel <https://spark.apache.org/downloads.html>`__.
Once you download the release, un-tar it first as below::

    tar xzvf spark-2.4.4-bin-hadoop2.7.tgz

After that, make sure set ``SPARK_HOME`` environment variable to indicate the directory you untar-ed::

    cd spark-2.4.4-bin-hadoop2.7
    export SPARK_HOME=`pwd`

Also, make sure your ``PYTHONPATH`` can find the PySpark and Py4J under ``$SPARK_HOME/python/lib``::

    export PYTHONPATH=$(ZIPS=("$SPARK_HOME"/python/lib/*.zip); IFS=:; echo "${ZIPS[*]}"):$PYTHONPATH


Installing with Conda
~~~~~~~~~~~~~~~~~~~~~~

PySpark can be installed via `Conda <https://anaconda.org/conda-forge/pyspark>`__::

    conda install -c conda-forge pyspark


Installing with PyPI
~~~~~~~~~~~~~~~~~~~~~~

PySpark can be installed via pip from `PyPI <https://pypi.org/project/pyspark>`__::

    pip install pyspark


Installing from source
~~~~~~~~~~~~~~~~~~~~~~

To install PySpark from source, refer `Building Spark <https://github.com/apache/spark#building-spark>`__.

Likewise, make sure you set ``SPARK_HOME`` environment variable to the git-cloned directory, and your
``PYTHONPATH`` environment variable can find the PySpark and Py4J under ``$SPARK_HOME/python/lib``::

    export PYTHONPATH=$(ZIPS=("$SPARK_HOME"/python/lib/*.zip); IFS=:; echo "${ZIPS[*]}"):$PYTHONPATH


Dependencies
------------

============= ================
Package       Required version
============= ================
`pandas`      >=0.23.2
`pyspark`     >=2.4.0
`pyarrow`     >=0.10
`matplotlib`  >=3.0.0
`numpy`       >=1.14
============= ================


Optional dependencies
~~~~~~~~~~~~~~~~~~~~~

============= ================
Package       Required version
============= ================
`mlflow`      >=1.0
============= ================
