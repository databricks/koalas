Quick Start
===========

Koalas needs Spark. You can install it via ``pip install pyspark`` or downloading the release.

After that, Koalas can be installed via ``pip`` as below:

.. code-block:: bash

    $ pip install koalas


After installing the package, you can import the package:

.. code-block:: python

    from databricks import koalas as ks


Now you can turn a pandas DataFrame into a Koalas DataFrame that is API-compliant with the former:

.. code-block:: python

    import pandas as pd
    pdf = pd.DataFrame({'x':range(3), 'y':['a','b','b'], 'z':['a','b','b']})

    # Create a Koalas DataFrame from pandas DataFrame
    df = ks.from_pandas(pdf)

    # Rename the columns
    df.columns = ['x', 'y', 'z1']

    # Do some operations in place:
    df['x2'] = df.x * df.x

You can also create Koalas DataFrame directly from regular Python data like Pandas:

.. code-block:: python

    from databricks import koalas as ks

    ks.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6]})

