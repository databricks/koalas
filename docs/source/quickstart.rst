Quick Start
===========

Koalas requires Spark. You can install it using ``pip install pyspark`` or downloading the release.

After that, you install Koalas using ``pip``:

.. code-block:: bash

    $ pip install koalas

After installing Koalas, you import the Koalas package:

.. code-block:: python

    import databricks.koalas as ks

Then you can easily turn a pandas DataFrame into a Koalas DataFrame that is `API-compliant` with the former:

.. code-block:: python

    import pandas as pd
    pdf = pd.DataFrame({'x':range(3), 'y':['a','b','b'], 'z':['a','b','b']})

    # Create a Koalas DataFrame from pandas DataFrame
    df = ks.from_pandas(pdf)

    # Rename the columns
    df.columns = ['x', 'y', 'z1']

    # Do some operations in place:
    df['x2'] = df.x * df.x

You can also create a Koalas DataFrame directly from a Python data structure:

.. code-block:: python

    import databricks.koalas as ks

    ks.DataFrame({'x': [1, 2], 'y': [3, 4], 'z': [5, 6]})
