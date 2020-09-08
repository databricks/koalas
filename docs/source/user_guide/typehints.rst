====================
Type Hints In Koalas
====================

.. currentmodule:: databricks.koalas

Koalas, by default, infers the schema by taking some top records from the output,
in particular, when you use APIs that allow users to apply a function against Koalas DataFrame
such as :func:`DataFrame.transform`, :func:`DataFrame.apply`, :func:`DataFrame.koalas.apply_batch`,
:func:`DataFrame.koalas.apply_batch`, :func:`Series.koalas.apply_batch`, etc.

However, this is potentially expensive. If there are several expensive operations such as a shuffle
in the upstream of the execution plan, Koalas will end up with executing the Spark job twice, once
for schema inference, and once for processing actual data with the schema.

To avoid the consequences, Koalas has its own type hinting style to specify the schema to avoid
schema inference. Koalas understands the type hints specified in the return type and converts it
as a Spark schema for pandas UDFs used internally. The way of type hinting has been evolved over
the time.

In this chapter, it covers the recommended way and the supported ways in details.


Koalas DataFrame and Pandas DataFrame
-------------------------------------

In the early Koalas version, it was introduced to specify a type hint in the function in order to use
it as a Spark schema. As an example, you can specify the return type hint as below by using Koalas
:class:`DataFrame`.

.. code-block:: python

    >>> def pandas_div(pdf) -> ks.DataFrame[float, float]:
    ...    # pdf is a pandas DataFrame.
    ...    return pdf[['B', 'C']] / pdf[['B', 'C']]
    ...
    >>> df = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': [1, 2, 3], 'C': [4, 6, 5]})
    >>> df.groupby('A').apply(pandas_div)

The function ``pandas_div`` actually takes and outputs a pandas DataFrame instead of Koalas :class:`DataFrame`.
However, Koalas has to force to set the mismatched type hints.

From Koalas 1.0 with Python 3.7+, now you can specify the type hints by using pandas instances.

.. code-block:: python

    >>> def pandas_div(pdf) -> pd.DataFrame[float, float]:
    ...    # pdf is a pandas DataFrame.
    ...    return pdf[['B', 'C']] / pdf[['B', 'C']]
    ...
    >>> df = ks.DataFrame({'A': ['a', 'a', 'b'], 'B': [1, 2, 3], 'C': [4, 6, 5]})
    >>> df.groupby('A').apply(pandas_div)

Likewise, pandas Series can be also used as a type hints:

.. code-block:: python

    >>> def sqrt(x) -> pd.Series[float]:
    ...     return np.sqrt(x)
    ...
    >>> df = ks.DataFrame([[4, 9]] * 3, columns=['A', 'B'])
    >>> df.apply(sqrt, axis=0)

Currently, both Koalas and pandas instances can be used to specify the type hints; however, Koalas
plans to move gradually towards using pandas instances only as the stability becomes proven.


Type Hinting with Names
-----------------------

In Koalas 1.0, the new style of type hinting was introduced to overcome the limitations in the existing type
hinting especially for DataFrame. When you use a DataFrame as the return type hint, for example,
``DataFrame[int, int]``, there is no way to specify the names of each Series. In the old way, Koalas just generates
the column names as ``c#`` and this easily leads users to lose or forgot the Series mappings. See the example below:

.. code-block:: python

    >>> def transform(pdf) -> pd.DataFrame[int, int]:
    ...     pdf['A'] = pdf.id + 1
    ...     return pdf
    ...
    >>> ks.range(5).koalas.apply_batch(transform)

.. code-block:: bash

       c0  c1
    0   0   1
    1   1   2
    2   2   3
    3   3   4
    4   4   5

The new style of type hinting in Koalas is similar with the regular Python type hints in variables. The Series name
is specified as a string, and the type is specified after a colon. The following example shows a simple case with
the Series names, ``id`` and ``A``, and ``int`` types respectively.

.. code-block:: python

    >>> def transform(pdf) -> pd.DataFrame["id": int, "A": int]:
    ...     pdf['A'] = pdf.id + 1
    ...     return pdf
    ...
    >>> ks.range(5).koalas.apply_batch(transform)

.. code-block:: bash

       id   A
    0   0   1
    1   1   2
    2   2   3
    3   3   4
    4   4   5

In addition, Koalas also dynamically supports ``dtype`` instance and the column index in pandas so that users can
programmatically generate the return type and schema.

.. code-block:: python

    >>> def transform(pdf) -> pd.DataFrame[zip(pdf.columns, pdf.dtypes)]:
    ...    return pdf + 1
    ...
    >>> kdf.koalas.apply_batch(transform)

Likewise, ``dtype`` instances from pandas DataFrame can be used alone and let Koalas generate column names.

.. code-block:: python

    >>> def transform(pdf) -> pd.DataFrame[pdf.dtypes]:
    ...     return pdf + 1
    ...
    >>> kdf.koalas.apply_batch(transform)

.. warning::
    This new style of type hinting is experimental. It could be changed or removed between minor
    releases without warnings.

