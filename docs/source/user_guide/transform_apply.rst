==============================
Transform and apply a function
==============================

.. NOTE: the images are stored at https://github.com/databricks/koalas/issues/1443. Feel free to edit and/or add.

.. currentmodule:: databricks.koalas

There are many APIs that allow users to apply a function against Koalas DataFrame such as
:func:`DataFrame.transform`, :func:`DataFrame.apply`, :func:`DataFrame.transform_batch`,
:func:`DataFrame.apply_batch`, :func:`Series.transform_batch`, etc. Each has a distinct
purpose and works differently internally. This section describes several differences among
them where users are confused often.

`transform` and `apply`
-----------------------

The main difference between :func:`DataFrame.transform` and :func:`DataFrame.apply` is that the former requires
to return the same length of the input and the latter does not require this. See the example below:

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'a': [1,2,3], 'b':[4,5,6]})
   >>> def pandas_plus(pser):
   ...     return pser + 1  # should always return the same length as input.
   >>> kdf.transform(pandas_plus)

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'a': [1,2,3], 'b':[5,6,7]})
   >>> def pandas_plus(pser):
   ...     return pser[pser % 2 == 1]  # should always return the same length as input.
   >>> kdf.apply(pandas_plus)

In this case, each function takes a pandas Series, and Koalas computes the functions in a distributed manner as below.

.. image:: https://user-images.githubusercontent.com/6477701/80076790-a1cf0680-8587-11ea-8b08-8dc694071ba0.png
  :alt: transform and apply
  :align: center
  :width: 550

In case of 'column' axis, the function takes each row as a pandas Seires.

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'a': [1,2,3], 'b':[4,5,6]})
   >>> def pandas_plus(pser):
   ...     return sum(pser)  # allow arbitrary length
   >>> kdf.apply(pandas_plus, axis='columns')

The example above calculates the summation of each row as a pandas Series. See below:

.. image:: https://user-images.githubusercontent.com/6477701/80076898-c2975c00-8587-11ea-9b2c-69c9729e9294.png
  :alt: apply axis
  :align: center
  :width: 600


`transform_batch` and `apply_batch`
-----------------------------------

In :func:`DataFrame.transform_batch`, :func:`DataFrame.apply_batch`, :func:`Series.transform_batch`, etc., the `_batch`
postfix means each chunk in Koalas DataFrame or Series. The APIs slice the Koalas DataFrame or Series, and
then applies the given function with pandas DataFrame or Series as input and output. See the examples below:

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'a': [1,2,3], 'b':[4,5,6]})
   >>> def pandas_plus(pdf):
   ...     return pdf + 1  # should always return the same length as input.
   >>> kdf.transform_batch(pandas_plus)

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'a': [1,2,3], 'b':[4,5,6]})
   >>> def pandas_plus(pdf):
   ...     return pdf[pdf.a > 1]  # allow arbitrary length
   >>> kdf.apply_batch(pandas_plus)

Note that :func:`DataFrame.transform_batch` has the length
resctriction whereas :func:`DataFrame.apply_batch` is not, and :func:`DataFrame.transform_batch` can return a Series which
can be usful to avoid a shuffle by the operations between different DataFrames, see also
`Operations on different DataFrames <options.rst#operations-on-different-dataframes>`_ for more details.

The functions in both examples take a pandas DataFrame as a chunk of Koalas DataFrame, and output a pandas DataFrame.
Koalas combines the pandas DataFrames as a Koalas DataFrame.

.. image:: https://user-images.githubusercontent.com/6477701/80076779-9f6cac80-8587-11ea-8c92-07d7b992733b.png
  :alt: transform_batch and apply_batch in Frame
  :align: center
  :width: 650

In case of :func:`Series.transform_batch`, it is also similar with :func:`DataFrame.transform_batch`; however, it takes
a pandas Series as a chunk of Koalas Series.

.. code-block:: python

   >>> import databricks.koalas as ks
   >>> kdf = ks.DataFrame({'a': [1,2,3], 'b':[4,5,6]})
   >>> def pandas_plus(pser):
   ...     return pser + 1  # should always return the same length as input.
   >>> kdf.a.transform_batch(pandas_plus)

It can be illustrated as below.

.. image:: https://user-images.githubusercontent.com/6477701/80076795-a3003380-8587-11ea-8b73-186e4047f8c0.png
  :alt: transform_batch in Series
  :width: 350
  :align: center
