======================
Type Support In Koalas
======================

.. currentmodule:: databricks.koalas

In this chapter, we will briefly show you how data types change when converting Koalas DataFrame from/to PySpark DataFrame or pandas DataFrame.


Type casting between PySpark and Koalas
---------------------------------------

When converting the Koalas DataFrame from/to PySpark DataFrame, the data types automatically casted to the appropriate type.

The example below shows how types are casted between PySpark DataFrame and Koalas DataFrame.

.. code-block:: python

    # 1. Create PySpark DataFrame
    >>> sdf = spark.createDataFrame([
    ...     (1, Decimal(1.0), 1., 1., 1, 1, 1, datetime(2020, 10, 27), "1", True),
    ... ], 'tinyint tinyint, decimal decimal, float float, double double, integer integer, long long, short short, timestamp timestamp, string string, boolean boolean')

    # 2. Check the PySpark data types
    >>> sdf
    DataFrame[tinyint: tinyint, decimal: decimal(10,0), float: float, double: double, integer: int, long: bigint, short: smallint, timestamp: timestamp, string: string]

    # 3. Convert PySpark DataFrame to Koalas DataFrame
    >>> kdf = sdf.to_koalas()

    # 4. Check the Koalas data types
    >>> kdf.dtypes
    tinyint                int8
    decimal              object
    float               float32
    double              float64
    integer               int32
    long                  int64
    short                 int16
    timestamp    datetime64[ns]
    string               object
    boolean                bool
    dtype: object

    # 5. Easily go back to the PySpark DataFrame
    >>> sdf = kdf.to_spark()

    # 6. Check the PySpark data types again
    >>> sdf
    DataFrame[tinyint: tinyint, decimal: decimal(10,0), float: float, double: double, integer: int, long: bigint, short: smallint, timestamp: timestamp, string: string]


Type casting between pandas and Koalas
--------------------------------------

We can easily convert Koalas DataFrame to pandas DataFrame, and the data types are basically same as pandas.

.. code-block:: python

    # Convert Koalas DataFrame to pandas DataFrame
    >>> pdf = kdf.to_pandas()

    # Check the pandas data types
    >>> pdf.dtypes
    tinyint                int8
    decimal              object
    float               float32
    double              float64
    integer               int32
    long                  int64
    short                 int16
    timestamp    datetime64[ns]
    string               object
    boolean                bool
    dtype: object


However, there are several types only provided by pandas.

.. code-block:: python

    # pd.Catrgorical type is not supported in Koalas yet.
    >>> ks.Series([pd.Categorical([1, 2, 3])])
    Traceback (most recent call last):
    ...
    pyarrow.lib.ArrowInvalid: Could not convert [1, 2, 3]
    Categories (3, int64): [1, 2, 3] with type Categorical: did not recognize Python value type when inferring an Arrow data type


These kind of pandas specific types below are not currently supported in Koalas, but planned to be supported in the future.

* pd.Timedelta
* pd.Categorical
* pd.CategoricalDtype


However, note that the pandas specific types below are not planned to be supported in Koalas yet.

* pd.SparseDtype
* pd.DatetimeTZDtype
* pd.UInt*Dtype
* pd.BooleanDtype
* pd.StringDtype
