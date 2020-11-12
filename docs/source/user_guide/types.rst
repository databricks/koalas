======================
Type Support In Koalas
======================

.. currentmodule:: databricks.koalas


This is based on Koalas 1.1.4.

Koalas supports various of types by mapping them to specific types in PySpark internally.

This chapter gives you an information of what types are supported and which types are not.


Types supported in Koalas
-------------------------

A table below shows which NumPy types are matched to which PySpark types internally in Koalas.

============= =======================
NumPy         PySpark
============= =======================
np.character  BinaryType
np.bytes\_    BinaryType
np.string\_   BinaryType
np.int8       ByteType
np.byte       ByteType
np.int16      ShortType
np.int32      IntegerType
np.int64      LongType
np.int        LongType
np.float32    FloatType
np.float      DoubleType
np.float64    DoubleType
np.str        StringType
np.unicode\_  StringType
np.bool       BooleanType
np.datetime64 TimestampType
np.ndarray    ArrayType(StringType())
============= =======================


For `np.ndarray`, it's casted to `ArrayType(StringType())`.

If you want to use `ArrayType` contains another types, use Python typing system as below.

======================= ==============================
Python typing           PySpark ArrayType
======================= ==============================
List[bytes]             ArrayType(BinaryType())
List[np.character]      ArrayType(BinaryType())
List[np.bytes\_]        ArrayType(BinaryType())
List[np.string\_]       ArrayType(BinaryType())
List[bool]              ArrayType(BooleanType())
List[np.bool]           ArrayType(BooleanType())
List[datetime.date]     ArrayType(DateType())
List[np.int8]           ArrayType(ByteType())
List[np.byte]           ArrayType(ByteType())
List[decimal.Decimal]   ArrayType(DecimalType(38, 18))
List[float]             ArrayType(DoubleType())
List[np.float]          ArrayType(DoubleType())
List[np.float64]        ArrayType(DoubleType())
List[np.float32]        ArrayType(FloatType())
List[np.int32]          ArrayType(IntegerType())
List[int]               ArrayType(LongType())
List[np.int]            ArrayType(LongType())
List[np.int64]          ArrayType(LongType())
List[np.int16]          ArrayType(ShortType())
List[str]               ArrayType(StringType())
List[np.unicode\_]      ArrayType(StringType())
List[datetime.datetime] ArrayType(TimestampType())
List[np.datetime64]     ArrayType(TimestampType())
======================= ==============================


A table below shows which Python types are matched to which PySpark types internally in Koalas.

================= ===================
Python            PySpark
================= ===================
bytes             BinaryType
int               LongType
float             DoubleType
str               StringType
bool              BooleanType
datetime.datetime TimestampType
datetime.date     DateType
decimal.Decimal   DecimalType(38, 18)
================= ===================

For decimal type, Koalas uses Spark's system default precision and scale.


You can easily check this mapping by using `as_spark_type` function.

.. code-block:: python

    >>> import typing
    >>> import numpy as np
    >>> from databricks.koalas.typedef import as_spark_type

    >>> as_spark_type(int)
    LongType

    >>> as_spark_type(np.int32)
    IntegerType

    >>> as_spark_type(typing.List[float])
    ArrayType(DoubleType,true)

You can also easily check the underlying PySpark type of `Series` by using Spark accessor.

.. code-block:: python

    >>> ks.Series([0.3, 0.1, 0.8]).spark.data_type
    DoubleType

    >>> ks.Series(["welcome", "to", "Koalas"]).spark.data_type
    StringType

    >>> ks.Series([[False, True, False]]).spark.data_type
    ArrayType(BooleanType,true)


pandas types not supported in Koalas
------------------------------------

For reference, this is based on pandas 1.1.4.

There are several types that are only provided by pandas.

The pandas specific types below are not currently supported in Koalas, but planned to be supported in the future.

* pd.Timedelta
* pd.Categorical
* pd.CategoricalDtype


The pandas specific types below are not planned to be supported in Koalas yet.

* pd.SparseDtype
* pd.DatetimeTZDtype
* pd.UInt*Dtype
* pd.BooleanDtype
* pd.StringDtype
