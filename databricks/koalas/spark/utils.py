#
# Copyright (C) 2019 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Helpers and utilities to deal with PySpark instances
"""
from pyspark.sql.types import DecimalType, StructType, MapType, ArrayType, StructField, DataType


def as_nullable_spark_type(dt: DataType) -> DataType:
    """
    Returns a nullable schema or data types.

    Examples
    --------
    >>> from pyspark.sql.types import *
    >>> as_nullable_spark_type(StructType([
    ...     StructField("A", IntegerType(), True),
    ...     StructField("B", FloatType(), False)]))  # doctest: +NORMALIZE_WHITESPACE
    StructType(List(StructField(A,IntegerType,true),StructField(B,FloatType,true)))

    >>> as_nullable_spark_type(StructType([
    ...     StructField("A",
    ...         StructType([
    ...             StructField('a',
    ...                 MapType(IntegerType(),
    ...                 ArrayType(IntegerType(), False), False), False),
    ...             StructField('b', StringType(), True)])),
    ...     StructField("B", FloatType(), False)]))  # doctest: +NORMALIZE_WHITESPACE
    StructType(List(StructField(A,StructType(List(StructField(a,MapType(IntegerType,ArrayType\
(IntegerType,true),true),true),StructField(b,StringType,true))),true),\
StructField(B,FloatType,true)))
    """
    if isinstance(dt, StructType):
        new_fields = []
        for field in dt.fields:
            new_fields.append(
                StructField(
                    field.name,
                    as_nullable_spark_type(field.dataType),
                    nullable=True,
                    metadata=field.metadata,
                )
            )
        return StructType(new_fields)
    elif isinstance(dt, ArrayType):
        return ArrayType(as_nullable_spark_type(dt.elementType), containsNull=True)
    elif isinstance(dt, MapType):
        return MapType(
            as_nullable_spark_type(dt.keyType),
            as_nullable_spark_type(dt.valueType),
            valueContainsNull=True,
        )
    else:
        return dt


def force_decimal_precision_scale(dt: DataType, precision: int = 38, scale: int = 18) -> DataType:
    """
    Returns a data type with a fixed decimal type.

    The precision and scale of the decimal type are fixed with the given values.

    Examples
    --------
    >>> from pyspark.sql.types import *
    >>> force_decimal_precision_scale(StructType([
    ...     StructField("A", DecimalType(10, 0), True),
    ...     StructField("B", DecimalType(14, 7), False)]))  # doctest: +NORMALIZE_WHITESPACE
    StructType(List(StructField(A,DecimalType(38,18),true),StructField(B,DecimalType(38,18),false)))

    >>> force_decimal_precision_scale(StructType([
    ...     StructField("A",
    ...         StructType([
    ...             StructField('a',
    ...                 MapType(DecimalType(5, 0),
    ...                 ArrayType(DecimalType(20, 0), False), False), False),
    ...             StructField('b', StringType(), True)])),
    ...     StructField("B", DecimalType(30, 15), False)]),
    ...     precision=30, scale=15)  # doctest: +NORMALIZE_WHITESPACE
    StructType(List(StructField(A,StructType(List(StructField(a,MapType(DecimalType(30,15),\
ArrayType(DecimalType(30,15),false),false),false),StructField(b,StringType,true))),true),\
StructField(B,DecimalType(30,15),false)))
    """
    if isinstance(dt, StructType):
        new_fields = []
        for field in dt.fields:
            new_fields.append(
                StructField(
                    field.name,
                    force_decimal_precision_scale(field.dataType, precision, scale),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
        return StructType(new_fields)
    elif isinstance(dt, ArrayType):
        return ArrayType(
            force_decimal_precision_scale(dt.elementType, precision, scale),
            containsNull=dt.containsNull,
        )
    elif isinstance(dt, MapType):
        return MapType(
            force_decimal_precision_scale(dt.keyType, precision, scale),
            force_decimal_precision_scale(dt.valueType, precision, scale),
            valueContainsNull=dt.valueContainsNull,
        )
    elif isinstance(dt, DecimalType):
        return DecimalType(precision=precision, scale=scale)
    else:
        return dt
