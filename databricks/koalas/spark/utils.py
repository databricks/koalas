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
from pyspark.sql.types import StructType, MapType, ArrayType, StructField, DataType


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
