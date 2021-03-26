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
import sys
import unittest
import datetime
import decimal
from typing import List

import pandas
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from pyspark.sql.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    ByteType,
    ShortType,
    DateType,
    DecimalType,
    DoubleType,
    TimestampType,
)

from databricks.koalas.typedef import (
    as_spark_type,
    extension_dtypes_available,
    extension_float_dtypes_available,
    extension_object_dtypes_available,
    infer_return_type,
    koalas_dtype,
)
from databricks import koalas as ks


class TypeHintTests(unittest.TestCase):
    @unittest.skipIf(
        sys.version_info < (3, 7),
        "Type inference from pandas instances is supported with Python 3.7+",
    )
    def test_infer_schema_from_pandas_instances(self):
        def func() -> pd.Series[int]:
            pass

        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtype, np.int64)
        self.assertEqual(inferred.spark_type, LongType())

        def func() -> pd.Series[np.float]:
            pass

        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtype, np.float64)
        self.assertEqual(inferred.spark_type, DoubleType())

        def func() -> "pd.DataFrame[np.float, str]":
            pass

        expected = StructType([StructField("c0", DoubleType()), StructField("c1", StringType())])
        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtypes, [np.float64, np.unicode_])
        self.assertEqual(inferred.spark_type, expected)

        def func() -> "pandas.DataFrame[np.float]":
            pass

        expected = StructType([StructField("c0", DoubleType())])
        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtypes, [np.float64])
        self.assertEqual(inferred.spark_type, expected)

        def func() -> "pd.Series[int]":
            pass

        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtype, np.int64)
        self.assertEqual(inferred.spark_type, LongType())

        def func() -> pd.DataFrame[np.float, str]:
            pass

        expected = StructType([StructField("c0", DoubleType()), StructField("c1", StringType())])
        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtypes, [np.float64, np.unicode_])
        self.assertEqual(inferred.spark_type, expected)

        def func() -> pd.DataFrame[np.float]:
            pass

        expected = StructType([StructField("c0", DoubleType())])
        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtypes, [np.float64])
        self.assertEqual(inferred.spark_type, expected)

        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})

        def func() -> pd.DataFrame[pdf.dtypes]:  # type: ignore
            pass

        expected = StructType([StructField("c0", LongType()), StructField("c1", LongType())])
        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtypes, [np.int64, np.int64])
        self.assertEqual(inferred.spark_type, expected)

        pdf = pd.DataFrame({"a": [1, 2, 3], "b": pd.Categorical(["a", "b", "c"])})

        def func() -> pd.Series[pdf.b.dtype]:  # type: ignore
            pass

        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtype, CategoricalDtype(categories=["a", "b", "c"]))
        self.assertEqual(inferred.spark_type, LongType())

        def func() -> pd.DataFrame[pdf.dtypes]:  # type: ignore
            pass

        expected = StructType([StructField("c0", LongType()), StructField("c1", LongType())])
        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtypes, [np.int64, CategoricalDtype(categories=["a", "b", "c"])])
        self.assertEqual(inferred.spark_type, expected)

    def test_if_pandas_implements_class_getitem(self):
        # the current type hint implementation of pandas DataFrame assumes pandas doesn't
        # implement '__class_getitem__'. This test case is to make sure pandas
        # doesn't implement them.
        assert not ks._frame_has_class_getitem
        assert not ks._series_has_class_getitem

    @unittest.skipIf(
        sys.version_info < (3, 7),
        "Type inference from pandas instances is supported with Python 3.7+",
    )
    def test_infer_schema_with_names_pandas_instances(self):
        def func() -> 'pd.DataFrame["a" : np.float, "b":str]':  # noqa: F821
            pass

        expected = StructType([StructField("a", DoubleType()), StructField("b", StringType())])
        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtypes, [np.float64, np.unicode_])
        self.assertEqual(inferred.spark_type, expected)

        def func() -> "pd.DataFrame['a': np.float, 'b': int]":  # noqa: F821
            pass

        expected = StructType([StructField("a", DoubleType()), StructField("b", LongType())])
        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtypes, [np.float64, np.int64])
        self.assertEqual(inferred.spark_type, expected)

        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})

        def func() -> pd.DataFrame[zip(pdf.columns, pdf.dtypes)]:
            pass

        expected = StructType([StructField("a", LongType()), StructField("b", LongType())])
        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtypes, [np.int64, np.int64])
        self.assertEqual(inferred.spark_type, expected)

        pdf = pd.DataFrame({("x", "a"): [1, 2, 3], ("y", "b"): [3, 4, 5]})

        def func() -> pd.DataFrame[zip(pdf.columns, pdf.dtypes)]:
            pass

        expected = StructType(
            [StructField("(x, a)", LongType()), StructField("(y, b)", LongType())]
        )
        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtypes, [np.int64, np.int64])
        self.assertEqual(inferred.spark_type, expected)

        pdf = pd.DataFrame({"a": [1, 2, 3], "b": pd.Categorical(["a", "b", "c"])})

        def func() -> pd.DataFrame[zip(pdf.columns, pdf.dtypes)]:
            pass

        expected = StructType([StructField("a", LongType()), StructField("b", LongType())])
        inferred = infer_return_type(func)
        self.assertEqual(inferred.dtypes, [np.int64, CategoricalDtype(categories=["a", "b", "c"])])
        self.assertEqual(inferred.spark_type, expected)

    @unittest.skipIf(
        sys.version_info < (3, 7),
        "Type inference from pandas instances is supported with Python 3.7+",
    )
    def test_infer_schema_with_names_pandas_instances_negative(self):
        def try_infer_return_type():
            def f() -> 'pd.DataFrame["a" : np.float : 1, "b":str:2]':  # noqa: F821
                pass

            infer_return_type(f)

        self.assertRaisesRegex(TypeError, "Type hints should be specified", try_infer_return_type)

        class A:
            pass

        def try_infer_return_type():
            def f() -> pd.DataFrame[A]:
                pass

            infer_return_type(f)

        self.assertRaisesRegex(TypeError, "not understood", try_infer_return_type)

        def try_infer_return_type():
            def f() -> 'pd.DataFrame["a" : np.float : 1, "b":str:2]':  # noqa: F821
                pass

            infer_return_type(f)

        self.assertRaisesRegex(TypeError, "Type hints should be specified", try_infer_return_type)

        # object type
        pdf = pd.DataFrame({"a": ["a", 2, None]})

        def try_infer_return_type():
            def f() -> pd.DataFrame[pdf.dtypes]:  # type: ignore
                pass

            infer_return_type(f)

        self.assertRaisesRegex(TypeError, "object.*not understood", try_infer_return_type)

        def try_infer_return_type():
            def f() -> pd.Series[pdf.a.dtype]:  # type: ignore
                pass

            infer_return_type(f)

        self.assertRaisesRegex(TypeError, "object.*not understood", try_infer_return_type)

    def test_infer_schema_with_names_negative(self):
        def try_infer_return_type():
            def f() -> 'ks.DataFrame["a" : np.float : 1, "b":str:2]':  # noqa: F821
                pass

            infer_return_type(f)

        self.assertRaisesRegex(TypeError, "Type hints should be specified", try_infer_return_type)

        class A:
            pass

        def try_infer_return_type():
            def f() -> ks.DataFrame[A]:
                pass

            infer_return_type(f)

        self.assertRaisesRegex(TypeError, "not understood", try_infer_return_type)

        def try_infer_return_type():
            def f() -> 'ks.DataFrame["a" : np.float : 1, "b":str:2]':  # noqa: F821
                pass

            infer_return_type(f)

        self.assertRaisesRegex(TypeError, "Type hints should be specified", try_infer_return_type)

        # object type
        pdf = pd.DataFrame({"a": ["a", 2, None]})

        def try_infer_return_type():
            def f() -> ks.DataFrame[pdf.dtypes]:  # type: ignore
                pass

            infer_return_type(f)

        self.assertRaisesRegex(TypeError, "object.*not understood", try_infer_return_type)

        def try_infer_return_type():
            def f() -> ks.Series[pdf.a.dtype]:  # type: ignore
                pass

            infer_return_type(f)

        self.assertRaisesRegex(TypeError, "object.*not understood", try_infer_return_type)

    def test_as_spark_type_koalas_dtype(self):
        type_mapper = {
            # binary
            np.character: (np.character, BinaryType()),
            np.bytes_: (np.bytes_, BinaryType()),
            np.string_: (np.bytes_, BinaryType()),
            bytes: (np.bytes_, BinaryType()),
            # integer
            np.int8: (np.int8, ByteType()),
            np.byte: (np.int8, ByteType()),
            np.int16: (np.int16, ShortType()),
            np.int32: (np.int32, IntegerType()),
            np.int64: (np.int64, LongType()),
            np.int: (np.int64, LongType()),
            int: (np.int64, LongType()),
            # floating
            np.float32: (np.float32, FloatType()),
            np.float: (np.float64, DoubleType()),
            np.float64: (np.float64, DoubleType()),
            float: (np.float64, DoubleType()),
            # string
            np.str: (np.unicode_, StringType()),
            np.unicode_: (np.unicode_, StringType()),
            str: (np.unicode_, StringType()),
            # bool
            np.bool: (np.bool, BooleanType()),
            bool: (np.bool, BooleanType()),
            # datetime
            np.datetime64: (np.datetime64, TimestampType()),
            datetime.datetime: (np.dtype("datetime64[ns]"), TimestampType()),
            # DateType
            datetime.date: (np.dtype("object"), DateType()),
            # DecimalType
            decimal.Decimal: (np.dtype("object"), DecimalType(38, 18)),
            # ArrayType
            np.ndarray: (np.dtype("object"), ArrayType(StringType())),
            List[bytes]: (np.dtype("object"), ArrayType(BinaryType())),
            List[np.character]: (np.dtype("object"), ArrayType(BinaryType())),
            List[np.bytes_]: (np.dtype("object"), ArrayType(BinaryType())),
            List[np.string_]: (np.dtype("object"), ArrayType(BinaryType())),
            List[bool]: (np.dtype("object"), ArrayType(BooleanType())),
            List[np.bool]: (np.dtype("object"), ArrayType(BooleanType())),
            List[datetime.date]: (np.dtype("object"), ArrayType(DateType())),
            List[np.int8]: (np.dtype("object"), ArrayType(ByteType())),
            List[np.byte]: (np.dtype("object"), ArrayType(ByteType())),
            List[decimal.Decimal]: (np.dtype("object"), ArrayType(DecimalType(38, 18))),
            List[float]: (np.dtype("object"), ArrayType(DoubleType())),
            List[np.float]: (np.dtype("object"), ArrayType(DoubleType())),
            List[np.float64]: (np.dtype("object"), ArrayType(DoubleType())),
            List[np.float32]: (np.dtype("object"), ArrayType(FloatType())),
            List[np.int32]: (np.dtype("object"), ArrayType(IntegerType())),
            List[int]: (np.dtype("object"), ArrayType(LongType())),
            List[np.int]: (np.dtype("object"), ArrayType(LongType())),
            List[np.int64]: (np.dtype("object"), ArrayType(LongType())),
            List[np.int16]: (np.dtype("object"), ArrayType(ShortType())),
            List[str]: (np.dtype("object"), ArrayType(StringType())),
            List[np.unicode_]: (np.dtype("object"), ArrayType(StringType())),
            List[datetime.datetime]: (np.dtype("object"), ArrayType(TimestampType())),
            List[np.datetime64]: (np.dtype("object"), ArrayType(TimestampType())),
            # CategoricalDtype
            CategoricalDtype(categories=["a", "b", "c"]): (
                CategoricalDtype(categories=["a", "b", "c"]),
                LongType(),
            ),
        }

        for numpy_or_python_type, (dtype, spark_type) in type_mapper.items():
            self.assertEqual(as_spark_type(numpy_or_python_type), spark_type)
            self.assertEqual(koalas_dtype(numpy_or_python_type), (dtype, spark_type))

        with self.assertRaisesRegex(TypeError, "Type uint64 was not understood."):
            as_spark_type(np.dtype("uint64"))

        with self.assertRaisesRegex(TypeError, "Type object was not understood."):
            as_spark_type(np.dtype("object"))

        with self.assertRaisesRegex(TypeError, "Type uint64 was not understood."):
            koalas_dtype(np.dtype("uint64"))

        with self.assertRaisesRegex(TypeError, "Type object was not understood."):
            koalas_dtype(np.dtype("object"))

    @unittest.skipIf(not extension_dtypes_available, "The pandas extension types are not available")
    def test_as_spark_type_extension_dtypes(self):
        from pandas import Int8Dtype, Int16Dtype, Int32Dtype, Int64Dtype

        type_mapper = {
            Int8Dtype(): ByteType(),
            Int16Dtype(): ShortType(),
            Int32Dtype(): IntegerType(),
            Int64Dtype(): LongType(),
        }

        for extension_dtype, spark_type in type_mapper.items():
            self.assertEqual(as_spark_type(extension_dtype), spark_type)
            self.assertEqual(koalas_dtype(extension_dtype), (extension_dtype, spark_type))

    @unittest.skipIf(
        not extension_object_dtypes_available, "The pandas extension object types are not available"
    )
    def test_as_spark_type_extension_object_dtypes(self):
        from pandas import BooleanDtype, StringDtype

        type_mapper = {
            BooleanDtype(): BooleanType(),
            StringDtype(): StringType(),
        }

        for extension_dtype, spark_type in type_mapper.items():
            self.assertEqual(as_spark_type(extension_dtype), spark_type)
            self.assertEqual(koalas_dtype(extension_dtype), (extension_dtype, spark_type))

    @unittest.skipIf(
        not extension_float_dtypes_available, "The pandas extension float types are not available"
    )
    def test_as_spark_type_extension_float_dtypes(self):
        from pandas import Float32Dtype, Float64Dtype

        type_mapper = {
            Float32Dtype(): FloatType(),
            Float64Dtype(): DoubleType(),
        }

        for extension_dtype, spark_type in type_mapper.items():
            self.assertEqual(as_spark_type(extension_dtype), spark_type)
            self.assertEqual(koalas_dtype(extension_dtype), (extension_dtype, spark_type))
