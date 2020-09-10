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

import pandas
import pandas as pd
import numpy as np
from pyspark.sql.types import FloatType, IntegerType, LongType, StringType, StructField, StructType

from databricks.koalas.typedef import infer_return_type
from databricks import koalas as ks


class TypeHintTests(unittest.TestCase):
    @unittest.skipIf(
        sys.version_info < (3, 7),
        "Type inference from pandas instances is supported with Python 3.7+",
    )
    def test_infer_schema_from_pandas_instances(self):
        def func() -> pd.Series[int]:
            pass

        self.assertEqual(infer_return_type(func).tpe, IntegerType())

        def func() -> pd.Series[np.float]:
            pass

        self.assertEqual(infer_return_type(func).tpe, FloatType())

        def func() -> "pd.DataFrame[np.float, str]":
            pass

        expected = StructType([StructField("c0", FloatType()), StructField("c1", StringType())])
        self.assertEqual(infer_return_type(func).tpe, expected)

        def func() -> "pandas.DataFrame[np.float]":
            pass

        expected = StructType([StructField("c0", FloatType())])
        self.assertEqual(infer_return_type(func).tpe, expected)

        def func() -> "pd.Series[int]":
            pass

        self.assertEqual(infer_return_type(func).tpe, IntegerType())

        def func() -> pd.DataFrame[np.float, str]:
            pass

        expected = StructType([StructField("c0", FloatType()), StructField("c1", StringType())])
        self.assertEqual(infer_return_type(func).tpe, expected)

        def func() -> pd.DataFrame[np.float]:
            pass

        expected = StructType([StructField("c0", FloatType())])
        self.assertEqual(infer_return_type(func).tpe, expected)

        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})

        def func() -> pd.DataFrame[pdf.dtypes]:  # type: ignore
            pass

        expected = StructType([StructField("c0", LongType()), StructField("c1", LongType())])
        self.assertEqual(infer_return_type(func).tpe, expected)

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

        expected = StructType([StructField("a", FloatType()), StructField("b", StringType())])
        self.assertEqual(infer_return_type(func).tpe, expected)

        def func() -> "pd.DataFrame['a': np.float, 'b': int]":  # noqa: F821
            pass

        expected = StructType([StructField("a", FloatType()), StructField("b", IntegerType())])
        self.assertEqual(infer_return_type(func).tpe, expected)

        pdf = pd.DataFrame({"a": [1, 2, 3], "b": [3, 4, 5]})

        def func() -> pd.DataFrame[zip(pdf.columns, pdf.dtypes)]:
            pass

        expected = StructType([StructField("a", LongType()), StructField("b", LongType())])
        self.assertEqual(infer_return_type(func).tpe, expected)

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
