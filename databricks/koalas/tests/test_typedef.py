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
from pyspark.sql.types import *

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

    def test_if_pandas_implements_class_getitem(self):
        # the current type hint implementation of pandas DataFrame assumes pandas doesn't
        # implement '__class_getitem__'. This test case is to make sure pandas
        # doesn't implement them.
        assert not ks._frame_has_class_getitem
        assert not ks._series_has_class_getitem
