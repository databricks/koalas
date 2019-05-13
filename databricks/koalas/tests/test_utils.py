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

import pandas as pd

from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils
from databricks.koalas.utils import lazy_property, validate_arguments_and_invoke_function

some_global_variable = 0


class UtilsTest(ReusedSQLTestCase, SQLTestUtils):

    # a dummy to_html version with an extra parameter that pandas does not support
    # used in test_validate_arguments_and_invoke_function
    def to_html(self, max_rows=None, unsupported_param=None):
        args = locals()

        pdf = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        }, index=[0, 1, 3])
        validate_arguments_and_invoke_function(pdf, self.to_html, pd.DataFrame.to_html, args)

    def to_clipboard(self, sep=',', **kwargs):
        args = locals()

        pdf = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
        }, index=[0, 1, 3])
        validate_arguments_and_invoke_function(pdf, self.to_clipboard,
                                               pd.DataFrame.to_clipboard, args)

    def test_validate_arguments_and_invoke_function(self):
        # This should pass and run fine
        self.to_html()
        self.to_html(unsupported_param=None)
        self.to_html(max_rows=5)

        # This should fail because we are explicitly setting an unsupported param
        # to a non-default value
        with self.assertRaises(TypeError):
            self.to_html(unsupported_param=1)

        # Support for **kwargs
        self.to_clipboard(sep=',', index=False)

    def test_lazy_property(self):
        obj = TestClassForLazyProp()
        # If lazy prop is not working, the second test would fail (because it'd be 2)
        self.assert_eq(obj.lazy_prop, 1)
        self.assert_eq(obj.lazy_prop, 1)


class TestClassForLazyProp:

    def __init__(self):
        self.some_variable = 0

    @lazy_property
    def lazy_prop(self):
        self.some_variable += 1
        return self.some_variable
