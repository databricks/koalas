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

from databricks import koalas as ks
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils

from pyspark.sql.utils import ParseException


class SQLTest(ReusedSQLTestCase, SQLTestUtils):
    def test_error_variable_not_exist(self):
        msg = "The key variable_foo in the SQL statement was not found.*"
        with self.assertRaisesRegex(ValueError, msg):
            ks.sql("select * from {variable_foo}")

    def test_error_unsupported_type(self):
        msg = "Unsupported variable type <class 'dict'>: {'a': 1}"
        with self.assertRaisesRegex(ValueError, msg):
            some_dict = {"a": 1}
            ks.sql("select * from {some_dict}")

    def test_error_bad_sql(self):
        with self.assertRaises(ParseException):
            ks.sql("this is not valid sql")
