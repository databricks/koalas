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
from databricks.koalas import config
from databricks.koalas.testing.utils import ReusedSQLTestCase


class ConfigTest(ReusedSQLTestCase):

    def setUp(self):
        config._registered_options['test.config'] = 'default'

    def tearDown(self):
        ks.reset_option('test.config')
        del config._registered_options['test.config']

    def test_get_set_reset_option(self):
        self.assertEqual(ks.get_option('test.config'), 'default')

        ks.set_option('test.config', 'value')
        self.assertEqual(ks.get_option('test.config'), 'value')

        ks.reset_option('test.config')
        self.assertEqual(ks.get_option('test.config'), 'default')

    def test_unknown_option(self):
        with self.assertRaisesRegex(config.OptionError, 'No such key'):
            ks.get_option('unknown')

        with self.assertRaisesRegex(config.OptionError, "No such key"):
            ks.set_option('unknown', 'value')

        with self.assertRaisesRegex(config.OptionError, "No such key"):
            ks.reset_option('unknows')
