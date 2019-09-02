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
import numpy as np

from databricks import koalas as ks
from databricks.koalas import config
from databricks.koalas.testing.utils import ReusedSQLTestCase


class ConfigTest(ReusedSQLTestCase):

    def setUp(self):
        config._registered_options['test.config'] = 'default'
        config._registered_options['test.config.list'] = []
        config._registered_options['test.config.float'] = 1.2
        config._registered_options['test.config.int'] = 1
        config._registered_options['test.config.none'] = None
        config._registered_options_default_none['test.config.none'] = int

    def tearDown(self):
        ks.reset_option('test.config')
        del config._registered_options['test.config']
        del config._registered_options['test.config.list']
        del config._registered_options['test.config.float']
        del config._registered_options['test.config.int']
        del config._registered_options['test.config.none']
        del config._registered_options_default_none['test.config.none']

    def test_get_set_reset_option(self):
        self.assertEqual(ks.get_option('test.config'), 'default')

        ks.set_option('test.config', 'value')
        self.assertEqual(ks.get_option('test.config'), 'value')

        ks.reset_option('test.config')
        self.assertEqual(ks.get_option('test.config'), 'default')

    def test_get_set_reset_option_different_types(self):
        ks.set_option('test.config.list', [1, 2, 3, 4])
        self.assertEqual(ks.get_option('test.config.list'), [1, 2, 3, 4])
        ks.set_option('test.config.list', None)
        self.assertEqual(ks.get_option('test.config.list'), None)

        ks.set_option('test.config.float', None)
        self.assertEqual(ks.get_option('test.config.float'), None)
        ks.set_option('test.config.float', 5.0)
        self.assertEqual(ks.get_option('test.config.float'), 5.0)

        ks.set_option('test.config.int', 123)
        self.assertEqual(ks.get_option('test.config.int'), 123)

        ks.set_option('test.config.none', 5)
        self.assertEqual(ks.get_option('test.config.none'), 5)

    def test_different_types(self):
        with self.assertRaisesRegex(TypeError, "The configuration value for 'test.config'"):
            ks.set_option('test.config', 1)

        with self.assertRaisesRegex(TypeError, "was <class 'int'>"):
            ks.set_option('test.config.list', 1)

        with self.assertRaisesRegex(TypeError, "however, <class 'float'> is expected."):
            ks.set_option('test.config.float', 'abc')

        with self.assertRaisesRegex(TypeError, "however, <class 'int'> is expected."):
            ks.set_option('test.config.int', 'abc')

    def test_unknown_option(self):
        with self.assertRaisesRegex(config.OptionError, 'No such option'):
            ks.get_option('unknown')

        with self.assertRaisesRegex(config.OptionError, "Available options"):
            ks.set_option('unknown', 'value')

        with self.assertRaisesRegex(config.OptionError, "test.config"):
            ks.reset_option('unknown')
