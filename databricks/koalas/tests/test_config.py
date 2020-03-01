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
from databricks.koalas.config import Option, DictWrapper
from databricks.koalas.testing.utils import ReusedSQLTestCase


class ConfigTest(ReusedSQLTestCase):
    def setUp(self):
        config._options_dict["test.config"] = Option(key="test.config", doc="", default="default")

        config._options_dict["test.config.list"] = Option(
            key="test.config.list", doc="", default=[], types=list
        )
        config._options_dict["test.config.float"] = Option(
            key="test.config.float", doc="", default=1.2, types=float
        )

        config._options_dict["test.config.int"] = Option(
            key="test.config.int",
            doc="",
            default=1,
            types=int,
            check_func=(lambda v: v > 0, "bigger then 0"),
        )
        config._options_dict["test.config.int.none"] = Option(
            key="test.config.int", doc="", default=None, types=(int, type(None))
        )

    def tearDown(self):
        ks.reset_option("test.config")
        del config._options_dict["test.config"]
        del config._options_dict["test.config.list"]
        del config._options_dict["test.config.float"]
        del config._options_dict["test.config.int"]
        del config._options_dict["test.config.int.none"]

    def test_get_set_reset_option(self):
        self.assertEqual(ks.get_option("test.config"), "default")

        ks.set_option("test.config", "value")
        self.assertEqual(ks.get_option("test.config"), "value")

        ks.reset_option("test.config")
        self.assertEqual(ks.get_option("test.config"), "default")

    def test_get_set_reset_option_different_types(self):
        ks.set_option("test.config.list", [1, 2, 3, 4])
        self.assertEqual(ks.get_option("test.config.list"), [1, 2, 3, 4])

        ks.set_option("test.config.float", 5.0)
        self.assertEqual(ks.get_option("test.config.float"), 5.0)

        ks.set_option("test.config.int", 123)
        self.assertEqual(ks.get_option("test.config.int"), 123)

        self.assertEqual(ks.get_option("test.config.int.none"), None)  # default None
        ks.set_option("test.config.int.none", 123)
        self.assertEqual(ks.get_option("test.config.int.none"), 123)
        ks.set_option("test.config.int.none", None)
        self.assertEqual(ks.get_option("test.config.int.none"), None)

    def test_different_types(self):
        with self.assertRaisesRegex(ValueError, "was <class 'int'>"):
            ks.set_option("test.config.list", 1)

        with self.assertRaisesRegex(ValueError, "however, expected types are"):
            ks.set_option("test.config.float", "abc")

        with self.assertRaisesRegex(ValueError, "[<class 'int'>]"):
            ks.set_option("test.config.int", "abc")

        with self.assertRaisesRegex(ValueError, "(<class 'int'>, <class 'NoneType'>)"):
            ks.set_option("test.config.int.none", "abc")

    def test_check_func(self):
        with self.assertRaisesRegex(ValueError, "bigger then 0"):
            ks.set_option("test.config.int", -1)

    def test_unknown_option(self):
        with self.assertRaisesRegex(config.OptionError, "No such option"):
            ks.get_option("unknown")

        with self.assertRaisesRegex(config.OptionError, "Available options"):
            ks.set_option("unknown", "value")

        with self.assertRaisesRegex(config.OptionError, "test.config"):
            ks.reset_option("unknown")

    def test_namespace_access(self):
        try:
            self.assertEqual(ks.options.compute.max_rows, ks.get_option("compute.max_rows"))
            ks.options.compute.max_rows = 0
            self.assertEqual(ks.options.compute.max_rows, 0)
            self.assertTrue(isinstance(ks.options.compute, DictWrapper))

            wrapper = ks.options.compute
            self.assertEqual(wrapper.max_rows, ks.get_option("compute.max_rows"))
            wrapper.max_rows = 1000
            self.assertEqual(ks.options.compute.max_rows, 1000)

            self.assertRaisesRegex(config.OptionError, "No such option", lambda: ks.options.compu)
            self.assertRaisesRegex(
                config.OptionError, "No such option", lambda: ks.options.compute.max
            )
            self.assertRaisesRegex(
                config.OptionError, "No such option", lambda: ks.options.max_rows1
            )

            with self.assertRaisesRegex(config.OptionError, "No such option"):
                ks.options.compute.max = 0
            with self.assertRaisesRegex(config.OptionError, "No such option"):
                ks.options.compute = 0
            with self.assertRaisesRegex(config.OptionError, "No such option"):
                ks.options.com = 0
        finally:
            ks.reset_option("compute.max_rows")

    def test_dir_options(self):
        self.assertTrue("compute.default_index_type" in dir(ks.options))
        self.assertTrue("plotting.sample_ratio" in dir(ks.options))

        self.assertTrue("default_index_type" in dir(ks.options.compute))
        self.assertTrue("sample_ratio" not in dir(ks.options.compute))

        self.assertTrue("default_index_type" not in dir(ks.options.plotting))
        self.assertTrue("sample_ratio" in dir(ks.options.plotting))
