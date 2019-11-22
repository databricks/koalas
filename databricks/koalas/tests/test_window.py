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

import inspect

from databricks import koalas as ks
from databricks.koalas.exceptions import PandasNotImplementedError
from databricks.koalas.missing.window import _MissingPandasLikeExpanding, \
    _MissingPandasLikeRolling, _MissingPandasLikeExpandingGroupby, \
    _MissingPandasLikeRollingGroupby
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils


class ExpandingRollingTest(ReusedSQLTestCase, TestUtils):
    def test_missing(self):
        kdf = ks.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9]})

        # Expanding functions
        missing_functions = inspect.getmembers(_MissingPandasLikeExpanding,
                                               inspect.isfunction)
        unsupported_functions = [name for (name, type_) in missing_functions
                                 if type_.__name__ == 'unsupported_function']
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*Expanding.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.expanding(1), name)()  # Frame

            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*Expanding.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.a.expanding(1), name)()  # Series

        deprecated_functions = [name for (name, type_) in missing_functions
                                if type_.__name__ == 'deprecated_function']
        for name in deprecated_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Expanding.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.expanding(1), name)()  # Frame

            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Expanding.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.a.expanding(1), name)()  # Series

        # Rolling functions
        missing_functions = inspect.getmembers(_MissingPandasLikeRolling,
                                               inspect.isfunction)
        unsupported_functions = [name for (name, type_) in missing_functions
                                 if type_.__name__ == 'unsupported_function']
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*Rolling.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.rolling(1), name)()  # Frame
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*Rolling.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.a.rolling(1), name)()  # Series

        deprecated_functions = [name for (name, type_) in missing_functions
                                if type_.__name__ == 'deprecated_function']
        for name in deprecated_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Rolling.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.rolling(1), name)()  # Frame
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Rolling.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.a.rolling(1), name)()  # Series

        # Expanding properties
        missing_properties = inspect.getmembers(_MissingPandasLikeExpanding,
                                                lambda o: isinstance(o, property))
        unsupported_properties = [name for (name, type_) in missing_properties
                                  if type_.fget.__name__ == 'unsupported_property']
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*Expanding.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.expanding(1), name)  # Frame
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*Expanding.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.a.expanding(1), name)  # Series

        deprecated_properties = [name for (name, type_) in missing_properties
                                 if type_.fget.__name__ == 'deprecated_property']
        for name in deprecated_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Expanding.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.expanding(1), name)  # Frame
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Expanding.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.a.expanding(1), name)  # Series

        # Rolling properties
        missing_properties = inspect.getmembers(_MissingPandasLikeRolling,
                                                lambda o: isinstance(o, property))
        unsupported_properties = [name for (name, type_) in missing_properties
                                  if type_.fget.__name__ == 'unsupported_property']
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*Rolling.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.rolling(1), name)()  # Frame
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*Rolling.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.a.rolling(1), name)()  # Series
        deprecated_properties = [name for (name, type_) in missing_properties
                                 if type_.fget.__name__ == 'deprecated_property']
        for name in deprecated_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Rolling.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.rolling(1), name)()  # Frame
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Rolling.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.a.rolling(1), name)()  # Series

    def test_missing_groupby(self):
        kdf = ks.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9]})

        # Expanding functions
        missing_functions = inspect.getmembers(_MissingPandasLikeExpandingGroupby,
                                               inspect.isfunction)
        unsupported_functions = [name for (name, type_) in missing_functions
                                 if type_.__name__ == 'unsupported_function']
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*Expanding.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.groupby("a").expanding(1), name)()  # Frame

            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*Expanding.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.a.groupby(kdf.a).expanding(1), name)()  # Series

        deprecated_functions = [name for (name, type_) in missing_functions
                                if type_.__name__ == 'deprecated_function']
        for name in deprecated_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Expanding.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.groupby("a").expanding(1), name)()  # Frame

            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Expanding.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.a.groupby(kdf.a).expanding(1), name)()  # Series

        # Rolling functions
        missing_functions = inspect.getmembers(_MissingPandasLikeRollingGroupby,
                                               inspect.isfunction)
        unsupported_functions = [name for (name, type_) in missing_functions
                                 if type_.__name__ == 'unsupported_function']
        for name in unsupported_functions:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*Rolling.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.groupby("a").rolling(1), name)()  # Frame
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "method.*Rolling.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.a.groupby(kdf.a).rolling(1), name)()  # Series

        deprecated_functions = [name for (name, type_) in missing_functions
                                if type_.__name__ == 'deprecated_function']
        for name in deprecated_functions:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Rolling.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.rolling(1), name)()  # Frame
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "method.*Rolling.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.a.rolling(1), name)()  # Series

        # Expanding properties
        missing_properties = inspect.getmembers(_MissingPandasLikeExpandingGroupby,
                                                lambda o: isinstance(o, property))
        unsupported_properties = [name for (name, type_) in missing_properties
                                  if type_.fget.__name__ == 'unsupported_property']
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*Expanding.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.groupby("a").expanding(1), name)()  # Frame
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*Expanding.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.a.groupby(kdf.a).expanding(1), name)()  # Series

        deprecated_properties = [name for (name, type_) in missing_properties
                                 if type_.fget.__name__ == 'deprecated_property']
        for name in deprecated_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Expanding.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.expanding(1), name)  # Frame
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Expanding.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.a.expanding(1), name)  # Series

        # Rolling properties
        missing_properties = inspect.getmembers(_MissingPandasLikeRollingGroupby,
                                                lambda o: isinstance(o, property))
        unsupported_properties = [name for (name, type_) in missing_properties
                                  if type_.fget.__name__ == 'unsupported_property']
        for name in unsupported_properties:
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*Rolling.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.groupby("a").rolling(1), name)()  # Frame
            with self.assertRaisesRegex(
                    PandasNotImplementedError,
                    "property.*Rolling.*{}.*not implemented( yet\\.|\\. .+)".format(name)):
                getattr(kdf.a.groupby(kdf.a).rolling(1), name)()  # Series
        deprecated_properties = [name for (name, type_) in missing_properties
                                 if type_.fget.__name__ == 'deprecated_property']
        for name in deprecated_properties:
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Rolling.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.rolling(1), name)()  # Frame
            with self.assertRaisesRegex(PandasNotImplementedError,
                                        "property.*Rolling.*{}.*is deprecated"
                                        .format(name)):
                getattr(kdf.a.rolling(1), name)()  # Series
