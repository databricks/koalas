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

from databricks.koalas.exceptions import PandasNotImplementedError


def _unsupported_function(class_name, method_name):

    def unsupported_function(*args, **kwargs):
        raise PandasNotImplementedError(class_name=class_name, method_name=method_name)

    unsupported_function.__doc__ = \
        """A stub for the equivalent method to `{0}.{1}()`.

        The method `{0}.{1}()` is not implemented yet.
        """.format(class_name, method_name)

    return unsupported_function
