#!/usr/bin/env python
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

"""
A script to generate the missing property stubs. Before running this,
make sure you install koalas from the current checkout by running:
pip install -e .
"""

import inspect

import pandas as pd

from databricks.koalas.frame import DataFrame
from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy
from databricks.koalas.indexes import Index, MultiIndex
from databricks.koalas.series import Series


def inspect_missing_properties(original_type, target_type):
    """
    Find properties which exist in original_type but not in target_type.

    :return: the missing property name.
    """
    missing = []
    deprecated = []

    for name, func in inspect.getmembers(original_type, lambda o: isinstance(o, property)):
        # Skip the private attributes
        if name.startswith('_'):
            continue

        if hasattr(target_type, name) and isinstance(getattr(target_type, name), property):
            continue

        docstring = func.fget.__doc__
        if docstring and ('.. deprecated::' in docstring):
            deprecated.append(name)
        else:
            missing.append(name)

    return missing, deprecated


def _main():
    for original_type, target_type in [(pd.DataFrame, DataFrame),
                                       (pd.Series, Series),
                                       (pd.core.groupby.DataFrameGroupBy, DataFrameGroupBy),
                                       (pd.core.groupby.SeriesGroupBy, SeriesGroupBy),
                                       (pd.Index, Index),
                                       (pd.MultiIndex, MultiIndex)]:
        missing, deprecated = inspect_missing_properties(original_type, target_type)

        print('MISSING properties for {}'.format(original_type.__name__))
        for name in missing:
            print("""    {0} = unsupported_property('{0}')""".format(name))

        print()
        print('DEPRECATED properties for {}'.format(original_type.__name__))
        for name in deprecated:
            print("""    {0} = unsupported_property('{0}', deprecated=True)""".format(name))
        print()


if __name__ == '__main__':
    _main()
