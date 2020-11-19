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
A script to check whether each function has return type annotated. Before running this,
make sure you install koalas from the current checkout by running:
pip install -e .
"""

import inspect
from collections import defaultdict
from inspect import Signature
from pprint import pprint
import sys

from databricks.koalas.frame import DataFrame
from databricks.koalas.groupby import DataFrameGroupBy, SeriesGroupBy
from databricks.koalas.indexes import Index, MultiIndex
from databricks.koalas.series import Series
from databricks.koalas.window import Expanding, ExpandingGroupby, Rolling, RollingGroupby


def inspect_functions_missing_return_types(inspect_obj):
    """
    Inspect functions which exist in inspect_obj but don't have return types annotated.

    :return: the tuple of the function (name, signature) which has no return types annotated,
        and missing rate of inspect_obj. Here, inspect_obj's missing rate is calculated by:
        count of functions missing return types / count of total functions
    """
    missing_funcs = []

    inspect_funcs = inspect.getmembers(inspect_obj, inspect.isfunction)

    if inspect_funcs:
        for name, func in inspect_funcs:
            # Skip the private functions
            if name.startswith('_'):
                continue

            signature = inspect.signature(func)

            if signature.return_annotation == Signature.empty:
                missing_funcs.append((name, signature))

        missing_rate = len(missing_funcs) / len(inspect_funcs)

        return missing_funcs, missing_rate
    else:
        return None, None


def _main():
    for inspect_type in [
        DataFrame,
        Series,
        DataFrameGroupBy,
        SeriesGroupBy,
        Index,
        MultiIndex,
        Expanding,
        ExpandingGroupby,
        Rolling,
        RollingGroupby,
    ]:

        missing = inspect_functions_missing_return_types(inspect_type)

        print(f"========={inspect_type}=========")
        for name, signature in missing:
            print(f"name = {name}    signature = {signature}")


def _main2():
    inspect_modules = [
        module for module in sys.modules if module.startswith("databricks")
    ]

    # class_name_to_missing_rate = defaultdict()
    # class_name_to_missing_funcs = defaultdict()
    for inspect_module in inspect_modules:
        for name, clss in inspect.getmembers(sys.modules[inspect_module], inspect.isclass):
            if clss.__module__ == inspect_module:
                # Gets only classes defined in inspect_module

                missing_funcs, missing_rate = inspect_functions_missing_return_types(clss)
                if missing_rate is not None:
                    class_name_to_missing_rate[name] = missing_rate
                    class_name_to_missing_funcs[name] = missing_funcs


def print_per_class_coverage():
    print("Class name : "
          "Coverage (Count of public functions with return types / Count of public functions)")
    for class_name in sorted(class_name_to_missing_rate, key=class_name_to_missing_rate.get):
        print(f"{class_name} : {round((1 - class_name_to_missing_rate[class_name]) * 100, 3)}%")


if __name__ == '__main__':
    class_name_to_missing_rate = defaultdict()
    class_name_to_missing_funcs = defaultdict()
    _main2()
    print_per_class_coverage()
    # pprint(class_name_to_missing_funcs)
    pprint(class_name_to_missing_funcs["KoalasSeriesMethods"])
