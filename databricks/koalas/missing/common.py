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


memory_usage = lambda f: f(
    'memory_usage',
    reason="Unlike pandas, most DataFrames are not materialized in memory in Spark "
           "(and Koalas), and as a result memory_usage() does not do what you intend it "
           "to do. Use Spark's web UI to monitor disk and memory usage of your application.")

values = lambda f: f(
    'values',
    reason="If you want to collect your data as an NumPy array, use 'to_numpy()' instead.")

array = lambda f: f(
    'array',
    reason="If you want to collect your data as an NumPy array, use 'to_numpy()' instead.")

to_pickle = lambda f: f(
    'to_pickle',
    reason="For storage, we encourage you to use Delta or Parquet, instead of Python pickle "
           "format.")

to_xarray = lambda f: f(
    'to_xarray',
    reason="If you want to collect your data as an NumPy array, use 'to_numpy()' instead.")

to_list = lambda f: f(
    'to_list',
    reason="If you want to collect your data as an NumPy array, use 'to_numpy()' instead.")

tolist = lambda f: f(
    'tolist',
    reason="If you want to collect your data as an NumPy array, use 'to_numpy()' instead.")

is_boolean = lambda f: f(
    'is_boolean',
    reason="If you want to know whether the type is boolean, use 'dtype' and check.")

is_categorical = lambda f: f(
    'is_categorical',
    reason="If you want to know whether the type is categorical, use 'dtype' and check.")

is_floating = lambda f: f(
    'is_floating',
    reason="If you want to know whether the type is float, use 'dtype' and check.")

is_integer = lambda f: f(
    'is_integer',
    reason="If you want to know whether the type is integer, use 'dtype' and check.")

is_interval = lambda f: f(
    'is_interval',
    reason="If you want to know whether the type is interval, use 'dtype' and check.")

is_numeric = lambda f: f(
    'is_numeric',
    reason="If you want to know whether the type is numeric, use 'dtype' and check.")

is_object = lambda f: f(
    'is_object',
    reason="If you want to know whether the type is object, use 'dtype' and check.")
