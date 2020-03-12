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

from databricks.koalas.missing import _unsupported_function, _unsupported_property


def unsupported_function_expanding(method_name, deprecated=False, reason=""):
    return _unsupported_function(
        class_name="pandas.core.window.Expanding",
        method_name=method_name,
        deprecated=deprecated,
        reason=reason,
    )


def unsupported_property_expanding(property_name, deprecated=False, reason=""):
    return _unsupported_property(
        class_name="pandas.core.window.Expanding",
        property_name=property_name,
        deprecated=deprecated,
        reason=reason,
    )


def unsupported_function_rolling(method_name, deprecated=False, reason=""):
    return _unsupported_function(
        class_name="pandas.core.window.Rolling",
        method_name=method_name,
        deprecated=deprecated,
        reason=reason,
    )


def unsupported_property_rolling(property_name, deprecated=False, reason=""):
    return _unsupported_property(
        class_name="pandas.core.window.Rolling",
        property_name=property_name,
        deprecated=deprecated,
        reason=reason,
    )


class _MissingPandasLikeExpanding(object):
    agg = unsupported_function_expanding("agg")
    aggregate = unsupported_function_expanding("aggregate")
    apply = unsupported_function_expanding("apply")
    corr = unsupported_function_expanding("corr")
    cov = unsupported_function_expanding("cov")
    kurt = unsupported_function_expanding("kurt")
    median = unsupported_function_expanding("median")
    quantile = unsupported_function_expanding("quantile")
    skew = unsupported_function_expanding("skew")
    validate = unsupported_function_expanding("validate")

    exclusions = unsupported_property_expanding("exclusions")
    is_datetimelike = unsupported_property_expanding("is_datetimelike")
    is_freq_type = unsupported_property_expanding("is_freq_type")
    ndim = unsupported_property_expanding("ndim")


class _MissingPandasLikeRolling(object):
    agg = unsupported_function_rolling("agg")
    aggregate = unsupported_function_rolling("aggregate")
    apply = unsupported_function_rolling("apply")
    corr = unsupported_function_rolling("corr")
    cov = unsupported_function_rolling("cov")
    kurt = unsupported_function_rolling("kurt")
    median = unsupported_function_rolling("median")
    quantile = unsupported_function_rolling("quantile")
    skew = unsupported_function_rolling("skew")
    validate = unsupported_function_rolling("validate")

    exclusions = unsupported_property_rolling("exclusions")
    is_datetimelike = unsupported_property_rolling("is_datetimelike")
    is_freq_type = unsupported_property_rolling("is_freq_type")
    ndim = unsupported_property_rolling("ndim")


class _MissingPandasLikeExpandingGroupby(object):
    agg = unsupported_function_expanding("agg")
    aggregate = unsupported_function_expanding("aggregate")
    apply = unsupported_function_expanding("apply")
    corr = unsupported_function_expanding("corr")
    cov = unsupported_function_expanding("cov")
    kurt = unsupported_function_expanding("kurt")
    median = unsupported_function_expanding("median")
    quantile = unsupported_function_expanding("quantile")
    skew = unsupported_function_expanding("skew")
    validate = unsupported_function_expanding("validate")

    exclusions = unsupported_property_expanding("exclusions")
    is_datetimelike = unsupported_property_expanding("is_datetimelike")
    is_freq_type = unsupported_property_expanding("is_freq_type")
    ndim = unsupported_property_expanding("ndim")


class _MissingPandasLikeRollingGroupby(object):
    agg = unsupported_function_rolling("agg")
    aggregate = unsupported_function_rolling("aggregate")
    apply = unsupported_function_rolling("apply")
    corr = unsupported_function_rolling("corr")
    cov = unsupported_function_rolling("cov")
    kurt = unsupported_function_rolling("kurt")
    median = unsupported_function_rolling("median")
    quantile = unsupported_function_rolling("quantile")
    skew = unsupported_function_rolling("skew")
    validate = unsupported_function_rolling("validate")

    exclusions = unsupported_property_rolling("exclusions")
    is_datetimelike = unsupported_property_rolling("is_datetimelike")
    is_freq_type = unsupported_property_rolling("is_freq_type")
    ndim = unsupported_property_rolling("ndim")
