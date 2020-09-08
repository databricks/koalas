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

from databricks.koalas.missing import unsupported_function, unsupported_property


def _unsupported_function_expanding(method_name, deprecated=False, reason=""):
    return unsupported_function(
        class_name="pandas.core.window.Expanding",
        method_name=method_name,
        deprecated=deprecated,
        reason=reason,
    )


def _unsupported_property_expanding(property_name, deprecated=False, reason=""):
    return unsupported_property(
        class_name="pandas.core.window.Expanding",
        property_name=property_name,
        deprecated=deprecated,
        reason=reason,
    )


def _unsupported_function_rolling(method_name, deprecated=False, reason=""):
    return unsupported_function(
        class_name="pandas.core.window.Rolling",
        method_name=method_name,
        deprecated=deprecated,
        reason=reason,
    )


def _unsupported_property_rolling(property_name, deprecated=False, reason=""):
    return unsupported_property(
        class_name="pandas.core.window.Rolling",
        property_name=property_name,
        deprecated=deprecated,
        reason=reason,
    )


class MissingPandasLikeExpanding(object):
    agg = _unsupported_function_expanding("agg")
    aggregate = _unsupported_function_expanding("aggregate")
    apply = _unsupported_function_expanding("apply")
    corr = _unsupported_function_expanding("corr")
    cov = _unsupported_function_expanding("cov")
    kurt = _unsupported_function_expanding("kurt")
    median = _unsupported_function_expanding("median")
    quantile = _unsupported_function_expanding("quantile")
    skew = _unsupported_function_expanding("skew")
    validate = _unsupported_function_expanding("validate")

    exclusions = _unsupported_property_expanding("exclusions")
    is_datetimelike = _unsupported_property_expanding("is_datetimelike")
    is_freq_type = _unsupported_property_expanding("is_freq_type")
    ndim = _unsupported_property_expanding("ndim")


class MissingPandasLikeRolling(object):
    agg = _unsupported_function_rolling("agg")
    aggregate = _unsupported_function_rolling("aggregate")
    apply = _unsupported_function_rolling("apply")
    corr = _unsupported_function_rolling("corr")
    cov = _unsupported_function_rolling("cov")
    kurt = _unsupported_function_rolling("kurt")
    median = _unsupported_function_rolling("median")
    quantile = _unsupported_function_rolling("quantile")
    skew = _unsupported_function_rolling("skew")
    validate = _unsupported_function_rolling("validate")

    exclusions = _unsupported_property_rolling("exclusions")
    is_datetimelike = _unsupported_property_rolling("is_datetimelike")
    is_freq_type = _unsupported_property_rolling("is_freq_type")
    ndim = _unsupported_property_rolling("ndim")


class MissingPandasLikeExpandingGroupby(object):
    agg = _unsupported_function_expanding("agg")
    aggregate = _unsupported_function_expanding("aggregate")
    apply = _unsupported_function_expanding("apply")
    corr = _unsupported_function_expanding("corr")
    cov = _unsupported_function_expanding("cov")
    kurt = _unsupported_function_expanding("kurt")
    median = _unsupported_function_expanding("median")
    quantile = _unsupported_function_expanding("quantile")
    skew = _unsupported_function_expanding("skew")
    validate = _unsupported_function_expanding("validate")

    exclusions = _unsupported_property_expanding("exclusions")
    is_datetimelike = _unsupported_property_expanding("is_datetimelike")
    is_freq_type = _unsupported_property_expanding("is_freq_type")
    ndim = _unsupported_property_expanding("ndim")


class MissingPandasLikeRollingGroupby(object):
    agg = _unsupported_function_rolling("agg")
    aggregate = _unsupported_function_rolling("aggregate")
    apply = _unsupported_function_rolling("apply")
    corr = _unsupported_function_rolling("corr")
    cov = _unsupported_function_rolling("cov")
    kurt = _unsupported_function_rolling("kurt")
    median = _unsupported_function_rolling("median")
    quantile = _unsupported_function_rolling("quantile")
    skew = _unsupported_function_rolling("skew")
    validate = _unsupported_function_rolling("validate")

    exclusions = _unsupported_property_rolling("exclusions")
    is_datetimelike = _unsupported_property_rolling("is_datetimelike")
    is_freq_type = _unsupported_property_rolling("is_freq_type")
    ndim = _unsupported_property_rolling("ndim")
