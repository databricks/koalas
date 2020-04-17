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


def unsupported_function(method_name, deprecated=False, reason=""):
    return _unsupported_function(
        class_name="pd.groupby.GroupBy",
        method_name=method_name,
        deprecated=deprecated,
        reason=reason,
    )


def unsupported_property(property_name, deprecated=False, reason=""):
    return _unsupported_property(
        class_name="pd.groupby.GroupBy",
        property_name=property_name,
        deprecated=deprecated,
        reason=reason,
    )


class _MissingPandasLikeDataFrameGroupBy(object):

    # Properties
    corr = unsupported_property("corr")
    corrwith = unsupported_property("corrwith")
    cov = unsupported_property("cov")
    dtypes = unsupported_property("dtypes")
    groups = unsupported_property("groups")
    hist = unsupported_property("hist")
    indices = unsupported_property("indices")
    mad = unsupported_property("mad")
    ngroups = unsupported_property("ngroups")
    plot = unsupported_property("plot")
    quantile = unsupported_property("quantile")
    skew = unsupported_property("skew")
    tshift = unsupported_property("tshift")

    # Deprecated properties
    take = unsupported_property("take", deprecated=True)

    # Functions
    boxplot = unsupported_function("boxplot")
    cumcount = unsupported_function("cumcount")
    get_group = unsupported_function("get_group")
    median = unsupported_function("median")
    ngroup = unsupported_function("ngroup")
    nth = unsupported_function("nth")
    ohlc = unsupported_function("ohlc")
    pct_change = unsupported_function("pct_change")
    pipe = unsupported_function("pipe")
    prod = unsupported_function("prod")
    resample = unsupported_function("resample")
    sem = unsupported_function("sem")
    tail = unsupported_function("tail")


class _MissingPandasLikeSeriesGroupBy(object):

    # Properties
    corr = unsupported_property("corr")
    cov = unsupported_property("cov")
    dtype = unsupported_property("dtype")
    groups = unsupported_property("groups")
    hist = unsupported_property("hist")
    indices = unsupported_property("indices")
    is_monotonic_decreasing = unsupported_property("is_monotonic_decreasing")
    is_monotonic_increasing = unsupported_property("is_monotonic_increasing")
    mad = unsupported_property("mad")
    ngroups = unsupported_property("ngroups")
    plot = unsupported_property("plot")
    quantile = unsupported_property("quantile")
    skew = unsupported_property("skew")
    tshift = unsupported_property("tshift")

    # Deprecated properties
    take = unsupported_property("take", deprecated=True)

    # Functions
    agg = unsupported_function("agg")
    aggregate = unsupported_function("aggregate")
    cumcount = unsupported_function("cumcount")
    describe = unsupported_function("describe")
    filter = unsupported_function("filter")
    get_group = unsupported_function("get_group")
    median = unsupported_function("median")
    ngroup = unsupported_function("ngroup")
    nth = unsupported_function("nth")
    ohlc = unsupported_function("ohlc")
    pct_change = unsupported_function("pct_change")
    pipe = unsupported_function("pipe")
    prod = unsupported_function("prod")
    resample = unsupported_function("resample")
    sem = unsupported_function("sem")
    tail = unsupported_function("tail")
