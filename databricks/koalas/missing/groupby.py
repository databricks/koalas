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


def _unsupported_function(method_name, deprecated=False, reason=""):
    return unsupported_function(
        class_name="pd.groupby.GroupBy",
        method_name=method_name,
        deprecated=deprecated,
        reason=reason,
    )


def _unsupported_property(property_name, deprecated=False, reason=""):
    return unsupported_property(
        class_name="pd.groupby.GroupBy",
        property_name=property_name,
        deprecated=deprecated,
        reason=reason,
    )


class MissingPandasLikeDataFrameGroupBy(object):

    # Properties
    corr = _unsupported_property("corr")
    corrwith = _unsupported_property("corrwith")
    cov = _unsupported_property("cov")
    dtypes = _unsupported_property("dtypes")
    groups = _unsupported_property("groups")
    hist = _unsupported_property("hist")
    indices = _unsupported_property("indices")
    mad = _unsupported_property("mad")
    ngroups = _unsupported_property("ngroups")
    plot = _unsupported_property("plot")
    quantile = _unsupported_property("quantile")
    skew = _unsupported_property("skew")
    tshift = _unsupported_property("tshift")

    # Deprecated properties
    take = _unsupported_property("take", deprecated=True)

    # Functions
    boxplot = _unsupported_function("boxplot")
    ngroup = _unsupported_function("ngroup")
    nth = _unsupported_function("nth")
    ohlc = _unsupported_function("ohlc")
    pct_change = _unsupported_function("pct_change")
    pipe = _unsupported_function("pipe")
    prod = _unsupported_function("prod")
    resample = _unsupported_function("resample")
    sem = _unsupported_function("sem")


class MissingPandasLikeSeriesGroupBy(object):

    # Properties
    corr = _unsupported_property("corr")
    cov = _unsupported_property("cov")
    dtype = _unsupported_property("dtype")
    groups = _unsupported_property("groups")
    hist = _unsupported_property("hist")
    indices = _unsupported_property("indices")
    is_monotonic_decreasing = _unsupported_property("is_monotonic_decreasing")
    is_monotonic_increasing = _unsupported_property("is_monotonic_increasing")
    mad = _unsupported_property("mad")
    ngroups = _unsupported_property("ngroups")
    plot = _unsupported_property("plot")
    quantile = _unsupported_property("quantile")
    skew = _unsupported_property("skew")
    tshift = _unsupported_property("tshift")

    # Deprecated properties
    take = _unsupported_property("take", deprecated=True)

    # Functions
    agg = _unsupported_function("agg")
    aggregate = _unsupported_function("aggregate")
    describe = _unsupported_function("describe")
    ngroup = _unsupported_function("ngroup")
    nth = _unsupported_function("nth")
    ohlc = _unsupported_function("ohlc")
    pct_change = _unsupported_function("pct_change")
    pipe = _unsupported_function("pipe")
    prod = _unsupported_function("prod")
    resample = _unsupported_function("resample")
    sem = _unsupported_function("sem")
