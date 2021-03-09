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
from distutils.version import LooseVersion

import pandas as pd

from databricks.koalas.missing import unsupported_function, unsupported_property, common


def _unsupported_function(method_name, deprecated=False, reason="", cls="Index"):
    return unsupported_function(
        class_name="pd.{}".format(cls),
        method_name=method_name,
        deprecated=deprecated,
        reason=reason,
    )


def _unsupported_property(property_name, deprecated=False, reason="", cls="Index"):
    return unsupported_property(
        class_name="pd.{}".format(cls),
        property_name=property_name,
        deprecated=deprecated,
        reason=reason,
    )


class MissingPandasLikeIndex(object):

    # Properties
    nbytes = _unsupported_property("nbytes")

    # Functions
    argsort = _unsupported_function("argsort")
    asof_locs = _unsupported_function("asof_locs")
    factorize = _unsupported_function("factorize")
    format = _unsupported_function("format")
    get_indexer = _unsupported_function("get_indexer")
    get_indexer_for = _unsupported_function("get_indexer_for")
    get_indexer_non_unique = _unsupported_function("get_indexer_non_unique")
    get_loc = _unsupported_function("get_loc")
    get_slice_bound = _unsupported_function("get_slice_bound")
    get_value = _unsupported_function("get_value")
    groupby = _unsupported_function("groupby")
    is_ = _unsupported_function("is_")
    is_lexsorted_for_tuple = _unsupported_function("is_lexsorted_for_tuple")
    join = _unsupported_function("join")
    map = _unsupported_function("map")
    putmask = _unsupported_function("putmask")
    ravel = _unsupported_function("ravel")
    reindex = _unsupported_function("reindex")
    searchsorted = _unsupported_function("searchsorted")
    slice_indexer = _unsupported_function("slice_indexer")
    slice_locs = _unsupported_function("slice_locs")
    sortlevel = _unsupported_function("sortlevel")
    to_flat_index = _unsupported_function("to_flat_index")
    to_native_types = _unsupported_function("to_native_types")
    where = _unsupported_function("where")

    # Deprecated functions
    is_mixed = _unsupported_function("is_mixed")
    get_values = _unsupported_function("get_values", deprecated=True)
    set_value = _unsupported_function("set_value")

    # Properties we won't support.
    array = common.array(_unsupported_property)
    duplicated = common.duplicated(_unsupported_property)

    # Functions we won't support.
    memory_usage = common.memory_usage(_unsupported_function)
    __iter__ = common.__iter__(_unsupported_function)

    if LooseVersion(pd.__version__) < LooseVersion("1.0"):
        # Deprecated properties
        strides = _unsupported_property("strides", deprecated=True)
        data = _unsupported_property("data", deprecated=True)
        itemsize = _unsupported_property("itemsize", deprecated=True)
        base = _unsupported_property("base", deprecated=True)
        flags = _unsupported_property("flags", deprecated=True)

        # Deprecated functions
        get_duplicates = _unsupported_function("get_duplicates", deprecated=True)
        summary = _unsupported_function("summary", deprecated=True)
        contains = _unsupported_function("contains", deprecated=True)


class MissingPandasLikeDatetimeIndex(MissingPandasLikeIndex):

    # Properties
    nanosecond = _unsupported_property("nanosecond", cls="DatetimeIndex")
    date = _unsupported_property("date", cls="DatetimeIndex")
    time = _unsupported_property("time", cls="DatetimeIndex")
    timetz = _unsupported_property("timetz", cls="DatetimeIndex")
    tz = _unsupported_property("tz", cls="DatetimeIndex")
    freq = _unsupported_property("freq", cls="DatetimeIndex")
    freqstr = _unsupported_property("freqstr", cls="DatetimeIndex")
    inferred_freq = _unsupported_property("inferred_freq", cls="DatetimeIndex")

    # Functions
    indexer_at_time = _unsupported_function("indexer_at_time", cls="DatetimeIndex")
    indexer_between_time = _unsupported_function("indexer_between_time", cls="DatetimeIndex")
    snap = _unsupported_function("snap", cls="DatetimeIndex")
    tz_convert = _unsupported_function("tz_convert", cls="DatetimeIndex")
    tz_localize = _unsupported_function("tz_localize", cls="DatetimeIndex")
    to_period = _unsupported_function("to_period", cls="DatetimeIndex")
    to_perioddelta = _unsupported_function("to_perioddelta", cls="DatetimeIndex")
    to_pydatetime = _unsupported_function("to_pydatetime", cls="DatetimeIndex")
    mean = _unsupported_function("mean", cls="DatetimeIndex")
    std = _unsupported_function("std", cls="DatetimeIndex")


class MissingPandasLikeCategoricalIndex(MissingPandasLikeIndex):

    # Functions
    rename_categories = _unsupported_function("rename_categories", cls="CategoricalIndex")
    reorder_categories = _unsupported_function("reorder_categories", cls="CategoricalIndex")
    add_categories = _unsupported_function("add_categories", cls="CategoricalIndex")
    remove_categories = _unsupported_function("remove_categories", cls="CategoricalIndex")
    remove_unused_categories = _unsupported_function(
        "remove_unused_categories", cls="CategoricalIndex"
    )
    set_categories = _unsupported_function("set_categories", cls="CategoricalIndex")
    as_ordered = _unsupported_function("as_ordered", cls="CategoricalIndex")
    as_unordered = _unsupported_function("as_unordered", cls="CategoricalIndex")
    map = _unsupported_function("map", cls="CategoricalIndex")


class MissingPandasLikeMultiIndex(object):

    # Deprecated properties
    strides = _unsupported_property("strides", deprecated=True)
    data = _unsupported_property("data", deprecated=True)
    itemsize = _unsupported_property("itemsize", deprecated=True)

    # Functions
    argsort = _unsupported_function("argsort")
    asof_locs = _unsupported_function("asof_locs")
    equal_levels = _unsupported_function("equal_levels")
    factorize = _unsupported_function("factorize")
    format = _unsupported_function("format")
    get_indexer = _unsupported_function("get_indexer")
    get_indexer_for = _unsupported_function("get_indexer_for")
    get_indexer_non_unique = _unsupported_function("get_indexer_non_unique")
    get_loc = _unsupported_function("get_loc")
    get_loc_level = _unsupported_function("get_loc_level")
    get_locs = _unsupported_function("get_locs")
    get_slice_bound = _unsupported_function("get_slice_bound")
    get_value = _unsupported_function("get_value")
    groupby = _unsupported_function("groupby")
    is_ = _unsupported_function("is_")
    is_lexsorted = _unsupported_function("is_lexsorted")
    is_lexsorted_for_tuple = _unsupported_function("is_lexsorted_for_tuple")
    join = _unsupported_function("join")
    map = _unsupported_function("map")
    putmask = _unsupported_function("putmask")
    ravel = _unsupported_function("ravel")
    reindex = _unsupported_function("reindex")
    remove_unused_levels = _unsupported_function("remove_unused_levels")
    reorder_levels = _unsupported_function("reorder_levels")
    searchsorted = _unsupported_function("searchsorted")
    set_codes = _unsupported_function("set_codes")
    set_levels = _unsupported_function("set_levels")
    slice_indexer = _unsupported_function("slice_indexer")
    slice_locs = _unsupported_function("slice_locs")
    sortlevel = _unsupported_function("sortlevel")
    to_flat_index = _unsupported_function("to_flat_index")
    to_native_types = _unsupported_function("to_native_types")
    truncate = _unsupported_function("truncate")
    where = _unsupported_function("where")

    # Deprecated functions
    is_mixed = _unsupported_function("is_mixed")
    get_duplicates = _unsupported_function("get_duplicates", deprecated=True)
    get_values = _unsupported_function("get_values", deprecated=True)
    set_value = _unsupported_function("set_value", deprecated=True)

    # Functions we won't support.
    array = common.array(_unsupported_property)
    duplicated = common.duplicated(_unsupported_property)
    codes = _unsupported_property(
        "codes",
        reason="'codes' requires to collect all data into the driver which is against the "
        "design principle of Koalas. Alternatively, you could call 'to_pandas()' and"
        " use 'codes' property in pandas.",
    )
    levels = _unsupported_property(
        "levels",
        reason="'levels' requires to collect all data into the driver which is against the "
        "design principle of Koalas. Alternatively, you could call 'to_pandas()' and"
        " use 'levels' property in pandas.",
    )
    __iter__ = common.__iter__(_unsupported_function)

    # Properties we won't support.
    memory_usage = common.memory_usage(_unsupported_function)

    if LooseVersion(pd.__version__) < LooseVersion("1.0"):
        # Deprecated properties
        base = _unsupported_property("base", deprecated=True)
        labels = _unsupported_property("labels", deprecated=True)
        flags = _unsupported_property("flags", deprecated=True)

        # Deprecated functions
        set_labels = _unsupported_function("set_labels")
        summary = _unsupported_function("summary", deprecated=True)
        to_hierarchical = _unsupported_function("to_hierarchical", deprecated=True)
        contains = _unsupported_function("contains", deprecated=True)
