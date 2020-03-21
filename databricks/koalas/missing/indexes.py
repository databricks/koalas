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

from databricks.koalas.missing import _unsupported_function, _unsupported_property, common


def unsupported_function(method_name, deprecated=False, reason=""):
    return _unsupported_function(
        class_name="pd.Index", method_name=method_name, deprecated=deprecated, reason=reason
    )


def unsupported_property(property_name, deprecated=False, reason=""):
    return _unsupported_property(
        class_name="pd.Index", property_name=property_name, deprecated=deprecated, reason=reason
    )


class _MissingPandasLikeIndex(object):

    # Properties
    nbytes = unsupported_property("nbytes")

    # Functions
    argsort = unsupported_function("argsort")
    asof_locs = unsupported_function("asof_locs")
    delete = unsupported_function("delete")
    factorize = unsupported_function("factorize")
    format = unsupported_function("format")
    get_indexer = unsupported_function("get_indexer")
    get_indexer_for = unsupported_function("get_indexer_for")
    get_indexer_non_unique = unsupported_function("get_indexer_non_unique")
    get_level_values = unsupported_function("get_level_values")
    get_loc = unsupported_function("get_loc")
    get_slice_bound = unsupported_function("get_slice_bound")
    get_value = unsupported_function("get_value")
    groupby = unsupported_function("groupby")
    holds_integer = unsupported_function("holds_integer")
    insert = unsupported_function("insert")
    intersection = unsupported_function("intersection")
    is_ = unsupported_function("is_")
    is_lexsorted_for_tuple = unsupported_function("is_lexsorted_for_tuple")
    is_mixed = unsupported_function("is_mixed")
    is_type_compatible = unsupported_function("is_type_compatible")
    join = unsupported_function("join")
    map = unsupported_function("map")
    putmask = unsupported_function("putmask")
    ravel = unsupported_function("ravel")
    reindex = unsupported_function("reindex")
    searchsorted = unsupported_function("searchsorted")
    slice_indexer = unsupported_function("slice_indexer")
    slice_locs = unsupported_function("slice_locs")
    sortlevel = unsupported_function("sortlevel")
    to_flat_index = unsupported_function("to_flat_index")
    to_native_types = unsupported_function("to_native_types")
    view = unsupported_function("view")
    where = unsupported_function("where")

    # Deprecated functions
    get_values = unsupported_function("get_values", deprecated=True)
    item = unsupported_function("item", deprecated=True)
    set_value = unsupported_function("set_value")

    # Properties we won't support.
    array = common.array(unsupported_property)
    duplicated = common.duplicated(unsupported_property)

    # Functions we won't support.
    memory_usage = common.memory_usage(unsupported_function)
    to_list = common.to_list(unsupported_function)
    tolist = common.tolist(unsupported_function)
    __iter__ = common.__iter__(unsupported_function)

    if LooseVersion(pd.__version__) < LooseVersion("1.0"):
        # Deprecated properties
        strides = unsupported_property("strides", deprecated=True)
        data = unsupported_property("data", deprecated=True)
        itemsize = unsupported_property("itemsize", deprecated=True)
        base = unsupported_property("base", deprecated=True)
        flags = unsupported_property("flags", deprecated=True)

        # Deprecated functions
        get_duplicates = unsupported_function("get_duplicates", deprecated=True)
        summary = unsupported_function("summary", deprecated=True)
        contains = unsupported_function("contains", deprecated=True)


class _MissingPandasLikeMultiIndex(object):

    # Deprecated properties
    strides = unsupported_property("strides", deprecated=True)
    data = unsupported_property("data", deprecated=True)
    itemsize = unsupported_property("itemsize", deprecated=True)

    # Functions
    argsort = unsupported_function("argsort")
    asof_locs = unsupported_function("asof_locs")
    delete = unsupported_function("delete")
    equal_levels = unsupported_function("equal_levels")
    factorize = unsupported_function("factorize")
    format = unsupported_function("format")
    get_indexer = unsupported_function("get_indexer")
    get_indexer_for = unsupported_function("get_indexer_for")
    get_indexer_non_unique = unsupported_function("get_indexer_non_unique")
    get_level_values = unsupported_function("get_level_values")
    get_loc = unsupported_function("get_loc")
    get_loc_level = unsupported_function("get_loc_level")
    get_locs = unsupported_function("get_locs")
    get_slice_bound = unsupported_function("get_slice_bound")
    get_value = unsupported_function("get_value")
    groupby = unsupported_function("groupby")
    holds_integer = unsupported_function("holds_integer")
    insert = unsupported_function("insert")
    intersection = unsupported_function("intersection")
    is_ = unsupported_function("is_")
    is_lexsorted = unsupported_function("is_lexsorted")
    is_lexsorted_for_tuple = unsupported_function("is_lexsorted_for_tuple")
    is_mixed = unsupported_function("is_mixed")
    is_type_compatible = unsupported_function("is_type_compatible")
    join = unsupported_function("join")
    map = unsupported_function("map")
    putmask = unsupported_function("putmask")
    ravel = unsupported_function("ravel")
    reindex = unsupported_function("reindex")
    remove_unused_levels = unsupported_function("remove_unused_levels")
    reorder_levels = unsupported_function("reorder_levels")
    searchsorted = unsupported_function("searchsorted")
    set_codes = unsupported_function("set_codes")
    set_levels = unsupported_function("set_levels")
    slice_indexer = unsupported_function("slice_indexer")
    slice_locs = unsupported_function("slice_locs")
    sortlevel = unsupported_function("sortlevel")
    to_flat_index = unsupported_function("to_flat_index")
    to_native_types = unsupported_function("to_native_types")
    truncate = unsupported_function("truncate")
    view = unsupported_function("view")
    where = unsupported_function("where")

    # Deprecated functions
    get_duplicates = unsupported_function("get_duplicates", deprecated=True)
    get_values = unsupported_function("get_values", deprecated=True)
    item = unsupported_function("item", deprecated=True)
    set_value = unsupported_function("set_value", deprecated=True)

    # Functions we won't support.
    array = common.array(unsupported_property)
    duplicated = common.duplicated(unsupported_property)
    codes = unsupported_property(
        "codes",
        reason="'codes' requires to collect all data into the driver which is against the "
        "design principle of Koalas. Alternatively, you could call 'to_pandas()' and"
        " use 'codes' property in pandas.",
    )
    levels = unsupported_property(
        "levels",
        reason="'levels' requires to collect all data into the driver which is against the "
        "design principle of Koalas. Alternatively, you could call 'to_pandas()' and"
        " use 'levels' property in pandas.",
    )
    __iter__ = common.__iter__(unsupported_function)

    # Properties we won't support.
    memory_usage = common.memory_usage(unsupported_function)
    to_list = common.to_list(unsupported_function)
    tolist = common.tolist(unsupported_function)

    if LooseVersion(pd.__version__) < LooseVersion("1.0"):
        # Deprecated properties
        base = unsupported_property("base", deprecated=True)
        labels = unsupported_property("labels", deprecated=True)
        flags = unsupported_property("flags", deprecated=True)

        # Deprecated functions
        set_labels = unsupported_function("set_labels")
        summary = unsupported_function("summary", deprecated=True)
        to_hierarchical = unsupported_function("to_hierarchical", deprecated=True)
        contains = unsupported_function("contains", deprecated=True)
