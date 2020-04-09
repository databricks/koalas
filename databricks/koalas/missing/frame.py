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
        class_name="pd.DataFrame", method_name=method_name, deprecated=deprecated, reason=reason
    )


def unsupported_property(property_name, deprecated=False, reason=""):
    return _unsupported_property(
        class_name="pd.DataFrame", property_name=property_name, deprecated=deprecated, reason=reason
    )


class _MissingPandasLikeDataFrame(object):

    # Functions
    align = unsupported_function("align")
    asfreq = unsupported_function("asfreq")
    asof = unsupported_function("asof")
    at_time = unsupported_function("at_time")
    between_time = unsupported_function("between_time")
    boxplot = unsupported_function("boxplot")
    combine = unsupported_function("combine")
    combine_first = unsupported_function("combine_first")
    corrwith = unsupported_function("corrwith")
    cov = unsupported_function("cov")
    dot = unsupported_function("dot")
    droplevel = unsupported_function("droplevel")
    ewm = unsupported_function("ewm")
    first = unsupported_function("first")
    infer_objects = unsupported_function("infer_objects")
    insert = unsupported_function("insert")
    interpolate = unsupported_function("interpolate")
    itertuples = unsupported_function("itertuples")
    last = unsupported_function("last")
    last_valid_index = unsupported_function("last_valid_index")
    lookup = unsupported_function("lookup")
    mad = unsupported_function("mad")
    mode = unsupported_function("mode")
    prod = unsupported_function("prod")
    product = unsupported_function("product")
    reindex_like = unsupported_function("reindex_like")
    rename_axis = unsupported_function("rename_axis")
    reorder_levels = unsupported_function("reorder_levels")
    resample = unsupported_function("resample")
    sem = unsupported_function("sem")
    set_axis = unsupported_function("set_axis")
    slice_shift = unsupported_function("slice_shift")
    swapaxes = unsupported_function("swapaxes")
    swaplevel = unsupported_function("swaplevel")
    tail = unsupported_function("tail")
    to_feather = unsupported_function("to_feather")
    to_gbq = unsupported_function("to_gbq")
    to_hdf = unsupported_function("to_hdf")
    to_period = unsupported_function("to_period")
    to_sql = unsupported_function("to_sql")
    to_stata = unsupported_function("to_stata")
    to_timestamp = unsupported_function("to_timestamp")
    tshift = unsupported_function("tshift")
    tz_convert = unsupported_function("tz_convert")
    tz_localize = unsupported_function("tz_localize")

    # Deprecated functions
    convert_objects = unsupported_function("convert_objects", deprecated=True)
    select = unsupported_function("select", deprecated=True)
    to_panel = unsupported_function("to_panel", deprecated=True)
    get_values = unsupported_function("get_values", deprecated=True)
    compound = unsupported_function("compound", deprecated=True)
    reindex_axis = unsupported_function("reindex_axis", deprecated=True)

    # Functions we won't support.
    to_pickle = common.to_pickle(unsupported_function)
    memory_usage = common.memory_usage(unsupported_function)
    to_xarray = common.to_xarray(unsupported_function)

    if LooseVersion(pd.__version__) < LooseVersion("1.0"):
        # Deprecated properties
        blocks = unsupported_property("blocks", deprecated=True)
        ftypes = unsupported_property("ftypes", deprecated=True)
        is_copy = unsupported_property("is_copy", deprecated=True)
        ix = unsupported_property("ix", deprecated=True)

        # Deprecated functions
        as_blocks = unsupported_function("as_blocks", deprecated=True)
        as_matrix = unsupported_function("as_matrix", deprecated=True)
        clip_lower = unsupported_function("clip_lower", deprecated=True)
        clip_upper = unsupported_function("clip_upper", deprecated=True)
        get_ftype_counts = unsupported_function("get_ftype_counts", deprecated=True)
        get_value = unsupported_function("get_value", deprecated=True)
        set_value = unsupported_function("set_value", deprecated=True)
        to_dense = unsupported_function("to_dense", deprecated=True)
        to_sparse = unsupported_function("to_sparse", deprecated=True)
        to_msgpack = unsupported_function("to_msgpack", deprecated=True)
