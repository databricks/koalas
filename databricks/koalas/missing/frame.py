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


def _unsupported_function(method_name, deprecated=False, reason=""):
    return unsupported_function(
        class_name="pd.DataFrame", method_name=method_name, deprecated=deprecated, reason=reason
    )


def _unsupported_property(property_name, deprecated=False, reason=""):
    return unsupported_property(
        class_name="pd.DataFrame", property_name=property_name, deprecated=deprecated, reason=reason
    )


class _MissingPandasLikeDataFrame(object):

    # Functions
    align = _unsupported_function("align")
    asfreq = _unsupported_function("asfreq")
    asof = _unsupported_function("asof")
    at_time = _unsupported_function("at_time")
    between_time = _unsupported_function("between_time")
    boxplot = _unsupported_function("boxplot")
    combine = _unsupported_function("combine")
    combine_first = _unsupported_function("combine_first")
    compare = _unsupported_function("compare")
    convert_dtypes = _unsupported_function("convert_dtypes")
    corrwith = _unsupported_function("corrwith")
    cov = _unsupported_function("cov")
    ewm = _unsupported_function("ewm")
    first = _unsupported_function("first")
    infer_objects = _unsupported_function("infer_objects")
    insert = _unsupported_function("insert")
    interpolate = _unsupported_function("interpolate")
    last = _unsupported_function("last")
    lookup = _unsupported_function("lookup")
    mode = _unsupported_function("mode")
    reorder_levels = _unsupported_function("reorder_levels")
    resample = _unsupported_function("resample")
    sem = _unsupported_function("sem")
    set_axis = _unsupported_function("set_axis")
    slice_shift = _unsupported_function("slice_shift")
    to_feather = _unsupported_function("to_feather")
    to_gbq = _unsupported_function("to_gbq")
    to_hdf = _unsupported_function("to_hdf")
    to_period = _unsupported_function("to_period")
    to_sql = _unsupported_function("to_sql")
    to_stata = _unsupported_function("to_stata")
    to_timestamp = _unsupported_function("to_timestamp")
    tshift = _unsupported_function("tshift")
    tz_convert = _unsupported_function("tz_convert")
    tz_localize = _unsupported_function("tz_localize")

    # Deprecated functions
    convert_objects = _unsupported_function("convert_objects", deprecated=True)
    select = _unsupported_function("select", deprecated=True)
    to_panel = _unsupported_function("to_panel", deprecated=True)
    get_values = _unsupported_function("get_values", deprecated=True)
    compound = _unsupported_function("compound", deprecated=True)
    reindex_axis = _unsupported_function("reindex_axis", deprecated=True)

    # Functions we won't support.
    to_pickle = common.to_pickle(_unsupported_function)
    memory_usage = common.memory_usage(_unsupported_function)
    to_xarray = common.to_xarray(_unsupported_function)

    if LooseVersion(pd.__version__) < LooseVersion("1.0"):
        # Deprecated properties
        blocks = _unsupported_property("blocks", deprecated=True)
        ftypes = _unsupported_property("ftypes", deprecated=True)
        is_copy = _unsupported_property("is_copy", deprecated=True)
        ix = _unsupported_property("ix", deprecated=True)

        # Deprecated functions
        as_blocks = _unsupported_function("as_blocks", deprecated=True)
        as_matrix = _unsupported_function("as_matrix", deprecated=True)
        clip_lower = _unsupported_function("clip_lower", deprecated=True)
        clip_upper = _unsupported_function("clip_upper", deprecated=True)
        get_ftype_counts = _unsupported_function("get_ftype_counts", deprecated=True)
        get_value = _unsupported_function("get_value", deprecated=True)
        set_value = _unsupported_function("set_value", deprecated=True)
        to_dense = _unsupported_function("to_dense", deprecated=True)
        to_sparse = _unsupported_function("to_sparse", deprecated=True)
        to_msgpack = _unsupported_function("to_msgpack", deprecated=True)
