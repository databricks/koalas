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
        class_name="pd.Series", method_name=method_name, deprecated=deprecated, reason=reason
    )


def unsupported_property(property_name, deprecated=False, reason=""):
    return _unsupported_property(
        class_name="pd.Series", property_name=property_name, deprecated=deprecated, reason=reason
    )


class _MissingPandasLikeSeries(object):

    # Functions
    align = unsupported_function("align")
    argsort = unsupported_function("argsort")
    asfreq = unsupported_function("asfreq")
    at_time = unsupported_function("at_time")
    autocorr = unsupported_function("autocorr")
    between_time = unsupported_function("between_time")
    bfill = unsupported_function("bfill")
    combine = unsupported_function("combine")
    cov = unsupported_function("cov")
    divmod = unsupported_function("divmod")
    droplevel = unsupported_function("droplevel")
    ewm = unsupported_function("ewm")
    factorize = unsupported_function("factorize")
    ffill = unsupported_function("ffill")
    filter = unsupported_function("filter")
    first = unsupported_function("first")
    infer_objects = unsupported_function("infer_objects")
    interpolate = unsupported_function("interpolate")
    item = unsupported_function("item")
    items = unsupported_function("items")
    iteritems = unsupported_function("iteritems")
    last = unsupported_function("last")
    last_valid_index = unsupported_function("last_valid_index")
    mad = unsupported_function("mad")
    prod = unsupported_function("prod")
    product = unsupported_function("product")
    rdivmod = unsupported_function("rdivmod")
    reindex = unsupported_function("reindex")
    reindex_like = unsupported_function("reindex_like")
    rename_axis = unsupported_function("rename_axis")
    reorder_levels = unsupported_function("reorder_levels")
    resample = unsupported_function("resample")
    searchsorted = unsupported_function("searchsorted")
    sem = unsupported_function("sem")
    set_axis = unsupported_function("set_axis")
    slice_shift = unsupported_function("slice_shift")
    swapaxes = unsupported_function("swapaxes")
    swaplevel = unsupported_function("swaplevel")
    tail = unsupported_function("tail")
    to_hdf = unsupported_function("to_hdf")
    to_period = unsupported_function("to_period")
    to_sql = unsupported_function("to_sql")
    to_timestamp = unsupported_function("to_timestamp")
    tshift = unsupported_function("tshift")
    tz_convert = unsupported_function("tz_convert")
    tz_localize = unsupported_function("tz_localize")
    unstack = unsupported_function("unstack")
    view = unsupported_function("view")

    # Deprecated functions
    convert_objects = unsupported_function("convert_objects", deprecated=True)
    nonzero = unsupported_function("nonzero", deprecated=True)
    reindex_axis = unsupported_function("reindex_axis", deprecated=True)
    select = unsupported_function("select", deprecated=True)
    get_values = unsupported_function("get_values", deprecated=True)

    # Properties we won't support.
    array = common.array(unsupported_property)
    duplicated = common.duplicated(unsupported_property)
    nbytes = unsupported_property(
        "nbytes",
        reason="'nbytes' requires to compute whole dataset. You can calculate manually it, "
        "with its 'itemsize', by explicitly executing its count. Use Spark's web UI "
        "to monitor disk and memory usage of your application in general.",
    )

    # Functions we won't support.
    memory_usage = common.memory_usage(unsupported_function)
    to_pickle = common.to_pickle(unsupported_function)
    to_xarray = common.to_xarray(unsupported_function)
    __iter__ = common.__iter__(unsupported_function)
    ravel = unsupported_function(
        "ravel",
        reason="If you want to collect your flattened underlying data as an NumPy array, "
        "use 'to_numpy().ravel()' instead.",
    )

    if LooseVersion(pd.__version__) < LooseVersion("1.0"):
        # Deprecated properties
        blocks = unsupported_property("blocks", deprecated=True)
        ftypes = unsupported_property("ftypes", deprecated=True)
        ftype = unsupported_property("ftype", deprecated=True)
        is_copy = unsupported_property("is_copy", deprecated=True)
        ix = unsupported_property("ix", deprecated=True)
        asobject = unsupported_property("asobject", deprecated=True)
        strides = unsupported_property("strides", deprecated=True)
        imag = unsupported_property("imag", deprecated=True)
        itemsize = unsupported_property("itemsize", deprecated=True)
        data = unsupported_property("data", deprecated=True)
        base = unsupported_property("base", deprecated=True)
        flags = unsupported_property("flags", deprecated=True)

        # Deprecated functions
        as_blocks = unsupported_function("as_blocks", deprecated=True)
        as_matrix = unsupported_function("as_matrix", deprecated=True)
        clip_lower = unsupported_function("clip_lower", deprecated=True)
        clip_upper = unsupported_function("clip_upper", deprecated=True)
        compress = unsupported_function("compress", deprecated=True)
        get_ftype_counts = unsupported_function("get_ftype_counts", deprecated=True)
        get_value = unsupported_function("get_value", deprecated=True)
        set_value = unsupported_function("set_value", deprecated=True)
        valid = unsupported_function("valid", deprecated=True)
        to_dense = unsupported_function("to_dense", deprecated=True)
        to_sparse = unsupported_function("to_sparse", deprecated=True)
        to_msgpack = unsupported_function("to_msgpack", deprecated=True)
        compound = unsupported_function("compound", deprecated=True)
        put = unsupported_function("put", deprecated=True)
        ptp = unsupported_function("ptp", deprecated=True)
        argmax = unsupported_function("argmax", deprecated=True)
        argmin = unsupported_function("argmin", deprecated=True)

        # Functions we won't support.
        real = unsupported_property(
            "real",
            reason="If you want to collect your data as an NumPy array, use 'to_numpy()' instead.",
        )
