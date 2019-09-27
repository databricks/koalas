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

from databricks.koalas.missing import _unsupported_function, _unsupported_property, common


def unsupported_function(method_name, deprecated=False, reason=""):
    return _unsupported_function(class_name='pd.Index', method_name=method_name,
                                 deprecated=deprecated, reason=reason)


def unsupported_property(property_name, deprecated=False, reason=""):
    return _unsupported_property(class_name='pd.Index', property_name=property_name,
                                 deprecated=deprecated, reason=reason)


class _MissingPandasLikeIndex(object):

    # Properties
    T = unsupported_property('T')
    has_duplicates = unsupported_property('has_duplicates')
    nbytes = unsupported_property('nbytes')
    ndim = unsupported_property('ndim')
    nlevels = unsupported_property('nlevels')
    shape = unsupported_property('shape')

    # Deprecated properties
    strides = unsupported_property('strides', deprecated=True)
    data = unsupported_property('data', deprecated=True)
    itemsize = unsupported_property('itemsize', deprecated=True)
    base = unsupported_property('base', deprecated=True)
    flags = unsupported_property('flags', deprecated=True)

    # Functions
    append = unsupported_function('append')
    argmax = unsupported_function('argmax')
    argmin = unsupported_function('argmin')
    argsort = unsupported_function('argsort')
    asof = unsupported_function('asof')
    asof_locs = unsupported_function('asof_locs')
    copy = unsupported_function('copy')
    delete = unsupported_function('delete')
    difference = unsupported_function('difference')
    drop = unsupported_function('drop')
    drop_duplicates = unsupported_function('drop_duplicates')
    droplevel = unsupported_function('droplevel')
    dropna = unsupported_function('dropna')
    duplicated = unsupported_function('duplicated')
    equals = unsupported_function('equals')
    factorize = unsupported_function('factorize')
    fillna = unsupported_function('fillna')
    format = unsupported_function('format')
    get_indexer = unsupported_function('get_indexer')
    get_indexer_for = unsupported_function('get_indexer_for')
    get_indexer_non_unique = unsupported_function('get_indexer_non_unique')
    get_level_values = unsupported_function('get_level_values')
    get_loc = unsupported_function('get_loc')
    get_slice_bound = unsupported_function('get_slice_bound')
    get_value = unsupported_function('get_value')
    groupby = unsupported_function('groupby')
    holds_integer = unsupported_function('holds_integer')
    identical = unsupported_function('identical')
    insert = unsupported_function('insert')
    intersection = unsupported_function('intersection')
    is_ = unsupported_function('is_')
    is_lexsorted_for_tuple = unsupported_function('is_lexsorted_for_tuple')
    is_mixed = unsupported_function('is_mixed')
    is_type_compatible = unsupported_function('is_type_compatible')
    join = unsupported_function('join')
    map = unsupported_function('map')
    max = unsupported_function('max')
    min = unsupported_function('min')
    nunique = unsupported_function('nunique')
    putmask = unsupported_function('putmask')
    ravel = unsupported_function('ravel')
    reindex = unsupported_function('reindex')
    repeat = unsupported_function('repeat')
    searchsorted = unsupported_function('searchsorted')
    set_names = unsupported_function('set_names')
    set_value = unsupported_function('set_value')
    slice_indexer = unsupported_function('slice_indexer')
    slice_locs = unsupported_function('slice_locs')
    sort = unsupported_function('sort')
    sort_values = unsupported_function('sort_values')
    sortlevel = unsupported_function('sortlevel')
    symmetric_difference = unsupported_function('symmetric_difference')
    take = unsupported_function('take')
    to_flat_index = unsupported_function('to_flat_index')
    to_frame = unsupported_function('to_frame')
    to_native_types = unsupported_function('to_native_types')
    to_numpy = unsupported_function('to_numpy')
    transpose = unsupported_function('transpose')
    union = unsupported_function('union')
    unique = unsupported_function('unique')
    value_counts = unsupported_function('value_counts')
    view = unsupported_function('view')
    where = unsupported_function('where')

    # Deprecated functions
    get_duplicates = unsupported_function('get_duplicates', deprecated=True)
    summary = unsupported_function('summary', deprecated=True)
    get_values = unsupported_function('get_values', deprecated=True)
    item = unsupported_function('item', deprecated=True)
    contains = unsupported_function('contains', deprecated=True)

    # Properties we won't support.
    values = common.values(unsupported_property)
    array = common.array(unsupported_property)

    # Functions we won't support.
    memory_usage = common.memory_usage(unsupported_function)
    to_list = common.to_list(unsupported_function)
    tolist = common.tolist(unsupported_function)
    __iter__ = common.__iter__(unsupported_function)


class _MissingPandasLikeMultiIndex(object):

    # Properties
    T = unsupported_property('T')
    codes = unsupported_property('codes')
    has_duplicates = unsupported_property('has_duplicates')
    is_all_dates = unsupported_property('is_all_dates')
    levels = unsupported_property('levels')
    levshape = unsupported_property('levshape')
    ndim = unsupported_property('ndim')
    nlevels = unsupported_property('nlevels')
    shape = unsupported_property('shape')

    # Deprecated properties
    strides = unsupported_property('strides', deprecated=True)
    data = unsupported_property('data', deprecated=True)
    base = unsupported_property('base', deprecated=True)
    itemsize = unsupported_property('itemsize', deprecated=True)
    labels = unsupported_property('labels', deprecated=True)
    flags = unsupported_property('flags', deprecated=True)

    # Functions
    append = unsupported_function('append')
    argmax = unsupported_function('argmax')
    argmin = unsupported_function('argmin')
    argsort = unsupported_function('argsort')
    asof = unsupported_function('asof')
    asof_locs = unsupported_function('asof_locs')
    copy = unsupported_function('copy')
    delete = unsupported_function('delete')
    difference = unsupported_function('difference')
    drop = unsupported_function('drop')
    drop_duplicates = unsupported_function('drop_duplicates')
    droplevel = unsupported_function('droplevel')
    dropna = unsupported_function('dropna')
    duplicated = unsupported_function('duplicated')
    equal_levels = unsupported_function('equal_levels')
    equals = unsupported_function('equals')
    factorize = unsupported_function('factorize')
    fillna = unsupported_function('fillna')
    format = unsupported_function('format')
    get_indexer = unsupported_function('get_indexer')
    get_indexer_for = unsupported_function('get_indexer_for')
    get_indexer_non_unique = unsupported_function('get_indexer_non_unique')
    get_level_values = unsupported_function('get_level_values')
    get_loc = unsupported_function('get_loc')
    get_loc_level = unsupported_function('get_loc_level')
    get_locs = unsupported_function('get_locs')
    get_slice_bound = unsupported_function('get_slice_bound')
    get_value = unsupported_function('get_value')
    groupby = unsupported_function('groupby')
    holds_integer = unsupported_function('holds_integer')
    identical = unsupported_function('identical')
    insert = unsupported_function('insert')
    intersection = unsupported_function('intersection')
    is_ = unsupported_function('is_')
    is_lexsorted = unsupported_function('is_lexsorted')
    is_lexsorted_for_tuple = unsupported_function('is_lexsorted_for_tuple')
    is_mixed = unsupported_function('is_mixed')
    is_type_compatible = unsupported_function('is_type_compatible')
    join = unsupported_function('join')
    map = unsupported_function('map')
    max = unsupported_function('max')
    min = unsupported_function('min')
    nunique = unsupported_function('nunique')
    putmask = unsupported_function('putmask')
    ravel = unsupported_function('ravel')
    reindex = unsupported_function('reindex')
    remove_unused_levels = unsupported_function('remove_unused_levels')
    reorder_levels = unsupported_function('reorder_levels')
    repeat = unsupported_function('repeat')
    searchsorted = unsupported_function('searchsorted')
    set_codes = unsupported_function('set_codes')
    set_labels = unsupported_function('set_labels')
    set_levels = unsupported_function('set_levels')
    set_names = unsupported_function('set_names')
    set_value = unsupported_function('set_value')
    slice_indexer = unsupported_function('slice_indexer')
    slice_locs = unsupported_function('slice_locs')
    sort = unsupported_function('sort')
    sort_values = unsupported_function('sort_values')
    sortlevel = unsupported_function('sortlevel')
    swaplevel = unsupported_function('swaplevel')
    symmetric_difference = unsupported_function('symmetric_difference')
    take = unsupported_function('take')
    to_flat_index = unsupported_function('to_flat_index')
    to_frame = unsupported_function('to_frame')
    to_native_types = unsupported_function('to_native_types')
    to_numpy = unsupported_function('to_numpy')
    transpose = unsupported_function('transpose')
    truncate = unsupported_function('truncate')
    union = unsupported_function('union')
    unique = unsupported_function('unique')
    value_counts = unsupported_function('value_counts')
    view = unsupported_function('view')
    where = unsupported_function('where')

    # Deprecated functions
    get_duplicates = unsupported_function('get_duplicates', deprecated=True)
    summary = unsupported_function('summary', deprecated=True)
    to_hierarchical = unsupported_function('to_hierarchical', deprecated=True)
    get_values = unsupported_function('get_values', deprecated=True)
    contains = unsupported_function('contains', deprecated=True)
    item = unsupported_function('item', deprecated=True)

    # Functions we won't support.
    values = common.values(unsupported_property)
    array = common.array(unsupported_property)
    __iter__ = common.__iter__(unsupported_function)

    # Properties we won't support.
    memory_usage = common.memory_usage(unsupported_function)
    to_list = common.to_list(unsupported_function)
    tolist = common.tolist(unsupported_function)
