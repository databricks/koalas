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


def unsupported_function(method_name):
    return _unsupported_function(class_name='pd.Index', method_name=method_name)


def unsupported_property(property_name):
    return _unsupported_property(class_name='pd.Index', property_name=property_name)


class _MissingPandasLikeIndex(object):

    # Properties
    T = unsupported_property('T')
    array = unsupported_property('array')
    base = unsupported_property('base')
    data = unsupported_property('data')
    empty = unsupported_property('empty')
    flags = unsupported_property('flags')
    has_duplicates = unsupported_property('has_duplicates')
    is_monotonic = unsupported_property('is_monotonic')
    is_monotonic_decreasing = unsupported_property('is_monotonic_decreasing')
    is_monotonic_increasing = unsupported_property('is_monotonic_increasing')
    itemsize = unsupported_property('itemsize')
    nbytes = unsupported_property('nbytes')
    ndim = unsupported_property('ndim')
    nlevels = unsupported_property('nlevels')
    shape = unsupported_property('shape')
    size = unsupported_property('size')
    strides = unsupported_property('strides')
    values = unsupported_property('values')

    # Functions
    all = unsupported_function('all')
    any = unsupported_function('any')
    append = unsupported_function('append')
    argmax = unsupported_function('argmax')
    argmin = unsupported_function('argmin')
    argsort = unsupported_function('argsort')
    asof = unsupported_function('asof')
    asof_locs = unsupported_function('asof_locs')
    astype = unsupported_function('astype')
    contains = unsupported_function('contains')
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
    get_duplicates = unsupported_function('get_duplicates')
    get_indexer = unsupported_function('get_indexer')
    get_indexer_for = unsupported_function('get_indexer_for')
    get_indexer_non_unique = unsupported_function('get_indexer_non_unique')
    get_level_values = unsupported_function('get_level_values')
    get_loc = unsupported_function('get_loc')
    get_slice_bound = unsupported_function('get_slice_bound')
    get_value = unsupported_function('get_value')
    get_values = unsupported_function('get_values')
    groupby = unsupported_function('groupby')
    holds_integer = unsupported_function('holds_integer')
    identical = unsupported_function('identical')
    insert = unsupported_function('insert')
    intersection = unsupported_function('intersection')
    is_ = unsupported_function('is_')
    is_boolean = unsupported_function('is_boolean')
    is_categorical = unsupported_function('is_categorical')
    is_floating = unsupported_function('is_floating')
    is_integer = unsupported_function('is_integer')
    is_interval = unsupported_function('is_interval')
    is_lexsorted_for_tuple = unsupported_function('is_lexsorted_for_tuple')
    is_mixed = unsupported_function('is_mixed')
    is_numeric = unsupported_function('is_numeric')
    is_object = unsupported_function('is_object')
    is_type_compatible = unsupported_function('is_type_compatible')
    isin = unsupported_function('isin')
    isna = unsupported_function('isna')
    isnull = unsupported_function('isnull')
    item = unsupported_function('item')
    join = unsupported_function('join')
    map = unsupported_function('map')
    max = unsupported_function('max')
    memory_usage = unsupported_function('memory_usage')
    min = unsupported_function('min')
    notna = unsupported_function('notna')
    notnull = unsupported_function('notnull')
    nunique = unsupported_function('nunique')
    putmask = unsupported_function('putmask')
    ravel = unsupported_function('ravel')
    reindex = unsupported_function('reindex')
    rename = unsupported_function('rename')
    repeat = unsupported_function('repeat')
    searchsorted = unsupported_function('searchsorted')
    set_names = unsupported_function('set_names')
    set_value = unsupported_function('set_value')
    shift = unsupported_function('shift')
    slice_indexer = unsupported_function('slice_indexer')
    slice_locs = unsupported_function('slice_locs')
    sort = unsupported_function('sort')
    sort_values = unsupported_function('sort_values')
    sortlevel = unsupported_function('sortlevel')
    summary = unsupported_function('summary')
    symmetric_difference = unsupported_function('symmetric_difference')
    take = unsupported_function('take')
    to_flat_index = unsupported_function('to_flat_index')
    to_frame = unsupported_function('to_frame')
    to_list = unsupported_function('to_list')
    to_native_types = unsupported_function('to_native_types')
    to_numpy = unsupported_function('to_numpy')
    tolist = unsupported_function('tolist')
    transpose = unsupported_function('transpose')
    union = unsupported_function('union')
    unique = unsupported_function('unique')
    value_counts = unsupported_function('value_counts')
    view = unsupported_function('view')
    where = unsupported_function('where')


class _MissingPandasLikeMultiIndex(object):

    # Properties
    T = unsupported_property('T')
    array = unsupported_property('array')
    base = unsupported_property('base')
    codes = unsupported_property('codes')
    data = unsupported_property('data')
    empty = unsupported_property('empty')
    flags = unsupported_property('flags')
    has_duplicates = unsupported_property('has_duplicates')
    is_all_dates = unsupported_property('is_all_dates')
    is_monotonic = unsupported_property('is_monotonic')
    itemsize = unsupported_property('itemsize')
    labels = unsupported_property('labels')
    levels = unsupported_property('levels')
    levshape = unsupported_property('levshape')
    ndim = unsupported_property('ndim')
    nlevels = unsupported_property('nlevels')
    shape = unsupported_property('shape')
    size = unsupported_property('size')
    strides = unsupported_property('strides')
    values = unsupported_property('values')

    # Functions
    all = unsupported_function('all')
    any = unsupported_function('any')
    append = unsupported_function('append')
    argmax = unsupported_function('argmax')
    argmin = unsupported_function('argmin')
    argsort = unsupported_function('argsort')
    asof = unsupported_function('asof')
    asof_locs = unsupported_function('asof_locs')
    astype = unsupported_function('astype')
    contains = unsupported_function('contains')
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
    get_duplicates = unsupported_function('get_duplicates')
    get_indexer = unsupported_function('get_indexer')
    get_indexer_for = unsupported_function('get_indexer_for')
    get_indexer_non_unique = unsupported_function('get_indexer_non_unique')
    get_level_values = unsupported_function('get_level_values')
    get_loc = unsupported_function('get_loc')
    get_loc_level = unsupported_function('get_loc_level')
    get_locs = unsupported_function('get_locs')
    get_slice_bound = unsupported_function('get_slice_bound')
    get_value = unsupported_function('get_value')
    get_values = unsupported_function('get_values')
    groupby = unsupported_function('groupby')
    holds_integer = unsupported_function('holds_integer')
    identical = unsupported_function('identical')
    insert = unsupported_function('insert')
    intersection = unsupported_function('intersection')
    is_ = unsupported_function('is_')
    is_boolean = unsupported_function('is_boolean')
    is_categorical = unsupported_function('is_categorical')
    is_floating = unsupported_function('is_floating')
    is_integer = unsupported_function('is_integer')
    is_interval = unsupported_function('is_interval')
    is_lexsorted = unsupported_function('is_lexsorted')
    is_lexsorted_for_tuple = unsupported_function('is_lexsorted_for_tuple')
    is_mixed = unsupported_function('is_mixed')
    is_numeric = unsupported_function('is_numeric')
    is_object = unsupported_function('is_object')
    is_type_compatible = unsupported_function('is_type_compatible')
    isin = unsupported_function('isin')
    isna = unsupported_function('isna')
    isnull = unsupported_function('isnull')
    item = unsupported_function('item')
    join = unsupported_function('join')
    map = unsupported_function('map')
    max = unsupported_function('max')
    memory_usage = unsupported_function('memory_usage')
    min = unsupported_function('min')
    notna = unsupported_function('notna')
    notnull = unsupported_function('notnull')
    nunique = unsupported_function('nunique')
    putmask = unsupported_function('putmask')
    ravel = unsupported_function('ravel')
    reindex = unsupported_function('reindex')
    remove_unused_levels = unsupported_function('remove_unused_levels')
    rename = unsupported_function('rename')
    reorder_levels = unsupported_function('reorder_levels')
    repeat = unsupported_function('repeat')
    searchsorted = unsupported_function('searchsorted')
    set_codes = unsupported_function('set_codes')
    set_labels = unsupported_function('set_labels')
    set_levels = unsupported_function('set_levels')
    set_names = unsupported_function('set_names')
    set_value = unsupported_function('set_value')
    shift = unsupported_function('shift')
    slice_indexer = unsupported_function('slice_indexer')
    slice_locs = unsupported_function('slice_locs')
    sort = unsupported_function('sort')
    sort_values = unsupported_function('sort_values')
    sortlevel = unsupported_function('sortlevel')
    summary = unsupported_function('summary')
    swaplevel = unsupported_function('swaplevel')
    symmetric_difference = unsupported_function('symmetric_difference')
    take = unsupported_function('take')
    to_flat_index = unsupported_function('to_flat_index')
    to_frame = unsupported_function('to_frame')
    to_hierarchical = unsupported_function('to_hierarchical')
    to_list = unsupported_function('to_list')
    to_native_types = unsupported_function('to_native_types')
    to_numpy = unsupported_function('to_numpy')
    tolist = unsupported_function('tolist')
    transpose = unsupported_function('transpose')
    truncate = unsupported_function('truncate')
    union = unsupported_function('union')
    unique = unsupported_function('unique')
    value_counts = unsupported_function('value_counts')
    view = unsupported_function('view')
    where = unsupported_function('where')
