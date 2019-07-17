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
    return _unsupported_function(class_name='pd.Series', method_name=method_name,
                                 deprecated=deprecated, reason=reason)


def unsupported_property(property_name, deprecated=False, reason=""):
    return _unsupported_property(class_name='pd.Series', property_name=property_name,
                                 deprecated=deprecated, reason=reason)


class _MissingPandasLikeSeries(object):

    # Properties
    array = unsupported_property('array')
    asobject = unsupported_property('asobject')
    axes = unsupported_property('axes')
    base = unsupported_property('base')
    flags = unsupported_property('flags')
    ftype = unsupported_property('ftype')
    ftypes = unsupported_property('ftypes')
    iat = unsupported_property('iat')
    imag = unsupported_property('imag')
    is_copy = unsupported_property('is_copy')
    ix = unsupported_property('ix')
    nbytes = unsupported_property('nbytes')
    real = unsupported_property('real')
    strides = unsupported_property('strides')

    # Deprecated properties
    blocks = unsupported_property('blocks', deprecated=True)

    # Functions
    agg = unsupported_function('agg')
    aggregate = unsupported_function('aggregate')
    align = unsupported_function('align')
    argmax = unsupported_function('argmax')
    argmin = unsupported_function('argmin')
    argsort = unsupported_function('argsort')
    asfreq = unsupported_function('asfreq')
    asof = unsupported_function('asof')
    at_time = unsupported_function('at_time')
    autocorr = unsupported_function('autocorr')
    between = unsupported_function('between')
    between_time = unsupported_function('between_time')
    bfill = unsupported_function('bfill')
    combine = unsupported_function('combine')
    combine_first = unsupported_function('combine_first')
    compound = unsupported_function('compound')
    copy = unsupported_function('copy')
    cov = unsupported_function('cov')
    divmod = unsupported_function('divmod')
    dot = unsupported_function('dot')
    drop = unsupported_function('drop')
    drop_duplicates = unsupported_function('drop_duplicates')
    droplevel = unsupported_function('droplevel')
    duplicated = unsupported_function('duplicated')
    ewm = unsupported_function('ewm')
    expanding = unsupported_function('expanding')
    factorize = unsupported_function('factorize')
    ffill = unsupported_function('ffill')
    filter = unsupported_function('filter')
    first = unsupported_function('first')
    first_valid_index = unsupported_function('first_valid_index')
    get = unsupported_function('get')
    get_values = unsupported_function('get_values')
    idxmax = unsupported_function('idxmax')
    idxmin = unsupported_function('idxmin')
    infer_objects = unsupported_function('infer_objects')
    interpolate = unsupported_function('interpolate')
    item = unsupported_function('item')
    items = unsupported_function('items')
    iteritems = unsupported_function('iteritems')
    keys = unsupported_function('keys')
    last = unsupported_function('last')
    last_valid_index = unsupported_function('last_valid_index')
    mad = unsupported_function('mad')
    mask = unsupported_function('mask')
    mode = unsupported_function('mode')
    pct_change = unsupported_function('pct_change')
    pop = unsupported_function('pop')
    prod = unsupported_function('prod')
    product = unsupported_function('product')
    ptp = unsupported_function('ptp')
    put = unsupported_function('put')
    ravel = unsupported_function('ravel')
    rdivmod = unsupported_function('rdivmod')
    reindex = unsupported_function('reindex')
    reindex_like = unsupported_function('reindex_like')
    rename_axis = unsupported_function('rename_axis')
    reorder_levels = unsupported_function('reorder_levels')
    repeat = unsupported_function('repeat')
    replace = unsupported_function('replace')
    resample = unsupported_function('resample')
    rolling = unsupported_function('rolling')
    searchsorted = unsupported_function('searchsorted')
    sem = unsupported_function('sem')
    set_axis = unsupported_function('set_axis')
    slice_shift = unsupported_function('slice_shift')
    squeeze = unsupported_function('squeeze')
    swapaxes = unsupported_function('swapaxes')
    swaplevel = unsupported_function('swaplevel')
    tail = unsupported_function('tail')
    take = unsupported_function('take')
    to_dense = unsupported_function('to_dense')
    to_hdf = unsupported_function('to_hdf')
    to_msgpack = unsupported_function('to_msgpack')
    to_period = unsupported_function('to_period')
    to_sparse = unsupported_function('to_sparse')
    to_sql = unsupported_function('to_sql')
    to_timestamp = unsupported_function('to_timestamp')
    to_xarray = unsupported_function('to_xarray')
    truncate = unsupported_function('truncate')
    tshift = unsupported_function('tshift')
    tz_convert = unsupported_function('tz_convert')
    tz_localize = unsupported_function('tz_localize')
    unstack = unsupported_function('unstack')
    update = unsupported_function('update')
    view = unsupported_function('view')
    where = unsupported_function('where')
    xs = unsupported_function('xs')

    # Deprecated functions
    itemsize = unsupported_property('itemsize', deprecated=True)
    data = unsupported_property('data', deprecated=True)
    as_blocks = unsupported_function('as_blocks', deprecated=True)
    as_matrix = unsupported_function('as_matrix', deprecated=True)
    clip_lower = unsupported_function('clip_lower', deprecated=True)
    clip_upper = unsupported_function('clip_upper', deprecated=True)
    compress = unsupported_function('compress', deprecated=True)
    convert_objects = unsupported_function('convert_objects', deprecated=True)
    get_ftype_counts = unsupported_function('get_ftype_counts', deprecated=True)
    get_value = unsupported_function('get_value', deprecated=True)
    nonzero = unsupported_function('nonzero', deprecated=True)
    reindex_axis = unsupported_function('reindex_axis', deprecated=True)
    select = unsupported_function('select', deprecated=True)
    set_value = unsupported_function('set_value', deprecated=True)
    valid = unsupported_function('valid', deprecated=True)

    # Functions and properties we won't support.
    values = common.values(unsupported_property)
    memory_usage = common.memory_usage(unsupported_function)
    # Functions and properties we won't support.
    to_pickle = unsupported_function(
        'to_pickle',
        reason="For storage, we encourage you to use Delta or Parquet, instead of Python pickle "
               "format.")
